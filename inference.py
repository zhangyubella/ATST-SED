import yaml
import torch
import scipy
import torch.nn as nn
import numpy as np
import math
from torchaudio.transforms import AmplitudeToDB
from desed_task.nnet.CRNN_e2e import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.dataio.datasets_atst_sed import SEDTransform, ATSTTransform, read_audio
from desed_task.utils.scaler import TorchScaler
from train.local.classes_dict import classes_labels

class ATSTSEDFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sed_feat_extractor = SEDTransform(config["feats"])
        self.scaler = TorchScaler(
                "instance",
                config["scaler"]["normtype"],
                config["scaler"]["dims"],
            )
        self.atst_feat_extractor = ATSTTransform()
    
    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code
    
    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)
        sed_feats = self.scaler(self.take_log(sed_feats))
        atst_feats = self.atst_feat_extractor(mixture)

        return sed_feats, atst_feats

class ATSTSEDInferencer(nn.Module):
    """Inference module for ATST-SED
    """
    def __init__(
        self, 
        pretrained_path, 
        model_config_path="./confs/stage2.yaml", 
        overlap_dur=3, 
        hard_threshold=0.5):
        super().__init__()

        # Load model configurations
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        # Initialize model
        self.model = self.load_from_pretrained(pretrained_path, config)
        # Initialize label encoder
        self.label_encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
        )
        # Initialize feature extractor
        self.feature_extractor = ATSTSEDFeatureExtractor(config)
    
        # Initial parameters
        self.audio_dur = 10 # this value is fixed because ATST-SED is trained on 10-second audio, if you want to change it, you need to retrain the model
        self.overlap_dur = overlap_dur
        self.fs = config["data"]["fs"]
        
        # Unfolder for splitting audio into chunks
        self.unfolder = nn.Unfold(kernel_size=(self.fs * self.audio_dur, 1), stride=(self.fs * self.overlap_dur, 1))
    
        self.hard_threshold = [hard_threshold] * len(self.label_encoder.labels) if not isinstance(hard_threshold, list) else hard_threshold
    
    def load_from_pretrained(self, pretrained_path: str, config: dict):
        # Initializign model
        model = CRNN(
            unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"], 
            **config["net"], 
            model_init=config["ultra"]["model_init"],
            atst_dropout=config["ultra"]["atst_dropout"],
            atst_init=config["ultra"]["atst_init"],
            mode="teacher")
        # Load pretrained ckpt
        state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
        ### get teacher model
        state_dict = {k.replace("sed_teacher.", ""): v for k, v in state_dict.items() if "teacher" in k}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    
    def get_logmel(self, wav_file):
        mixture, onset_s, offset_s, padded_indx = read_audio(
            wav_file, False, False, None
        )
        sed_feats, _ = self.feature_extractor(mixture)
        return sed_feats[0].detach().cpu().numpy()
        
    def forward(self, wav_file):
        mixture, onset_s, offset_s, padded_indx = read_audio(
                wav_file, False, False, None
            )
        # split wav into chunks with overlap
        if (mixture.numel() // self.fs) <= self.audio_dur:
            inference_chunks = [mixture]
            padding_frames = 0
            mixture_pad = mixture.clone()
        else:
            # pad the mixtures
            mixture = mixture.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            total_chunks = (mixture.numel() - ((self.audio_dur - self.overlap_dur) * self.fs)) // (self.overlap_dur * self.fs) + 1
            total_length = total_chunks * self.overlap_dur * self.fs + (self.audio_dur - self.overlap_dur) * self.fs
            mixture_pad = torch.nn.functional.pad(mixture, (0, 0, 0, total_length - mixture.numel()))
            padding_frames = self.time2frame(total_length - mixture.numel())
            inference_chunks = self.unfolder(mixture_pad)
            inference_chunks = inference_chunks.squeeze(0).T
            
        
        # inference result for each chunk
        sed_results = []
        for chunk in inference_chunks:
            sed_feats, atst_feats = self.feature_extractor(chunk)
            # sed_feats = sed_feats.transpose(1, 2)
            chunk_result, _ = self.model(sed_feats, atst_feats)
            sed_results.append(chunk_result)

        if self.hard_threshold is None:
            return sed_results  # If no threshold given, return soft results and you can customize the threshold later
        else:
            chunk_decisions = []
            for i, chunk_result in enumerate(sed_results):
                hard_chunk_result = self.post_process(chunk_result.detach().cpu())
                chunk_decisions.append(hard_chunk_result)
            return self.decision_unify(chunk_decisions, self.time2frame(mixture_pad.numel()), padding_frames)
        
    def post_process(self, strong_preds):
        strong_preds = strong_preds[0]  # only support single input (bsz=1)
        smoothed_preds = []
        for score, fil_val in zip(strong_preds, self.config["training"]["median_window"]):
            score = scipy.ndimage.filters.median_filter(score[:, np.newaxis], (fil_val, 1))
            smoothed_preds.append(score)
        smoothed_preds = np.concatenate(smoothed_preds, axis=1)
        decisions = []
        for score, c_th in zip(smoothed_preds.T, self.hard_threshold):
            pred = score > c_th
            decisions.append(pred[np.newaxis, :] )
        decisions = np.concatenate(decisions, axis=0)
        return decisions
    
    def time2frame(self, time):
        return int(int((time / self.label_encoder.frame_hop)) / self.label_encoder.net_pooling)
    
    def decision_unify(self, chunk_decisions, total_frames, padding_frames):
        # decision: [C, T]
        C, T = chunk_decisions[0].shape
        if len(chunk_decisions) == 1:
            return chunk_decisions[0]
        else:
            decisions = np.zeros((C, total_frames))
            hop_frame = self.time2frame(self.overlap_dur * self.fs)
            for i in range(len(chunk_decisions)):
                decisions[:, i * hop_frame: i * hop_frame + T] += chunk_decisions[i]
            return (decisions > 0).astype(float)[:, :-padding_frames]
        
if __name__ == "__main__":
    import soundfile as sf
    import matplotlib.pyplot as plt
    test_file = "YOUR_TEST_FILE_HERE"
    test_name = ".".join(test_file.split("/")[-1].split(".")[:-1])
    sed_classes = [x.split("_")[0] for x in classes_labels.keys()]
    inference_model = ATSTSEDInferencer(
        "YOUR_CKPT_HERE",
        "YOUR_CONFIG_HERE",
        overlap_dur=3)
    mel_spec = inference_model.get_logmel(test_file)
    sed_results = inference_model(test_file)

    
    if sed_results.sum():
        # Give sed results colors
        sed_results = sed_results * np.arange(1, len(sed_results) + 1).reshape(-1, 1) * 10
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6))
        ax1.imshow(mel_spec, aspect="auto", origin="lower", interpolation="none", cmap="jet")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_ylabel("Mel-bands")
        # pad zeros between columns and turn zero to nan for better visualization
        sed_results = np.concatenate([np.zeros_like(sed_results), sed_results], axis=1)
        sed_results = sed_results.reshape(-1, sed_results.shape[1] // 2).astype(float)
        sed_results[sed_results == 0] = np.nan
        ax2.imshow(sed_results, aspect="auto", origin="lower", interpolation="none", vmax=math.log(1e2), vmin=math.log(1e-7), cmap='jet')
        xticks = np.linspace(0, sed_results.shape[1], 11)
        xticks_performed = ["{:.1f}".format(x) for x in np.linspace(0, sf.info(test_file).duration, 11)]
        ax2.set_xticks(xticks, xticks_performed)
        ax2.set_yticks(np.linspace(1, 2 * len(sed_classes) - 1, 10), sed_classes)
        ax2.set_xlabel("Time (s)")
        print(f"SED results plotted in ./inference_{test_name}.png")
        plt.suptitle(f"SED inference results for {test_file}")
        plt.savefig(f"inference_{test_name}.png")
    else:
        print("no valid event detected")
    