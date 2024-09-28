import os

import pandas as pd
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from desed_task.dataio.datasets_atst_sed import ATSTTransform


def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture

if __name__ == "__main__":
    audio_dir = '/20A021/DESED_dataset/strong_label_real(3373)'
    audio_dir = '/20A021/DESED_dataset/SynthDataset/Train/soundscapes_16k'
    for audio_file in tqdm(os.listdir(audio_dir)):
        mixture, fs = torchaudio.load(os.path.join(audio_dir, audio_file))
        # mixture = to_mono(mixture)
        # transform = ATSTTransform()
        # output = transform(mixture)
