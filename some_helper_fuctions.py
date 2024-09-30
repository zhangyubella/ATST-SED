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

def resample():
    import subprocess
    audio_dir = '/20A021/DESED_dataset/DESED_public_eval/audio/eval/resample/'
    save_dir = '/20A021/DESED_dataset/DESED_public_eval/audio/eval/resample_16k/'
    row_list = []
    for audio_file in tqdm(os.listdir(audio_dir)):
        try:
            # Construct ffmpeg command to resample the audio file
            command = [
                'ffmpeg', '-i', os.path.join(audio_dir, audio_file),  # Input file
                '-ar', str(16000),  # Resample to target sample rate
                '-loglevel', 'quiet',
                os.path.join(save_dir, audio_file)  # Output file path
            ]
            # Run the command
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during resampling: {e}")

if __name__ == "__main__":
    resample()