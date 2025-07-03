from pathlib import Path
from mixing import mix_with_multiple_noises
import numpy as np
import random

CLEAN_DIR = Path("data/clean_speech")
NOISE_DIR = Path("data/noise")
MIXED_DIR = Path("data/mixed/balanced")
MIXED_DIR.mkdir(parents=True, exist_ok=True)

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
N_CLEAN = 100  # number of clean samples
NUM_NON_WHITE = 4

clean_files = list(CLEAN_DIR.glob("*.wav"))
np.random.shuffle(clean_files)

# Identify white and non-white noise files
all_noises = list(NOISE_DIR.glob("*.wav"))
white_noises = [f for f in all_noises if "white" in f.stem.lower()]
nonwhite_noises = [f for f in all_noises if "white" not in f.stem.lower()]

for i, clean_file in enumerate(clean_files[:N_CLEAN]):
    # Select 1 white and 2 non-white noises for this clean file (fixed for all SNRs)
    if len(white_noises) < 1 or len(nonwhite_noises) < NUM_NON_WHITE:
        raise ValueError("Not enough white or non-white noise files in noise directory.")
    selected_white = random.sample(white_noises, 1)
    selected_nonwhite = random.sample(nonwhite_noises, NUM_NON_WHITE)
    selected_noises = selected_white + selected_nonwhite
    for snr in SNR_LEVELS:
        out_path = MIXED_DIR / f"{clean_file.stem}_mixed_white1_nonwhite{NUM_NON_WHITE}_snr{snr}.wav"
        mix_with_multiple_noises(clean_file, selected_noises, snr, out_path, normalize=True) 