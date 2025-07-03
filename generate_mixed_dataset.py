from pathlib import Path
from mixing import mix_with_noise
import numpy as np

# Paths
CLEAN_DIR = Path("data/clean_speech")
NOISE_DIR = Path("data/noise")
MIXED_DIR = Path("data/mixed/test")
MIXED_DIR.mkdir(parents=True, exist_ok=True)

# SNR levels (in dB)
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

# List files
clean_files = list(CLEAN_DIR.glob("*.wav"))[:10]
noise_files = list(NOISE_DIR.glob("*.wav"))[:10]

for clean_file in clean_files:
    for noise_file in noise_files:
        for snr in SNR_LEVELS:
            out_name = f"{clean_file.stem}_noise_{noise_file.stem}_snr{snr}.wav"
            out_path = MIXED_DIR / out_name
            if not out_path.exists():
                mix_with_noise(clean_file, noise_file, snr, out_path)
                print(f"Created: {out_path}")