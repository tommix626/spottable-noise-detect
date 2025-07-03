import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Literal
import random

def scale_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to achieve desired SNR relative to clean signal."""
    clean_rms = np.sqrt(np.mean(clean**2))
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-8
    target_noise_rms = clean_rms / (10**(snr_db / 20))
    return noise * (target_noise_rms / noise_rms)

def mix_with_noise(clean_path: Path, noise_path: Path, snr_db: float, out_path: Path):
    """Mix a clean file with noise at given SNR and save to out_path."""
    clean, sr = sf.read(clean_path)
    noise, _ = sf.read(noise_path)
    # Ensure noise is at least as long as clean
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]
    scaled_noise = scale_noise(clean, noise, snr_db)
    mixture = clean + scaled_noise
    sf.write(out_path, mixture, sr)

def mix_with_multiple_noises(
    clean_path: Path,
    noise_paths: List[Path],
    snr_db: float,
    out_path: Path,
    normalize: bool = True
):
    """
    Mix a clean file with multiple noise files at a given SNR and save to out_path.
    The noises are merged (summed) and optionally normalized before mixing.
    """
    clean, sr = sf.read(clean_path)
    noises = []
    max_len = 0
    for npth in noise_paths:
        noise, _ = sf.read(npth)
        noises.append(noise)
        max_len = max(max_len, len(noise))
    # Pad noises to same length
    padded_noises = []
    for noise in noises:
        if len(noise) < max_len:
            noise = np.pad(noise, (0, max_len - len(noise)), mode='constant')
        padded_noises.append(noise)
    # Merge noises
    merged_noise = np.sum(padded_noises, axis=0)
    if normalize:
        merged_noise = merged_noise / (np.max(np.abs(merged_noise)) + 1e-8)
    # Repeat noise if too short
    if len(merged_noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(merged_noise)))
        merged_noise = np.tile(merged_noise, repeats)
    merged_noise = merged_noise[:len(clean)]
    scaled_noise = scale_noise(clean, merged_noise, snr_db)
    mixture = clean + scaled_noise
    sf.write(out_path, mixture, sr)

def mix_clean_with_random_noises(
    clean_path: Path,
    noise_dir: Path,
    snr_db: float,
    out_path: Path,
    n_noises: int = 1,
    noise_type: Literal['white', 'nonwhite', 'both'] = 'both',
    normalize: bool = True,
    white_keywords: List[str] = ['white']
):
    """
    Mix a clean file with N random noise files (white, nonwhite, or both) from noise_dir.
    - noise_type: 'white', 'nonwhite', or 'both'
    - white_keywords: list of substrings to identify white noise files
    """
    all_noises = list(noise_dir.glob('*.wav'))
    if noise_type == 'white':
        noises = [f for f in all_noises if any(kw in f.stem.lower() for kw in white_keywords)]
    elif noise_type == 'nonwhite':
        noises = [f for f in all_noises if not any(kw in f.stem.lower() for kw in white_keywords)]
    else:
        noises = all_noises
    if len(noises) < n_noises:
        raise ValueError(f"Not enough noise files of type '{noise_type}' in {noise_dir}")
    selected_noises = random.sample(noises, n_noises)
    mix_with_multiple_noises(clean_path, selected_noises, snr_db, out_path, normalize=normalize) 