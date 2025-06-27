#!/usr/bin/env python3
"""
Dataset Download Script for Noise Detection Evaluation

This script downloads and prepares the following datasets:
1. Clean Speech: LibriSpeech test-clean (small, high-quality speech)
2. Noise: ESC-50 (Environmental Sound Classification) and MUSAN noise
3. Optional: VoiceBank-DEMAND for additional clean speech

Usage:
    python download_datasets.py
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
def create_directories():
    """Create the required directories for datasets."""
    dirs = ['data/clean_speech', 'data/noise', 'data/mixed']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def download_librispeech():
    """Download LibriSpeech test-clean dataset."""
    logger.info("Downloading LibriSpeech test-clean...")
    
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    filename = "data/test-clean.tar.gz"
    
    if not os.path.exists(filename):
        download_file(url, filename)
    else:
        logger.info("LibriSpeech already downloaded")
    
    # Extract
    if not os.path.exists("data/LibriSpeech"):
        logger.info("Extracting LibriSpeech...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall("data")
        
        # Move files to clean_speech directory
        libri_path = Path("data/LibriSpeech/test-clean")
        clean_speech_path = Path("data/clean_speech")
        
        if libri_path.exists():
            # Copy all .flac files to clean_speech
            for flac_file in libri_path.rglob("*.flac"):
                # Convert to wav and copy
                import librosa
                import soundfile as sf
                
                audio, sr = librosa.load(flac_file, sr=16000)
                wav_filename = clean_speech_path / f"{flac_file.stem}.wav"
                sf.write(wav_filename, audio, sr)
        
        logger.info("LibriSpeech processing complete")

def download_esc50():
    """Download ESC-50 environmental sounds dataset."""
    logger.info("Downloading ESC-50 dataset...")
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    filename = "data/ESC-50-master.zip"
    
    if not os.path.exists(filename):
        download_file(url, filename)
    else:
        logger.info("ESC-50 already downloaded")
    
    # Extract
    if not os.path.exists("data/ESC-50-master"):
        logger.info("Extracting ESC-50...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall("data")
    
    # Copy audio files to noise directory
    esc50_audio_path = Path("data/ESC-50-master/audio")
    noise_path = Path("data/noise")
    
    if esc50_audio_path.exists():
        logger.info("Copying ESC-50 audio files to noise directory...")
        for wav_file in esc50_audio_path.glob("*.wav"):
            # Copy and resample to 16kHz if needed
            import librosa
            import soundfile as sf
            
            audio, sr = librosa.load(wav_file, sr=16000)
            output_filename = noise_path / f"esc50_{wav_file.stem}.wav"
            sf.write(output_filename, audio, sr)
    
    logger.info("ESC-50 processing complete")

def download_musan():
    """Download MUSAN noise dataset."""
    logger.info("Downloading MUSAN noise dataset...")
    
    # MUSAN is quite large, so we'll download a subset
    # You can download the full dataset from: https://www.openslr.org/17/
    
    # For now, let's create some synthetic noise samples
    logger.info("Creating synthetic noise samples...")
    
    import numpy as np
    import soundfile as sf
    
    noise_path = Path("data/noise")
    
    # Generate different types of synthetic noise
    sample_rate = 16000
    duration = 10  # seconds
    
    # White noise
    for i in range(5):
        white_noise = np.random.normal(0, 0.1, sample_rate * duration)
        sf.write(noise_path / f"synthetic_white_{i:02d}.wav", white_noise, sample_rate)
    
    # Pink noise (simplified)
    for i in range(5):
        pink_noise = np.random.normal(0, 0.1, sample_rate * duration)
        # Apply simple low-pass filter to simulate pink noise
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 0.1, btype='low')
        pink_noise = filtfilt(b, a, pink_noise)
        sf.write(noise_path / f"synthetic_pink_{i:02d}.wav", pink_noise, sample_rate)
    
    # Traffic-like noise (simplified)
    for i in range(5):
        traffic_noise = np.random.normal(0, 0.15, sample_rate * duration)
        # Add some low-frequency components
        b, a = butter(2, 0.05, btype='low')
        low_freq = filtfilt(b, a, np.random.normal(0, 0.1, sample_rate * duration))
        traffic_noise += low_freq
        sf.write(noise_path / f"synthetic_traffic_{i:02d}.wav", traffic_noise, sample_rate)
    
    logger.info("Synthetic noise samples created")

def download_voicebank_demand():
    """Download VoiceBank-DEMAND dataset (optional)."""
    logger.info("VoiceBank-DEMAND is a large dataset. Skipping for now.")
    logger.info("You can download it manually from: https://datashare.ed.ac.uk/handle/10283/2791")
    logger.info("Or use the LibriSpeech data we already downloaded.")

def verify_datasets():
    """Verify that datasets are properly downloaded and accessible."""
    logger.info("Verifying datasets...")
    
    clean_speech_path = Path("data/clean_speech")
    noise_path = Path("data/noise")
    
    clean_files = list(clean_speech_path.glob("*.wav"))
    noise_files = list(noise_path.glob("*.wav"))
    
    logger.info(f"Found {len(clean_files)} clean speech files")
    logger.info(f"Found {len(noise_files)} noise files")
    
    if len(clean_files) == 0:
        logger.warning("No clean speech files found!")
    if len(noise_files) == 0:
        logger.warning("No noise files found!")
    
    # Test loading a few files
    try:
        import librosa
        import soundfile as sf
        
        if clean_files:
            test_clean = clean_files[0]
            audio, sr = sf.read(test_clean)
            logger.info(f"Test clean file: {test_clean.name}, duration: {len(audio)/sr:.2f}s, sr: {sr}")
        
        if noise_files:
            test_noise = noise_files[0]
            audio, sr = sf.read(test_noise)
            logger.info(f"Test noise file: {test_noise.name}, duration: {len(audio)/sr:.2f}s, sr: {sr}")
            
    except Exception as e:
        logger.error(f"Error testing file loading: {e}")

def main():
    """Main function to download all datasets."""
    logger.info("Starting dataset download...")
    
    # Create directories
    create_directories()
    
    # Download datasets
    download_librispeech()
    download_esc50()
    download_musan()
    download_voicebank_demand()
    
    # Verify
    verify_datasets()
    
    logger.info("Dataset download complete!")

if __name__ == "__main__":
    main() 