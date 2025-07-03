from pathlib import Path
import pandas as pd
import soundfile as sf
from detection_algorithms import run_vad, compute_spectral_features, compute_rms
from whisper_confidence import whisper_confidence
from tqdm import tqdm

MIXED_DIR = Path("data/mixed/balanced")
rows = []

wav_files = list(MIXED_DIR.glob("*.wav"))
for wav_file in tqdm(wav_files, desc="Extracting features"):
    audio, sr = sf.read(wav_file)
    # Features
    rms = compute_rms(audio)
    zcr, sfm = compute_spectral_features(audio)
    vad_ratio = run_vad(audio, sr)
    whisper_conf = whisper_confidence(wav_file)
    # Parse SNR from filename
    if "snr" in wav_file.stem:
        try:
            snr = int(wav_file.stem.split("snr")[-1].split('_')[0])
        except Exception:
            snr = 30
    else:
        snr = 30
    is_noisy = snr < 15
    rows.append({
        "file": wav_file.name,
        "snr": snr,
        "is_noisy": is_noisy,
        "rms_score": rms,
        "zcr_score": zcr,
        "spectral_score": sfm,
        "vad_score": 1 - vad_ratio,
        "whisper_score": -whisper_conf
    })

# Save as CSV
pd.DataFrame(rows).to_csv("feature_dataset_backup.csv", index=False) 