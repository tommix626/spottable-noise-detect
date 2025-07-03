import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from detection_algorithms import run_vad, compute_spectral_features, compute_rms
from typing import List

def evaluate_mixed_directory(mixed_dir: Path, methods: List[str] = None) -> pd.DataFrame:
    """
    Evaluate all audio files in a mixed directory using all heuristics.
    Returns DataFrame with columns: file, snr, is_noisy, <method scores>
    """
    records = []
    files = list(mixed_dir.glob("*.wav"))
    if not files:
        print(f"No .wav files found in {mixed_dir}")
        return pd.DataFrame()
    for wav_file in tqdm(files, desc=f"Evaluating {mixed_dir.name}"):
        try:
            snr_value = int(wav_file.stem.split("snr")[-1])
        except Exception:
            print(f"Warning: Could not parse SNR from {wav_file.name}")
            continue
        is_noisy = snr_value < 20
        audio, sr = sf.read(wav_file)
        speech_ratio = run_vad(audio, sr)
        zcr, sfm = compute_spectral_features(audio)
        rms_val = compute_rms(audio)
        # Placeholder for whisper_confidence, can be added if needed
        records.append({
            'file': wav_file.name,
            'snr': snr_value,
            'is_noisy': is_noisy,
            'vad_score': 1 - speech_ratio,
            'spectral_score': sfm,
            'zcr_score': zcr,
            'rms_score': rms_val
        })
    df = pd.DataFrame(records)
    return df

def plot_roc_curve(df: pd.DataFrame, title: str = "ROC Curve", methods: List[str] = None, output_dir: Path = None):
    if methods is None:
        methods = [col for col in df.columns if col.endswith('_score')]
    plt.figure(figsize=(10, 8))
    for method in methods:
        try:
            fpr, tpr, _ = roc_curve(df['is_noisy'], df[method])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Skipping {method}: {e}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "roc.png", bbox_inches='tight')
    plt.show() 