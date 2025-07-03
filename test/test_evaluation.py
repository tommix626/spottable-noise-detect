import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
from evaluation import evaluate_mixed_directory, plot_roc_curve
import tempfile
import matplotlib
matplotlib.use('Agg')  # For headless testing

def test_evaluate_mixed_directory():
    sr = 16000
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        # Create dummy wav files with SNR in name
        for snr in [0, 10, 20]:
            audio = np.random.randn(sr)
            sf.write(d / f"test_snr{snr}.wav", audio, sr)
        df = evaluate_mixed_directory(d)
        assert isinstance(df, pd.DataFrame)
        assert set(['file','snr','is_noisy','vad_score','spectral_score','zcr_score','rms_score']).issubset(df.columns)

def test_plot_roc_curve_runs():
    # Use a small dummy DataFrame
    df = pd.DataFrame({
        'is_noisy': [True, False, True, False],
        'vad_score': [0.9, 0.1, 0.8, 0.2],
        'spectral_score': [0.7, 0.2, 0.6, 0.3],
        'zcr_score': [0.5, 0.4, 0.6, 0.3],
        'rms_score': [0.8, 0.2, 0.7, 0.1]
    })
    plot_roc_curve(df, title="Test ROC Curve") 