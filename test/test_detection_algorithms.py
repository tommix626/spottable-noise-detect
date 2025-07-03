import numpy as np
from detection_algorithms import run_vad, compute_spectral_features, compute_rms

def test_run_vad_silence():
    sr = 16000
    audio = np.zeros(sr)
    ratio = run_vad(audio, sr)
    assert ratio == 0.0

def test_compute_spectral_features_white_noise():
    sr = 16000
    audio = np.random.randn(sr)
    zcr, sfm = compute_spectral_features(audio)
    assert 0 <= zcr <= 1
    assert 0 <= sfm <= 1

def test_compute_rms_constant():
    sr = 16000
    audio = np.ones(sr)
    rms = compute_rms(audio)
    assert rms > 0 