import numpy as np
import soundfile as sf
from pathlib import Path
from mixing import scale_noise, mix_with_noise
import tempfile

def test_scale_noise_identity():
    clean = np.ones(1000)
    noise = np.ones(1000)
    snr_db = 0
    scaled = scale_noise(clean, noise, snr_db)
    assert np.allclose(np.std(clean), np.std(scaled), atol=1e-2)

def test_mix_with_noise_creates_file():
    clean = np.ones(1000)
    noise = np.ones(1000)
    sr = 16000
    with tempfile.TemporaryDirectory() as tmpdir:
        clean_path = Path(tmpdir) / "clean.wav"
        noise_path = Path(tmpdir) / "noise.wav"
        out_path = Path(tmpdir) / "mixed.wav"
        sf.write(clean_path, clean, sr)
        sf.write(noise_path, noise, sr)
        mix_with_noise(clean_path, noise_path, 0, out_path)
        assert out_path.exists()
        mixed, _ = sf.read(out_path)
        assert len(mixed) == 1000 