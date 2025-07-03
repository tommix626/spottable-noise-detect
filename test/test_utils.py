from pathlib import Path
from utils import sample_files, parse_snr_from_filename

def test_sample_files():
    files = [Path(f"file{i}.wav") for i in range(5)]
    sampled = sample_files(files, 3)
    assert len(sampled) == 3
    assert all(f in files for f in sampled)

def test_parse_snr_from_filename():
    assert parse_snr_from_filename("audio_snr10.wav") == 10
    assert parse_snr_from_filename("audio_snr-5.wav") == -5
    assert parse_snr_from_filename("audio.wav") is None