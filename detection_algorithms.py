import numpy as np
import librosa
import webrtcvad

def run_vad(audio: np.ndarray, sr: int, mode: int = 2) -> float:
    """Compute speech ratio using WebRTC VAD."""
    vad = webrtcvad.Vad(mode)
    frame_ms = 30
    samples_per_frame = int(sr * frame_ms / 1000)
    n_frames = len(audio) // samples_per_frame
    speech = 0
    for idx in range(n_frames):
        start = idx * samples_per_frame
        frame = audio[start:start + samples_per_frame]
        pcm = (frame * 32768).astype(np.int16).tobytes()
        if vad.is_speech(pcm, sr):
            speech += 1
    return speech / n_frames if n_frames > 0 else 0.0

def compute_spectral_features(audio: np.ndarray) -> tuple:
    """Compute zero-crossing rate and spectral flatness."""
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    sfm = np.mean(librosa.feature.spectral_flatness(y=audio))
    return zcr, sfm

def compute_rms(audio: np.ndarray) -> float:
    """Compute root-mean-square energy."""
    return float(np.mean(librosa.feature.rms(y=audio))) 