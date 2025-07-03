import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel

# Load Whisper model once (global)
device = 'cuda' if 'CUDA_VISIBLE_DEVICES' in __import__('os').environ else 'cpu'
whisper_model = WhisperModel('base', device=device)

def whisper_confidence(audio_path: Path) -> float:
    """Compute average log-probability confidence from Whisper transcription."""
    segments, _ = whisper_model.transcribe(str(audio_path), beam_size=1)
    confidences = [seg.avg_logprob for seg in segments if seg.avg_logprob is not None]
    return float(np.mean(confidences)) if confidences else -np.inf 