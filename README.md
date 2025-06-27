# Noise Detection System

A comprehensive system for detecting when the audio environment is too noisy for effective speech recognition. This system implements multiple heuristics to determine when users should be prompted to move to a quieter environment.

## Features

### Detection Methods

1. **Low Speech Ratio (VAD)**
   - Uses WebRTC VAD to detect speech activity
   - If speech is detected in less than 20-30% of the duration but audio energy is high, background noise is likely dominating

2. **Spectral Features**
   - Computes spectral flatness and zero-crossing rate
   - High values indicate noisy environments (fans, traffic, etc.)
   - Uses spectral centroid as additional feature

3. **RMS Threshold**
   - Simple energy-based detection using root-mean-square
   - Detects when audio is loud but contains no speech
   - Also considers energy variance over time

4. **STT Confidence (Whisper)**
   - Uses Whisper's confidence scores
   - If the model struggles despite loud input, suggests noisy environment
   - Provides transcription quality assessment

5. **Combined Approach**
   - Ensemble of all methods with weighted voting
   - Configurable weights for each method
   - Provides most robust detection

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets:**
   ```bash
   python download_datasets.py
   ```

## Usage

### Quick Test

Test the system on sample files:
```bash
python test_noise_detection.py
```

### Full Evaluation

Run comprehensive evaluation on all datasets:
```bash
python noise_detection_system.py
```

### Programmatic Usage

```python
from noise_detection_system import NoiseDetector

# Initialize detector
detector = NoiseDetector(
    vad_aggressiveness=2,
    speech_ratio_threshold=0.3,
    rms_threshold=0.1,
    spectral_flatness_threshold=0.7,
    zcr_threshold=0.15,
    whisper_confidence_threshold=-1.0
)

# Analyze audio file
result = detector.analyze_audio("path/to/audio.wav", method="combined")

# Check results
if result.is_noisy:
    print(f"Environment is too noisy! Confidence: {result.confidence:.3f}")
    print("Please move to a quieter location.")
else:
    print("Environment is suitable for speech recognition.")
```

## Configuration

### Detector Parameters

- `vad_aggressiveness`: VAD sensitivity (0-3, higher = more aggressive)
- `speech_ratio_threshold`: Minimum speech ratio to consider audio "clean"
- `rms_threshold`: RMS energy threshold for noise detection
- `spectral_flatness_threshold`: Threshold for spectral flatness
- `zcr_threshold`: Threshold for zero-crossing rate
- `whisper_confidence_threshold`: Minimum Whisper confidence

### Method Weights (Combined Approach)

Default weights for the combined method:
- VAD: 0.3
- Spectral: 0.25
- RMS: 0.2
- Whisper: 0.25

## Datasets

The system uses the following datasets:

1. **Clean Speech**: LibriSpeech test-clean
2. **Noise**: ESC-50 environmental sounds + synthetic noise
3. **Mixed**: Generated mixtures at various SNR levels

## Output

### Evaluation Results

The system generates:
- `noise_detection_results.csv`: Detailed results for each file and method
- `noise_detection_results.png`: Visualization of results including ROC curves

### Metrics

For each method, the system computes:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: (True positives + True negatives) / Total samples
- **AUC**: Area under ROC curve

## Real-time Usage

For real-time applications, you can use the detector in a streaming fashion:

```python
import sounddevice as sd
import numpy as np
from noise_detection_system import NoiseDetector

detector = NoiseDetector()

def audio_callback(indata, frames, time, status):
    """Callback for real-time audio processing."""
    audio = indata[:, 0]  # Take first channel
    
    # Analyze in chunks
    result = detector.analyze_audio_chunk(audio, sample_rate=16000)
    
    if result.is_noisy:
        print("⚠️  Environment is noisy - consider moving to a quieter location")

# Start recording
with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
    print("Recording... Press Ctrl+C to stop")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Recording stopped")
```

## Performance

### Speed
- VAD: ~1ms per 30ms frame
- Spectral: ~5ms per second of audio
- RMS: ~1ms per second of audio
- Whisper: ~500ms per second of audio (depends on model size)
- Combined: Sum of all methods

### Accuracy
Based on evaluation results:
- Combined method typically achieves >80% F1-score
- VAD method is fastest but less accurate
- Whisper method is most accurate but slowest

## Troubleshooting

### Common Issues

1. **CUDA not available**: The system will automatically fall back to CPU
2. **Memory issues**: Reduce batch size or use smaller Whisper model
3. **Audio format issues**: Ensure audio is 16kHz, mono, WAV format

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To add new detection methods:

1. Implement the method in `NoiseDetector` class
2. Add it to the `analyze_audio` method
3. Update the evaluation pipeline
4. Test on benchmark datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{noise_detection_system,
  title={Noise Detection System for Speech Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/noise-detect}
}
``` 