#!/usr/bin/env python3
"""
Noise Detection System

This module implements various noise detection heuristics to determine when
the environment is too noisy for effective speech recognition.

Methods implemented:
1. Low Speech Ratio (VAD) - Detect when speech is minimal but audio energy is high
2. Spectral Features - Use spectral flatness and zero-crossing rate
3. STT Confidence - Use Whisper's confidence scores
4. RMS Threshold - Simple energy-based detection
5. Combined Approach - Ensemble of all methods

Usage:
    python noise_detection_system.py
"""

import os
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from faster_whisper import WhisperModel
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NoiseDetectionResult:
    """Container for noise detection results."""
    is_noisy: bool
    confidence: float
    method: str
    details: Dict
    audio_path: str

class NoiseDetector:
    """Main noise detection class implementing multiple heuristics."""
    
    def __init__(self, 
                 vad_aggressiveness: int = 2,
                 speech_ratio_threshold: float = 0.3,
                 rms_threshold: float = 0.1,
                 spectral_flatness_threshold: float = 0.7,
                 zcr_threshold: float = 0.15,
                 whisper_confidence_threshold: float = -1.0,
                 sample_rate: int = 16000):
        """
        Initialize the noise detector.
        
        Args:
            vad_aggressiveness: VAD sensitivity (0-3, higher = more aggressive)
            speech_ratio_threshold: Minimum speech ratio to consider audio "clean"
            rms_threshold: RMS energy threshold for noise detection
            spectral_flatness_threshold: Threshold for spectral flatness
            zcr_threshold: Threshold for zero-crossing rate
            whisper_confidence_threshold: Minimum Whisper confidence
            sample_rate: Audio sample rate
        """
        self.vad_aggressiveness = vad_aggressiveness
        self.speech_ratio_threshold = speech_ratio_threshold
        self.rms_threshold = rms_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        self.zcr_threshold = zcr_threshold
        self.whisper_confidence_threshold = whisper_confidence_threshold
        self.sample_rate = sample_rate
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Initialize Whisper model (lazy loading)
        self.whisper_model = None
        
        logger.info("NoiseDetector initialized")
    
    def _load_whisper_model(self):
        """Lazy load Whisper model."""
        if self.whisper_model is None:
            logger.info("Loading Whisper model...")
            device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
            self.whisper_model = WhisperModel('base', device=device)
            logger.info(f"Whisper model loaded on {device}")
    
    def detect_vad_speech_ratio(self, audio: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Method 1: Low Speech Ratio Detection
        
        Compute VAD for the audio segment. If speech is detected in less than
        the threshold percentage of the duration, and audio energy is non-negligible,
        background noise is likely dominating.
        
        Args:
            audio: Audio array
            
        Returns:
            (is_noisy, confidence, details)
        """
        frame_ms = 30
        samples_per_frame = int(self.sample_rate * frame_ms / 1000)
        n_frames = len(audio) // samples_per_frame
        
        if n_frames == 0:
            return False, 0.0, {"speech_ratio": 0.0, "n_frames": 0}
        
        speech_frames = 0
        for idx in range(n_frames):
            start = idx * samples_per_frame
            frame = audio[start:start + samples_per_frame]
            
            # Convert to 16-bit PCM
            pcm = (frame * 32768).astype(np.int16).tobytes()
            
            if self.vad.is_speech(pcm, self.sample_rate):
                speech_frames += 1
        
        speech_ratio = speech_frames / n_frames
        is_noisy = speech_ratio < self.speech_ratio_threshold
        
        # Confidence based on how far below threshold
        confidence = max(0, (self.speech_ratio_threshold - speech_ratio) / self.speech_ratio_threshold)
        
        details = {
            "speech_ratio": speech_ratio,
            "speech_frames": speech_frames,
            "total_frames": n_frames,
            "threshold": self.speech_ratio_threshold
        }
        
        return is_noisy, confidence, details
    
    def detect_spectral_features(self, audio: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Method 2: Spectral Features Detection
        
        High spectral flatness or high zero-crossing rate indicates noisy environment
        (like fans, traffic, etc.).
        
        Args:
            audio: Audio array
            
        Returns:
            (is_noisy, confidence, details)
        """
        # Compute spectral flatness
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        # Compute zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Compute spectral centroid (additional feature)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        
        # Determine if noisy based on thresholds
        is_flat_noisy = spectral_flatness > self.spectral_flatness_threshold
        is_zcr_noisy = zcr > self.zcr_threshold
        
        is_noisy = is_flat_noisy or is_zcr_noisy
        
        # Confidence based on how much features exceed thresholds
        flat_confidence = max(0, (spectral_flatness - self.spectral_flatness_threshold) / self.spectral_flatness_threshold)
        zcr_confidence = max(0, (zcr - self.zcr_threshold) / self.zcr_threshold)
        confidence = max(flat_confidence, zcr_confidence)
        
        details = {
            "spectral_flatness": spectral_flatness,
            "zero_crossing_rate": zcr,
            "spectral_centroid": spectral_centroid,
            "flat_threshold": self.spectral_flatness_threshold,
            "zcr_threshold": self.zcr_threshold,
            "is_flat_noisy": is_flat_noisy,
            "is_zcr_noisy": is_zcr_noisy
        }
        
        return is_noisy, confidence, details
    
    def detect_rms_threshold(self, audio: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Method 3: RMS Threshold Detection
        
        Use root-mean-square energy. If loud but no VAD speech, flag as environmental noise.
        
        Args:
            audio: Audio array
            
        Returns:
            (is_noisy, confidence, details)
        """
        # Compute RMS energy
        rms = np.sqrt(np.mean(audio**2))
        
        # Compute RMS over time (frame-wise)
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        rms_frames = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
        rms_mean = np.mean(rms_frames)
        rms_std = np.std(rms_frames)
        
        # Check if RMS is high
        is_high_rms = rms > self.rms_threshold
        
        # Also check for high variance (indicating inconsistent energy)
        high_variance = rms_std > (rms_mean * 0.5)
        
        is_noisy = is_high_rms or high_variance
        
        # Confidence based on how much RMS exceeds threshold
        confidence = max(0, (rms - self.rms_threshold) / self.rms_threshold) if is_high_rms else 0
        
        details = {
            "rms": rms,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "threshold": self.rms_threshold,
            "is_high_rms": is_high_rms,
            "high_variance": high_variance
        }
        
        return is_noisy, confidence, details
    
    def detect_whisper_confidence(self, audio_path: str) -> Tuple[bool, float, Dict]:
        """
        Method 4: STT Confidence Detection
        
        Use Whisper's confidence scores. If the model struggles despite loud input,
        that suggests a noisy environment.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            (is_noisy, confidence, details)
        """
        self._load_whisper_model()
        
        try:
            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=1)
            
            # Extract confidence scores
            confidences = []
            transcriptions = []
            
            for segment in segments:
                if segment.avg_logprob is not None:
                    confidences.append(segment.avg_logprob)
                    transcriptions.append(segment.text.strip())
            
            if not confidences:
                # No speech detected or very low confidence
                return True, 1.0, {"avg_confidence": -np.inf, "transcription": "", "segments": 0}
            
            avg_confidence = np.mean(confidences)
            transcription = " ".join(transcriptions)
            
            # Determine if noisy based on confidence threshold
            is_noisy = avg_confidence < self.whisper_confidence_threshold
            
            # Confidence based on how much below threshold
            confidence = max(0, (self.whisper_confidence_threshold - avg_confidence) / abs(self.whisper_confidence_threshold))
            
            details = {
                "avg_confidence": avg_confidence,
                "transcription": transcription,
                "segments": len(confidences),
                "threshold": self.whisper_confidence_threshold
            }
            
            return is_noisy, confidence, details
            
        except Exception as e:
            logger.error(f"Error in Whisper detection: {e}")
            return True, 1.0, {"error": str(e)}
    
    def detect_combined(self, audio: np.ndarray, audio_path: str) -> Tuple[bool, float, Dict]:
        """
        Method 5: Combined Approach
        
        Ensemble of all methods with weighted voting.
        
        Args:
            audio: Audio array
            audio_path: Path to audio file
            
        Returns:
            (is_noisy, confidence, details)
        """
        # Run all detection methods
        vad_result = self.detect_vad_speech_ratio(audio)
        spectral_result = self.detect_spectral_features(audio)
        rms_result = self.detect_rms_threshold(audio)
        whisper_result = self.detect_whisper_confidence(audio_path)
        
        # Weighted voting (can be tuned)
        weights = {
            'vad': 0.3,
            'spectral': 0.25,
            'rms': 0.2,
            'whisper': 0.25
        }
        
        # Calculate weighted score
        weighted_score = 0
        total_weight = 0
        
        if vad_result[0]:  # is_noisy
            weighted_score += weights['vad'] * vad_result[1]  # confidence
        total_weight += weights['vad']
        
        if spectral_result[0]:
            weighted_score += weights['spectral'] * spectral_result[1]
        total_weight += weights['spectral']
        
        if rms_result[0]:
            weighted_score += weights['rms'] * rms_result[1]
        total_weight += weights['rms']
        
        if whisper_result[0]:
            weighted_score += weights['whisper'] * whisper_result[1]
        total_weight += weights['whisper']
        
        # Normalize score
        final_confidence = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine if noisy (threshold can be tuned)
        is_noisy = final_confidence > 0.3
        
        details = {
            "vad_result": vad_result,
            "spectral_result": spectral_result,
            "rms_result": rms_result,
            "whisper_result": whisper_result,
            "weighted_score": final_confidence,
            "weights": weights
        }
        
        return is_noisy, final_confidence, details
    
    def analyze_audio(self, audio_path: str, method: str = "combined") -> NoiseDetectionResult:
        """
        Analyze audio file using specified method.
        
        Args:
            audio_path: Path to audio file
            method: Detection method to use
            
        Returns:
            NoiseDetectionResult object
        """
        logger.info(f"Analyzing {audio_path} with method: {method}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Ensure mono and correct sample rate
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Run detection based on method
        if method == "vad":
            is_noisy, confidence, details = self.detect_vad_speech_ratio(audio)
        elif method == "spectral":
            is_noisy, confidence, details = self.detect_spectral_features(audio)
        elif method == "rms":
            is_noisy, confidence, details = self.detect_rms_threshold(audio)
        elif method == "whisper":
            is_noisy, confidence, details = self.detect_whisper_confidence(audio_path)
        elif method == "combined":
            is_noisy, confidence, details = self.detect_combined(audio, audio_path)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return NoiseDetectionResult(
            is_noisy=is_noisy,
            confidence=confidence,
            method=method,
            details=details,
            audio_path=audio_path
        )

class NoiseDetectionEvaluator:
    """Evaluator for testing noise detection methods on benchmark datasets."""
    
    def __init__(self, detector: NoiseDetector):
        self.detector = detector
        self.results = []
    
    def evaluate_dataset(self, 
                        clean_dir: str = "data/clean_speech",
                        noise_dir: str = "data/noise",
                        mixed_dir: str = "data/mixed",
                        methods: List[str] = None) -> pd.DataFrame:
        """
        Evaluate noise detection methods on a dataset.
        
        Args:
            clean_dir: Directory with clean speech files
            noise_dir: Directory with noise files
            mixed_dir: Directory with mixed files (if available)
            methods: List of methods to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        if methods is None:
            methods = ["vad", "spectral", "rms", "whisper", "combined"]
        
        results = []
        
        # Test on clean speech (should not be detected as noisy)
        logger.info("Testing on clean speech...")
        clean_files = list(Path(clean_dir).glob("*.wav"))
        for file_path in clean_files[:10]:  # Limit for speed
            for method in methods:
                try:
                    result = self.detector.analyze_audio(str(file_path), method)
                    results.append({
                        'file': file_path.name,
                        'type': 'clean',
                        'method': method,
                        'is_noisy': result.is_noisy,
                        'confidence': result.confidence,
                        'ground_truth': False  # Clean speech should not be noisy
                    })
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Test on noise files (should be detected as noisy)
        logger.info("Testing on noise files...")
        noise_files = list(Path(noise_dir).glob("*.wav"))
        for file_path in noise_files[:10]:  # Limit for speed
            for method in methods:
                try:
                    result = self.detector.analyze_audio(str(file_path), method)
                    results.append({
                        'file': file_path.name,
                        'type': 'noise',
                        'method': method,
                        'is_noisy': result.is_noisy,
                        'confidence': result.confidence,
                        'ground_truth': True  # Noise should be detected as noisy
                    })
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Test on mixed files if available
        if Path(mixed_dir).exists():
            logger.info("Testing on mixed files...")
            mixed_files = list(Path(mixed_dir).glob("*.wav"))
            for file_path in mixed_files[:10]:  # Limit for speed
                # Determine ground truth from filename (assuming SNR in filename)
                snr = self._extract_snr_from_filename(file_path.name)
                ground_truth = snr < 15 if snr is not None else None  # SNR < 15dB is noisy
                
                for method in methods:
                    try:
                        result = self.detector.analyze_audio(str(file_path), method)
                        results.append({
                            'file': file_path.name,
                            'type': 'mixed',
                            'method': method,
                            'is_noisy': result.is_noisy,
                            'confidence': result.confidence,
                            'ground_truth': ground_truth,
                            'snr': snr
                        })
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
        
        return pd.DataFrame(results)
    
    def _extract_snr_from_filename(self, filename: str) -> Optional[float]:
        """Extract SNR value from filename."""
        import re
        match = re.search(r'snr(-?\d+)', filename.lower())
        return float(match.group(1)) if match else None
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute evaluation metrics."""
        metrics = {}
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            
            # Filter out rows with missing ground truth
            valid_df = method_df.dropna(subset=['ground_truth'])
            
            if len(valid_df) == 0:
                continue
            
            # Compute basic metrics
            tp = ((valid_df['is_noisy'] == True) & (valid_df['ground_truth'] == True)).sum()
            tn = ((valid_df['is_noisy'] == False) & (valid_df['ground_truth'] == False)).sum()
            fp = ((valid_df['is_noisy'] == True) & (valid_df['ground_truth'] == False)).sum()
            fn = ((valid_df['is_noisy'] == False) & (valid_df['ground_truth'] == True)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(valid_df) if len(valid_df) > 0 else 0
            
            # Compute ROC curve and AUC
            try:
                fpr, tpr, _ = roc_curve(valid_df['ground_truth'], valid_df['confidence'])
                roc_auc = auc(fpr, tpr)
            except:
                roc_auc = 0.5
            
            metrics[method] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'auc': roc_auc,
                'samples': len(valid_df)
            }
        
        return metrics
    
    def plot_results(self, df: pd.DataFrame, save_path: str = "noise_detection_results.png"):
        """Plot evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC curves
        ax1 = axes[0, 0]
        for method in df['method'].unique():
            method_df = df[df['method'] == method].dropna(subset=['ground_truth'])
            if len(method_df) > 0:
                fpr, tpr, _ = roc_curve(method_df['ground_truth'], method_df['confidence'])
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Confidence distributions
        ax2 = axes[0, 1]
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            ax2.hist(method_df['confidence'], alpha=0.7, label=method, bins=20)
        
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distributions')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Performance by audio type
        ax3 = axes[1, 0]
        type_performance = df.groupby(['type', 'method'])['is_noisy'].mean().unstack()
        type_performance.plot(kind='bar', ax=ax3)
        ax3.set_ylabel('Noise Detection Rate')
        ax3.set_title('Performance by Audio Type')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        
        # 4. SNR analysis (if available)
        ax4 = axes[1, 1]
        snr_df = df.dropna(subset=['snr'])
        if len(snr_df) > 0:
            for method in snr_df['method'].unique():
                method_df = snr_df[snr_df['method'] == method]
                ax4.scatter(method_df['snr'], method_df['confidence'], alpha=0.7, label=method)
            
            ax4.set_xlabel('SNR (dB)')
            ax4.set_ylabel('Confidence')
            ax4.set_title('Confidence vs SNR')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No SNR data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('SNR Analysis')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run noise detection evaluation."""
    logger.info("Starting noise detection evaluation...")
    
    # Initialize detector
    detector = NoiseDetector(
        vad_aggressiveness=2,
        speech_ratio_threshold=0.3,
        rms_threshold=0.1,
        spectral_flatness_threshold=0.7,
        zcr_threshold=0.15,
        whisper_confidence_threshold=-1.0
    )
    
    # Initialize evaluator
    evaluator = NoiseDetectionEvaluator(detector)
    
    # Check if datasets exist
    if not Path("data/clean_speech").exists() or not Path("data/noise").exists():
        logger.error("Datasets not found! Please run download_datasets.py first.")
        return
    
    # Run evaluation
    logger.info("Running evaluation...")
    results_df = evaluator.evaluate_dataset()
    
    # Compute metrics
    metrics = evaluator.compute_metrics(results_df)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("NOISE DETECTION EVALUATION RESULTS")
    logger.info("="*60)
    
    for method, metric in metrics.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Precision: {metric['precision']:.3f}")
        logger.info(f"  Recall: {metric['recall']:.3f}")
        logger.info(f"  F1-Score: {metric['f1']:.3f}")
        logger.info(f"  Accuracy: {metric['accuracy']:.3f}")
        logger.info(f"  AUC: {metric['auc']:.3f}")
        logger.info(f"  Samples: {metric['samples']}")
    
    # Save results
    results_df.to_csv("noise_detection_results.csv", index=False)
    logger.info("\nResults saved to noise_detection_results.csv")
    
    # Plot results
    evaluator.plot_results(results_df)
    
    # Find best method
    best_method = max(metrics.keys(), key=lambda x: metrics[x]['f1'])
    logger.info(f"\nBest method by F1-score: {best_method}")
    
    # Example usage
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE USAGE")
    logger.info("="*60)
    
    # Test on a sample file
    clean_files = list(Path("data/clean_speech").glob("*.wav"))
    noise_files = list(Path("data/noise").glob("*.wav"))
    
    if clean_files and noise_files:
        logger.info(f"\nTesting on clean speech: {clean_files[0].name}")
        result = detector.analyze_audio(str(clean_files[0]), "combined")
        logger.info(f"  Is noisy: {result.is_noisy}")
        logger.info(f"  Confidence: {result.confidence:.3f}")
        
        logger.info(f"\nTesting on noise: {noise_files[0].name}")
        result = detector.analyze_audio(str(noise_files[0]), "combined")
        logger.info(f"  Is noisy: {result.is_noisy}")
        logger.info(f"  Confidence: {result.confidence:.3f}")

if __name__ == "__main__":
    main() 