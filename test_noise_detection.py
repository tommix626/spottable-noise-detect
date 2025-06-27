#!/usr/bin/env python3
"""
Simple test script for the noise detection system.

This script demonstrates how to use the NoiseDetector class to analyze
audio files and determine if the environment is too noisy.
"""

import os
from pathlib import Path
from noise_detection_system import NoiseDetector, NoiseDetectionResult
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_file(audio_path: str, detector: NoiseDetector):
    """Test noise detection on a single audio file."""
    logger.info(f"\nTesting: {audio_path}")
    
    # Test all methods
    methods = ["vad", "spectral", "rms", "whisper", "combined"]
    
    for method in methods:
        try:
            result = detector.analyze_audio(audio_path, method)
            
            status = "🔴 NOISY" if result.is_noisy else "🟢 CLEAN"
            logger.info(f"  {method.upper():10} | {status} | Confidence: {result.confidence:.3f}")
            
            # Print some details for combined method
            if method == "combined" and result.details:
                details = result.details
                if "vad_result" in details:
                    vad_details = details["vad_result"][2]
                    logger.info(f"    VAD: Speech ratio = {vad_details.get('speech_ratio', 0):.3f}")
                
                if "spectral_result" in details:
                    spec_details = details["spectral_result"][2]
                    logger.info(f"    Spectral: Flatness = {spec_details.get('spectral_flatness', 0):.3f}")
                
                if "rms_result" in details:
                    rms_details = details["rms_result"][2]
                    logger.info(f"    RMS: Energy = {rms_details.get('rms', 0):.3f}")
                
                if "whisper_result" in details:
                    whisper_details = details["whisper_result"][2]
                    logger.info(f"    Whisper: Confidence = {whisper_details.get('avg_confidence', 0):.3f}")
                    
        except Exception as e:
            logger.error(f"  {method.upper():10} | ERROR: {e}")

def main():
    """Main test function."""
    logger.info("Noise Detection System Test")
    logger.info("=" * 50)
    
    # Initialize detector with default settings
    detector = NoiseDetector()
    
    # Check if datasets exist
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("Data directory not found! Please run download_datasets.py first.")
        return
    
    # Test on clean speech files
    clean_dir = data_dir / "clean_speech"
    if clean_dir.exists():
        clean_files = list(clean_dir.glob("*.wav"))
        if clean_files:
            logger.info(f"\nTesting on {len(clean_files[:3])} clean speech files:")
            for file_path in clean_files[:3]:  # Test first 3 files
                test_single_file(str(file_path), detector)
    
    # Test on noise files
    noise_dir = data_dir / "noise"
    if noise_dir.exists():
        noise_files = list(noise_dir.glob("*.wav"))
        if noise_files:
            logger.info(f"\nTesting on {len(noise_files[:3])} noise files:")
            for file_path in noise_files[:3]:  # Test first 3 files
                test_single_file(str(file_path), detector)
    
    # Test on mixed files if available
    mixed_dir = data_dir / "mixed"
    if mixed_dir.exists():
        mixed_files = list(mixed_dir.glob("*.wav"))
        if mixed_files:
            logger.info(f"\nTesting on {len(mixed_files[:2])} mixed files:")
            for file_path in mixed_files[:2]:  # Test first 2 files
                test_single_file(str(file_path), detector)
    
    logger.info("\n" + "=" * 50)
    logger.info("Test completed!")
    logger.info("\nTo run full evaluation, use: python noise_detection_system.py")

if __name__ == "__main__":
    main() 