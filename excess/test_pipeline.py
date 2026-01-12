#!/usr/bin/env python
"""
Test the full BCI pipeline with simulated noisy EEG data.

This tests:
1. Simulated EEG generation with SSVEP
2. Preprocessing (filtering)
3. Classification (CCA and FFT)
4. Calibration system
5. Score generation and playback
"""

import sys
import numpy as np
import time

# Add project to path
sys.path.insert(0, '.')

from src.bci.preprocessing import EEGPreprocessor, SimulatedEEGSource
from src.bci.classifier import SSVEPClassifier, AttentionTarget
from src.bci.calibration import CalibrationData
from src.bci.controller import BCICursorController
from src.bci.score import BCIScore, play_score

def generate_noisy_ssvep(
    frequency: float,
    duration: float,
    sample_rate: float = 250,
    n_channels: int = 8,
    snr: float = 0.3,  # Low SNR = very noisy
    phase: float = 0.0
) -> np.ndarray:
    """Generate noisy EEG with embedded SSVEP signal."""
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    # Base noise - pink noise approximation
    noise = np.random.randn(n_samples, n_channels)
    
    # Add some low-frequency drift
    for ch in range(n_channels):
        drift = 0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi)
        noise[:, ch] += drift
    
    # Add 50Hz powerline noise
    powerline = 0.2 * np.sin(2 * np.pi * 50 * t)
    noise += powerline[:, np.newaxis]
    
    # Add SSVEP signal to occipital channels (5, 6, 7)
    ssvep = snr * np.sin(2 * np.pi * frequency * t + phase)
    
    # Also add first harmonic
    ssvep += (snr * 0.5) * np.sin(2 * np.pi * 2 * frequency * t + 2 * phase)
    
    # SSVEP strongest in occipital channels
    noise[:, 5] += ssvep * 0.7   # PO7
    noise[:, 6] += ssvep * 1.0   # Oz (strongest)
    noise[:, 7] += ssvep * 0.7   # PO8
    
    # Small amount in parietal
    noise[:, 4] += ssvep * 0.3   # Pz
    
    return noise


def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("\n" + "="*60)
    print("TEST 1: Preprocessing Pipeline")
    print("="*60)
    
    preprocessor = EEGPreprocessor(sample_rate=250, n_channels=8)
    
    # Generate noisy data with 15Hz SSVEP
    data = generate_noisy_ssvep(15.0, duration=2.0, snr=0.3)
    
    print(f"Input data shape: {data.shape}")
    print(f"Input range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Process
    processed = preprocessor.process_chunk(data)
    
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Check band power
    power_15hz = preprocessor.get_band_power(6, 14, 16)  # Oz channel
    power_10hz = preprocessor.get_band_power(6, 9, 11)
    
    print(f"15Hz band power (Oz): {power_15hz:.4f}")
    print(f"10Hz band power (Oz): {power_10hz:.4f}")
    print(f"Ratio (should be > 1): {power_15hz / (power_10hz + 1e-10):.2f}")
    
    return preprocessor


def test_classifier():
    """Test the SSVEP classifier."""
    print("\n" + "="*60)
    print("TEST 2: SSVEP Classifier")
    print("="*60)
    
    classifier = SSVEPClassifier(sample_rate=250)
    preprocessor = EEGPreprocessor(sample_rate=250, n_channels=8)
    
    # Test with 15Hz signal
    print("\n--- Testing with 15Hz SSVEP (should classify as UP) ---")
    data_15hz = generate_noisy_ssvep(15.0, duration=1.0, snr=0.5, phase=0.0)
    processed = preprocessor.process_chunk(data_15hz)
    
    result_fft = classifier.classify(processed, method="fft")
    print(f"FFT: {result_fft.target.name}, confidence={result_fft.confidence:.2f}")
    print(f"     15Hz power={result_fft.power_15hz:.4f}, 10Hz power={result_fft.power_10hz:.4f}")
    
    classifier.reset()
    result_cca = classifier.classify(processed, method="cca")
    print(f"CCA: {result_cca.target.name}, confidence={result_cca.confidence:.2f}")
    print(f"     15Hz corr={result_cca.power_15hz:.4f}, 10Hz corr={result_cca.power_10hz:.4f}")
    
    # Test with 10Hz signal
    print("\n--- Testing with 10Hz SSVEP (should classify as DOWN) ---")
    preprocessor.reset()
    classifier.reset()
    
    data_10hz = generate_noisy_ssvep(10.0, duration=1.0, snr=0.5, phase=np.pi)
    processed = preprocessor.process_chunk(data_10hz)
    
    result_fft = classifier.classify(processed, method="fft")
    print(f"FFT: {result_fft.target.name}, confidence={result_fft.confidence:.2f}")
    
    classifier.reset()
    result_cca = classifier.classify(processed, method="cca")
    print(f"CCA: {result_cca.target.name}, confidence={result_cca.confidence:.2f}")
    
    # Test with pure noise (no SSVEP)
    print("\n--- Testing with pure noise (should be uncertain) ---")
    preprocessor.reset()
    classifier.reset()
    
    noise_data = np.random.randn(250, 8) * 0.5
    processed = preprocessor.process_chunk(noise_data)
    
    result_fft = classifier.classify(processed, method="fft")
    print(f"FFT: {result_fft.target.name}, confidence={result_fft.confidence:.2f}")
    
    return classifier


def test_calibration():
    """Test the calibration system."""
    print("\n" + "="*60)
    print("TEST 3: Calibration System")
    print("="*60)
    
    cal_data = CalibrationData(sample_rate=250, n_channels=8)
    
    # Simulate calibration trials
    print("Simulating calibration trials...")
    
    for i in range(3):
        # 15Hz trial
        eeg_15 = generate_noisy_ssvep(15.0, duration=5.0, snr=0.4, phase=0.0)
        timestamps = np.arange(len(eeg_15)) / 250
        cal_data.add_trial(15.0, eeg_15, timestamps, 5.0)
        print(f"  Added 15Hz trial {i+1}")
        
        # 10Hz trial
        eeg_10 = generate_noisy_ssvep(10.0, duration=5.0, snr=0.4, phase=np.pi)
        timestamps = np.arange(len(eeg_10)) / 250
        cal_data.add_trial(10.0, eeg_10, timestamps, 5.0)
        print(f"  Added 10Hz trial {i+1}")
    
    # Compute templates
    print("\nComputing templates...")
    cal_data.compute_templates()
    
    stats = cal_data.get_statistics()
    print(f"Calibration stats: {stats}")
    
    # Get references
    ref_15, ref_10 = cal_data.get_cca_references()
    print(f"15Hz template shape: {ref_15.shape}")
    print(f"10Hz template shape: {ref_10.shape}")
    
    # Test classifier with calibration
    print("\n--- Testing classifier WITH calibration ---")
    classifier = SSVEPClassifier(sample_rate=250)
    classifier.load_calibration(cal_data)
    
    preprocessor = EEGPreprocessor(sample_rate=250, n_channels=8)
    
    # Test 15Hz
    data_15hz = generate_noisy_ssvep(15.0, duration=0.5, snr=0.3)
    processed = preprocessor.process_chunk(data_15hz)
    result = classifier.classify(processed, method="cca")
    print(f"15Hz test: {result.target.name}, confidence={result.confidence:.2f}")
    
    # Test 10Hz
    preprocessor.reset()
    classifier.reset()
    data_10hz = generate_noisy_ssvep(10.0, duration=0.5, snr=0.3, phase=np.pi)
    processed = preprocessor.process_chunk(data_10hz)
    result = classifier.classify(processed, method="cca")
    print(f"10Hz test: {result.target.name}, confidence={result.confidence:.2f}")
    
    return cal_data


def test_full_composition():
    """Test a full composition session with simulated data."""
    print("\n" + "="*60)
    print("TEST 4: Full Composition Simulation")
    print("="*60)
    
    # Setup - first create calibration for better accuracy
    print("Creating calibration data first...")
    cal_data = CalibrationData(sample_rate=250, n_channels=8)
    for i in range(2):
        eeg_15 = generate_noisy_ssvep(15.0, duration=3.0, snr=0.5, phase=0.0)
        timestamps = np.arange(len(eeg_15)) / 250
        cal_data.add_trial(15.0, eeg_15, timestamps, 3.0)
        
        eeg_10 = generate_noisy_ssvep(10.0, duration=3.0, snr=0.5, phase=np.pi)
        timestamps = np.arange(len(eeg_10)) / 250
        cal_data.add_trial(10.0, eeg_10, timestamps, 3.0)
    cal_data.compute_templates()
    
    preprocessor = EEGPreprocessor(sample_rate=250, n_channels=8)
    classifier = SSVEPClassifier(sample_rate=250)
    classifier.load_calibration(cal_data)
    
    controller = BCICursorController(duration=5.0, vertical_speed=0.8)
    
    # Pre-fill the preprocessor buffer
    print("Pre-filling buffer...")
    warmup = generate_noisy_ssvep(15.0, duration=1.0, snr=0.5)
    preprocessor.process_chunk(warmup)
    
    # Simulate a 5-second composition
    print("\nSimulating 5-second composition...")
    print("Pattern: UP (0-2s) -> DOWN (2-4s) -> UP (4-5s)")
    
    controller.start()
    start_time = time.perf_counter()
    
    sample_rate = 250
    chunk_size = 16
    dt = chunk_size / sample_rate
    last_print = -1
    
    while controller.is_running:
        # Use real elapsed time
        real_elapsed = time.perf_counter() - start_time
        
        # Generate data based on simulated attention
        if real_elapsed < 2.0:
            # Looking at 15Hz (UP)
            chunk = generate_noisy_ssvep(15.0, duration=dt, snr=0.6, phase=0.0)
        elif real_elapsed < 4.0:
            # Looking at 10Hz (DOWN)
            chunk = generate_noisy_ssvep(10.0, duration=dt, snr=0.6, phase=np.pi)
        else:
            # Back to 15Hz (UP)
            chunk = generate_noisy_ssvep(15.0, duration=dt, snr=0.6, phase=0.0)
        
        # Process and classify
        processed = preprocessor.process_chunk(chunk)
        buffer = preprocessor.get_recent_data(0.5)
        result = classifier.classify(buffer, method="cca")
        
        # Update controller
        pos = controller.update(result)
        
        # Print progress every 0.5s
        if int(real_elapsed * 2) > last_print:
            last_print = int(real_elapsed * 2)
            print(f"  t={real_elapsed:.1f}s: pitch={pos.pitch:.2f}, target={result.target.name}, conf={result.confidence:.2f}")
        
        # Small sleep to not spin too fast
        time.sleep(0.01)
    
    # Get trail
    trail = controller.get_trail_as_tuples()
    print(f"\nComposition complete! Trail has {len(trail)} points")
    
    # Check trail shape
    pitches = [p[1] for p in trail]
    print(f"Pitch range: [{min(pitches):.2f}, {max(pitches):.2f}]")
    print(f"Start pitch: {pitches[0]:.2f}")
    print(f"End pitch: {pitches[-1]:.2f}")
    
    # Analyze the pattern
    mid_pitch = pitches[len(pitches)//2]
    print(f"Mid pitch (should be low ~0.3): {mid_pitch:.2f}")
    
    # Create score
    score = BCIScore(
        trail=trail,
        duration=5.0,
        waveform_name="Sine"
    )
    
    return score


def test_playback(score: BCIScore):
    """Test score playback."""
    print("\n" + "="*60)
    print("TEST 5: Score Playback")
    print("="*60)
    
    print(f"Score duration: {score.duration}s")
    print(f"Trail points: {len(score.trail)}")
    
    stats = score.get_statistics()
    print(f"Statistics: {stats}")
    
    print("\nPlaying score (you should hear pitch changes)...")
    try:
        play_score(score)
        print("Playback complete!")
    except Exception as e:
        print(f"Playback error: {e}")


def main():
    print("="*60)
    print("BCI PIPELINE TEST - Simulated Noisy Data")
    print("="*60)
    
    # Run tests
    test_preprocessing()
    test_classifier()
    cal_data = test_calibration()
    score = test_full_composition()
    
    # Play automatically in test mode
    print("\n" + "="*60)
    print("Playing generated score...")
    test_playback(score)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
