"""
Offline testing script for validation data.

Loads saved raw EEG data and tests different preprocessing/classification approaches
without needing to run the GUI.

Usage:
    python test_validation_offline.py [timestamp]
    
If no timestamp provided, uses the most recent validation data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bci.preprocessing import EEGPreprocessor
from src.bci.classifier import SSVEPClassifier


def load_validation_data(timestamp: Optional[str] = None) -> dict:
    """Load validation data from saved files."""
    validation_dir = Path("validation_plots")
    
    if timestamp is None:
        # Find most recent metadata file
        metadata_files = sorted(validation_dir.glob("validation_metadata_*.json"), reverse=True)
        if not metadata_files:
            raise FileNotFoundError("No validation data found!")
        metadata_file = metadata_files[0]
        timestamp = metadata_file.stem.replace("validation_metadata_", "")
        print(f"Using most recent validation: {timestamp}")
    else:
        metadata_file = validation_dir / f"validation_metadata_{timestamp}.json"
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load raw data
    data = {'metadata': metadata}
    
    if metadata['top_raw_shape']:
        top_raw = np.load(validation_dir / f"top_raw_{timestamp}.npy")
        data['top_raw'] = top_raw
        print(f"Loaded top_raw: {top_raw.shape}")
    
    if metadata['bottom_raw_shape']:
        bottom_raw = np.load(validation_dir / f"bottom_raw_{timestamp}.npy")
        data['bottom_raw'] = bottom_raw
        print(f"Loaded bottom_raw: {bottom_raw.shape}")
    
    if metadata['baseline_raw_shape']:
        baseline_raw = np.load(validation_dir / f"baseline_raw_{timestamp}.npy")
        data['baseline_raw'] = baseline_raw
        print(f"Loaded baseline_raw: {baseline_raw.shape}")
    
    # Load processed data for comparison
    top_processed = np.load(validation_dir / f"top_processed_{timestamp}.npy")
    bottom_processed = np.load(validation_dir / f"bottom_processed_{timestamp}.npy")
    data['top_processed'] = top_processed
    data['bottom_processed'] = bottom_processed
    
    return data


def test_preprocessing(data: dict, preprocessor: EEGPreprocessor, occipital_channels: list, use_fbcca: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Test preprocessing on raw data."""
    print(f"\nTesting preprocessing:")
    if use_fbcca:
        print(f"  FBCCA mode: Only CAR + notch filter (no main bandpass)")
        print(f"  CAR: {preprocessor.use_car}")
        # For FBCCA, only apply CAR and notch (no bandpass)
        # We'll need to manually apply CAR and notch
        top_raw = data['top_raw']
        bottom_raw = data['bottom_raw']
        
        # Apply CAR manually
        if preprocessor.use_car:
            top_car = top_raw - np.mean(top_raw, axis=1, keepdims=True)
            bottom_car = bottom_raw - np.mean(bottom_raw, axis=1, keepdims=True)
        else:
            top_car = top_raw
            bottom_car = bottom_raw
        
        # Apply notch filter manually (50/60Hz)
        nyquist = preprocessor.sample_rate / 2
        notch_freq = preprocessor.notch_freq
        notch_low = (notch_freq - 2) / nyquist
        notch_high = (notch_freq + 2) / nyquist
        notch_low = max(0.001, min(notch_low, 0.99))
        notch_high = max(notch_low + 0.01, min(notch_high, 0.99))
        from scipy import signal
        b_notch, a_notch = signal.butter(2, [notch_low, notch_high], btype='bandstop')
        
        top_processed_all = np.zeros_like(top_car)
        bottom_processed_all = np.zeros_like(bottom_car)
        for ch in range(top_car.shape[1]):
            top_processed_all[:, ch] = signal.lfilter(b_notch, a_notch, top_car[:, ch])
            bottom_processed_all[:, ch] = signal.lfilter(b_notch, a_notch, bottom_car[:, ch])
    else:
        print(f"  Bandpass: {preprocessor.bandpass_low}-{preprocessor.bandpass_high}Hz")
        print(f"  CAR: {preprocessor.use_car}")
        # Process raw data (all 8 channels) with full preprocessing
        top_processed_all = preprocessor.process_chunk(data['top_raw'])
        bottom_processed_all = preprocessor.process_chunk(data['bottom_raw'])
    
    # Extract occipital channels only (PO7, Oz, PO8 = indices 5, 6, 7)
    top_processed = top_processed_all[:, occipital_channels]
    bottom_processed = bottom_processed_all[:, occipital_channels]
    
    print(f"  Top processed (occipital): {top_processed.shape}")
    print(f"  Bottom processed (occipital): {bottom_processed.shape}")
    
    return top_processed, bottom_processed


def test_classification(top_processed: np.ndarray, bottom_processed: np.ndarray, 
                       classifier: SSVEPClassifier, window_seconds: float = 0.3,
                       step_seconds: float = 0.05, method: str = "cca") -> dict:
    """Test classification on processed data."""
    print(f"\nTesting classification (method: {method}):")
    print(f"  Window: {window_seconds}s, Step: {step_seconds}s")
    if method == "fbcca":
        print(f"  Filter Bank CCA: 3 bands (fundamental, 1st harmonic, 2nd harmonic)")
    else:
        print(f"  Harmonics: {classifier.n_harmonics}")
        print(f"  Min correlation diff: {classifier.min_correlation_diff}")
    
    sample_rate = classifier.sample_rate
    window_samples = int(window_seconds * sample_rate)
    step_samples = int(step_seconds * sample_rate)
    
    results = {
        'top': {'predictions': [], 'n_up': 0, 'n_down': 0, 'n_none': 0},
        'bottom': {'predictions': [], 'n_up': 0, 'n_down': 0, 'n_none': 0}
    }
    
    # Classify top target chunks (should predict UP)
    for start in range(0, len(top_processed) - window_samples + 1, step_samples):
        chunk = top_processed[start:start + window_samples]
        result = classifier.classify(chunk, method=method)
        results['top']['predictions'].append({
            'target': result.target.name,
            'confidence': result.confidence,
            'corr_up': result.power_higher_freq,
            'corr_down': result.power_lower_freq,
            'raw_score': result.raw_score
        })
        if result.target.name == 'UP':
            results['top']['n_up'] += 1
        elif result.target.name == 'DOWN':
            results['top']['n_down'] += 1
        else:
            results['top']['n_none'] += 1
    
    # Classify bottom target chunks (should predict DOWN)
    classifier.reset()  # Reset history
    for start in range(0, len(bottom_processed) - window_samples + 1, step_samples):
        chunk = bottom_processed[start:start + window_samples]
        result = classifier.classify(chunk, method=method)
        results['bottom']['predictions'].append({
            'target': result.target.name,
            'confidence': result.confidence,
            'corr_up': result.power_higher_freq,
            'corr_down': result.power_lower_freq,
            'raw_score': result.raw_score
        })
        if result.target.name == 'UP':
            results['bottom']['n_up'] += 1
        elif result.target.name == 'DOWN':
            results['bottom']['n_down'] += 1
        else:
            results['bottom']['n_none'] += 1
    
    # Calculate accuracies
    top_total = len(results['top']['predictions'])
    bottom_total = len(results['bottom']['predictions'])
    results['top']['accuracy'] = results['top']['n_up'] / top_total if top_total > 0 else 0
    results['bottom']['accuracy'] = results['bottom']['n_down'] / bottom_total if bottom_total > 0 else 0
    results['overall_accuracy'] = (results['top']['n_up'] + results['bottom']['n_down']) / (top_total + bottom_total) if (top_total + bottom_total) > 0 else 0
    
    return results


def main():
    """Main testing function."""
    import argparse
    parser = argparse.ArgumentParser(description="Test validation data offline")
    parser.add_argument('timestamp', nargs='?', help='Validation timestamp (YYYYMMDD_HHMMSS)')
    parser.add_argument('--method', default='fbcca', choices=['cca', 'fbcca', 'fft'],
                       help='Classification method (default: fbcca)')
    args = parser.parse_args()
    
    # Load data
    print("=" * 60)
    print("OFFLINE VALIDATION TESTING")
    print("=" * 60)
    data = load_validation_data(args.timestamp)
    metadata = data['metadata']
    
    # Create preprocessor with current settings
    # Note: Raw data has 8 channels, but processed data has 3 (occipital)
    # We need to process all 8 channels first, then extract occipital
    raw_n_channels = data['top_raw'].shape[1] if 'top_raw' in data else 8
    preprocessor = EEGPreprocessor(
        sample_rate=metadata['sample_rate'],
        n_channels=raw_n_channels,  # Use raw data channel count (8)
        bandpass_low=metadata['preprocessing']['bandpass_low'],
        bandpass_high=metadata['preprocessing']['bandpass_high'],
        notch_freq=metadata['preprocessing']['notch_freq'],
        use_car=metadata['preprocessing']['use_car']
    )
    
    # Create classifier with current settings
    classifier = SSVEPClassifier(
        sample_rate=metadata['sample_rate'],
        target_frequencies=tuple(metadata['classifier']['target_frequencies']),
        target_phases=tuple(metadata['classifier']['target_phases']),
        n_harmonics=metadata['classifier']['n_harmonics'],
        occipital_channels=metadata['occipital_channels']
    )
    
    # Test preprocessing (skip main bandpass if using FBCCA)
    occipital_channels = metadata['occipital_channels']
    use_fbcca = (args.method == 'fbcca')
    top_processed, bottom_processed = test_preprocessing(data, preprocessor, occipital_channels, use_fbcca=use_fbcca)
    
    # Test classification with specified method
    results = test_classification(top_processed, bottom_processed, classifier,
                                 window_seconds=metadata['window_seconds'],
                                 step_seconds=metadata['update_interval_seconds'],
                                 method=args.method)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Top target (should be UP):")
    print(f"  Accuracy: {results['top']['accuracy']:.1%} ({results['top']['n_up']}/{len(results['top']['predictions'])})")
    print(f"  UP: {results['top']['n_up']}, DOWN: {results['top']['n_down']}, NONE: {results['top']['n_none']}")
    
    print(f"\nBottom target (should be DOWN):")
    print(f"  Accuracy: {results['bottom']['accuracy']:.1%} ({results['bottom']['n_down']}/{len(results['bottom']['predictions'])})")
    print(f"  UP: {results['bottom']['n_up']}, DOWN: {results['bottom']['n_down']}, NONE: {results['bottom']['n_none']}")
    
    print(f"\nOverall accuracy: {results['overall_accuracy']:.1%}")
    
    # Calculate mean correlations
    top_mean_up = np.mean([p['corr_up'] for p in results['top']['predictions']])
    top_mean_down = np.mean([p['corr_down'] for p in results['top']['predictions']])
    bottom_mean_up = np.mean([p['corr_up'] for p in results['bottom']['predictions']])
    bottom_mean_down = np.mean([p['corr_down'] for p in results['bottom']['predictions']])
    
    print(f"\nMean correlations:")
    print(f"  Top target: corr_up={top_mean_up:.4f}, corr_down={top_mean_down:.4f}")
    print(f"  Bottom target: corr_up={bottom_mean_up:.4f}, corr_down={bottom_mean_down:.4f}")


if __name__ == "__main__":
    main()
