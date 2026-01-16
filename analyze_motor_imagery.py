#!/usr/bin/env python
"""
Offline Analysis Script for Motor Imagery BCI Data

This script loads saved motor imagery session data and replicates the exact
preprocessing pipeline used during live classification. It provides comprehensive
visualization of:
- Raw EEG signals
- Filtered mu/beta bands
- CSP features
- C3/C4 power differences
- Classification results
- Cursor movement

Based on MI-PLVGAT visualization approaches for motor imagery analysis.

Usage:
    python analyze_motor_imagery.py <session_dir>
    
Example:
    python analyze_motor_imagery.py motor_imagery_sessions/20260116_153022
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.bci.motor_imagery_classifier import (
    MotorImageryClassifier,
    AttentionTarget,
    DEFAULT_SENSORIMOTOR_CHANNELS,
    UNICORN_CHANNEL_NAMES
)
from src.bci.preprocessing import LSLPreprocessor


def load_session_data(session_dir: Path) -> Dict:
    """Load all data from a motor imagery session."""
    data = {}
    
    # Load metadata
    metadata_file = session_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            data['metadata'] = json.load(f)
        print(f"[LOAD] Loaded metadata from {metadata_file}")
    else:
        print(f"[WARNING] No metadata.json found in {session_dir}")
        data['metadata'] = {}
    
    # Load baseline data
    baseline_file = session_dir / "baseline_data.npy"
    if baseline_file.exists():
        data['baseline'] = np.load(baseline_file)
        print(f"[LOAD] Baseline data: shape={data['baseline'].shape}")
    else:
        print(f"[WARNING] No baseline_data.npy found")
        data['baseline'] = None
    
    # Load baseline statistics
    baseline_stats_file = session_dir / "baseline_stats.json"
    if baseline_stats_file.exists():
        with open(baseline_stats_file, 'r') as f:
            data['baseline_stats'] = json.load(f)
        print(f"[LOAD] Baseline statistics loaded")
    else:
        data['baseline_stats'] = None
    
    # Load composition EEG data
    eeg_file = session_dir / "eeg_data.npy"
    if eeg_file.exists():
        data['eeg'] = np.load(eeg_file)
        print(f"[LOAD] EEG data: shape={data['eeg'].shape}")
    else:
        print(f"[WARNING] No eeg_data.npy found")
        data['eeg'] = None
    
    # Load EEG timestamps
    timestamps_file = session_dir / "eeg_timestamps.npy"
    if timestamps_file.exists():
        data['eeg_timestamps'] = np.load(timestamps_file)
        print(f"[LOAD] EEG timestamps: {len(data['eeg_timestamps'])} timestamps")
    else:
        data['eeg_timestamps'] = None
    
    # Load classifications
    classifications_file = session_dir / "classifications.json"
    if classifications_file.exists():
        with open(classifications_file, 'r') as f:
            data['classifications'] = json.load(f)
        print(f"[LOAD] Classifications: {len(data['classifications'])} results")
    else:
        print(f"[WARNING] No classifications.json found")
        data['classifications'] = None
    
    # Load cursor trail
    trail_file = session_dir / "cursor_trail.npy"
    if trail_file.exists():
        data['trail'] = np.load(trail_file)
        print(f"[LOAD] Cursor trail: {len(data['trail'])} points")
    else:
        data['trail'] = None
    
    return data


def replicate_preprocessing_pipeline(
    eeg_data: np.ndarray,
    classifier: MotorImageryClassifier,
    sample_rate: float = 250.0
) -> Dict[str, np.ndarray]:
    """
    Replicate the exact preprocessing pipeline used during live classification.
    
    Returns:
        Dictionary with:
        - sensorimotor_data: Extracted sensorimotor channels
        - mu_data: Mu band filtered data
        - beta_data: Beta band filtered data
        - mu_features: CSP features from mu band
        - beta_features: CSP features from beta band
        - c3_mu_power: C3 mu power over time
        - c4_mu_power: C4 mu power over time
        - c3_beta_power: C3 beta power over time
        - c4_beta_power: C4 beta power over time
    """
    results = {}
    
    # Extract sensorimotor channels (C3, Cz, C4)
    sensorimotor_data = eeg_data[:, classifier.sensorimotor_channels]
    results['sensorimotor_data'] = sensorimotor_data
    print(f"[PREPROCESS] Sensorimotor data shape: {sensorimotor_data.shape}")
    
    # Apply bandpass filters
    mu_data = classifier._bandpass_filter(
        sensorimotor_data,
        classifier.mu_band[0],
        classifier.mu_band[1]
    )
    beta_data = classifier._bandpass_filter(
        sensorimotor_data,
        classifier.beta_band[0],
        classifier.beta_band[1]
    )
    results['mu_data'] = mu_data
    results['beta_data'] = beta_data
    print(f"[PREPROCESS] Mu data shape: {mu_data.shape}, Beta data shape: {beta_data.shape}")
    
    # Apply baseline normalization if available
    if classifier.has_baseline:
        mu_data = (mu_data - classifier._baseline_mu_mean) / classifier._baseline_mu_std
        beta_data = (beta_data - classifier._baseline_beta_mean) / classifier._baseline_beta_std
        results['mu_data_normalized'] = mu_data
        results['beta_data_normalized'] = beta_data
        print(f"[PREPROCESS] Applied baseline normalization")
    
    # Extract CSP features in sliding windows (replicate live classification)
    window_samples = int(classifier.window_seconds * sample_rate)
    step_samples = int(0.1 * sample_rate)  # 0.1 second steps
    
    mu_features_list = []
    beta_features_list = []
    feature_times = []
    
    for i in range(0, len(mu_data) - window_samples + 1, step_samples):
        window_mu = mu_data[i:i+window_samples]
        window_beta = beta_data[i:i+window_samples]
        
        mu_feat = classifier._extract_csp_features(window_mu)
        beta_feat = classifier._extract_csp_features(window_beta)
        
        mu_features_list.append(mu_feat)
        beta_features_list.append(beta_feat)
        feature_times.append(i / sample_rate)
    
    if len(mu_features_list) > 0:
        mu_features_array = np.array(mu_features_list)
        beta_features_array = np.array(beta_features_list)
        results['mu_features'] = mu_features_array
        results['beta_features'] = beta_features_array
        results['feature_times'] = np.array(feature_times)
        print(f"[PREPROCESS] Mu features: {mu_features_array.shape}, Beta features: {beta_features_array.shape}")
    else:
        # Fallback: extract from full data if window too large
        mu_features = classifier._extract_csp_features(mu_data)
        beta_features = classifier._extract_csp_features(beta_data)
        results['mu_features'] = mu_features
        results['beta_features'] = beta_features
        print(f"[PREPROCESS] Mu features: {mu_features.shape}, Beta features: {beta_features.shape} (full data)")
    
    # Compute C3/C4 power over time (for visualization)
    c3_idx = 0  # C3 is first in sensorimotor_channels
    c4_idx = 2  # C4 is third in sensorimotor_channels
    
    # Compute power in sliding windows
    window_samples = int(1.0 * sample_rate)  # 1 second windows
    step_samples = int(0.1 * sample_rate)  # 0.1 second steps
    
    c3_mu_power = []
    c4_mu_power = []
    c3_beta_power = []
    c4_beta_power = []
    power_times = []
    
    for i in range(0, len(mu_data) - window_samples + 1, step_samples):
        window_mu = mu_data[i:i+window_samples]
        window_beta = beta_data[i:i+window_samples]
        
        c3_mu_power.append(np.var(window_mu[:, c3_idx]))
        c4_mu_power.append(np.var(window_mu[:, c4_idx]))
        c3_beta_power.append(np.var(window_beta[:, c3_idx]))
        c4_beta_power.append(np.var(window_beta[:, c4_idx]))
        power_times.append(i / sample_rate)
    
    results['c3_mu_power'] = np.array(c3_mu_power)
    results['c4_mu_power'] = np.array(c4_mu_power)
    results['c3_beta_power'] = np.array(c3_beta_power)
    results['c4_beta_power'] = np.array(c4_beta_power)
    results['power_times'] = np.array(power_times)
    
    return results


def plot_comprehensive_analysis(
    data: Dict,
    preprocessed: Dict,
    classifier: MotorImageryClassifier,
    output_dir: Path
) -> None:
    """Create comprehensive visualization plots."""
    
    sample_rate = classifier.sample_rate
    n_channels = 8
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # ===== PLOT 1: Raw EEG Signals (All Channels) =====
    ax1 = fig.add_subplot(gs[0, :])
    if data['eeg'] is not None:
        time_axis = np.arange(len(data['eeg'])) / sample_rate
        for ch in range(min(8, data['eeg'].shape[1])):
            ch_name = UNICORN_CHANNEL_NAMES[ch] if ch < len(UNICORN_CHANNEL_NAMES) else f"Ch{ch}"
            offset = ch * 50  # Offset for visualization
            ax1.plot(time_axis, data['eeg'][:, ch] + offset, label=ch_name, alpha=0.7, linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (μV, offset)')
        ax1.set_title('Raw EEG Signals (All 8 Channels)')
        ax1.legend(ncol=4, loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # ===== PLOT 2: Sensorimotor Channels (C3, Cz, C4) =====
    ax2 = fig.add_subplot(gs[1, 0])
    if 'sensorimotor_data' in preprocessed:
        sensorimotor = preprocessed['sensorimotor_data']
        time_axis = np.arange(len(sensorimotor)) / sample_rate
        ch_names = ['C3', 'Cz', 'C4']
        for i, ch_name in enumerate(ch_names):
            offset = i * 30
            ax2.plot(time_axis, sensorimotor[:, i] + offset, label=ch_name, linewidth=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (μV, offset)')
        ax2.set_title('Sensorimotor Channels')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # ===== PLOT 3: Mu Band Filtered (C3, C4) =====
    ax3 = fig.add_subplot(gs[1, 1])
    if 'mu_data' in preprocessed:
        mu_data = preprocessed['mu_data']
        time_axis = np.arange(len(mu_data)) / sample_rate
        ax3.plot(time_axis, mu_data[:, 0], label='C3', alpha=0.7, linewidth=0.8)
        ax3.plot(time_axis, mu_data[:, 2], label='C4', alpha=0.7, linewidth=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude (μV)')
        ax3.set_title(f'Mu Band Filtered (8-13 Hz)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # ===== PLOT 4: Beta Band Filtered (C3, C4) =====
    ax4 = fig.add_subplot(gs[1, 2])
    if 'beta_data' in preprocessed:
        beta_data = preprocessed['beta_data']
        time_axis = np.arange(len(beta_data)) / sample_rate
        ax4.plot(time_axis, beta_data[:, 0], label='C3', alpha=0.7, linewidth=0.8)
        ax4.plot(time_axis, beta_data[:, 2], label='C4', alpha=0.7, linewidth=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Amplitude (μV)')
        ax4.set_title(f'Beta Band Filtered (13-30 Hz)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # ===== PLOT 5: C3/C4 Power Comparison (Mu Band) =====
    ax5 = fig.add_subplot(gs[2, 0])
    if 'c3_mu_power' in preprocessed and 'c4_mu_power' in preprocessed:
        times = preprocessed['power_times']
        ax5.plot(times, preprocessed['c3_mu_power'], label='C3', linewidth=2)
        ax5.plot(times, preprocessed['c4_mu_power'], label='C4', linewidth=2)
        ax5.fill_between(times, preprocessed['c3_mu_power'], preprocessed['c4_mu_power'], 
                         where=(preprocessed['c3_mu_power'] < preprocessed['c4_mu_power']),
                         alpha=0.3, color='red', label='C3 < C4 (Left)')
        ax5.fill_between(times, preprocessed['c3_mu_power'], preprocessed['c4_mu_power'],
                         where=(preprocessed['c3_mu_power'] > preprocessed['c4_mu_power']),
                         alpha=0.3, color='blue', label='C3 > C4 (Right)')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Power (Variance)')
        ax5.set_title('C3 vs C4 Power (Mu Band)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # ===== PLOT 6: C3/C4 Power Comparison (Beta Band) =====
    ax6 = fig.add_subplot(gs[2, 1])
    if 'c3_beta_power' in preprocessed and 'c4_beta_power' in preprocessed:
        times = preprocessed['power_times']
        ax6.plot(times, preprocessed['c3_beta_power'], label='C3', linewidth=2)
        ax6.plot(times, preprocessed['c4_beta_power'], label='C4', linewidth=2)
        ax6.fill_between(times, preprocessed['c3_beta_power'], preprocessed['c4_beta_power'],
                         where=(preprocessed['c3_beta_power'] < preprocessed['c4_beta_power']),
                         alpha=0.3, color='red', label='C3 < C4 (Left)')
        ax6.fill_between(times, preprocessed['c3_beta_power'], preprocessed['c4_beta_power'],
                         where=(preprocessed['c3_beta_power'] > preprocessed['c4_beta_power']),
                         alpha=0.3, color='blue', label='C3 > C4 (Right)')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Power (Variance)')
        ax6.set_title('C3 vs C4 Power (Beta Band)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # ===== PLOT 7: CSP Features Over Time =====
    ax7 = fig.add_subplot(gs[2, 2])
    if 'mu_features' in preprocessed and 'feature_times' in preprocessed:
        mu_features = preprocessed['mu_features']
        feature_times = preprocessed['feature_times']
        
        if len(mu_features.shape) == 2:  # (n_windows, n_features)
            for comp in range(min(4, mu_features.shape[1])):
                ax7.plot(feature_times, mu_features[:, comp], label=f'CSP {comp+1}', alpha=0.7, linewidth=1.5)
        elif len(mu_features.shape) == 1:  # Single feature vector
            ax7.plot(feature_times, mu_features, label='CSP Features', alpha=0.7, linewidth=1.5)
        
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Log Variance')
        ax7.set_title('CSP Features Over Time (Mu Band)')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
    
    # ===== PLOT 8: Classification Results =====
    ax8 = fig.add_subplot(gs[3, 0])
    if data['classifications'] is not None:
        times = [c['time'] for c in data['classifications']]
        targets = [c['target'] for c in data['classifications']]
        confidences = [c['confidence'] for c in data['classifications']]
        left_scores = [c['left_score'] for c in data['classifications']]
        right_scores = [c['right_score'] for c in data['classifications']]
        
        # Plot confidence
        ax8.plot(times, confidences, 'k-', label='Confidence', linewidth=1, alpha=0.5)
        ax8.fill_between(times, 0, confidences, alpha=0.3, color='gray')
        
        # Mark UP/DOWN predictions
        up_times = [t for t, tg in zip(times, targets) if tg == 'UP']
        down_times = [t for t, tg in zip(times, targets) if tg == 'DOWN']
        up_conf = [c for t, c in zip(targets, confidences) if t == 'UP']
        down_conf = [c for t, c in zip(targets, confidences) if t == 'DOWN']
        
        if up_times:
            ax8.scatter(up_times, up_conf, color='green', marker='^', s=30, label='UP', alpha=0.7, zorder=5)
        if down_times:
            ax8.scatter(down_times, down_conf, color='red', marker='v', s=30, label='DOWN', alpha=0.7, zorder=5)
        
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Confidence')
        ax8.set_title('Classification Results')
        ax8.set_ylim([0, 1.1])
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # ===== PLOT 9: Left/Right Scores =====
    ax9 = fig.add_subplot(gs[3, 1])
    if data['classifications'] is not None:
        times = [c['time'] for c in data['classifications']]
        left_scores = [c['left_score'] for c in data['classifications']]
        right_scores = [c['right_score'] for c in data['classifications']]
        
        ax9.plot(times, left_scores, 'g-', label='Left Score', linewidth=2, alpha=0.7)
        ax9.plot(times, right_scores, 'r-', label='Right Score', linewidth=2, alpha=0.7)
        ax9.fill_between(times, left_scores, right_scores, where=(np.array(left_scores) > np.array(right_scores)),
                         alpha=0.2, color='green', label='Left Dominant')
        ax9.fill_between(times, left_scores, right_scores, where=(np.array(left_scores) < np.array(right_scores)),
                         alpha=0.2, color='red', label='Right Dominant')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Probability')
        ax9.set_title('Left vs Right Scores')
        ax9.set_ylim([0, 1])
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    # ===== PLOT 10: Cursor Trail =====
    ax10 = fig.add_subplot(gs[3, 2])
    if data['trail'] is not None:
        trail = data['trail']
        if len(trail.shape) == 2 and trail.shape[1] == 2:
            times = trail[:, 0]
            pitches = trail[:, 1]
            ax10.plot(times, pitches, 'b-', linewidth=2, alpha=0.7)
            ax10.scatter(times[::10], pitches[::10], c=pitches[::10], cmap='viridis', s=20, alpha=0.6)
            ax10.set_xlabel('Time (s)')
            ax10.set_ylabel('Pitch (0-1)')
            ax10.set_title('Cursor Trail (Composition)')
            ax10.set_ylim([0, 1])
            ax10.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Motor Imagery BCI - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file = output_dir / "analysis_plots.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved comprehensive analysis to: {output_file}")
    
    # Also create a summary text report
    create_summary_report(data, preprocessed, classifier, output_dir)
    
    plt.close()


def create_summary_report(
    data: Dict,
    preprocessed: Dict,
    classifier: MotorImageryClassifier,
    output_dir: Path
) -> None:
    """Create a text summary report of the analysis."""
    report_file = output_dir / "analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MOTOR IMAGERY BCI - OFFLINE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        if 'metadata' in data:
            f.write("SESSION METADATA:\n")
            f.write("-" * 80 + "\n")
            for key, value in data['metadata'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Data summary
        f.write("DATA SUMMARY:\n")
        f.write("-" * 80 + "\n")
        if data['baseline'] is not None:
            f.write(f"  Baseline: {data['baseline'].shape[0]} samples, "
                   f"{data['baseline'].shape[0] / classifier.sample_rate:.2f}s\n")
        if data['eeg'] is not None:
            f.write(f"  Composition EEG: {data['eeg'].shape[0]} samples, "
                   f"{data['eeg'].shape[0] / classifier.sample_rate:.2f}s\n")
        if data['classifications'] is not None:
            f.write(f"  Classifications: {len(data['classifications'])} results\n")
        if data['trail'] is not None:
            f.write(f"  Cursor trail: {len(data['trail'])} points\n")
        f.write("\n")
        
        # Preprocessing summary
        f.write("PREPROCESSING SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Sample rate: {classifier.sample_rate} Hz\n")
        f.write(f"  Window size: {classifier.window_seconds}s ({int(classifier.window_seconds * classifier.sample_rate)} samples)\n")
        f.write(f"  Mu band: {classifier.mu_band[0]}-{classifier.mu_band[1]} Hz\n")
        f.write(f"  Beta band: {classifier.beta_band[0]}-{classifier.beta_band[1]} Hz\n")
        f.write(f"  Sensorimotor channels: {classifier.sensorimotor_channels} (C3, Cz, C4)\n")
        f.write(f"  CSP components: {classifier.n_csp_components}\n")
        f.write(f"  Baseline normalization: {'Yes' if classifier.has_baseline else 'No'}\n")
        f.write("\n")
        
        # Classification summary
        if data['classifications'] is not None:
            f.write("CLASSIFICATION SUMMARY:\n")
            f.write("-" * 80 + "\n")
            targets = [c['target'] for c in data['classifications']]
            confidences = [c['confidence'] for c in data['classifications']]
            left_scores = [c['left_score'] for c in data['classifications']]
            right_scores = [c['right_score'] for c in data['classifications']]
            
            up_count = sum(1 for t in targets if t == 'UP')
            down_count = sum(1 for t in targets if t == 'DOWN')
            none_count = sum(1 for t in targets if t == 'NONE')
            
            f.write(f"  Total classifications: {len(targets)}\n")
            f.write(f"  UP: {up_count} ({100*up_count/len(targets):.1f}%)\n")
            f.write(f"  DOWN: {down_count} ({100*down_count/len(targets):.1f}%)\n")
            f.write(f"  NONE: {none_count} ({100*none_count/len(targets):.1f}%)\n")
            f.write(f"  Mean confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}\n")
            f.write(f"  Mean left score: {np.mean(left_scores):.3f} ± {np.std(left_scores):.3f}\n")
            f.write(f"  Mean right score: {np.mean(right_scores):.3f} ± {np.std(right_scores):.3f}\n")
            f.write("\n")
        
        # Power analysis
        if 'c3_mu_power' in preprocessed and 'c4_mu_power' in preprocessed:
            f.write("POWER ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            c3_mu = preprocessed['c3_mu_power']
            c4_mu = preprocessed['c4_mu_power']
            c3_beta = preprocessed['c3_beta_power']
            c4_beta = preprocessed['c4_beta_power']
            
            f.write(f"  C3 mu power: {np.mean(c3_mu):.6f} ± {np.std(c3_mu):.6f}\n")
            f.write(f"  C4 mu power: {np.mean(c4_mu):.6f} ± {np.std(c4_mu):.6f}\n")
            f.write(f"  C3 beta power: {np.mean(c3_beta):.6f} ± {np.std(c3_beta):.6f}\n")
            f.write(f"  C4 beta power: {np.mean(c4_beta):.6f} ± {np.std(c4_beta):.6f}\n")
            
            mu_diff = np.mean(c3_mu) - np.mean(c4_mu)
            beta_diff = np.mean(c3_beta) - np.mean(c4_beta)
            f.write(f"  C3-C4 mu difference: {mu_diff:.6f} ({'C3 > C4' if mu_diff > 0 else 'C3 < C4'})\n")
            f.write(f"  C3-C4 beta difference: {beta_diff:.6f} ({'C3 > C4' if beta_diff > 0 else 'C3 < C4'})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"[REPORT] Saved analysis report to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline analysis of motor imagery BCI session data"
    )
    parser.add_argument(
        'session_dir',
        type=str,
        help='Path to session directory (e.g., motor_imagery_sessions/20260116_153022)'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=None,
        help='Path to baseline session directory (if different from session_dir)'
    )
    
    args = parser.parse_args()
    
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"[ERROR] Session directory not found: {session_dir}")
        return
    
    # Determine baseline directory
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
    else:
        # Look for baseline in parent directory (same timestamp prefix)
        session_name = session_dir.name
        parent_dir = session_dir.parent
        # Find baseline session (usually session_name - 1)
        baseline_candidates = list(parent_dir.glob(f"{session_name[:-1]}*"))
        baseline_dir = None
        for cand in baseline_candidates:
            if (cand / "baseline_data.npy").exists():
                baseline_dir = cand
                break
    
    print(f"[ANALYSIS] Session directory: {session_dir}")
    if baseline_dir:
        print(f"[ANALYSIS] Baseline directory: {baseline_dir}")
    else:
        print(f"[WARNING] No baseline directory found - analysis will proceed without baseline normalization")
    
    # Load data
    print("\n[ANALYSIS] Loading session data...")
    data = load_session_data(session_dir)
    
    if data['eeg'] is None:
        print("[ERROR] No EEG data found in session. Cannot perform analysis.")
        return
    
    # Initialize classifier (replicate live setup)
    print("\n[ANALYSIS] Initializing classifier...")
    classifier = MotorImageryClassifier(
        sample_rate=250.0,
        window_seconds=1.0,
        mu_band=(8.0, 13.0),
        beta_band=(13.0, 30.0)
    )
    
    # Load baseline if available
    if baseline_dir and (baseline_dir / "baseline_data.npy").exists():
        print("\n[ANALYSIS] Loading baseline data...")
        baseline_data = np.load(baseline_dir / "baseline_data.npy")
        success = classifier.capture_baseline(baseline_data)
        if success:
            print(f"[ANALYSIS] Baseline loaded: {baseline_data.shape[0]} samples")
        else:
            print("[WARNING] Failed to load baseline")
    elif data['baseline'] is not None:
        print("\n[ANALYSIS] Using baseline from same session...")
        success = classifier.capture_baseline(data['baseline'])
        if success:
            print(f"[ANALYSIS] Baseline loaded: {data['baseline'].shape[0]} samples")
    
    # Replicate preprocessing pipeline
    print("\n[ANALYSIS] Replicating preprocessing pipeline...")
    preprocessed = replicate_preprocessing_pipeline(
        data['eeg'],
        classifier,
        sample_rate=classifier.sample_rate
    )
    
    # Create visualizations
    print("\n[ANALYSIS] Creating visualizations...")
    plot_comprehensive_analysis(data, preprocessed, classifier, session_dir)
    
    print("\n[ANALYSIS] Analysis complete!")
    print(f"  - Plots saved to: {session_dir / 'analysis_plots.png'}")
    print(f"  - Report saved to: {session_dir / 'analysis_report.txt'}")


if __name__ == "__main__":
    main()
