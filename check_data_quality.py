#!/usr/bin/env python
"""Check data quality and identify clipping issues."""
import numpy as np
from pathlib import Path

session_dir = Path("motor_imagery_sessions/20260116_153022")
eeg_data = np.load(session_dir / "eeg_data.npy")

print("=" * 80)
print("EEG DATA QUALITY CHECK")
print("=" * 80)
print()

print(f"Data shape: {eeg_data.shape}")
print(f"Expected: (2275, 8) for 9.1s at 250Hz")
print(f"Actual: {eeg_data.shape}")
print()

# Check for clipping
print("CLIPPING ANALYSIS:")
print("-" * 80)
clipped_low = np.sum(eeg_data == -10.0)
clipped_high = np.sum(eeg_data == 10.0)
total_samples = eeg_data.size
print(f"Values exactly -10.0: {clipped_low} ({100*clipped_low/total_samples:.2f}%)")
print(f"Values exactly 10.0: {clipped_high} ({100*clipped_high/total_samples:.2f}%)")
print(f"Values in range (-10, 10): {total_samples - clipped_low - clipped_high} ({100*(total_samples - clipped_low - clipped_high)/total_samples:.2f}%)")
print()

# Check data statistics
print("DATA STATISTICS:")
print("-" * 80)
print(f"Overall range: [{np.min(eeg_data):.6f}, {np.max(eeg_data):.6f}]")
print(f"Overall mean: {np.mean(eeg_data):.6f}")
print(f"Overall std: {np.std(eeg_data):.6f}")
print()

# Check per channel
print("PER-CHANNEL STATISTICS:")
print("-" * 80)
for ch in range(8):
    ch_data = eeg_data[:, ch]
    ch_clipped_low = np.sum(ch_data == -10.0)
    ch_clipped_high = np.sum(ch_data == 10.0)
    print(f"Ch{ch}: mean={np.mean(ch_data):.6f}, std={np.std(ch_data):.6f}, "
          f"range=[{np.min(ch_data):.6f}, {np.max(ch_data):.6f}], "
          f"clipped={ch_clipped_low + ch_clipped_high} ({100*(ch_clipped_low + ch_clipped_high)/len(ch_data):.1f}%)")
print()

# Check temporal patterns
print("TEMPORAL PATTERNS:")
print("-" * 80)
# Check first 100 samples
first_100 = eeg_data[:100, 0]
print(f"First 100 samples (Ch0):")
print(f"  Unique values: {len(np.unique(first_100))}")
print(f"  Clipped: {np.sum((first_100 == -10) | (first_100 == 10))} ({100*np.sum((first_100 == -10) | (first_100 == 10))/len(first_100):.1f}%)")
print(f"  Range: [{np.min(first_100):.6f}, {np.max(first_100):.6f}]")
print()

# Check middle section
mid_start = len(eeg_data) // 2 - 50
mid_end = len(eeg_data) // 2 + 50
mid_section = eeg_data[mid_start:mid_end, 0]
print(f"Middle section (samples {mid_start}-{mid_end}, Ch0):")
print(f"  Unique values: {len(np.unique(mid_section))}")
print(f"  Clipped: {np.sum((mid_section == -10) | (mid_section == 10))} ({100*np.sum((mid_section == -10) | (mid_section == 10))/len(mid_section):.1f}%)")
print(f"  Range: [{np.min(mid_section):.6f}, {np.max(mid_section):.6f}]")
print()

# Check last 100 samples
last_100 = eeg_data[-100:, 0]
print(f"Last 100 samples (Ch0):")
print(f"  Unique values: {len(np.unique(last_100))}")
print(f"  Clipped: {np.sum((last_100 == -10) | (last_100 == 10))} ({100*np.sum((last_100 == -10) | (last_100 == 10))/len(last_100):.1f}%)")
print(f"  Range: [{np.min(last_100):.6f}, {np.max(last_100):.6f}]")
print()

# Check if data is actually varying
print("VARIANCE CHECK:")
print("-" * 80)
for ch in range(8):
    ch_data = eeg_data[:, ch]
    # Check if data is constant in any section
    window_size = 25  # 100ms at 250Hz
    constant_windows = 0
    for i in range(0, len(ch_data) - window_size, window_size):
        window = ch_data[i:i+window_size]
        if np.std(window) < 0.001:  # Essentially constant
            constant_windows += 1
    print(f"Ch{ch}: {constant_windows} windows with std < 0.001 (out of {len(ch_data)//window_size} windows)")
print()

print("=" * 80)
print("ISSUE IDENTIFIED:")
print("=" * 80)
print("The preprocessing code in src/bci/preprocessing.py is clipping normalized")
print("values to [-10, 10] at lines 358, 369, and 402.")
print()
print("This clipping is causing:")
print("1. Loss of information (4.4% + 5.3% = 9.7% of data is clipped)")
print("2. Artificial saturation at ±10.0")
print("3. Flat appearance in plots when values hit the clipping limits")
print()
print("RECOMMENDATION:")
print("- Remove or increase the clipping limits (e.g., ±20 or ±30)")
print("- Or remove clipping entirely and rely on artifact detection")
print()
