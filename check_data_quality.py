#!/usr/bin/env python
"""Check data quality and identify clipping issues."""
import numpy as np
import argparse
from pathlib import Path


def find_most_recent_session() -> Path:
    """Find the most recent motor imagery session with EEG data."""
    base_dir = Path("motor_imagery_sessions")
    if not base_dir.exists():
        return None
    
    sessions = []
    for session_dir in sorted(base_dir.iterdir(), reverse=True):
        if session_dir.is_dir():
            eeg_file = session_dir / "eeg_data.npy"
            if eeg_file.exists():
                sessions.append(session_dir)
    
    return sessions[0] if sessions else None


def main():
    parser = argparse.ArgumentParser(
        description="Check EEG data quality and identify clipping issues"
    )
    parser.add_argument(
        'session_dir',
        type=str,
        nargs='?',
        default=None,
        help='Path to session directory (e.g., motor_imagery_sessions/20260116_153022). If not provided, uses most recent session.'
    )
    
    args = parser.parse_args()
    
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = find_most_recent_session()
        if session_dir is None:
            print("ERROR: No session directory provided and no recent sessions found.")
            print("Usage: python check_data_quality.py [session_dir]")
            return
        print(f"Using most recent session: {session_dir}")
    
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        return
    
    eeg_file = session_dir / "eeg_data.npy"
    if not eeg_file.exists():
        print(f"ERROR: eeg_data.npy not found in {session_dir}")
        return
    
    eeg_data = np.load(eeg_file)

    print("=" * 80)
    print("EEG DATA QUALITY CHECK")
    print(f"Session: {session_dir.name}")
    print("=" * 80)
    print()

    print(f"Data shape: {eeg_data.shape}")
    print(f"Expected: (samples, 8) for N seconds at 250Hz")
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
    print("DATA QUALITY SUMMARY:")
    print("=" * 80)
    
    clipped_total = clipped_low + clipped_high
    clipped_pct = 100 * clipped_total / total_samples
    
    if clipped_pct > 5:
        print(f"[WARNING] Significant clipping detected: {clipped_pct:.1f}% of data is clipped")
        print("This may indicate:")
        print("1. Artifact contamination")
        print("2. Incorrect normalization")
        print("3. Hardware issues")
    else:
        print(f"[OK] Minimal clipping: {clipped_pct:.2f}% of data is clipped")
    
    print()
    print("RECOMMENDATIONS:")
    if clipped_pct > 5:
        print("- Check for artifacts in the raw data")
        print("- Verify baseline normalization is working correctly")
        print("- Consider adjusting artifact detection thresholds")
    else:
        print("- Data quality appears acceptable")
        print("- Continue with analysis")
    print()


if __name__ == "__main__":
    main()
