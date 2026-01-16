#!/usr/bin/env python
"""
Replay recorded EEG data through LSL stream for testing.

This script loads saved EEG session data and streams it through LSL,
allowing you to test the BCI system without the actual hardware.

Usage:
    python replay_eeg_lsl.py --composition --loop
    python replay_eeg_lsl.py --composition --selection 20260116_153022 --loop
    python replay_eeg_lsl.py --composition --loop --count 160  # Loop 160 times
    
Options:
    --composition: Use composition session data (default: baseline)
    --selection <N>: Select session by index or timestamp pattern (optional)
    --loop: Loop the data infinitely (or use --count for finite loops)
    --count <N>: Number of times to loop (requires --loop)
    --session-dir <path>: Direct path to session directory
"""

import sys
import argparse
import numpy as np
import time
from pathlib import Path
from typing import Optional

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    print("ERROR: pylsl not installed. Install with: pip install pylsl")
    sys.exit(1)


def find_session_dirs(session_type: str = "composition") -> list:
    """Find available session directories.
    
    Supports both:
    1. New motor imagery structure: motor_imagery_sessions/TIMESTAMP/eeg_data.npy
    2. Old P300 structure: melody_outputs/composition_data/raw_eeg_selection_{id}.npy
    """
    sessions = []
    
    # Try new motor imagery structure first
    base_dir = Path("motor_imagery_sessions")
    if base_dir.exists():
        for session_dir in sorted(base_dir.iterdir()):
            if session_dir.is_dir():
                eeg_file = session_dir / "eeg_data.npy"
                baseline_file = session_dir / "baseline_data.npy"
                
                if session_type == "composition" and eeg_file.exists():
                    sessions.append(("motor_imagery", session_dir))
                elif session_type == "baseline" and baseline_file.exists():
                    sessions.append(("motor_imagery", session_dir))
    
    # Try old P300 structure
    p300_data_dir = Path("melody_outputs")
    if p300_data_dir.exists():
        comp_dir = p300_data_dir / "composition_data"
        cal_dir = p300_data_dir / "calibration_data"
        
        if session_type == "composition" and comp_dir.exists():
            # Find all selection files
            for sel_file in sorted(comp_dir.glob("raw_eeg_selection_*.npy")):
                # Extract selection ID from filename
                sel_id = sel_file.stem.replace("raw_eeg_selection_", "")
                sessions.append(("p300_composition", sel_file, sel_id))
        
        elif session_type == "baseline" and cal_dir.exists():
            # Find all trial files
            for trial_file in sorted(cal_dir.glob("raw_eeg_trial_*.npy")):
                # Extract trial number from filename
                trial_num = trial_file.stem.replace("raw_eeg_trial_", "")
                sessions.append(("p300_calibration", trial_file, trial_num))
    
    return sessions


def load_session_data(session_info, session_type: str = "composition") -> Optional[tuple]:
    """Load EEG data from session.
    
    Args:
        session_info: Tuple of (type, path, [id]) where:
            - type: "motor_imagery" or "p300_composition" or "p300_calibration"
            - path: Path to directory (motor_imagery) or file (p300)
            - id: Selection/trial ID for P300 (optional)
    """
    data_type, path_info = session_info[0], session_info[1]
    
    if data_type == "motor_imagery":
        # New motor imagery structure
        session_dir = path_info
        if session_type == "composition":
            data_file = session_dir / "eeg_data.npy"
        else:  # baseline
            data_file = session_dir / "baseline_data.npy"
        
        if not data_file.exists():
            print(f"ERROR: {data_file} not found")
            return None
        
        data = np.load(data_file)
        
        # Load metadata if available
        metadata_file = session_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Get sample rate from metadata or default
        # Motor imagery data is recorded at 250Hz, then downsampled to 64Hz
        if 'sample_rate' in metadata:
            sample_rate = float(metadata['sample_rate'])
        elif 'duration' in metadata and metadata['duration'] > 0:
            # Calculate from duration, but round to nearest standard rate
            calculated_rate = data.shape[0] / float(metadata['duration'])
            # Round to nearest standard rate (64Hz or 250Hz)
            if abs(calculated_rate - 64.0) < abs(calculated_rate - 250.0):
                sample_rate = 64.0
            else:
                sample_rate = 250.0
            # Note if there's a significant difference
            if abs(calculated_rate - sample_rate) > 2.0:
                print(f"NOTE: Calculated {calculated_rate:.1f}Hz from duration, using {sample_rate}Hz (standard rate)")
        else:
            # Estimate based on data size
            # At 250Hz: 10s = 2500 samples, at 64Hz: 10s = 640 samples
            # 2275 samples in ~10s suggests it's close to 250Hz (before downsampling)
            # But since we downsample to 64Hz, check if it's already downsampled
            estimated_duration_250 = data.shape[0] / 250.0
            estimated_duration_64 = data.shape[0] / 64.0
            
            # If data looks like it's at 64Hz (fewer samples), use 64Hz
            # Otherwise use 250Hz (original recording rate)
            if estimated_duration_64 < 1.0 or data.shape[0] < 1000:
                sample_rate = 64.0
            else:
                sample_rate = 250.0
            print(f"WARNING: Sample rate not in metadata, assuming {sample_rate}Hz based on data size")
        
        print(f"Loaded {data.shape[0]} samples, {data.shape[1]} channels from {session_dir.name}")
        
    elif data_type in ["p300_composition", "p300_calibration"]:
        # Old P300 structure - path_info is the file path
        data_file = path_info
        if not data_file.exists():
            print(f"ERROR: {data_file} not found")
            return None
        
        data = np.load(data_file)
        
        # Ensure shape is (samples, channels)
        if len(data.shape) == 2 and data.shape[1] != 8:
            if data.shape[0] == 8:
                data = data.T
        
        # P300 data was recorded at 250Hz
        sample_rate = 250.0
        metadata = {}
        
        sel_id = session_info[2] if len(session_info) > 2 else "unknown"
        print(f"Loaded P300 {session_type} data (selection/trial {sel_id}): {data.shape[0]} samples, {data.shape[1]} channels")
    
    else:
        print(f"ERROR: Unknown session type: {data_type}")
        return None
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {data.shape[0] / sample_rate:.2f} seconds")
    
    return data, sample_rate, metadata


def stream_data(data: np.ndarray, sample_rate: float, loop: bool = False, loop_count: Optional[int] = None):
    """Stream EEG data through LSL.
    
    Args:
        data: EEG data array (n_samples, n_channels)
        sample_rate: Sample rate in Hz
        loop: If True, loop infinitely (or loop_count times if specified)
        loop_count: Number of times to loop (None = infinite if loop=True)
    """
    n_samples, n_channels = data.shape
    
    # Create LSL stream info
    stream_name = "ReplayEEG"
    stream_type = "EEG"
    
    info = StreamInfo(
        name=stream_name,
        type=stream_type,
        channel_count=n_channels,
        nominal_srate=sample_rate,
        channel_format='float32',
        source_id='replay-eeg'
    )
    
    # Add channel labels
    channel_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    channels = info.desc().append_child("channels")
    for i, ch_name in enumerate(channel_names[:n_channels]):
        ch = channels.append_child("channel")
        ch.append_child_value("label", ch_name)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    
    # Create outlet
    outlet = StreamOutlet(info)
    
    if loop:
        if loop_count:
            print(f"\nStreaming '{stream_name}' at {sample_rate} Hz (looping {loop_count} times)...")
        else:
            print(f"\nStreaming '{stream_name}' at {sample_rate} Hz (looping infinitely)...")
    else:
        print(f"\nStreaming '{stream_name}' at {sample_rate} Hz (single pass)...")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        sample_interval = 1.0 / sample_rate
        iteration = 0
        
        while True:
            for i in range(n_samples):
                # Get sample (convert to list for LSL)
                sample = data[i, :].astype(np.float32).tolist()
                
                # Push sample
                outlet.push_sample(sample)
                
                # Sleep to maintain sample rate
                time.sleep(sample_interval)
            
            if not loop:
                break
            
            iteration += 1
            if loop_count and iteration >= loop_count:
                print(f"\nCompleted {loop_count} loops")
                break
            
            if loop_count:
                print(f"Loop {iteration}/{loop_count}...")
            else:
                print(f"Loop {iteration}...")
    
    except KeyboardInterrupt:
        print(f"\n\nStopped streaming (after {iteration} loop(s))")
    
    finally:
        del outlet
        print("Stream closed")


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded EEG data through LSL stream"
    )
    parser.add_argument(
        '--composition',
        action='store_true',
        help='Use composition session data (default: baseline)'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Use baseline session data'
    )
    parser.add_argument(
        '--selection',
        type=str,
        default=None,
        help='Select session by index (0-based) or timestamp pattern (e.g., "160" or "20260116_153022")'
    )
    parser.add_argument(
        '--session-dir',
        type=str,
        default=None,
        help='Direct path to session directory'
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Loop the data continuously (infinitely, or use --count for finite loops)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Number of times to loop (requires --loop). If not specified with --loop, loops infinitely.'
    )
    
    args = parser.parse_args()
    
    # Determine session type
    session_type = "composition" if args.composition else "baseline"
    
    # Find or use specified session
    if args.session_dir:
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            print(f"ERROR: Session directory not found: {session_dir}")
            sys.exit(1)
    else:
        # Find available sessions
        sessions = find_session_dirs(session_type)
        
        if not sessions:
            print(f"ERROR: No {session_type} sessions found in motor_imagery_sessions/")
            print(f"Make sure you have session directories with eeg_data.npy or baseline_data.npy files")
            sys.exit(1)
        
        # Debug: print found sessions
        if args.selection:
            print(f"Found {len(sessions)} {session_type} session(s)")
            for i, s in enumerate(sessions):
                print(f"  {i}: {s.name}")
        
        # Select session
        if args.selection:
            selection = str(args.selection)
            
            # First, try to match P300 selection/trial ID (for old structure)
            p300_matched = [s for s in sessions if len(s) > 2 and str(s[2]) == selection]
            if p300_matched:
                session_info = p300_matched[0]
                print(f"Matched P300 selection/trial: {selection}")
            else:
                # Try to match by timestamp pattern (for motor imagery)
                motor_matched = [s for s in sessions if s[0] == "motor_imagery" and selection in str(s[1].name)]
                if motor_matched:
                    session_info = motor_matched[0]
                    print(f"Matched motor imagery session: {session_info[1].name}")
                else:
                    # Try as index
                    try:
                        idx = int(selection)
                        if 0 <= idx < len(sessions):
                            session_info = sessions[idx]
                            if session_info[0] == "motor_imagery":
                                print(f"Selected motor imagery session by index {idx}: {session_info[1].name}")
                            else:
                                sel_id = session_info[2] if len(session_info) > 2 else "unknown"
                                print(f"Selected P300 session by index {idx}: {sel_id}")
                        else:
                            print(f"ERROR: Selection index {idx} out of range (0-{len(sessions)-1})")
                            print(f"Available sessions:")
                            for i, s in enumerate(sessions):
                                if s[0] == "motor_imagery":
                                    print(f"  {i}: {s[1].name} (motor imagery)")
                                else:
                                    sel_id = s[2] if len(s) > 2 else "unknown"
                                    print(f"  {i}: {sel_id} (P300 {s[0]})")
                            sys.exit(1)
                    except ValueError:
                        print(f"ERROR: Could not find session matching '{selection}'")
                        print(f"Available sessions:")
                        for i, s in enumerate(sessions):
                            if s[0] == "motor_imagery":
                                print(f"  {i}: {s[1].name} (motor imagery)")
                            else:
                                sel_id = s[2] if len(s) > 2 else "unknown"
                                print(f"  {i}: {sel_id} (P300 {s[0]})")
                        sys.exit(1)
        else:
            # Use most recent
            session_info = sessions[-1]
            if session_info[0] == "motor_imagery":
                print(f"Using most recent motor imagery session: {session_info[1].name}")
            else:
                sel_id = session_info[2] if len(session_info) > 2 else "unknown"
                print(f"Using most recent P300 session: {sel_id}")
    
    # Load data
    result = load_session_data(session_info, session_type)
    if result is None:
        sys.exit(1)
    
    data, sample_rate, metadata = result
    
    # Validate loop arguments
    if args.count is not None and not args.loop:
        print("WARNING: --count specified without --loop. Ignoring --count.")
        args.count = None
    
    # Stream data
    stream_data(data, sample_rate, loop=args.loop, loop_count=args.count)


if __name__ == "__main__":
    main()
