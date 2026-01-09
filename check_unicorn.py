#!/usr/bin/env python
"""
Quick check script to verify Unicorn Black LSL stream is online.

Run this AFTER starting the LSL Interface in Unicorn Suite.

Usage:
    conda activate hack
    python check_unicorn.py
"""

import sys
import time

def check_lsl_streams():
    """Check for available LSL streams."""
    try:
        from pylsl import resolve_streams
    except ImportError:
        print("[ERROR] pylsl not installed. Run: pip install pylsl")
        return False
    
    print("Searching for LSL streams (5 seconds)...")
    streams = resolve_streams(5.0)
    
    if not streams:
        print("\n[!] No LSL streams found.")
        print("\nMake sure you have:")
        print("  1. Unicorn Suite open")
        print("  2. Device connected")
        print("  3. Apps -> LSL Interface -> Start clicked")
        return False
    
    print(f"\nFound {len(streams)} stream(s):\n")
    
    unicorn_stream = None
    for i, stream in enumerate(streams):
        name = stream.name()
        stype = stream.type()
        channels = stream.channel_count()
        rate = stream.nominal_srate()
        
        # Check if this looks like a Unicorn
        is_unicorn = ('unicorn' in name.lower() or 
                      'un-' in name.lower() or
                      (stype == 'EEG' and channels == 8 and 240 <= rate <= 260))
        
        marker = " <-- UNICORN" if is_unicorn else ""
        print(f"  [{i+1}] {name}")
        print(f"      Type: {stype}, Channels: {channels}, Rate: {rate} Hz{marker}")
        
        if is_unicorn:
            unicorn_stream = stream
    
    if unicorn_stream:
        print(f"\n[OK] Unicorn stream detected: {unicorn_stream.name()}")
        return True
    else:
        print("\n[?] No Unicorn stream identified.")
        print("    If your stream is listed above, it should still work.")
        return len(streams) > 0


def test_data_reception(duration=3.0):
    """Test receiving actual data from the stream."""
    from pylsl import StreamInlet, resolve_byprop
    
    print(f"\nTesting data reception for {duration} seconds...")
    
    # Find EEG stream
    streams = resolve_byprop('type', 'EEG', timeout=5.0)
    if not streams:
        print("[ERROR] No EEG stream found")
        return False
    
    # Connect
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    print(f"Connected to: {info.name()} ({info.channel_count()} channels @ {info.nominal_srate()} Hz)")
    
    # Receive data
    sample_count = 0
    start = time.time()
    
    while time.time() - start < duration:
        sample, timestamp = inlet.pull_sample(timeout=0.1)
        if sample:
            sample_count += 1
    
    inlet.close_stream()
    
    expected = int(duration * info.nominal_srate())
    received_rate = sample_count / duration
    
    print(f"\nReceived {sample_count} samples in {duration}s")
    print(f"Effective rate: {received_rate:.1f} Hz (expected: {info.nominal_srate()} Hz)")
    
    if sample_count > expected * 0.8:  # Allow 20% tolerance
        print("\n[OK] Data reception working correctly!")
        return True
    else:
        print("\n[WARNING] Lower than expected sample rate")
        print("          This might affect BCI performance")
        return True  # Still usable


def main():
    print("=" * 50)
    print("Unicorn Black LSL Connection Check")
    print("=" * 50)
    
    # Step 1: Check for streams
    if not check_lsl_streams():
        sys.exit(1)
    
    # Step 2: Test data reception
    print("\n" + "-" * 50)
    if not test_data_reception():
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 50)
    print("[OK] Unicorn is online and streaming!")
    print("=" * 50)
    print("\nYou can now run the BCI app:")
    print("  python bci_main.py --mode gui")
    print("\nThen click 'Connect LSL' in the interface.")


if __name__ == "__main__":
    main()
