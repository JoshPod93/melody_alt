#!/usr/bin/env python
"""
Live EEG Test - See your brain signals in real-time!

Shows big, obvious visual feedback from the Unicorn Black.
Focus on the occipital channels (PO7, Oz, PO8) for SSVEP.
"""

import sys
import time
import numpy as np

# Check for pylsl
try:
    from pylsl import StreamInlet, resolve_streams
except ImportError:
    print("ERROR: pylsl not installed. Run: pip install pylsl")
    sys.exit(1)

# Channel names
CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
OCCIPITAL = [5, 6, 7]  # PO7, Oz, PO8

def make_bar(value, width=50, max_val=100):
    """Create a text-based bar visualization."""
    # Clamp value
    value = max(-max_val, min(max_val, value))
    
    # Calculate bar position (center is middle)
    center = width // 2
    bar_len = int(abs(value) / max_val * center)
    
    # Build bar
    bar = [' '] * width
    bar[center] = '|'
    
    if value >= 0:
        for i in range(center + 1, min(center + 1 + bar_len, width)):
            bar[i] = '█'
    else:
        for i in range(max(center - bar_len, 0), center):
            bar[i] = '█'
    
    return ''.join(bar)

def main():
    print("=" * 60)
    print("  LIVE EEG TEST - Unicorn Black")
    print("=" * 60)
    
    # Find stream
    print("\nSearching for Unicorn stream...")
    streams = resolve_streams(5.0)
    
    unicorn_stream = None
    for s in streams:
        if s.channel_count() >= 8:
            unicorn_stream = s
            break
    
    if not unicorn_stream:
        print("ERROR: No Unicorn stream found!")
        print("Make sure Unicorn LSL is running and streaming.")
        sys.exit(1)
    
    print(f"Found: {unicorn_stream.name()} ({unicorn_stream.channel_count()} ch @ {unicorn_stream.nominal_srate()} Hz)")
    
    # Connect
    inlet = StreamInlet(unicorn_stream)
    print("\nConnected! Starting live display...")
    print("\n" + "-" * 60)
    print("TRY THESE:")
    print("  - Blink your eyes (big spike on frontal channels)")
    print("  - Clench your jaw (muscle artifact)")
    print("  - Close eyes and relax (alpha waves on occipital)")
    print("  - Move your head (accelerometer will go crazy)")
    print("-" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    time.sleep(1)
    
    # Buffer for smoothing
    buffer_size = 25  # 100ms at 250Hz
    buffers = [[] for _ in range(8)]
    
    try:
        while True:
            # Get sample
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            
            if sample:
                # Add to buffers (only EEG channels 0-7)
                for i in range(8):
                    buffers[i].append(sample[i])
                    if len(buffers[i]) > buffer_size:
                        buffers[i].pop(0)
                
                # Calculate RMS for each channel
                rms_values = []
                for i in range(8):
                    if buffers[i]:
                        rms = np.sqrt(np.mean(np.array(buffers[i])**2))
                        rms_values.append(rms)
                    else:
                        rms_values.append(0)
                
                # Clear screen and display
                print("\033[H\033[J", end="")  # Clear screen
                
                print("=" * 60)
                print("  LIVE EEG - Unicorn Black")
                print("  (Blink, clench jaw, close eyes to see changes)")
                print("=" * 60)
                print()
                
                # Display each channel
                max_val = max(rms_values) if max(rms_values) > 10 else 50
                
                for i in range(8):
                    ch_name = CHANNELS[i]
                    rms = rms_values[i]
                    
                    # Highlight occipital channels
                    if i in OCCIPITAL:
                        marker = " *** SSVEP"
                    else:
                        marker = ""
                    
                    bar = make_bar(rms, width=40, max_val=max_val)
                    print(f"  {ch_name:4s} [{bar}] {rms:6.1f} µV{marker}")
                
                print()
                print("-" * 60)
                
                # Show accelerometer if available (channels 8-10)
                if len(sample) > 10:
                    acc_x, acc_y, acc_z = sample[8], sample[9], sample[10]
                    print(f"  HEAD MOTION: X={acc_x:+.2f}g  Y={acc_y:+.2f}g  Z={acc_z:+.2f}g")
                
                # Show battery if available (channel 15)
                if len(sample) > 15:
                    battery = sample[15]
                    print(f"  BATTERY: {battery:.0f}%")
                
                print("-" * 60)
                print("\n  Press Ctrl+C to stop")
                
                time.sleep(0.05)  # ~20 FPS update
                
    except KeyboardInterrupt:
        print("\n\nStopped.")
        inlet.close_stream()

if __name__ == "__main__":
    main()
