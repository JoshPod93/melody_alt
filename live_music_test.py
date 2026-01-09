#!/usr/bin/env python
"""
LIVE BRAIN-CONTROLLED MUSIC TEST

Your occipital EEG (PO7, Oz, PO8) controls the pitch in real-time!
- More occipital activity = higher pitch
- Less activity = lower pitch

Try:
- Close your eyes (alpha waves = pitch changes)
- Open eyes and look at something flickering
- Relax vs concentrate
"""

import sys
import time
import numpy as np
from threading import Thread, Event

# Check imports
try:
    from pylsl import StreamInlet, resolve_streams
except ImportError:
    print("ERROR: pip install pylsl")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice")
    sys.exit(1)

# Audio settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

# Frequency range (Hz) - big range so you can hear the difference!
MIN_FREQ = 150   # Low pitch
MAX_FREQ = 800   # High pitch
CENTER_FREQ = 400

# Global state
current_freq = CENTER_FREQ
current_amplitude = 0.3
phase = 0.0
running = True

def audio_callback(outdata, frames, time_info, status):
    """Generate audio based on current brain-controlled frequency."""
    global phase, current_freq, current_amplitude
    
    t = (np.arange(frames) + phase) / SAMPLE_RATE
    
    # Generate sine wave at current frequency
    wave = current_amplitude * np.sin(2 * np.pi * current_freq * t)
    
    # Update phase for continuity
    phase += frames
    
    # Output stereo
    outdata[:, 0] = wave
    outdata[:, 1] = wave

def eeg_thread_func(stop_event):
    """Read EEG and update frequency."""
    global current_freq, current_amplitude, running
    
    # Find Unicorn stream
    print("Searching for Unicorn stream...")
    streams = resolve_streams(5.0)
    
    unicorn = None
    for s in streams:
        if s.channel_count() >= 8:
            unicorn = s
            break
    
    if not unicorn:
        print("ERROR: No Unicorn stream found!")
        running = False
        return
    
    print(f"Connected to: {unicorn.name()}")
    inlet = StreamInlet(unicorn)
    
    # Buffer for power calculation
    buffer_size = 64  # ~250ms at 250Hz
    occ_buffer = []
    
    # Baseline tracking
    baseline_samples = []
    baseline = None
    baseline_std = None
    
    print("\nCalibrating baseline (2 seconds)...")
    print("Keep your eyes OPEN and look at something static.\n")
    
    cal_start = time.time()
    while time.time() - cal_start < 2.0:
        sample, _ = inlet.pull_sample(timeout=0.1)
        if sample:
            # Occipital channels: 5, 6, 7 (PO7, Oz, PO8)
            occ_power = np.mean([abs(sample[5]), abs(sample[6]), abs(sample[7])])
            baseline_samples.append(occ_power)
    
    if baseline_samples:
        baseline = np.mean(baseline_samples)
        baseline_std = np.std(baseline_samples) + 1  # Avoid div by zero
        print(f"Baseline: {baseline:.1f} µV (std: {baseline_std:.1f})")
    else:
        baseline = 50
        baseline_std = 20
    
    print("\n" + "=" * 50)
    print("NOW PLAYING! Your brain controls the pitch!")
    print("=" * 50)
    print("\nTry:")
    print("  - CLOSE your eyes -> pitch should change (alpha waves)")
    print("  - OPEN eyes -> returns toward baseline")
    print("  - BLINK hard -> spike in pitch")
    print("  - RELAX vs CONCENTRATE")
    print("\nPress Ctrl+C to stop\n")
    
    last_print = time.time()
    
    while not stop_event.is_set() and running:
        sample, _ = inlet.pull_sample(timeout=0.1)
        
        if sample:
            # Get occipital power (channels 5, 6, 7)
            occ_power = np.mean([abs(sample[5]), abs(sample[6]), abs(sample[7])])
            
            # Add to buffer
            occ_buffer.append(occ_power)
            if len(occ_buffer) > buffer_size:
                occ_buffer.pop(0)
            
            # Calculate smoothed power
            smoothed_power = np.mean(occ_buffer)
            
            # Convert to z-score relative to baseline
            z_score = (smoothed_power - baseline) / baseline_std
            
            # Map z-score to frequency (big jumps!)
            # z=0 -> center, z=+2 -> max, z=-2 -> min
            freq_range = MAX_FREQ - MIN_FREQ
            freq_offset = (z_score / 3.0) * (freq_range / 2)  # ±3 std = full range
            
            new_freq = CENTER_FREQ + freq_offset
            new_freq = max(MIN_FREQ, min(MAX_FREQ, new_freq))
            
            # Smooth frequency changes
            current_freq = current_freq * 0.9 + new_freq * 0.1
            
            # Print status every 0.5 seconds
            if time.time() - last_print > 0.5:
                bar_len = int((current_freq - MIN_FREQ) / (MAX_FREQ - MIN_FREQ) * 30)
                bar = "#" * bar_len + "-" * (30 - bar_len)
                
                print(f"\r  Power: {smoothed_power:6.1f} µV | Z: {z_score:+5.2f} | Freq: {current_freq:5.0f} Hz [{bar}]", end="", flush=True)
                last_print = time.time()
    
    inlet.close_stream()

def main():
    global running
    
    print("=" * 60)
    print("  LIVE BRAIN-CONTROLLED MUSIC")
    print("  Your occipital EEG controls the pitch!")
    print("=" * 60)
    
    # Start audio stream
    print("\nStarting audio...")
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        blocksize=BUFFER_SIZE,
        callback=audio_callback
    )
    stream.start()
    print("Audio started!")
    
    # Start EEG thread
    stop_event = Event()
    eeg_thread = Thread(target=eeg_thread_func, args=(stop_event,))
    eeg_thread.start()
    
    # Wait for Ctrl+C
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    # Cleanup
    stop_event.set()
    eeg_thread.join(timeout=2.0)
    stream.stop()
    stream.close()
    
    print("Done!")

if __name__ == "__main__":
    main()
