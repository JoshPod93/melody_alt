"""
Lightweight Unicorn Black system testing via LSL.
Connects to existing UnicornLSL.exe stream for validation.
"""

import os
import time
import numpy as np
from typing import Optional, Dict

# Suppress LSL verbose logging
os.environ['LSL_LOG_LEVEL'] = 'ERROR'

try:
    from pylsl import StreamInlet, resolve_streams
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

SAMPLE_RATE = 250
N_EEG_CHANNELS = 8
N_TOTAL_CHANNELS = 17
EEG_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
BATTERY_CHANNEL = 15


def find_stream(timeout: float = 5.0) -> Optional[StreamInlet]:
    """Find and connect to Unicorn LSL stream."""
    if not LSL_AVAILABLE:
        return None
    streams = resolve_streams(timeout)
    for stream in streams:
        if stream.channel_count() >= 8:
            return StreamInlet(stream)
    return None


def verify_lsl_connection(timeout: float = 5.0) -> Dict:
    """Verify LSL stream is available."""
    if not LSL_AVAILABLE:
        return {'status': 'error', 'message': 'pylsl not available'}
    streams = resolve_streams(timeout)
    for s in streams:
        if s.channel_count() >= 8:
            return {
                'status': 'success',
                'stream_found': True,
                'name': str(s.name()),
                'channels': int(s.channel_count()),
                'rate': float(s.nominal_srate())
            }
    return {
        'status': 'not_found',
        'stream_found': False,
        'available': [s.name() for s in streams]
    }


def check_electrodes(inlet: StreamInlet, duration: float = 2.0) -> Dict:
    """Check all electrodes are working."""
    target_samples = int(SAMPLE_RATE * duration)
    samples = []
    start = time.time()
    
    while len(samples) < target_samples and (time.time() - start) < (duration + 0.5):
        chunk, timestamps = inlet.pull_chunk(timeout=0.1)
        if chunk:
            for sample in chunk:
                samples.append(sample[:N_EEG_CHANNELS])
                if len(samples) >= target_samples:
                    break
    
    if not samples:
        return {'status': 'error', 'message': 'No data received'}
    
    data = np.array(samples)
    working = []
    not_working = []
    
    for i, name in enumerate(EEG_CHANNELS):
        ch_data = data[:, i]
        std = np.std(ch_data)
        rng = np.max(ch_data) - np.min(ch_data)
        if std < 0.1 or rng < 1.0:
            not_working.append(name)
        else:
            working.append(name)
    
    return {
        'status': 'success',
        'working': working,
        'not_working': not_working,
        'samples': int(len(samples))
    }


def check_bandwidth(inlet: StreamInlet, duration: float = 5.0) -> Dict:
    """Check data throughput."""
    count = 0
    start = time.time()
    while time.time() - start < duration:
        chunk, _ = inlet.pull_chunk(timeout=0.1)
        if chunk:
            count += len(chunk)
    
    elapsed = time.time() - start
    rate = count / elapsed if elapsed > 0 else 0
    
    return {
        'status': 'success',
        'expected_rate': int(SAMPLE_RATE),
        'actual_rate': float(round(rate, 2)),
        'meets_spec': bool(rate >= SAMPLE_RATE * 0.95),
        'samples': int(count)
    }


def check_battery(inlet: StreamInlet) -> Dict:
    """Check battery level by searching all channels for battery-like values."""
    samples = []
    for _ in range(5):
        chunk, _ = inlet.pull_chunk(timeout=0.5)
        if chunk:
            samples.extend(chunk)
            if len(samples) >= 30:
                break
    
    if not samples:
        return {'status': 'unavailable', 'message': 'No data received'}
    
    data = np.array(samples)
    n_channels = len(samples[0]) if samples else 0
    
    # Check all channels for battery-like characteristics
    candidates = []
    for ch_idx in range(n_channels):
        ch_data = data[:, ch_idx]
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        abs_mean = abs(mean_val)
        
        # Battery: 0-100 range, very stable (std < 2), not in EEG range
        if 0 <= mean_val <= 100 and std_val < 2.0 and abs_mean > 0.1:
            candidates.append((ch_idx, mean_val, std_val))
    
    # Also check expected battery channel (15) and nearby channels
    check_channels = [BATTERY_CHANNEL, 14, 16] if n_channels > BATTERY_CHANNEL else list(range(n_channels))
    for ch_idx in check_channels:
        if ch_idx < n_channels:
            ch_data = data[:, ch_idx]
            mean_val = np.mean(ch_data)
            std_val = np.std(ch_data)
            if 0 <= mean_val <= 100 and std_val < 10.0:
                candidates.append((ch_idx, mean_val, std_val))
    
    if not candidates:
        return {'status': 'unavailable', 'message': 'Battery channel not found in stream'}
    
    # Select most stable candidate
    candidates.sort(key=lambda x: x[2])  # Sort by std (most stable first)
    battery_channel, battery_val, _ = candidates[0]
    
    return {
        'status': 'success',
        'battery_percent': float(round(battery_val, 1)),
        'is_ok': bool(battery_val > 20.0),
        'channel': int(battery_channel)
    }


def check_impedance(inlet: StreamInlet, duration: float = 2.0) -> Dict:
    """Estimate electrode impedance from signal quality metrics."""
    target_samples = int(SAMPLE_RATE * duration)
    samples = []
    start = time.time()
    
    while len(samples) < target_samples and (time.time() - start) < (duration + 0.5):
        chunk, _ = inlet.pull_chunk(timeout=0.1)
        if chunk:
            for sample in chunk:
                samples.append(sample[:N_EEG_CHANNELS])
                if len(samples) >= target_samples:
                    break
    
    if not samples:
        return {'status': 'error', 'message': 'No data received'}
    
    data = np.array(samples)
    impedance_results = {}
    
    for i, name in enumerate(EEG_CHANNELS):
        ch_data = data[:, i]
        std_val = np.std(ch_data)
        peak_to_peak = np.max(ch_data) - np.min(ch_data)
        rms = np.sqrt(np.mean(ch_data ** 2))
        
        # Estimate impedance from signal characteristics
        # Low std + low range = poor contact (high impedance)
        # Very high std = excessive noise (poor contact)
        # Moderate std + good range = good contact (low impedance)
        
        if std_val < 0.5 or peak_to_peak < 2.0:
            quality = 'poor'
            estimated_kohm = 75.0
            status = 'high'
        elif std_val > 100 or peak_to_peak > 500:
            quality = 'poor'
            estimated_kohm = 60.0
            status = 'high'
        elif std_val < 5.0:
            quality = 'acceptable'
            estimated_kohm = 30.0
            status = 'moderate'
        elif std_val < 15.0:
            quality = 'good'
            estimated_kohm = 8.0
            status = 'low'
        else:
            quality = 'good'
            estimated_kohm = 5.0
            status = 'low'
        
        impedance_results[name] = {
            'quality': quality,
            'impedance_kohm': float(round(estimated_kohm, 1)),
            'status': status,
            'std': float(round(std_val, 2)),
            'peak_to_peak': float(round(peak_to_peak, 2))
        }
    
    counts = {q: int(sum(1 for r in impedance_results.values() if r['quality'] == q)) 
              for q in ['good', 'acceptable', 'poor']}
    
    return {
        'status': 'success',
        'channels': impedance_results,
        'summary': {**counts, 'total': int(len(EEG_CHANNELS))}
    }


def capture_data(inlet: StreamInlet, duration: float = 2.0) -> Dict:
    """Capture EEG data - wait for full duration and use uniform time axis."""
    target_samples = int(SAMPLE_RATE * duration)
    all_samples = []
    start_wall_time = time.time()
    end_wall_time = start_wall_time + duration
    
    # Wait for the full duration, pulling chunks continuously
    while time.time() < end_wall_time:
        chunk, _ = inlet.pull_chunk(timeout=0.1)
        if chunk and len(chunk) > 0:
            chunk_array = np.array(chunk)
            # Extract EEG channels only (first 8)
            eeg_chunk = chunk_array[:, :N_EEG_CHANNELS]
            all_samples.append(eeg_chunk)
    
    if not all_samples:
        return {'status': 'error', 'message': 'No data captured'}
    
    # Concatenate all chunks
    data = np.vstack(all_samples)
    
    # Trim to exact target samples (in case we got more)
    if len(data) > target_samples:
        data = data[:target_samples]
    
    # Generate uniform time axis based on sample rate (most reliable for plotting)
    # This ensures smooth, continuous plots regardless of LSL timestamp issues
    time_axis = np.arange(len(data)) / SAMPLE_RATE
    
    return {
        'status': 'success',
        'data': data,
        'time': time_axis,
        'channels': EEG_CHANNELS,
        'samples': int(len(data))
    }
