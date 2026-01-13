"""
Unicorn Black System Test
Connects to existing UnicornLSL.exe stream for validation.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress LSL verbose logging by redirecting stderr temporarily
_original_stderr = sys.stderr
_suppress_lsl_logs = True

class LSLFilter:
    def write(self, text):
        if _suppress_lsl_logs and ('INFO|' in text or 'netif' in text or 'api_config' in text or 'common.cpp' in text):
            return
        _original_stderr.write(text)
    def flush(self):
        _original_stderr.flush()

sys.stderr = LSLFilter()

from unicorn_functions import (
    find_stream,
    verify_lsl_connection,
    check_electrodes,
    check_bandwidth,
    check_battery,
    check_impedance,
    capture_data
)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_test():
    # Clear reports folder at start
    report_dir = Path(__file__).parent / "reports"
    if report_dir.exists():
        for file in report_dir.iterdir():
            if file.is_file():
                file.unlink()
    
    print("=" * 60)
    print("  UNICORN BLACK SYSTEM TEST")
    print("=" * 60)
    print("\nConnecting to Unicorn LSL stream...")
    print("(Please ensure UnicornLSL.exe is running)\n")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # 1. Verify LSL connection
    print_section("1. LSL Stream Verification")
    lsl_check = verify_lsl_connection(timeout=5.0)
    if not lsl_check.get('stream_found'):
        print("ERROR: No Unicorn LSL stream found")
        print("Ensure UnicornLSL.exe is running and streaming")
        if lsl_check.get('available'):
            print(f"Available streams: {lsl_check['available']}")
        report['tests']['lsl_connection'] = {'status': 'failed', 'error': 'No stream found'}
        return save_report(report)
    
    print(f"Stream: {lsl_check['name']}")
    print(f"Channels: {lsl_check['channels']}")
    print(f"Rate: {lsl_check['rate']} Hz")
    report['tests']['lsl_connection'] = {
        'status': 'passed',
        'stream_name': lsl_check['name'],
        'channels': lsl_check['channels'],
        'sample_rate': lsl_check['rate']
    }
    
    # Connect to stream once and reuse
    inlet = find_stream()
    if not inlet:
        print("ERROR: Failed to connect to stream")
        report['tests']['connection'] = {'status': 'failed', 'error': 'Connection failed'}
        return save_report(report)
    
    # 2. Check electrodes
    print_section("2. Electrode Validation")
    electrode_result = check_electrodes(inlet, duration=2.0)
    if electrode_result.get('status') == 'success':
        working = electrode_result.get('working', [])
        not_working = electrode_result.get('not_working', [])
        total = len(working) + len(not_working)
        print(f"Working: {len(working)}/{total}")
        if working:
            print(f"  {', '.join(working)}")
        if not_working:
            print(f"WARNING - Non-functional: {', '.join(not_working)}")
        report['tests']['electrodes'] = {
            'status': 'passed' if len(not_working) == 0 else 'warning',
            'working': working,
            'not_working': not_working,
            'working_count': len(working),
            'total_count': total
        }
    else:
        print(f"ERROR: {electrode_result.get('message')}")
        report['tests']['electrodes'] = {'status': 'failed', 'error': electrode_result.get('message')}
    
    # 3. Check bandwidth
    print_section("3. Bandwidth Check")
    bandwidth_result = check_bandwidth(inlet, duration=5.0)
    if bandwidth_result.get('status') == 'success':
        print(f"Expected: {bandwidth_result['expected_rate']} Hz")
        print(f"Actual: {bandwidth_result['actual_rate']} Hz")
        if bandwidth_result.get('meets_spec'):
            print("PASS: Meets specification")
        else:
            print("WARNING: Below expected rate")
        report['tests']['bandwidth'] = {
            'status': 'passed' if bandwidth_result.get('meets_spec') else 'warning',
            'expected_rate': bandwidth_result['expected_rate'],
            'actual_rate': bandwidth_result['actual_rate'],
            'meets_specification': bandwidth_result.get('meets_spec')
        }
    else:
        print(f"ERROR: {bandwidth_result.get('message')}")
        report['tests']['bandwidth'] = {'status': 'failed', 'error': bandwidth_result.get('message')}
    
    # 4. Check battery
    print_section("4. Battery Status")
    battery_result = check_battery(inlet)
    if battery_result.get('status') == 'success':
        level = battery_result.get('battery_percent', 0)
        is_ok = battery_result.get('is_ok', False)
        channel = battery_result.get('channel', 'unknown')
        print(f"Battery: {level}% (channel {channel})")
        if is_ok:
            print("Status: OK")
        else:
            print("WARNING: Battery low")
        report['tests']['battery'] = {
            'status': 'passed' if is_ok else 'warning',
            'battery_percent': level,
            'is_ok': is_ok,
            'channel': channel
        }
    elif battery_result.get('status') == 'unavailable':
        print(f"NOTE: {battery_result.get('message')}")
        report['tests']['battery'] = {'status': 'unavailable', 'message': battery_result.get('message')}
    else:
        print(f"ERROR: {battery_result.get('message')}")
        report['tests']['battery'] = {'status': 'failed', 'error': battery_result.get('message')}
    
    # 5. Check impedance
    print_section("5. Electrode Impedance")
    impedance_result = check_impedance(inlet, duration=2.0)
    if impedance_result.get('status') == 'success':
        summary = impedance_result.get('summary', {})
        print(f"Good: {summary.get('good', 0)}/{summary.get('total', 0)}")
        print(f"Acceptable: {summary.get('acceptable', 0)}/{summary.get('total', 0)}")
        print(f"Poor: {summary.get('poor', 0)}/{summary.get('total', 0)}")
        print("\nChannel Details:")
        for ch_name, ch_data in impedance_result.get('channels', {}).items():
            quality = ch_data['quality'].upper()
            imp_kohm = ch_data['impedance_kohm']
            print(f"  {ch_name}: {quality} ({imp_kohm} kOhm)")
        
        poor_channels = [name for name, data in impedance_result.get('channels', {}).items() 
                        if data['quality'] == 'poor']
        if poor_channels:
            print(f"\nWARNING: High impedance on: {', '.join(poor_channels)}")
            print("  Check electrode contact and apply conductive gel if needed")
        
        report['tests']['impedance'] = {
            'status': 'passed' if summary.get('poor', 0) == 0 else 'warning',
            'channels': impedance_result.get('channels', {}),
            'summary': summary
        }
    else:
        print(f"ERROR: {impedance_result.get('message')}")
        report['tests']['impedance'] = {'status': 'failed', 'error': impedance_result.get('message')}
    
    # 6. Capture and plot data
    print_section("6. Data Capture and Visualization")
    print("Capturing 2 seconds of EEG data...")
    data_result = capture_data(inlet, duration=2.0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(__file__).parent / "reports"
    plot_path = None
    if data_result.get('status') == 'success':
        plot_path = plot_data(data_result, report_dir=report_dir, timestamp=timestamp)
        print(f"Data plot saved: {plot_path}")
        report['tests']['data_capture'] = {
            'status': 'passed',
            'samples': data_result.get('samples', 0),
            'plot_path': str(plot_path) if plot_path else None
        }
    else:
        print(f"ERROR: {data_result.get('message')}")
        report['tests']['data_capture'] = {'status': 'failed', 'error': data_result.get('message')}
    
    print_section("Test Complete")
    return save_report(report, plot_path=plot_path, timestamp=timestamp)


def plot_data(data_result: dict, report_dir: Path, timestamp: str) -> Path:
    """Create and save EEG data plot with CAR applied."""
    report_dir.mkdir(exist_ok=True)
    plot_path = report_dir / f"eeg_plot_{timestamp}.png"
    
    data = data_result['data']
    time_axis = data_result['time']
    channels = data_result['channels']
    sample_rate = 250.0  # Unicorn Black sample rate
    
    # Apply Common Average Reference (CAR) - vectorized for efficiency
    # CAR removes common-mode noise by subtracting the mean across all channels from each channel
    data_car = data.copy()
    car_ref = np.mean(data_car, axis=1, keepdims=True)
    data_car = data_car - car_ref
    
    # Apply 50 Hz notch filter - use iirnotch for better notch filter design
    # This creates a proper notch filter at exactly 50 Hz
    notch_freq = 50.0
    quality = 30.0  # Quality factor - higher = narrower notch
    b_notch, a_notch = signal.iirnotch(notch_freq, quality, sample_rate)
    
    # Apply notch filter using filtfilt for zero-phase (but with proper handling)
    data_filtered = np.zeros_like(data_car)
    for ch in range(data_car.shape[1]):
        # Use filtfilt but with edge handling - pad with mean to avoid transients
        ch_data = data_car[:, ch]
        # Pad with mean value to avoid edge artifacts
        pad_len = int(sample_rate * 0.1)  # 0.1 second padding
        mean_val = np.mean(ch_data)
        padded = np.pad(ch_data, pad_len, mode='constant', constant_values=mean_val)
        # Apply filter
        filtered = signal.filtfilt(b_notch, a_notch, padded)
        # Remove padding
        data_filtered[:, ch] = filtered[pad_len:-pad_len]
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 10), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    
    for i, (ax, ch_name) in enumerate(zip(axes, channels)):
        # Plot filtered data (CAR + notch filter applied)
        ax.plot(time_axis, data_filtered[:, i], linewidth=0.5, marker='', linestyle='-')
        ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=9)
        ax.grid(True, alpha=0.3)
        p1, p99 = np.percentile(data_filtered[:, i], [1, 99])
        ax.set_ylim([p1, p99])
        
    # Ensure x-axis uses actual time range
    if len(time_axis) > 0:
        axes[-1].set_xlim([time_axis[0], time_axis[-1]])
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    fig.suptitle('EEG Signal Capture - All Channels (CAR + 50Hz Notch Applied)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj


def save_report(report: dict, plot_path: Path = None, timestamp: str = None) -> str:
    """Save test report to file and return path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"test_report_{timestamp}.json"
    
    if plot_path:
        report['plot_path'] = str(plot_path)
    
    # Convert numpy types to native Python types
    report_clean = convert_to_native_types(report)
    
    with open(report_path, 'w') as f:
        json.dump(report_clean, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  TEST REPORT")
    print("=" * 60)
    print(f"Report saved to: {report_path}")
    if plot_path:
        print(f"Plot saved to: {plot_path}")
    print("\nReport Summary:")
    print("-" * 60)
    
    for test_name, result in report['tests'].items():
        status = result.get('status', 'unknown')
        status_display = status.upper()
        print(f"{test_name.replace('_', ' ').title()}: {status_display}")
        if status == 'passed' or status == 'warning':
            if 'working_count' in result:
                print(f"  Working electrodes: {result['working_count']}/{result['total_count']}")
            if 'actual_rate' in result:
                print(f"  Sample rate: {result['actual_rate']} Hz")
            if 'battery_percent' in result:
                print(f"  Battery: {result['battery_percent']}%")
            if 'channels' in result and isinstance(result['channels'], dict):
                avg_impedance = np.mean([ch['impedance_kohm'] for ch in result['channels'].values()])
                print(f"  Impedance: {round(avg_impedance, 1)} kOhm average")
            if 'samples' in result:
                print(f"  Samples captured: {result['samples']}")
        elif status == 'failed':
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("-" * 60)
    return str(report_path)


if __name__ == "__main__":
    try:
        run_test()
    finally:
        # Restore stderr
        sys.stderr = _original_stderr
