"""
Screen Calibration Script for BCI-UPIC.

Measures:
1. Actual monitor refresh rate
2. Actual flicker frequencies displayed (15Hz and 12Hz targets)
3. Phase relationships

Saves calibration data to screen_calibration.json for use by the classifier.
"""

from __future__ import annotations

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Tuple

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QScreen

from src.bci.stimulus import FlickerTarget


def get_monitor_refresh_rate() -> Optional[float]:
    """
    Get monitor refresh rate using PyQt6's QScreen API.
    
    This queries the system for the monitor's configured refresh rate.
    Used for compatibility checking (ensuring flicker frequencies are factors).
    
    Note: For actual displayed flicker frequencies, we use FFT analysis
    of intensity samples collected in paintEvent (see FlickerFrequencyDetector).
    The system refresh rate tells us what the monitor should do, while FFT
    tells us what's actually being displayed.
    
    Returns:
        Refresh rate in Hz, or None if unavailable
    """
    app = QApplication.instance()
    if app is None:
        return None
    
    # Get the primary screen
    screen = app.primaryScreen()
    if screen is None:
        return None
    
    # Get refresh rate from screen
    refresh_rate = screen.refreshRate()
    
    # QScreen.refreshRate() returns -1 if unavailable, 0 if unknown
    if refresh_rate > 0:
        return float(refresh_rate)
    
    return None


class RefreshRateDetector(QWidget):
    """Widget that displays detected screen refresh rate."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._refresh_rate: Optional[float] = None
        self._update_refresh_rate()
        
    def _update_refresh_rate(self):
        """Update refresh rate from system."""
        self._refresh_rate = get_monitor_refresh_rate()
        self.update()
        
    def paintEvent(self, event):
        """Display refresh rate."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        # Display refresh rate
        if self._refresh_rate:
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           f"Monitor Refresh Rate: {self._refresh_rate:.2f} Hz")
        else:
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "Refresh Rate: Unable to detect\n(Using system default)")
    
    def get_refresh_rate(self) -> Optional[float]:
        """Get detected refresh rate."""
        return self._refresh_rate


class FlickerFrequencyDetector(QWidget):
    """Widget that measures actual flicker frequency by tracking intensity transitions."""
    
    def __init__(self, frequency: float, phase_offset: float, parent=None):
        super().__init__(parent)
        self.frequency = frequency
        self.phase_offset = phase_offset
        self.target = FlickerTarget(
            frequency=frequency,
            phase_offset=phase_offset,
            size=(200, 200)
        )
        # Don't start flickering until calibration begins
        
        # Track intensity samples for FFT-based frequency detection
        self._intensity_samples = deque(maxlen=2000)  # Store intensity values
        self._sample_times = deque(maxlen=2000)  # Store timestamps
        self._sample_rate = 125.0  # Approximate sample rate (8ms updates = 125Hz)
        
        # Colors
        self.color_on = QColor(255, 255, 255)
        self.color_off = QColor(30, 30, 30)
        
        self.setMinimumSize(200, 200)
        
        # Timer for continuous updates - don't start until calibration begins
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self.update)
        self._is_calibrating = False  # Don't flicker until calibration starts
    
    def start_calibration(self):
        """Start flickering and measurement."""
        self._is_calibrating = True
        self.target.start()
        self._update_timer.start(8)  # Start fast updates
        self.update()
    
    def stop_calibration(self):
        """Stop flickering."""
        self._is_calibrating = False
        self._update_timer.stop()
        self.update()
    
    def paintEvent(self, event):
        """Paint flickering target and track frequency."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Only flicker if calibrating
        if self._is_calibrating:
            # Get current intensity
            current_time = time.perf_counter()
            intensity = self.target.get_intensity(None)
            
            # Store samples for FFT analysis
            self._intensity_samples.append(intensity)
            self._sample_times.append(current_time)
            
            # Draw flickering rectangle
            color = QColor(
                int(self.color_off.red() + intensity * (self.color_on.red() - self.color_off.red())),
                int(self.color_off.green() + intensity * (self.color_on.green() - self.color_off.green())),
                int(self.color_off.blue() + intensity * (self.color_on.blue() - self.color_off.blue()))
            )
        else:
            # Draw static (off) rectangle when not calibrating
            color = self.color_off
            intensity = 0.0
        
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(rect, 10, 10)
        
        # Display frequency info
        painter.setPen(QPen(QColor(150, 150, 150)))
        target_freq = self.frequency
        actual_freq = self.get_measured_frequency()
        
        if self._is_calibrating:
            if actual_freq:
                text = f"Target: {target_freq:.1f} Hz\nActual: {actual_freq:.3f} Hz"
            else:
                text = f"Target: {target_freq:.1f} Hz\nMeasuring..."
        else:
            text = f"Target: {target_freq:.1f} Hz\nReady"
        
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    
    def get_measured_frequency(self) -> Optional[float]:
        """
        Calculate actual flicker frequency using FFT-based detection.
        
        This measures what's actually being displayed on screen by analyzing
        the intensity signal, accounting for any timing variations, frame drops,
        or display hardware differences. This is different from the system refresh
        rate (which we get from QScreen.refreshRate()) - this tells us what
        frequency is actually being displayed.
        """
        if len(self._intensity_samples) < 100:  # Need at least 100 samples
            return None
        
        # Convert to numpy array
        samples = np.array(self._intensity_samples)
        
        # Calculate actual sample rate from timestamps
        if len(self._sample_times) >= 2:
            time_diffs = np.diff(list(self._sample_times))
            actual_sample_rate = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else self._sample_rate
        else:
            actual_sample_rate = self._sample_rate
        
        # Compute FFT
        n_samples = len(samples)
        freqs = np.fft.rfftfreq(n_samples, 1/actual_sample_rate)
        fft_vals = np.abs(np.fft.rfft(samples))
        
        # Find peak frequency near target frequency
        # Search in range: target ± 2Hz
        target_range = (self.frequency - 2.0, self.frequency + 2.0)
        idx_range = np.where((freqs >= target_range[0]) & (freqs <= target_range[1]))[0]
        
        if len(idx_range) == 0:
            return None
        
        # Find peak in this range
        fft_in_range = fft_vals[idx_range]
        freqs_in_range = freqs[idx_range]
        
        peak_idx = np.argmax(fft_in_range)
        peak_frequency = freqs_in_range[peak_idx]
        
        return peak_frequency
    
    def get_frequency_statistics(self) -> Optional[Dict]:
        """Get detailed frequency statistics using FFT."""
        if len(self._intensity_samples) < 100:
            return None
        
        # Get measured frequency
        measured_freq = self.get_measured_frequency()
        if measured_freq is None:
            return None
        
        # Calculate actual sample rate
        if len(self._sample_times) >= 2:
            time_diffs = np.diff(list(self._sample_times))
            actual_sample_rate = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else self._sample_rate
        else:
            actual_sample_rate = self._sample_rate
        
        # Compute FFT for statistics
        samples = np.array(self._intensity_samples)
        n_samples = len(samples)
        freqs = np.fft.rfftfreq(n_samples, 1/actual_sample_rate)
        fft_vals = np.abs(np.fft.rfft(samples))
        
        # Get power at measured frequency (with small bandwidth)
        bandwidth = 0.5  # Hz
        idx = np.where((freqs >= measured_freq - bandwidth) & (freqs <= measured_freq + bandwidth))[0]
        power = np.sum(fft_vals[idx] ** 2) if len(idx) > 0 else 0.0
        
        return {
            'frequency': measured_freq,
            'power': float(power),
            'n_samples': n_samples,
            'sample_rate': actual_sample_rate
        }
    
    def reset_measurement(self):
        """Reset measurement for a new calibration."""
        self._intensity_samples.clear()
        self._sample_times.clear()
        self.target.start()


class ScreenCalibrationWindow(QWidget):
    """Main calibration window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Calibration - BCI-UPIC")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Instructions
        info_label = QLabel(
            "Screen Calibration\n\n"
            "This will measure:\n"
            "1. Your monitor's refresh rate\n"
            "2. Actual flicker frequencies (15Hz and 12Hz)\n"
            "3. Phase relationships\n\n"
            "The calibration takes ~5 seconds. Please ensure:\n"
            "- Full screen mode (F11)\n"
            "- No other applications interfering\n"
            "- Monitor at native refresh rate"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: #ccc; font-size: 14px; padding: 20px;")
        layout.addWidget(info_label)
        
        # Refresh rate detector - uses system API to detect refresh rate
        self.refresh_detector = RefreshRateDetector()
        self.refresh_detector.setMinimumHeight(100)
        layout.addWidget(self.refresh_detector)
        
        # Flicker frequency detectors - use horizontal layout with proper spacing
        from PyQt6.QtWidgets import QHBoxLayout
        
        flicker_container = QWidget()
        flicker_layout = QHBoxLayout(flicker_container)
        flicker_layout.setSpacing(40)  # Large spacing between targets
        flicker_layout.setContentsMargins(20, 20, 20, 20)
        
        # 15Hz target (left side)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_label = QLabel("15Hz Target (Top):")
        left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_label.setStyleSheet("color: #ccc; font-size: 12px;")
        left_layout.addWidget(left_label)
        self.freq_15hz = FlickerFrequencyDetector(15.0, 0.0)
        self.freq_15hz.setMinimumSize(200, 200)
        self.freq_15hz.setMaximumSize(200, 200)
        left_layout.addWidget(self.freq_15hz, alignment=Qt.AlignmentFlag.AlignHCenter)
        left_layout.addStretch()
        flicker_layout.addWidget(left_widget)
        
        # 12Hz target (right side)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_label = QLabel("12Hz Target (Bottom):")
        right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_label.setStyleSheet("color: #ccc; font-size: 12px;")
        right_layout.addWidget(right_label)
        self.freq_12hz = FlickerFrequencyDetector(12.0, np.pi)
        self.freq_12hz.setMinimumSize(200, 200)
        self.freq_12hz.setMaximumSize(200, 200)
        right_layout.addWidget(self.freq_12hz, alignment=Qt.AlignmentFlag.AlignHCenter)
        right_layout.addStretch()
        flicker_layout.addWidget(right_widget)
        
        layout.addWidget(flicker_container)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Calibration")
        self.start_btn.clicked.connect(self.start_calibration)
        button_layout.addWidget(self.start_btn)
        
        self.save_btn = QPushButton("Save Calibration")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        self.status_label = QLabel("Ready to start calibration")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; padding: 10px;")
        button_layout.addWidget(self.status_label)
        
        layout.addLayout(button_layout)
        
        # Calibration results
        self.calibration_data: Dict = {}
        
        # Timer for calibration duration - increased to 15 seconds for better averaging
        self._cal_timer = QTimer()
        self._cal_timer.timeout.connect(self._finish_calibration)
        self._cal_timer.setSingleShot(True)
    
    def start_calibration(self):
        """Start the calibration process."""
        self.status_label.setText("Calibrating... Please wait 5 seconds")
        self.start_btn.setEnabled(False)
        
        # Reset measurements
        self.freq_15hz.reset_measurement()
        self.freq_12hz.reset_measurement()
        
        # Start flickering targets
        self.freq_15hz.start_calibration()
        self.freq_12hz.start_calibration()
        
        # Start calibration timer (5 seconds - FFT needs ~100 samples, at 125Hz that's <1 second)
        self._cal_timer.start(5000)
    
    def _finish_calibration(self):
        """Finish calibration and collect results."""
        # Stop flickering
        self.freq_15hz.stop_calibration()
        self.freq_12hz.stop_calibration()
        
        # Measure refresh rate
        refresh_rate = self.refresh_detector.get_refresh_rate()
        
        # Measure actual frequencies
        freq_15hz_actual = self.freq_15hz.get_measured_frequency()
        freq_12hz_actual = self.freq_12hz.get_measured_frequency()
        
        # Get detailed statistics
        stats_15hz = self.freq_15hz.get_frequency_statistics()
        stats_12hz = self.freq_12hz.get_frequency_statistics()
        
        # Store results with statistics (both new and old format for compatibility)
        self.calibration_data = {
            'refresh_rate_hz': float(refresh_rate) if refresh_rate else None,
            # New generic format
            'target_higher_freq': 15.0,
            'actual_higher_freq': float(freq_15hz_actual) if freq_15hz_actual is not None else None,
            'target_lower_freq': 12.0,
            'actual_lower_freq': float(freq_12hz_actual) if freq_12hz_actual is not None else None,
            'phase_higher_freq': 0.0,
            'phase_lower_freq': float(np.pi),
            # Old format for backward compatibility
            'target_15hz': 15.0,
            'actual_15hz': float(freq_15hz_actual) if freq_15hz_actual is not None else None,
            'target_12hz': 12.0,
            'actual_12hz': float(freq_12hz_actual) if freq_12hz_actual is not None else None,
            'phase_15hz': 0.0,
            'phase_12hz': float(np.pi),
            'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'monitor_info': 'Unknown',  # Could be enhanced with system info
            'statistics_15hz': stats_15hz,
            'statistics_12hz': stats_12hz
        }
        
        # Check frequency compatibility with refresh rate
        from src.bci.screen_config import ScreenCalibration
        temp_cal = ScreenCalibration(
            refresh_rate_hz=refresh_rate,
            actual_higher_freq=freq_15hz_actual if freq_15hz_actual else 15.0,
            actual_lower_freq=freq_12hz_actual if freq_12hz_actual else 12.0
        )
        is_compatible, warnings = temp_cal.check_frequency_compatibility()
        
        # Update status with variance information and warnings
        if refresh_rate and freq_15hz_actual and freq_12hz_actual:
            status_text = (
                f"Calibration Complete!\n"
                f"Refresh Rate: {refresh_rate:.2f} Hz\n"
                f"15Hz Actual: {freq_15hz_actual:.3f} Hz"
            )
            if stats_15hz:
                status_text += f" (±{stats_15hz.get('std_frequency', 0):.3f} Hz)"
            status_text += f"\n12Hz Actual: {freq_12hz_actual:.3f} Hz"
            if stats_12hz:
                status_text += f" (±{stats_12hz.get('std_frequency', 0):.3f} Hz)"
            
            # Add compatibility warnings
            if warnings:
                status_text += "\n\n⚠ WARNING:\n" + "\n".join(warnings)
            elif is_compatible:
                status_text += "\n\n✓ Frequencies compatible with refresh rate"
            
            status_text += "\n\nClick 'Save Calibration' to save."
            
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet("color: #ffa500;" if warnings else "color: #888; padding: 10px;")
            self.save_btn.setEnabled(True)
        else:
            self.status_label.setText(
                "Calibration incomplete. Please try again.\n"
                "Make sure the window is visible and not minimized."
            )
            self.start_btn.setEnabled(True)
    
    def save_calibration(self):
        """Save calibration data to file."""
        if not self.calibration_data:
            QMessageBox.warning(self, "Error", "No calibration data to save!")
            return
        
        try:
            # Ensure all values are JSON-serializable (convert numpy types)
            cal_data = {}
            for key, value in self.calibration_data.items():
                if key in ['statistics_15hz', 'statistics_12hz']:
                    # Statistics are optional, skip if None or convert dict values
                    if value is not None and isinstance(value, dict):
                        cal_data[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                         for k, v in value.items()}
                    else:
                        cal_data[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    cal_data[key] = float(value)
                elif value is None:
                    cal_data[key] = None
                else:
                    cal_data[key] = value
            
            # Save to file - save in the screen_calibration folder (where script is)
            script_dir = Path(__file__).parent
            cal_file = script_dir / "screen_calibration.json"
            
            # Ensure directory exists
            cal_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(cal_file, 'w') as f:
                json.dump(cal_data, f, indent=2)
            
            print(f"[CALIBRATION] Saved to: {cal_file.absolute()}")
            
        except Exception as e:
            error_msg = f"Failed to save calibration data:\n{str(e)}"
            print(f"[CALIBRATION ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Save Error",
                error_msg + "\n\nPlease check the console for details."
            )
            return
        
        # Build success message with proper field access
        refresh_rate = cal_data.get('refresh_rate_hz', 'N/A')
        higher_freq = cal_data.get('actual_higher_freq', cal_data.get('actual_15hz', 'N/A'))
        lower_freq = cal_data.get('actual_lower_freq', cal_data.get('actual_12hz', 'N/A'))
        
        msg = f"Calibration saved successfully!\n\nFile: {cal_file}\n\n"
        if refresh_rate and isinstance(refresh_rate, (int, float)):
            msg += f"Refresh Rate: {refresh_rate:.2f} Hz\n"
        if isinstance(higher_freq, (int, float)):
            msg += f"Higher freq: {higher_freq:.3f} Hz\n"
        if isinstance(lower_freq, (int, float)):
            msg += f"Lower freq: {lower_freq:.3f} Hz"
        
        QMessageBox.information(
            self, 
            "Saved", 
            msg
        )
        
        self.status_label.setText("Calibration saved! You can close this window.")


def main():
    """Run screen calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Screen Calibration Tool for BCI-UPIC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python screen_calibration.py              # Run calibration GUI
  python screen_calibration.py --check     # Check if calibration exists
  python screen_calibration.py --info      # Show current calibration info

The calibration tool measures:
  - Monitor refresh rate
  - Actual flicker frequencies (15Hz and 12Hz targets)
  - Phase relationships

Results are saved to screen_calibration.json and automatically
loaded by the BCI classifier for improved SSVEP accuracy.
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if calibration file exists'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display current calibration information'
    )
    
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete existing calibration file'
    )
    
    args = parser.parse_args()
    
    # Handle CLI-only commands
    # Check for calibration file in screen_calibration folder or root
    def find_cal_file():
        cal_file = Path("screen_calibration/screen_calibration.json")
        if cal_file.exists():
            return cal_file
        cal_file = Path("screen_calibration.json")
        if cal_file.exists():
            return cal_file
        return None
    
    if args.check:
        cal_file = find_cal_file()
        if cal_file:
            print(f"✓ Calibration file exists: {cal_file}")
            sys.exit(0)
        else:
            print("✗ No calibration file found")
            print("  Run: python screen_calibration/screen_calibration.py")
            sys.exit(1)
    
    if args.info:
        cal_file = find_cal_file()
        if cal_file is None:
            print("No calibration file found.")
            print("Run: python screen_calibration/screen_calibration.py")
            sys.exit(1)
        
        try:
            with open(cal_file, 'r') as f:
                cal_data = json.load(f)
            
            print("Screen Calibration Information:")
            print("=" * 50)
            print(f"Calibrated at: {cal_data.get('calibrated_at', 'Unknown')}")
            print(f"Refresh Rate: {cal_data.get('refresh_rate_hz', 'N/A'):.2f} Hz")
            print()
            print("15Hz Target:")
            print(f"  Target: {cal_data.get('target_15hz', 'N/A'):.3f} Hz")
            print(f"  Actual: {cal_data.get('actual_15hz', 'N/A'):.3f} Hz")
            print(f"  Offset: {cal_data.get('actual_15hz', 0) - cal_data.get('target_15hz', 0):.3f} Hz")
            print()
            print("12Hz Target:")
            # Support both old (10Hz) and new (12Hz) format for backward compatibility
            target_12hz = cal_data.get('target_12hz', cal_data.get('target_10hz', 'N/A'))
            actual_12hz = cal_data.get('actual_12hz', cal_data.get('actual_10hz', 'N/A'))
            print(f"  Target: {target_12hz:.3f} Hz" if isinstance(target_12hz, (int, float)) else f"  Target: {target_12hz}")
            print(f"  Actual: {actual_12hz:.3f} Hz" if isinstance(actual_12hz, (int, float)) else f"  Actual: {actual_12hz}")
            if isinstance(target_12hz, (int, float)) and isinstance(actual_12hz, (int, float)):
                print(f"  Offset: {actual_12hz - target_12hz:.3f} Hz")
            
            if 'statistics_15hz' in cal_data or 'statistics_12hz' in cal_data or 'statistics_10hz' in cal_data:
                print()
                print("Statistics:")
                if 'statistics_15hz' in cal_data:
                    stats = cal_data['statistics_15hz']
                    print(f"  15Hz: {stats.get('n_samples', 'N/A')} samples, "
                          f"power: {stats.get('power', 'N/A'):.2e}")
                # Support both old and new format
                stats_12hz = cal_data.get('statistics_12hz', cal_data.get('statistics_10hz', None))
                if stats_12hz:
                    print(f"  12Hz: {stats_12hz.get('n_samples', 'N/A')} samples, "
                          f"power: {stats_12hz.get('power', 'N/A'):.2e}")
            
        except Exception as e:
            print(f"Error reading calibration file: {e}")
            sys.exit(1)
        
        sys.exit(0)
    
    if args.delete:
        cal_file = find_cal_file()
        if cal_file:
            cal_file.unlink()
            print(f"✓ Calibration file deleted: {cal_file}")
            sys.exit(0)
        else:
            print("✗ No calibration file found to delete")
            sys.exit(1)
    
    # Run GUI calibration
    try:
        app = QApplication(sys.argv)
        
        # Apply dark theme
        app.setStyle('Fusion')
        palette = app.palette()
        palette.setColor(palette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
        app.setPalette(palette)
        
        window = ScreenCalibrationWindow()
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting calibration GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
