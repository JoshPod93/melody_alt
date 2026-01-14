"""
BCI Composition Interface for BCI-UPIC.

Main real-time interface for BCI-controlled music composition.
Features:
- Two flickering SSVEP targets (higher frequency top, lower frequency bottom)
- Automatic horizontal playhead movement
- Real-time cursor position visualization
- Score display and playback controls
"""

from __future__ import annotations

import sys
import time
import json
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
from datetime import datetime

# Matplotlib for plotting (use Agg backend for non-interactive)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QComboBox, QSlider, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

from .stimulus import SSVEPStimulus, FlickerState
from .p300_stimulus import P300Stimulus, P300FlashTarget, FlashState
from .preprocessing import EEGPreprocessor, SimulatedEEGSource, LSLPreprocessor
from .classifier import SSVEPClassifier, ClassificationResult as SSVEPClassificationResult
from .p300_classifier import P300Classifier, AttentionTarget, ClassificationResult
from .controller import BCICursorController, ControllerState, CursorPosition
from .score import BCIScore, play_score, synthesize_score
from .calibration import CalibrationData, CalibrationSession

try:
    from .lsl_stream import LSLReceiver, LSLMarkerSender, LSL_AVAILABLE
except ImportError:
    LSL_AVAILABLE = False


class SessionMode(Enum):
    """Mode of the BCI session."""
    IDLE = 0
    COMPOSING = 1
    PLAYBACK = 2
    CALIBRATING = 3


class P300FlashWidget(QWidget):
    """
    Widget displaying a P300 flash target.
    
    Renders a rectangle that flashes discretely for P300 ERP paradigm.
    """
    
    def __init__(
        self,
        position: str = "top",
        target: Optional[P300FlashTarget] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.position = position
        self._is_active = False
        
        # Use provided target or create new one
        if target is not None:
            self.target = target
        else:
            self.target = P300FlashTarget(position=position)
        
        # Colors
        self.color_on = QColor(255, 255, 255)
        self.color_off = QColor(30, 30, 30)
        self.border_color = QColor(100, 100, 100)
        
        # Size
        self.setMinimumSize(100, 100)
        self.setMaximumHeight(100)
        self.setMaximumWidth(100)
        
        # Qt optimizations
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        
        # Timer for updates
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self.update)
    
    def paintEvent(self, event) -> None:
        """Paint the flash target - discrete color changes, no flickering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get color from target (returns RGB tuple)
        # get_color() uses peek_state() so it doesn't update the state machine
        # The state machine is updated by update_flash() from the stimulus system
        if self._is_active:
            color_tuple = self.target.get_color()
        else:
            color_tuple = self.target.color_off
        
        # Convert to QColor
        if isinstance(color_tuple, tuple) and len(color_tuple) == 3:
            color = QColor(*color_tuple)
        else:
            # Fallback to default off color
            color = QColor(*self.target.color_off)
        
        # Draw rectangle with solid color (no flickering, just discrete color changes)
        rect = self.rect()
        painter.fillRect(rect, color)
        
        # Draw border
        painter.setPen(QPen(self.border_color, 2))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))
    
    def set_active(self, active: bool) -> None:
        """Set whether the target is active."""
        self._is_active = active
        if active:
            # Don't call target.start() here - the stimulus system handles starting
            # This prevents resetting timing when stimulus is already running
            self._update_timer.start(16)  # ~60Hz update rate
        else:
            self._update_timer.stop()
        self.update()
    
    def update_flash(self) -> None:
        """Update flash state (called by stimulus system)."""
        if self._is_active:
            self.target.update()
            self.update()


class FlickerWidget(QWidget):
    """
    Widget displaying a flickering SSVEP target.
    
    Renders a rectangle that flickers at the specified frequency.
    Uses real-time calculation in paintEvent for frame-synchronized rendering.
    
    EXACTLY matches the protocol used in screen calibration FlickerFrequencyDetector:
    - Creates its own FlickerTarget instance
    - Uses internal timer to call update()
    - paintEvent calculates intensity in real-time using target.get_intensity(None)
    """
    
    def __init__(
        self,
        frequency: float,
        phase_offset: float,
        position: str = "top",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.frequency = frequency
        self.phase_offset = phase_offset
        self.position = position
        self._is_active = False
        
        # Create our own FlickerTarget instance (same as screen calibration)
        from .stimulus import FlickerTarget
        self.target = FlickerTarget(
            frequency=frequency,
            phase_offset=phase_offset,
            position=position,
            size=(100, 100)
        )
        
        # Colors - fixed border (no dynamic changes to avoid interfering with flickering)
        self.color_on = QColor(255, 255, 255)
        self.color_off = QColor(30, 30, 30)
        self.border_color = QColor(100, 100, 100)  # Fixed - never changes during flickering
        
        # Phase monitoring for debugging
        self._last_paint_time = None
        self._paint_count = 0
        self._expected_phase = 0.0
        self._actual_phase = 0.0
        
        # Size - square targets centered on screen
        self.setMinimumSize(100, 100)
        self.setMaximumHeight(100)
        self.setMaximumWidth(100)
        
        # Qt performance optimizations for old hardware
        # 1. Opaque paint event - skip background clearing (faster)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        # 2. No system background - we draw everything ourselves
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        
        # Timer for continuous updates - EXACTLY match screen calibration (no PreciseTimer)
        # Try to use high-precision timer on Windows, fallback to QTimer
        try:
            from .high_precision_timer import HighPrecisionTimer, is_available
            self._use_high_precision = is_available()
        except (ImportError, RuntimeError):
            self._use_high_precision = False
        
        if self._use_high_precision:
            # Use Windows multimedia timer for better precision
            self._update_timer = None  # Will be created in set_active
            self._high_precision_timer = None
        else:
            # Fallback to QTimer
            self._update_timer = QTimer()
            self._high_precision_timer = None
        
        # Timer monitoring to detect if timer is actually firing
        self._timer_fire_count = 0
        self._last_timer_fire_time = None
        
        # Flicker rate accuracy tracking - track actual flicker cycles, not paint events
        self._flicker_state_changes = []  # Store timestamps when flicker state changes (ON->OFF or OFF->ON)
        self._last_flicker_state = None
        self._flicker_rate_samples = 50  # Number of flicker cycles to use for rate calculation
        
        if not self._use_high_precision:
            # Connect QTimer after monitoring vars are initialized
            self._update_timer.timeout.connect(self._on_timer_fire)
    
    def _on_timer_fire(self) -> None:
        """Wrapper to detect if timer is actually firing vs paintEvent being throttled."""
        self._timer_fire_count += 1
        current_time = time.perf_counter()
        if self._last_timer_fire_time is not None:
            dt = current_time - self._last_timer_fire_time
            if self._timer_fire_count % 125 == 0:  # Log every ~1 second
                if dt > 0.020:  # More than 20ms between timer fires
                    print(f"[TIMER DEBUG] {self.position}: Timer slow! dt={dt*1000:.1f}ms (expected ~8ms), fires={self._timer_fire_count}")
        self._last_timer_fire_time = current_time
        
        # Force immediate repaint - update() was being throttled too much
        # repaint() forces immediate painting which is necessary for accurate flickering
        # The Qt optimizations (WA_OpaquePaintEvent, etc.) reduce the cost of repaint()
        if self._use_high_precision:
            QTimer.singleShot(0, self.repaint)  # Force immediate repaint on main thread
        else:
            self.repaint()  # Direct call for QTimer (already on main thread)
    
    def set_active(self, active: bool) -> None:
        """
        Set whether the target is active (flickering).
        
        EXACTLY matches screen calibration FlickerFrequencyDetector protocol:
        - When activating: calls target.start() ONCE, starts timer with 8ms interval, calls update()
        - When deactivating: stops timer, calls update()
        """
        was_active = self._is_active
        self._is_active = active
        
        if active:
            # Call target.start() ONCE to set start time (don't reset timing!)
            if not was_active:
                self.target.start()
            
            # Start timer with 8ms interval
            if self._use_high_precision:
                # Use Windows multimedia timer for better precision
                try:
                    from .high_precision_timer import HighPrecisionTimer
                    if self._high_precision_timer is None:
                        self._high_precision_timer = HighPrecisionTimer(
                            interval_ms=8,
                            callback=self._on_timer_fire
                        )
                    if not self._high_precision_timer.is_running:
                        self._high_precision_timer.start()
                except Exception as e:
                    print(f"[FLICKER] High-precision timer failed, using QTimer: {e}")
                    self._use_high_precision = False
                    if self._update_timer is None:
                        self._update_timer = QTimer()
                        self._update_timer.timeout.connect(self._on_timer_fire)
                    self._update_timer.start(8)
            else:
                # Use QTimer (fallback)
                if self._update_timer is None:
                    self._update_timer = QTimer()
                    self._update_timer.timeout.connect(self._on_timer_fire)
                if not self._update_timer.isActive():
                    self._update_timer.start(8)
            
            self.update()  # Immediate update
        else:
            # Stop timer
            if self._use_high_precision and self._high_precision_timer:
                self._high_precision_timer.stop()
            elif self._update_timer:
                self._update_timer.stop()
            self.update()  # Final update to show inactive state
    
    def paintEvent(self, event) -> None:
        """Paint the flickering target with real-time intensity calculation."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate intensity in real-time based on current time for frame synchronization
        # EXACTLY matches screen calibration FlickerFrequencyDetector.paintEvent
        if self._is_active:
            # Get current time for phase monitoring
            current_time = time.perf_counter()
            
            # Calculate expected phase based on elapsed time since target started
            elapsed = current_time - self.target._start_time
            self._expected_phase = (2 * np.pi * self.frequency * elapsed + self.phase_offset) % (2 * np.pi)
            
            # Call get_intensity without time parameter - it will use perf_counter internally
            # This ensures frame-synchronized rendering regardless of timer updates
            intensity = self.target.get_intensity(None)  # None triggers internal time calculation
            
            # Calculate actual phase directly from time (same as expected, but using target's internal calculation)
            # This is more accurate than trying to invert intensity, which loses quadrant information
            # The target calculates: phase = 2*π*f*t + phase_offset
            target_elapsed = time.perf_counter() - self.target._start_time
            self._actual_phase = (2 * np.pi * self.frequency * target_elapsed + self.phase_offset) % (2 * np.pi)
            
            # Determine current flicker state (ON or OFF) - track actual flicker cycles, not paint events
            current_state = FlickerState.ON if intensity > 0.5 else FlickerState.OFF
            
            # Track actual flicker cycles (state changes), not paint events
            if self._last_flicker_state is not None and current_state != self._last_flicker_state:
                # State changed - record timestamp for flicker rate calculation
                self._flicker_state_changes.append(current_time)
                if len(self._flicker_state_changes) > self._flicker_rate_samples * 2:  # Keep enough for rate calc
                    self._flicker_state_changes.pop(0)
            
            self._last_flicker_state = current_state
            
            # Monitor paint rate and phase drift
            self._paint_count += 1
            if self._last_paint_time is not None:
                dt = current_time - self._last_paint_time
                phase_drift = abs(self._expected_phase - self._actual_phase)
                
                # Log if we detect significant drift or slow updates
                if self._paint_count % 150 == 0:  # Log every ~2.5 seconds at 60Hz
                    if dt > 0.020:  # More than 20ms between paints (should be ~16.7ms for 60Hz)
                        print(f"[FLICKER DEBUG] {self.position}: Slow paint! dt={dt*1000:.1f}ms (expected ~16.7ms for 60Hz)")
                    if phase_drift > 0.5:  # More than 0.5 radians drift
                        print(f"[FLICKER DEBUG] {self.position}: Phase drift! {phase_drift:.3f} rad, expected={self._expected_phase:.3f}, actual={self._actual_phase:.3f}")
                    
                    # Calculate and log actual flicker rate from state changes (not paint events)
                    if len(self._flicker_state_changes) >= 20:  # Need enough state changes (at least 10 cycles)
                        # Calculate time between state changes (each cycle has 2 state changes: ON->OFF and OFF->ON)
                        intervals = []
                        for i in range(1, len(self._flicker_state_changes)):
                            intervals.append(self._flicker_state_changes[i] - self._flicker_state_changes[i-1])
                        
                        if intervals:
                            # Average interval between state changes, then multiply by 2 to get full cycle period
                            mean_interval = np.mean(intervals)
                            cycle_period = mean_interval * 2  # Full cycle = 2 state changes (ON->OFF->ON)
                            actual_rate = 1.0 / cycle_period if cycle_period > 0 else 0
                            target_rate = self.frequency
                            rate_error = abs(actual_rate - target_rate)
                            rate_error_pct = (rate_error / target_rate * 100) if target_rate > 0 else 0
                            print(f"[FLICKER RATE] {self.position}: Target={target_rate:.2f}Hz, Actual={actual_rate:.2f}Hz, Error={rate_error:.3f}Hz ({rate_error_pct:.1f}%)")
            
            self._last_paint_time = current_time
        else:
            intensity = 0.0
            self._last_paint_time = None
            self._paint_count = 0
            self._flicker_state_changes.clear()
            self._last_flicker_state = None
        
        # Calculate color - EXACTLY match screen calibration FlickerFrequencyDetector.paintEvent
        if self._is_active:
            # Draw flickering rectangle - EXACTLY same calculation as screen calibration
            color = QColor(
                int(self.color_off.red() + intensity * (self.color_on.red() - self.color_off.red())),
                int(self.color_off.green() + intensity * (self.color_on.green() - self.color_off.green())),
                int(self.color_off.blue() + intensity * (self.color_on.blue() - self.color_off.blue()))
            )
        else:
            # Draw static (off) rectangle when not active (same as screen calibration)
            color = self.color_off
        
        # Draw rounded rectangle - use fixed border color (never changes during flickering)
        # Use exact rect to avoid unnecessary clipping
        rect = QRectF(5, 5, self.width() - 10, self.height() - 10)
        painter.setPen(QPen(QColor(100, 100, 100), 2))  # Fixed border - never changes
        painter.setBrush(QBrush(color))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # Disable AA for speed
        painter.drawRoundedRect(rect, 10, 10)
        
        # Draw frequency label on two lines
        painter.setPen(QPen(QColor(150, 150, 150)))
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        # First line: frequency
        freq_label = f"{self.frequency:.0f} Hz"
        # Second line: direction
        if self.position == "top":
            dir_label = "▲ UP"
        else:
            dir_label = "▼ DOWN"
        
        # Calculate text rectangles for two lines
        font_metrics = painter.fontMetrics()
        freq_height = font_metrics.height()
        dir_height = font_metrics.height()
        total_height = freq_height + dir_height + 2  # 2px spacing
        
        # Center vertically
        start_y = rect.center().y() - total_height // 2
        
        # Draw first line (frequency)
        freq_rect = QRectF(rect.left(), start_y, rect.width(), freq_height)
        painter.drawText(freq_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, freq_label)
        
        # Draw second line (direction)
        dir_rect = QRectF(rect.left(), start_y + freq_height + 2, rect.width(), dir_height)
        painter.drawText(dir_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, dir_label)


class IndicatorLight(QWidget):
    """Separate indicator light widget - doesn't interfere with flickering."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self._is_active = False
    
    def set_active(self, active: bool):
        """Set indicator state (green = active, gray = inactive)."""
        self._is_active = active
        self.update()
    
    def paintEvent(self, event):
        """Paint the indicator light."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw circle - green if active, gray if inactive
        color = QColor(0, 255, 0) if self._is_active else QColor(100, 100, 100)
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self.rect().adjusted(2, 2, -2, -2))


class CompositionCanvas(QWidget):
    """
    Canvas showing the cursor trail and composition progress.
    
    Displays:
    - Vertical frequency axis
    - Horizontal time axis
    - Cursor position
    - Trail of cursor movement
    - Playhead position
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(800, 300)
        
        # Composition state
        self._trail: List[Tuple[float, float]] = []
        self._cursor_pos: Optional[CursorPosition] = None
        self._duration: float = 10.0
        self._progress: float = 0.0
        self._is_composing: bool = False
        
        # Playback state
        self._playback_time: float = 0.0
        self._is_playing: bool = False
        
        # Colors
        self.bg_color = QColor(20, 20, 30)
        self.grid_color = QColor(50, 50, 60)
        self.trail_color = QColor(66, 200, 135)
        self.cursor_color = QColor(255, 100, 100)
        self.playhead_color = QColor(255, 200, 50)
        self.axis_color = QColor(100, 100, 110)
    
    def set_trail(self, trail: List[Tuple[float, float]]) -> None:
        """Set the trail points."""
        self._trail = trail
        self.update()
    
    def set_cursor_position(self, pos: CursorPosition) -> None:
        """Set current cursor position."""
        self._cursor_pos = pos
        self.update()
    
    def set_duration(self, duration: float) -> None:
        """Set composition duration."""
        self._duration = duration
        self.update()
    
    def set_progress(self, progress: float) -> None:
        """Set composition progress (0-1)."""
        self._progress = progress
        self.update()
    
    def set_composing(self, composing: bool) -> None:
        """Set whether currently composing."""
        self._is_composing = composing
        self.update()
    
    def set_playback_time(self, time: float) -> None:
        """Set playback position."""
        self._playback_time = time
        self.update()
    
    def set_playing(self, playing: bool) -> None:
        """Set whether currently playing back."""
        self._is_playing = playing
        self.update()
    
    def clear(self) -> None:
        """Clear the canvas."""
        self._trail = []
        self._cursor_pos = None
        self._progress = 0.0
        self._playback_time = 0.0
        self.update()
    
    def _time_to_x(self, t: float) -> float:
        """Convert time to x coordinate."""
        margin = 60
        width = self.width() - 2 * margin
        return margin + (t / self._duration) * width
    
    def _pitch_to_y(self, pitch: float) -> float:
        """Convert pitch (0-1) to y coordinate (inverted)."""
        margin = 40
        height = self.height() - 2 * margin
        return margin + (1 - pitch) * height
    
    def paintEvent(self, event) -> None:
        """Paint the composition canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw axes
        self._draw_axes(painter)
        
        # Draw trail
        self._draw_trail(painter)
        
        # Draw cursor
        if self._is_composing and self._cursor_pos:
            self._draw_cursor(painter)
        
        # Draw playhead during playback
        if self._is_playing:
            self._draw_playhead(painter)
    
    def _draw_grid(self, painter: QPainter) -> None:
        """Draw background grid."""
        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.DotLine))
        
        margin = 60
        margin_y = 40
        width = self.width() - 2 * margin
        height = self.height() - 2 * margin_y
        
        # Vertical lines (time)
        for i in range(int(self._duration) + 1):
            x = self._time_to_x(i)
            painter.drawLine(int(x), margin_y, int(x), self.height() - margin_y)
        
        # Horizontal lines (pitch)
        for i in range(11):
            y = self._pitch_to_y(i / 10)
            painter.drawLine(margin, int(y), self.width() - margin, int(y))
    
    def _draw_axes(self, painter: QPainter) -> None:
        """Draw axis labels."""
        painter.setPen(QPen(self.axis_color))
        font = QFont("Arial", 10)
        painter.setFont(font)
        
        margin = 60
        margin_y = 40
        
        # Time axis labels
        for i in range(int(self._duration) + 1):
            x = self._time_to_x(i)
            painter.drawText(int(x) - 10, self.height() - 15, f"{i}s")
        
        # Pitch axis labels
        painter.drawText(5, margin_y + 5, "High")
        painter.drawText(5, self.height() - margin_y, "Low")
        
        # Center line label
        y_center = self._pitch_to_y(0.5)
        painter.drawText(5, int(y_center) + 5, "Center")
    
    def _draw_trail(self, painter: QPainter) -> None:
        """Draw the cursor trail."""
        if len(self._trail) < 2:
            return
        
        # Create path
        path = QPainterPath()
        first_point = self._trail[0]
        path.moveTo(self._time_to_x(first_point[0]), self._pitch_to_y(first_point[1]))
        
        for t, pitch in self._trail[1:]:
            path.lineTo(self._time_to_x(t), self._pitch_to_y(pitch))
        
        # Draw trail
        pen = QPen(self.trail_color, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)
    
    def _draw_cursor(self, painter: QPainter) -> None:
        """Draw the current cursor position."""
        if not self._cursor_pos:
            return
        
        x = self._time_to_x(self._cursor_pos.time)
        y = self._pitch_to_y(self._cursor_pos.pitch)
        
        # Draw cursor circle
        painter.setPen(QPen(self.cursor_color, 2))
        painter.setBrush(QBrush(self.cursor_color))
        painter.drawEllipse(int(x) - 8, int(y) - 8, 16, 16)
        
        # Draw vertical line (playhead)
        painter.setPen(QPen(self.cursor_color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(x), 40, int(x), self.height() - 40)
    
    def _draw_playhead(self, painter: QPainter) -> None:
        """Draw playhead during playback."""
        x = self._time_to_x(self._playback_time)
        
        painter.setPen(QPen(self.playhead_color, 2))
        painter.drawLine(int(x), 40, int(x), self.height() - 40)


class BCICompositionWindow(QMainWindow):
    """
    Main window for BCI composition interface.
    """
    
    # =================================================================
    # EXPERIMENT MODE FLAG - Set to True for real experiments!
    # When True: Simulation is DISABLED, LSL connection is REQUIRED
    # =================================================================
    EXPERIMENT_MODE = True  # <-- SET THIS TO True FOR REAL EXPERIMENTS
    # =================================================================
    
    def __init__(self):
        super().__init__()
        
        if self.EXPERIMENT_MODE:
            self.setWindowTitle("BCI-UPIC Composer [EXPERIMENT MODE - Real Data Only]")
        else:
            self.setWindowTitle("BCI-UPIC Composer [DEV MODE - Simulation Allowed]")
        
        self.setMinimumSize(1000, 700)
        
        # BCI components - P300 paradigm (optimized timing: 62ms flash, 25ms ISI = 87ms cycle, ~11.5 flashes/sec)
        self.stimulus = P300Stimulus(duration=10.0, flash_duration_ms=62, isi_ms=25)
        self.classifier = P300Classifier(sample_rate=250.0)
        self.controller = BCICursorController(duration=10.0)
        
        # Flash onset tracking for epoching
        self._flash_onsets: List[Tuple[str, str, float]] = []  # (position, color, timestamp)
        self._eeg_buffer: List[Tuple[NDArray, float]] = []  # (samples, timestamp)
        self._buffer_max_seconds: float = 2.0  # Keep 2 seconds of data for epoching
        
        # Check screen calibration compatibility
        self._check_screen_compatibility()
        
        # LSL connection state
        self._use_lsl = False
        self._lsl_connected = False
        
        # LSL preprocessor (handles both LSL and simulated)
        self.preprocessor = LSLPreprocessor(sample_rate=250, n_channels=8)
        
        # Simulated EEG source - ONLY available in dev mode
        if not self.EXPERIMENT_MODE:
            self.eeg_source = SimulatedEEGSource(sample_rate=250, n_channels=8)
        else:
            self.eeg_source = None  # Disabled in experiment mode
        
        # Marker sender for LSL
        self.marker_sender = None
        if LSL_AVAILABLE:
            try:
                self.marker_sender = LSLMarkerSender()
            except:
                pass
        
        # Current score
        self.current_score: Optional[BCIScore] = None
        
        # Calibration
        self._calibration_data: Optional[CalibrationData] = None
        self._calibration_session: Optional[CalibrationSession] = None
        self._cal_trial_timer: Optional[QTimer] = None
        self._cal_eeg_buffer: List = []
        self._cal_timestamp_buffer: List = []
        
        # Session state
        self._mode = SessionMode.IDLE
        
        # Timers
        # NOTE: We do NOT use _stimulus_timer anymore - FlickerWidget's internal timer handles updates
        # This matches the screen calibration protocol exactly (single timer per widget)
        
        self._composition_timer = QTimer()
        self._composition_timer.timeout.connect(self._update_composition)
        # Use 50ms interval to reduce event loop blocking
        # This allows flickering to run smoothly while still providing responsive classification
        # 20Hz update rate is sufficient for BCI control
        self._composition_timer.setInterval(50)  # 20Hz update rate
        
        # Setup UI
        self._setup_ui()
        
        # Apply dark theme
        self._apply_theme()
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # P300 flash targets
        # Top target (UP) with separate indicator
        top_container = QHBoxLayout()
        top_container.setSpacing(10)
        self.top_indicator = IndicatorLight()
        # Use stimulus's target instance so color updates are synchronized
        self.top_target = P300FlashWidget("top", target=self.stimulus.top_target)
        top_container.addWidget(self.top_indicator)
        top_container.addWidget(self.top_target)
        top_container.addStretch()
        top_widget = QWidget()
        top_widget.setLayout(top_container)
        layout.addWidget(top_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Composition canvas
        self.canvas = CompositionCanvas()
        layout.addWidget(self.canvas, stretch=1)
        
        # Bottom target (DOWN) with separate indicator
        bottom_container = QHBoxLayout()
        bottom_container.setSpacing(10)
        self.bottom_indicator = IndicatorLight()
        # Use stimulus's target instance so color updates are synchronized
        self.bottom_target = P300FlashWidget("bottom", target=self.stimulus.bottom_target)
        bottom_container.addWidget(self.bottom_indicator)
        bottom_container.addWidget(self.bottom_target)
        bottom_container.addStretch()
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_container)
        layout.addWidget(bottom_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Status bar
        self.status_label = QLabel("Ready - Press 'Start Composition' to begin")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %vs remaining")
        layout.addWidget(self.progress_bar)
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(panel)
        
        # Composition controls
        comp_group = QGroupBox("Composition")
        comp_layout = QHBoxLayout(comp_group)
        
        self.start_btn = QPushButton("▶ Start Composition")
        self.start_btn.clicked.connect(self._start_composition)
        comp_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⬛ Stop")
        self.stop_btn.clicked.connect(self._stop_composition)
        self.stop_btn.setEnabled(False)
        comp_layout.addWidget(self.stop_btn)
        
        layout.addWidget(comp_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 60)
        self.duration_spin.setValue(10)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.valueChanged.connect(self._update_duration)
        settings_layout.addWidget(self.duration_spin)
        
        settings_layout.addWidget(QLabel("Waveform:"))
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(["Sine", "Triangle", "Sawtooth", "Square"])
        settings_layout.addWidget(self.waveform_combo)
        
        layout.addWidget(settings_group)
        
        # Playback controls
        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout(playback_group)
        
        self.play_btn = QPushButton("🔊 Play Score")
        self.play_btn.clicked.connect(self._play_score)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self._save_score)
        self.save_btn.setEnabled(False)
        playback_layout.addWidget(self.save_btn)
        
        self.export_btn = QPushButton("📤 Export WAV")
        self.export_btn.clicked.connect(self._export_wav)
        self.export_btn.setEnabled(False)
        playback_layout.addWidget(self.export_btn)
        
        layout.addWidget(playback_group)
        
        # Calibration
        cal_group = QGroupBox("Calibration")
        cal_layout = QHBoxLayout(cal_group)
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self._start_calibration)
        self.calibrate_btn.setToolTip("Record your brain's response to each frequency")
        cal_layout.addWidget(self.calibrate_btn)
        
        self.cal_status = QLabel("Not calibrated")
        self.cal_status.setStyleSheet("color: #ff6b6b;")
        cal_layout.addWidget(self.cal_status)
        
        self.load_cal_btn = QPushButton("Load")
        self.load_cal_btn.clicked.connect(self._load_calibration)
        self.load_cal_btn.setToolTip("Load saved calibration")
        cal_layout.addWidget(self.load_cal_btn)
        
        layout.addWidget(cal_group)
        
        # Data Validation
        validation_group = QGroupBox("Data Validation")
        validation_layout = QHBoxLayout(validation_group)
        
        self.validate_btn = QPushButton("🔍 Check Data")
        self.validate_btn.clicked.connect(self._start_data_validation)
        self.validate_btn.setToolTip("Run 10-second test with flickering, record data, and generate diagnostic plots")
        validation_layout.addWidget(self.validate_btn)
        
        layout.addWidget(validation_group)
        
        # LSL Connection
        lsl_group = QGroupBox("EEG Source")
        lsl_layout = QHBoxLayout(lsl_group)
        
        if self.EXPERIMENT_MODE:
            self.lsl_status = QLabel("NOT CONNECTED")
            self.lsl_status.setStyleSheet("color: #ff4444; font-weight: bold;")  # Red = must connect
        else:
            self.lsl_status = QLabel("Simulated (DEV)")
            self.lsl_status.setStyleSheet("color: #ffa500;")  # Orange for dev mode
        lsl_layout.addWidget(self.lsl_status)
        
        self.connect_lsl_btn = QPushButton("Connect LSL")
        self.connect_lsl_btn.clicked.connect(self._toggle_lsl_connection)
        if not LSL_AVAILABLE:
            self.connect_lsl_btn.setEnabled(False)
            self.connect_lsl_btn.setToolTip("pylsl not installed")
        lsl_layout.addWidget(self.connect_lsl_btn)
        
        layout.addWidget(lsl_group)
        
        return panel
    
    def _apply_theme(self) -> None:
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #3a3a4e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3a3a5e;
                border: 1px solid #5a5a7e;
                border-radius: 5px;
                padding: 8px 15px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4a4a6e;
            }
            QPushButton:pressed {
                background-color: #2a2a4e;
            }
            QPushButton:disabled {
                background-color: #2a2a3e;
                color: #666;
            }
            QComboBox, QSpinBox {
                background-color: #2a2a4e;
                border: 1px solid #4a4a6e;
                border-radius: 3px;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #3a3a4e;
                border-radius: 5px;
                text-align: center;
                background-color: #2a2a3e;
            }
            QProgressBar::chunk {
                background-color: #42c887;
                border-radius: 4px;
            }
            QFrame {
                border: 1px solid #3a3a4e;
                border-radius: 5px;
            }
        """)
    
    def _update_duration(self, value: int) -> None:
        """Update composition duration."""
        self.stimulus.duration = float(value)
        self.controller.duration = float(value)
        self.canvas.set_duration(float(value))
    
    def _start_composition(self) -> None:
        """Start BCI composition session."""
        
        # EXPERIMENT MODE: Require LSL connection and calibration
        if self.EXPERIMENT_MODE:
            if not self._lsl_connected:
                QMessageBox.critical(
                    self,
                    "EXPERIMENT MODE - LSL Required",
                    "EXPERIMENT MODE is enabled.\n\n"
                    "You MUST connect to a real EEG device via LSL before starting.\n\n"
                    "Simulated data is DISABLED to ensure data integrity."
                )
                return
            
            if not self.classifier.is_calibrated:
                reply = QMessageBox.warning(
                    self,
                    "EXPERIMENT MODE - Calibration Recommended",
                    "EXPERIMENT MODE is enabled but classifier is NOT calibrated.\n\n"
                    "Calibration is STRONGLY recommended for accurate results.\n\n"
                    "Continue without calibration?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
        
        self._mode = SessionMode.COMPOSING
        
        # Send marker if LSL available
        if self.marker_sender:
            self.marker_sender.send("Composition Start")
        
        # Reset components
        self.controller.reset()
        self.classifier.reset()
        self.preprocessor.reset()
        self.canvas.clear()
        
        # Initialize performance metrics tracking
        self._classification_results = []  # Store all classification results
        self._classification_times = []  # Store timing for each classification
        self._composition_start_time = time.perf_counter()
        
        # Configure
        duration = self.duration_spin.value()
        self.controller.duration = float(duration)
        self.stimulus.duration = float(duration)
        self.canvas.set_duration(float(duration))
        
        # Generate and verify color sequences BEFORE starting
        # Duration must be set first since sequence length depends on it
        print("[P300] Generating and verifying color sequences...")
        if not self.stimulus.generate_and_verify_sequences():
            QMessageBox.critical(
                self,
                "Sequence Generation Failed",
                "Failed to generate valid color sequences.\n\n"
                "This should not happen. Please restart the application."
            )
            return
        
        # Create data logging directory
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_log_dir = Path("p300_sessions") / timestamp_str
        self._session_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Start
        self.controller.start()
        # Start P300 stimulus (handles flash timing)
        self.stimulus.start()
        self._flash_onsets.clear()
        self._eeg_buffer.clear()
        
        # Initialize logging
        self._log_trigger_file = self._session_log_dir / "triggers.jsonl"
        self._log_eeg_file = self._session_log_dir / "eeg_data.npy"
        self._log_metadata_file = self._session_log_dir / "metadata.json"
        
        # Save metadata
        metadata = {
            "timestamp": timestamp_str,
            "duration": duration,
            "flash_duration_ms": self.stimulus.flash_duration_ms,
            "isi_ms": self.stimulus.isi_ms,
            "target_probability": self.stimulus.target_probability,
            "expected_top_sequence": self.stimulus._top_color_sequence,
            "expected_bottom_sequence": self.stimulus._bottom_color_sequence,
            "lsl_connected": self._lsl_connected,
            "device": self.preprocessor._lsl_receiver.stream_name if hasattr(self.preprocessor, '_lsl_receiver') and self.preprocessor._lsl_receiver else None
        }
        with open(self._log_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[P300] Session data logging to: {self._session_log_dir}")
        
        # Update UI first
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        # self.random_btn.setEnabled(False)  # Removed - use calibration instead
        self.duration_spin.setEnabled(False)
        self.waveform_combo.setEnabled(False)
        self.canvas.set_composing(True)
        
        # Use same pattern as calibration: small delay before starting flickering
        # This ensures clean startup and prevents event loop blocking
        QTimer.singleShot(200, lambda: self._start_composition_flickering())
        
        # Start composition timer AFTER flickering starts (same pattern as calibration)
        # This prevents the composition timer from interfering with flickering startup
        
        self.status_label.setText("Composing... Focus on TOP target to move UP, BOTTOM target to move DOWN")
    
    def _stop_composition(self) -> None:
        """Stop composition and finalize score."""
        # Send marker if LSL available
        if self.marker_sender:
            self.marker_sender.send("Composition End")
        
        # Verify trigger alignment before stopping
        is_aligned, verification_report = self.stimulus.verify_trigger_alignment()
        if not is_aligned:
            print(f"[P300 WARNING] Trigger alignment mismatch detected!")
            print(f"  Expected: {verification_report['total_expected']} flashes")
            print(f"  Actual: {verification_report['total_actual']} flashes")
            print(f"  Top matches: {verification_report['top_matches']}/{len(self.stimulus._top_color_sequence)}")
            print(f"  Bottom matches: {verification_report['bottom_matches']}/{len(self.stimulus._bottom_color_sequence)}")
            if verification_report['top_mismatches'] > 0:
                print(f"  Top mismatches: {verification_report['top_mismatches']}")
                for mismatch in verification_report['top_mismatch_details'][:5]:  # Show first 5
                    print(f"    Index {mismatch['index']}: expected {mismatch['expected']}, got {mismatch.get('actual', 'MISSING')}")
            if verification_report['bottom_mismatches'] > 0:
                print(f"  Bottom mismatches: {verification_report['bottom_mismatches']}")
                for mismatch in verification_report['bottom_mismatch_details'][:5]:  # Show first 5
                    print(f"    Index {mismatch['index']}: expected {mismatch['expected']}, got {mismatch.get('actual', 'MISSING')}")
        else:
            print(f"[P300] Trigger alignment verified: {verification_report['total_actual']} flashes match expected sequence")
        
        # Stop flickering - EXACTLY match screen calibration protocol
        # Widgets are independent, no need to call stimulus.stop()
        self.top_target.set_active(False)
        self.bottom_target.set_active(False)
        self._composition_timer.stop()
        
        self.controller.stop()
        # NOTE: No stimulus.stop() - widgets are independent (same as screen calibration)
        self.bottom_target.set_active(False)
        self.canvas.set_composing(False)
        
        self._finalize_score()
        
        self._mode = SessionMode.IDLE
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # self.random_btn.setEnabled(True)  # Removed - use calibration instead
        self.duration_spin.setEnabled(True)
        self.waveform_combo.setEnabled(True)
        
        self.status_label.setText("Composition complete! Play or save your score.")
    
    def _check_screen_compatibility(self) -> None:
        """Screen compatibility check - simplified: using fixed 15Hz/12Hz frequencies."""
        # No longer checking screen calibration - using fixed frequencies
        pass
    
    # REMOVED: _update_stimulus() - no longer needed
    # FlickerWidget's internal timer handles all updates (same as screen calibration)
    
    def _toggle_lsl_connection(self) -> None:
        """Toggle LSL connection."""
        if self._lsl_connected:
            # Disconnect
            self.preprocessor.disconnect_lsl()
            self._lsl_connected = False
            self._use_lsl = False
            self.connect_lsl_btn.setText("Connect LSL")
            self.lsl_status.setText("Simulated")
            self.lsl_status.setStyleSheet("color: #ffa500;")
            self.status_label.setText("Disconnected from LSL - using simulated EEG")
        else:
            # Try to connect
            self.status_label.setText("Searching for LSL streams...")
            QApplication.processEvents()
            
            if self.preprocessor.connect_lsl():
                self._lsl_connected = True
                self._use_lsl = True
                self.connect_lsl_btn.setText("Disconnect")
                stream_info = self.preprocessor._lsl_receiver.stream_info
                self.lsl_status.setText(f"Connected: {stream_info.name}")
                self.lsl_status.setStyleSheet("color: #00ff00;")  # Green
                
                # Update classifier sample rate to match preprocessor
                self.classifier.sample_rate = self.preprocessor.sample_rate
                self.status_label.setText(f"Connected to {stream_info.name} @ {stream_info.sample_rate}Hz")
            else:
                if self.EXPERIMENT_MODE:
                    self.status_label.setText("No LSL streams found - CANNOT proceed in experiment mode")
                    QMessageBox.critical(
                        self,
                        "EXPERIMENT MODE - LSL Required",
                        "No EEG streams found!\n\n"
                        "EXPERIMENT MODE requires a real EEG connection.\n\n"
                        "Make sure your g.tec Unicorn Black is:\n"
                        "1. Powered on and paired via Bluetooth\n"
                        "2. LSL streaming started in Unicorn Suite\n\n"
                        "Cannot proceed without real EEG data."
                    )
                else:
                    self.status_label.setText("No LSL streams found - using simulated EEG (DEV MODE)")
                    QMessageBox.information(
                        self,
                        "LSL Connection (DEV MODE)",
                        "No EEG streams found.\n\n"
                        "DEV MODE: Using simulated EEG data.\n\n"
                        "For real experiments, set EXPERIMENT_MODE = True"
                    )
    
    def _update_composition(self) -> None:
        """Update composition state - P300 paradigm."""
        if self.controller.state != ControllerState.RUNNING:
            if self.controller.state == ControllerState.COMPLETED:
                self._stop_composition()
            return
        
        # Update P300 stimulus (handles flash timing)
        self.stimulus.update()
        
        # Update flash widgets
        self.top_target.update_flash()
        self.bottom_target.update_flash()
        
        # Track flash onsets (for epoching)
        flash_onsets = self.stimulus.get_flash_onsets()
        if flash_onsets:
            # Get new flash onsets since last check
            if len(flash_onsets) > len(self._flash_onsets):
                new_onsets = flash_onsets[len(self._flash_onsets):]
                self._flash_onsets.extend(new_onsets)
                
                # Send markers for new flashes (include color info)
                if self.marker_sender:
                    for position, color, flash_time in new_onsets:
                        # Ensure color is a string (not being iterated)
                        color_str = str(color) if isinstance(color, str) else color
                        is_target = "TARGET" if color_str == "red" else "NONTARGET"
                        marker = f"P300_{position.upper()}_{color_str.upper()}_{is_target}"
                        self.marker_sender.send(marker)
                        
                        # Log trigger to file
                        if hasattr(self, '_log_trigger_file'):
                            trigger_data = {
                                "timestamp": flash_time,
                                "position": position,
                                "color": color_str,
                                "is_target": is_target == "TARGET",
                                "marker": marker
                            }
                            with open(self._log_trigger_file, 'a') as f:
                                f.write(json.dumps(trigger_data) + '\n')
        
        # Get EEG data and classify using P300 epoching
        if self._use_lsl and self._lsl_connected:
            # Pull chunk and add to buffer
            n_samples_per_update = int(0.050 * self.preprocessor.sample_rate)  # 50ms worth
            processed = self.preprocessor.pull_and_process(n_samples=n_samples_per_update)
            
            # Log EEG data
            if hasattr(self, '_log_eeg_file') and processed is not None and len(processed) > 0:
                current_time = time.perf_counter()
                # Append to buffer for batch saving
                if not hasattr(self, '_eeg_log_buffer'):
                    self._eeg_log_buffer = []
                self._eeg_log_buffer.append({
                    "data": processed.tolist(),
                    "timestamp": current_time
                })
            
            if len(processed) > 0:
                # Add to buffer with LSL-synchronized timestamp
                try:
                    from pylsl import local_clock
                    current_time = local_clock()  # Use LSL clock for synchronization
                except ImportError:
                    current_time = time.perf_counter()  # Fallback if LSL not available
                self._eeg_buffer.append((processed, current_time))
                
                # Trim buffer to max duration
                buffer_start_time = current_time - self._buffer_max_seconds
                self._eeg_buffer = [(data, ts) for data, ts in self._eeg_buffer if ts >= buffer_start_time]
                
                # Classify using P300 epoching if we have flash onsets
                if len(self._flash_onsets) >= self.classifier.n_epochs_to_average:
                    # Reconstruct continuous EEG data with timestamps
                    if len(self._eeg_buffer) > 0:
                        all_data = np.vstack([data for data, _ in self._eeg_buffer])
                        # Create timestamps array - CRITICAL: Use LSL-synchronized timestamps
                        buffer_times = []
                        sample_rate = self.preprocessor.sample_rate
                        for i, (data, ts) in enumerate(self._eeg_buffer):
                            n_samples = len(data)
                            # Timestamps are already in LSL time domain
                            timestamps = np.linspace(ts, ts + (n_samples - 1) / sample_rate, n_samples)
                            buffer_times.extend(timestamps)
                        buffer_times = np.array(buffer_times)
                        
                        # Classify using P300 oddball paradigm
                        # Flash onsets now include color: (position, color, time)
                        result = self.classifier.classify_averaged(
                            all_data,
                            self._flash_onsets,
                            buffer_times
                        )
                        
                        # Track classification metrics
                        classify_time = time.perf_counter()
                        self._classification_results.append(result)
                        self._classification_times.append(classify_time - self._composition_start_time)
                        
                        # Debug: Log classification results periodically
                        if not hasattr(self, '_last_classify_log_time'):
                            self._last_classify_log_time = 0
                        current_time_log = time.time()
                        if current_time_log - self._last_classify_log_time > 1.0:  # Log every second
                            print(f"[P300 CLASSIFY] Target={result.target.name}, Confidence={result.confidence:.2f}, "
                                  f"Top_amp={result.p300_amplitude_top:.3f}, Bottom_amp={result.p300_amplitude_bottom:.3f}, "
                                  f"Score={result.raw_score:.3f}")
                            self._last_classify_log_time = current_time_log
                    else:
                        result = ClassificationResult(
                            target=AttentionTarget.NONE,
                            confidence=0.0,
                            p300_amplitude_top=0.0,
                            p300_amplitude_bottom=0.0,
                            raw_score=0.0
                        )
                else:
                    # Not enough epochs yet - wait
                    result = ClassificationResult(
                        target=AttentionTarget.NONE,
                        confidence=0.0,
                        p300_amplitude_top=0.0,
                        p300_amplitude_bottom=0.0,
                        raw_score=0.0
                    )
            else:
                # No data available - hold position
                result = ClassificationResult(
                    target=AttentionTarget.NONE,
                    confidence=0.0,
                    p300_amplitude_top=0.0,
                    p300_amplitude_bottom=0.0,
                    raw_score=0.0
                )
        elif self.EXPERIMENT_MODE:
            # EXPERIMENT MODE: No LSL = ERROR, should not reach here
            # This is a safety check - _start_composition should block this
            self._stop_composition()
            QMessageBox.critical(
                self,
                "EXPERIMENT MODE ERROR",
                "Lost LSL connection during experiment!\n\n"
                "Composition stopped to prevent invalid data."
            )
            return
        else:
            # DEV MODE ONLY: Simulated data (disabled in experiments)
            if self.eeg_source is None:
                # Safety: should not happen, but handle gracefully
                result = ClassificationResult(
                    target=AttentionTarget.NONE,
                    confidence=0.0,
                    power_higher_freq=0.0,
                    power_lower_freq=0.0,
                    raw_score=0.0
                )
            else:
                # Fixed frequencies: 15Hz (UP) and 12Hz (DOWN)
                higher_freq, lower_freq = 15.0, 12.0
                
                # Simulate user attention based on cursor position
                current_pitch = self.controller.position.pitch
                if current_pitch < 0.4:
                    self.eeg_source.set_target(higher_freq)  # Attend to UP
                elif current_pitch > 0.6:
                    self.eeg_source.set_target(lower_freq)  # Attend to DOWN
                else:
                    self.eeg_source.set_target(None)  # No strong attention
                
                # Generate and process EEG
                eeg_chunk = self.eeg_source.generate_chunk(16)
                processed = self.preprocessor.process_chunk(eeg_chunk)
                
                # Classify with CCA
                eeg_buffer = self.preprocessor.get_recent_data(0.5)
                result = self.classifier.classify(eeg_buffer, method="cca")
        
        # Update controller with classification
        pos = self.controller.update(result)
        
        # Update canvas
        self.canvas.set_cursor_position(pos)
        self.canvas.set_trail(self.controller.get_trail_as_tuples())
        
        # Update progress
        progress = self.controller.progress
        self.progress_bar.setValue(int(progress * 100))
        remaining = self.controller.duration * (1 - progress)
        self.progress_bar.setFormat(f"{int(progress * 100)}% - {remaining:.1f}s remaining")
    
    def _finalize_score(self) -> None:
        """Create score from completed composition."""
        try:
            trail = self.controller.get_trail_as_tuples()
            
            # Calculate performance metrics
            try:
                performance_metrics = self._calculate_performance_metrics()
            except Exception as e:
                print(f"[ERROR] Failed to calculate performance metrics: {e}")
                import traceback
                traceback.print_exc()
                performance_metrics = {}
            
            try:
                # P300 doesn't have frequencies, use flash timing instead
                metadata = {
                    'simulated': not self._use_lsl,
                    'paradigm': 'P300',
                    'flash_duration_ms': self.stimulus.flash_duration_ms,
                    'isi_ms': self.stimulus.isi_ms,
                    'performance_metrics': performance_metrics
                }
                # Add frequency info only if it exists (for SSVEP compatibility)
                if hasattr(self.stimulus, 'top_frequency'):
                    metadata['top_frequency'] = self.stimulus.top_frequency
                    metadata['bottom_frequency'] = self.stimulus.bottom_frequency
                
                self.current_score = BCIScore(
                    trail=trail,
                    duration=self.controller.duration,
                    waveform_name=self.waveform_combo.currentText(),
                    metadata=metadata
                )
            except Exception as e:
                print(f"[ERROR] Failed to create BCIScore: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Enable playback controls
            self.play_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Show statistics with performance summary
            try:
                stats = self.current_score.get_statistics()
                perf_summary = self._format_performance_summary(performance_metrics)
                self.status_label.setText(
                    f"Score complete: {stats['num_points']} points, "
                    f"pitch range: {stats['pitch_range']:.2f}, "
                    f"total movement: {stats['total_movement']:.2f}\n{perf_summary}"
                )
            except Exception as e:
                print(f"[ERROR] Failed to get statistics: {e}")
                import traceback
                traceback.print_exc()
                self.status_label.setText("Score complete! Play or save your score.")
            
            # Print detailed performance report (also saves to file)
            try:
                self._print_performance_report(performance_metrics)
            except Exception as e:
                print(f"[ERROR] Failed to print performance report: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to finalize score: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error Finalizing Score",
                f"Failed to finalize score:\n\n{str(e)}\n\nCheck console for details."
            )
            self.status_label.setText(f"Error: {str(e)}")
    
    def _calculate_performance_metrics(self) -> dict:
        """Calculate performance metrics from composition session."""
        metrics = {}
        
        # Flicker rate accuracy - calculate from actual flicker cycles
        if hasattr(self.top_target, '_flicker_state_changes') and len(self.top_target._flicker_state_changes) >= 20:
            top_changes = self.top_target._flicker_state_changes
            intervals = [top_changes[i] - top_changes[i-1] for i in range(1, len(top_changes))]
            if intervals:
                mean_interval = np.mean(intervals)
                cycle_period = mean_interval * 2  # Full cycle = 2 state changes
                top_actual_rate = 1.0 / cycle_period if cycle_period > 0 else 0
                top_target_rate = self.top_target.frequency
                top_error = abs(top_actual_rate - top_target_rate)
                top_error_pct = (top_error / top_target_rate * 100) if top_target_rate > 0 else 0
                metrics['top_flicker'] = {
                    'target_hz': top_target_rate,
                    'actual_hz': top_actual_rate,
                    'error_hz': top_error,
                    'error_pct': top_error_pct
                }
        
        if hasattr(self.bottom_target, '_flicker_state_changes') and len(self.bottom_target._flicker_state_changes) >= 20:
            bottom_changes = self.bottom_target._flicker_state_changes
            intervals = [bottom_changes[i] - bottom_changes[i-1] for i in range(1, len(bottom_changes))]
            if intervals:
                mean_interval = np.mean(intervals)
                cycle_period = mean_interval * 2  # Full cycle = 2 state changes
                bottom_actual_rate = 1.0 / cycle_period if cycle_period > 0 else 0
                bottom_target_rate = self.bottom_target.frequency
                bottom_error = abs(bottom_actual_rate - bottom_target_rate)
                bottom_error_pct = (bottom_error / bottom_target_rate * 100) if bottom_target_rate > 0 else 0
                metrics['bottom_flicker'] = {
                    'target_hz': bottom_target_rate,
                    'actual_hz': bottom_actual_rate,
                    'error_hz': bottom_error,
                    'error_pct': bottom_error_pct
                }
        
        # Classification performance
        if hasattr(self, '_classification_results') and len(self._classification_results) > 0:
            confidences = [r.confidence for r in self._classification_results]
            targets = [r.target for r in self._classification_results]
            
            # P300 uses amplitudes, not power
            if hasattr(self._classification_results[0], 'p300_amplitude_top'):
                # P300 results
                p300_top = [r.p300_amplitude_top for r in self._classification_results]
                p300_bottom = [r.p300_amplitude_bottom for r in self._classification_results]
                metrics['classification'] = {
                    'n_classifications': len(self._classification_results),
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'min_confidence': np.min(confidences),
                    'max_confidence': np.max(confidences),
                    'target_distribution': {
                        'UP': sum(1 for t in targets if t == AttentionTarget.UP),
                        'DOWN': sum(1 for t in targets if t == AttentionTarget.DOWN),
                        'NONE': sum(1 for t in targets if t == AttentionTarget.NONE)
                    },
                    'mean_p300_top': np.mean(p300_top),
                    'mean_p300_bottom': np.mean(p300_bottom)
                }
            else:
                # SSVEP results (backward compatibility)
                power_higher = [r.power_higher_freq for r in self._classification_results]
                power_lower = [r.power_lower_freq for r in self._classification_results]
                metrics['classification'] = {
                    'n_classifications': len(self._classification_results),
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'min_confidence': np.min(confidences),
                    'max_confidence': np.max(confidences),
                    'target_distribution': {
                        'UP': sum(1 for t in targets if t == AttentionTarget.UP),
                        'DOWN': sum(1 for t in targets if t == AttentionTarget.DOWN),
                        'NONE': sum(1 for t in targets if t == AttentionTarget.NONE)
                    },
                    'mean_power_higher': np.mean(power_higher),
                    'mean_power_lower': np.mean(power_lower)
                }
            
            # Classification rate (classifications per second)
            if hasattr(self, '_composition_start_time') and hasattr(self, '_classification_times'):
                total_time = self.controller.duration
                if total_time > 0:
                    metrics['classification']['rate_per_sec'] = len(self._classification_results) / total_time
        
        return metrics
    
    def _format_performance_summary(self, metrics: dict) -> str:
        """Format a brief performance summary for status label."""
        parts = []
        
        if 'top_flicker' in metrics:
            tf = metrics['top_flicker']
            parts.append(f"Top: {tf['actual_hz']:.2f}Hz ({tf['error_pct']:.1f}% error)")
        
        if 'bottom_flicker' in metrics:
            bf = metrics['bottom_flicker']
            parts.append(f"Bottom: {bf['actual_hz']:.2f}Hz ({bf['error_pct']:.1f}% error)")
        
        if 'classification' in metrics:
            cf = metrics['classification']
            parts.append(f"Conf: {cf['mean_confidence']:.2f}")
        
        return " | ".join(parts) if parts else ""
    
    def _print_performance_report(self, metrics: dict) -> None:
        """Print detailed performance report to console."""
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT - During Live Data Capture + Preprocessing + Classification")
        print("=" * 60)
        print("NOTE: These rates show flicker accuracy WHILE LSL pulling, preprocessing,")
        print("      and classification were running simultaneously.")
        print("=" * 60)
        
        # Flicker rate accuracy
        if 'top_flicker' in metrics:
            tf = metrics['top_flicker']
            print(f"\nTop Flicker Rate (during composition):")
            print(f"  Target: {tf['target_hz']:.3f} Hz")
            print(f"  Actual: {tf['actual_hz']:.3f} Hz")
            print(f"  Error:  {tf['error_hz']:.3f} Hz ({tf['error_pct']:.1f}%)")
            if tf['error_pct'] > 5.0:
                print(f"  ⚠️  WARNING: High error rate! Flicker degraded during composition.")
            elif tf['error_pct'] > 2.0:
                print(f"  ⚠️  Moderate error - flicker rate slowed during composition.")
            else:
                print(f"  ✓ Good accuracy - flicker rate maintained during composition.")
        
        if 'bottom_flicker' in metrics:
            bf = metrics['bottom_flicker']
            print(f"\nBottom Flicker Rate (during composition):")
            print(f"  Target: {bf['target_hz']:.3f} Hz")
            print(f"  Actual: {bf['actual_hz']:.3f} Hz")
            print(f"  Error:  {bf['error_hz']:.3f} Hz ({bf['error_pct']:.1f}%)")
            if bf['error_pct'] > 5.0:
                print(f"  ⚠️  WARNING: High error rate! Flicker degraded during composition.")
            elif bf['error_pct'] > 2.0:
                print(f"  ⚠️  Moderate error - flicker rate slowed during composition.")
            else:
                print(f"  ✓ Good accuracy - flicker rate maintained during composition.")
        
        # Classification performance
        if 'classification' in metrics:
            cf = metrics['classification']
            print(f"\nClassification Performance:")
            print(f"  Total classifications: {cf['n_classifications']}")
            if 'rate_per_sec' in cf:
                print(f"  Classification rate: {cf['rate_per_sec']:.1f} Hz")
            print(f"  Mean confidence: {cf['mean_confidence']:.3f} (std: {cf['std_confidence']:.3f})")
            print(f"  Confidence range: [{cf['min_confidence']:.3f}, {cf['max_confidence']:.3f}]")
            print(f"  Target distribution:")
            dist = cf['target_distribution']
            for target, count in dist.items():
                pct = (count / cf['n_classifications'] * 100) if cf['n_classifications'] > 0 else 0
                print(f"    {target}: {count} ({pct:.1f}%)")
            # P300 uses amplitudes, not power
            if 'mean_p300_top' in cf:
                print(f"  Mean P300 amplitude (top): {cf['mean_p300_top']:.3f} μV")
                print(f"  Mean P300 amplitude (bottom): {cf['mean_p300_bottom']:.3f} μV")
            elif 'mean_power_higher' in cf:
                print(f"  Mean power (higher): {cf['mean_power_higher']:.3f}")
                print(f"  Mean power (lower):  {cf['mean_power_lower']:.3f}")
        
        print("=" * 60 + "\n")
        
        # Also save to file for easy access
        self._save_performance_report(metrics)
    
    def _save_performance_report(self, metrics: dict) -> None:
        """Save performance report to file for easy access."""
        from datetime import datetime
        report_file = Path("performance_report.txt")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("PERFORMANCE REPORT - During Live Data Capture + Preprocessing + Classification\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                if 'top_flicker' in metrics:
                    tf = metrics['top_flicker']
                    f.write(f"Top Flicker Rate:\n")
                    f.write(f"  Target: {tf['target_hz']:.3f} Hz\n")
                    f.write(f"  Actual: {tf['actual_hz']:.3f} Hz\n")
                    f.write(f"  Error:  {tf['error_hz']:.3f} Hz ({tf['error_pct']:.1f}%)\n\n")
                
                if 'bottom_flicker' in metrics:
                    bf = metrics['bottom_flicker']
                    f.write(f"Bottom Flicker Rate:\n")
                    f.write(f"  Target: {bf['target_hz']:.3f} Hz\n")
                    f.write(f"  Actual: {bf['actual_hz']:.3f} Hz\n")
                    f.write(f"  Error:  {bf['error_hz']:.3f} Hz ({bf['error_pct']:.1f}%)\n\n")
                
                if 'classification' in metrics:
                    cf = metrics['classification']
                    f.write(f"Classification Performance:\n")
                    f.write(f"  Total classifications: {cf['n_classifications']}\n")
                    if 'rate_per_sec' in cf:
                        f.write(f"  Classification rate: {cf['rate_per_sec']:.1f} Hz\n")
                    f.write(f"  Mean confidence: {cf['mean_confidence']:.3f}\n")
                    f.write(f"  Target distribution:\n")
                    dist = cf['target_distribution']
                    for target, count in dist.items():
                        pct = (count / cf['n_classifications'] * 100) if cf['n_classifications'] > 0 else 0
                        f.write(f"    {target}: {count} ({pct:.1f}%)\n")
            
            print(f"[PERFORMANCE] Report saved to: {report_file.absolute()}")
        except Exception as e:
            print(f"[ERROR] Failed to save performance report: {e}")
            import traceback
            traceback.print_exc()
    
    def _play_score(self) -> None:
        """Play the current score."""
        if not self.current_score:
            return
        
        self.status_label.setText("Playing score...")
        self.play_btn.setEnabled(False)
        
        try:
            play_score(self.current_score)
            self.status_label.setText("Playback complete")
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", str(e))
            self.status_label.setText(f"Playback error: {e}")
        finally:
            self.play_btn.setEnabled(True)
    
    def _save_score(self) -> None:
        """Save the current score to file."""
        if not self.current_score:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Score",
            "",
            "BCI Score (*.json)"
        )
        
        if filepath:
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            self.current_score.save(filepath)
            self.status_label.setText(f"Score saved to {filepath}")
    
    def _export_wav(self) -> None:
        """Export the current score as WAV."""
        if not self.current_score:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export WAV",
            "",
            "WAV Audio (*.wav)"
        )
        
        if filepath:
            if not filepath.endswith('.wav'):
                filepath += '.wav'
            
            self.status_label.setText("Exporting WAV...")
            try:
                synthesize_score(self.current_score, filepath)
                self.status_label.setText(f"Exported to {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))
                self.status_label.setText(f"Export error: {e}")
    
    def _run_random_test(self) -> None:
        """Run a test with random cursor movement."""
        from .controller import RandomController
        
        self._mode = SessionMode.COMPOSING
        
        # Create random controller
        duration = self.duration_spin.value()
        random_ctrl = RandomController(duration=float(duration))
        
        # Reset canvas
        self.canvas.clear()
        self.canvas.set_duration(float(duration))
        
        # Start random test
        random_ctrl.start()
        # NOTE: No stimulus.start() - widgets are independent (same as screen calibration)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        # self.random_btn.setEnabled(False)  # Removed - use calibration instead
        self.duration_spin.setEnabled(False)
        
        self.top_target.set_active(True)
        self.bottom_target.set_active(True)
        self.canvas.set_composing(True)
        
        self.status_label.setText("Running random test...")
        
        # Run test in timer
        def update_random():
            if random_ctrl.is_running:
                pos = random_ctrl.update()
                self.canvas.set_cursor_position(pos)
                self.canvas.set_trail([p.to_tuple() for p in random_ctrl.trail])
                
                # Flicker is updated via FlickerWidget's internal timer -> paintEvent (same as screen calibration)
                # No need to manually set intensity - it's calculated in real-time in paintEvent
                
                # Update progress
                progress = random_ctrl._controller.progress
                self.progress_bar.setValue(int(progress * 100))
            else:
                # Test complete
                random_timer.stop()
                # NOTE: No stimulus.stop() - widgets are independent (same as screen calibration)
                
                self.top_target.set_active(False)
                self.bottom_target.set_active(False)
                self.canvas.set_composing(False)
                
                # Create score from random trail
                trail = [p.to_tuple() for p in random_ctrl.trail]
                self.current_score = BCIScore(
                    trail=trail,
                    duration=float(duration),
                    waveform_name=self.waveform_combo.currentText(),
                    metadata={'test': True, 'random': True}
                )
                
                # Enable controls
                self.start_btn.setEnabled(True)
                # self.random_btn.setEnabled(True)  # Removed - use calibration instead
                self.duration_spin.setEnabled(True)
                self.play_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
                
                self._mode = SessionMode.IDLE
                self.status_label.setText("Random test complete! Play or save the result.")
        
        random_timer = QTimer()
        random_timer.timeout.connect(update_random)
        random_timer.start(16)
    
    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Stop flickering widgets (their internal timers)
        self.top_target.set_active(False)
        self.bottom_target.set_active(False)
        self._composition_timer.stop()
        self.controller.stop()
        # NOTE: No stimulus.stop() - widgets are independent (same as screen calibration)
        event.accept()
    
    # ==================== CALIBRATION ====================
    
    def _start_calibration(self) -> None:
        """Start the calibration process."""
        if not self._use_lsl or not self._lsl_connected:
            QMessageBox.warning(
                self,
                "LSL Required",
                "Please connect to LSL first.\n\n"
                "Calibration requires real EEG data from your headset."
            )
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Start Calibration",
            "Calibration will record your brain's response to each flickering target.\n\n"
            "You will see:\n"
            "- 3 trials of 15Hz (look at TOP)\n"
            "- 3 trials of 12Hz (look at BOTTOM)\n"
            "- Each trial is 5 seconds with 2 second rest\n\n"
            "Total time: ~40 seconds\n\n"
            "Ready to begin?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Clear any existing calibration file to ensure fresh data
        cal_path = Path("calibration_data.json")
        if cal_path.exists():
            try:
                cal_path.unlink()
                print(f"[CALIBRATION] Cleared existing calibration file: {cal_path}")
            except Exception as e:
                print(f"[CALIBRATION] Warning: Could not delete old calibration file: {e}")
        
        # Setup calibration
        self._mode = SessionMode.CALIBRATING
        self._calibration_data = CalibrationData(
            sample_rate=self.preprocessor.sample_rate,
            n_channels=8
        )
        
        print(f"[CALIBRATION] Starting new calibration session")
        print(f"[CALIBRATION] Sample rate: {self.preprocessor.sample_rate}")
        
        # Disable controls
        self.start_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(False)
        # self.random_btn.setEnabled(False)  # Removed - use calibration instead
        self.connect_lsl_btn.setEnabled(False)
        
        # Start calibration sequence - use CalibrationSession to get dynamic sequence
        from .calibration import CalibrationSession
        cal_session = CalibrationSession(n_trials_per_frequency=3, trial_duration=5.0)
        self._cal_trial_index = 0
        self._cal_sequence = cal_session.get_trial_sequence()  # Dynamic frequencies from screen calibration
        self._cal_trial_duration = 5.0
        self._cal_rest_duration = 2.0
        
        # Fixed frequencies: 15Hz (UP) and 12Hz (DOWN)
        self._cal_higher_freq, self._cal_lower_freq = 15.0, 12.0
        
        self.status_label.setText("Calibration starting... Get ready!")
        self.canvas.clear()
        
        # Initialize indicators (both off at start - flickering will start when indicator lights)
        self.top_indicator.set_active(False)
        self.bottom_indicator.set_active(False)
        
        # Do NOT start flickering yet - wait for indicator to light up first
        # Flickering will start when trial begins (after indicator lights)
        
        # Start with rest period
        QTimer.singleShot(1000, self._cal_start_rest)
    
    def _cal_start_rest(self) -> None:
        """Start rest period before trial."""
        try:
            if self._cal_trial_index >= len(self._cal_sequence):
                self._cal_finish()
                return
            
            freq = self._cal_sequence[self._cal_trial_index]
            # Compare against higher/lower frequencies (with tolerance)
            is_higher = abs(freq - self._cal_higher_freq) < 0.5
            target = f"TOP ({self._cal_higher_freq:.1f}Hz)" if is_higher else f"BOTTOM ({self._cal_lower_freq:.1f}Hz)"
            
            self.status_label.setText(
                f"REST - Next: Look at {target} - Trial {self._cal_trial_index + 1}/{len(self._cal_sequence)}"
            )
            
            # IMPORTANT: Keep flickering running during rest periods!
            # Do NOT call set_active(False) - this would reset the start time and cause drift
            # The flickering must remain active throughout the entire calibration session
            # This matches screen calibration behavior - flickering runs continuously
            
            # After rest, start trial
            QTimer.singleShot(int(self._cal_rest_duration * 1000), self._cal_start_trial)
            
        except Exception as e:
            print(f"Error in _cal_start_rest: {e}")
            import traceback
            traceback.print_exc()
            self._cal_finish()
    
    def _cal_start_trial(self) -> None:
        """Start a calibration trial."""
        try:
            freq = self._cal_sequence[self._cal_trial_index]
            # Compare against higher/lower frequencies (with tolerance)
            is_higher = abs(freq - self._cal_higher_freq) < 0.5
            target = "TOP" if is_higher else "BOTTOM"
            
            print(f"[CALIBRATION] Starting trial {self._cal_trial_index + 1}: {freq}Hz ({target})")
            self.status_label.setText(f"LOOK AT {target}! Recording {freq}Hz response...")
            
            # Clear buffers and counters
            self._cal_eeg_buffer = []
            self._cal_timestamp_buffer = []
            self._cal_no_data_count = 0
            
            # CORRECT SEQUENCE: Indicator lights FIRST, then flickering starts
            # This gives user time to see which target to look at before flickering begins
            is_higher = abs(freq - self._cal_higher_freq) < 0.5
            if is_higher:
                self.top_indicator.set_active(True)  # Green indicator - lights FIRST
                self.bottom_indicator.set_active(False)
            else:
                self.bottom_indicator.set_active(True)  # Green indicator - lights FIRST
                self.top_indicator.set_active(False)
            
            # Small delay to let user see the indicator, then start flickering
            # Both targets flicker (for consistent timing), but indicator shows which to focus on
            QTimer.singleShot(200, lambda: self._start_trial_flickering())
            
            # End trial after duration
            QTimer.singleShot(int(self._cal_trial_duration * 1000), self._cal_end_trial)
            
        except Exception as e:
            print(f"[CALIBRATION] Error in _cal_start_trial: {e}")
            import traceback
            traceback.print_exc()
            self._cal_finish()
    
    def _start_trial_flickering(self) -> None:
        """Start flickering for the current trial (called after indicator lights)."""
        # Start flickering targets - this ensures consistent timing
        # Both targets flicker, but indicator shows which one to focus on
        if not self.top_target._is_active:
            self.top_target.set_active(True)
        if not self.bottom_target._is_active:
            self.bottom_target.set_active(True)
        
        # Start recording timer (16ms = ~62.5Hz) - reduced frequency to avoid blocking event loop
        # The flickering timer (8ms) needs priority, so recording can be slower
        self._cal_record_timer = QTimer()
        self._cal_record_timer.timeout.connect(self._cal_record_sample)
        self._cal_record_timer.start(16)  # Reduced from 4ms to avoid blocking flickering updates
    
    def _start_composition_flickering(self) -> None:
        """Start flickering for main composition (same pattern as calibration)."""
        # Use EXACTLY the same pattern as calibration for consistent flickering
        # Start flickering targets
        if not self.top_target._is_active:
            self.top_target.set_active(True)
        if not self.bottom_target._is_active:
            self.bottom_target.set_active(True)
        
        # Start composition timer AFTER flickering is running
        # This prevents the composition timer from interfering with flickering startup
        self._composition_timer.start()
    
    def _cal_record_sample(self) -> None:
        """Record EEG sample during calibration trial."""
        try:
            if not self._use_lsl or not self._lsl_connected:
                print("[CALIBRATION] Warning: LSL not connected during recording")
                return
            
            # Pull data from LSL
            processed = self.preprocessor.pull_and_process(n_samples=8)
            
            if processed is not None and len(processed) > 0:
                for sample in processed:
                    self._cal_eeg_buffer.append(sample)
                    self._cal_timestamp_buffer.append(time.perf_counter())
                
                # Log occasionally to confirm data is coming in
                if len(self._cal_eeg_buffer) % 50 == 0:
                    print(f"[CALIBRATION] Recorded {len(self._cal_eeg_buffer)} samples")
            else:
                # Log if no data received
                if not hasattr(self, '_cal_no_data_count'):
                    self._cal_no_data_count = 0
                self._cal_no_data_count += 1
                if self._cal_no_data_count % 25 == 0:
                    print(f"[CALIBRATION] Warning: No data received ({self._cal_no_data_count} empty pulls)")
        except Exception as e:
            print(f"[CALIBRATION] Error in _cal_record_sample: {e}")
            import traceback
            traceback.print_exc()
    
    def _cal_end_trial(self) -> None:
        """End calibration trial and save data."""
        try:
            # Stop recording
            if hasattr(self, '_cal_record_timer') and self._cal_record_timer is not None:
                self._cal_record_timer.stop()
            
            # Stop recording timer
            # Stop flickering at end of trial - will restart when next trial begins
            # This matches user expectation: indicator lights, then flickering starts
            self.top_target.set_active(False)
            self.bottom_target.set_active(False)
            
            # Reset indicators
            self.top_indicator.set_active(False)
            self.bottom_indicator.set_active(False)
            
            # Save trial data
            freq = self._cal_sequence[self._cal_trial_index]
            
            if self._cal_eeg_buffer:
                eeg_data = np.array(self._cal_eeg_buffer)
                timestamps = np.array(self._cal_timestamp_buffer)
                
                print(f"[CALIBRATION] Trial {self._cal_trial_index + 1} ended: {len(eeg_data)} samples, shape: {eeg_data.shape}")
                
                self._calibration_data.add_trial(
                    frequency=freq,
                    eeg_data=eeg_data,
                    timestamps=timestamps,
                    duration=self._cal_trial_duration
                )
                
                self.status_label.setText(
                    f"Trial {self._cal_trial_index + 1} complete! Recorded {len(eeg_data)} samples"
                )
            else:
                print(f"[CALIBRATION] Trial {self._cal_trial_index + 1} ended: NO SAMPLES RECORDED!")
                self.status_label.setText(
                    f"Trial {self._cal_trial_index + 1} complete! (No samples recorded)"
                )
            
            # Move to next trial
            self._cal_trial_index += 1
            
        except Exception as e:
            print(f"[CALIBRATION] Error in _cal_end_trial: {e}")
            import traceback
            traceback.print_exc()
            # Still try to continue even on error
            if self._cal_trial_index < len(self._cal_sequence):
                self._cal_trial_index += 1
        
        # Continue or finish - check if we're done before starting rest
        if self._cal_trial_index >= len(self._cal_sequence):
            print(f"[CALIBRATION] All trials complete, finishing...")
            QTimer.singleShot(500, self._cal_finish)
        else:
            # Continue with next rest period
            QTimer.singleShot(500, self._cal_start_rest)
    
    def _cal_finish(self) -> None:
        """Finish calibration and compute templates."""
        try:
            # Stop flickering now that calibration is complete
            # This is the ONLY place we stop flickering during calibration
            self.top_target.set_active(False)
            self.bottom_target.set_active(False)
            
            self._mode = SessionMode.IDLE
            
            # Reset preprocessor filter state to prevent numerical issues
            print("[CALIBRATION] Resetting preprocessor state...")
            self.preprocessor.reset()
            
            stats = self._calibration_data.get_statistics()
            print(f"[CALIBRATION] Finishing calibration")
            print(f"[CALIBRATION] Stats: {stats}")
            
            # Check if we got any data
            if stats['n_trials_15hz'] == 0 and stats['n_trials_12hz'] == 0:
                print("[CALIBRATION] ERROR: No trials recorded!")
                QMessageBox.warning(
                    self,
                    "Calibration Failed",
                    "No EEG data was recorded during calibration.\n\n"
                    "Please check:\n"
                    "1. LSL stream is connected\n"
                    "2. Unicorn headset is on and transmitting\n"
                    "3. Try reconnecting to LSL"
                )
                # Re-enable controls
                self.start_btn.setEnabled(True)
                self.calibrate_btn.setEnabled(True)
                self.connect_lsl_btn.setEnabled(True)
                return
            
            # Compute templates
            print("[CALIBRATION] Computing templates...")
            self._calibration_data.compute_templates()
            
            # Load into classifier
            print("[CALIBRATION] Loading into classifier...")
            if self.classifier.load_calibration(self._calibration_data):
                self.cal_status.setText("Calibrated!")
                self.cal_status.setStyleSheet("color: #00ff00;")
                
                # Save calibration
                cal_path = Path("calibration_data.json")
                self._calibration_data.save(cal_path)
                
                print(f"[CALIBRATION] Saved to {cal_path.absolute()}")
                
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Calibration successful!\n\n"
                    f"Higher freq trials: {stats['n_trials_15hz']}\n"
                    f"Lower freq trials: {stats['n_trials_12hz']}\n"
                    f"Total samples: {stats['total_samples_15hz'] + stats['total_samples_12hz']}\n\n"
                    f"Saved to: {cal_path.absolute()}\n\n"
                    f"The classifier will now use your personalized brain responses!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Calibration Failed",
                    "Could not compute templates from calibration data.\n"
                    "Please try again."
                )
            
            # Re-enable controls
            self.start_btn.setEnabled(True)
            self.calibrate_btn.setEnabled(True)
            self.connect_lsl_btn.setEnabled(True)
            
            self.status_label.setText("Calibration complete! Ready to compose.")
            print("[CALIBRATION] Calibration finish complete")
            
        except Exception as e:
            print(f"[CALIBRATION] Error in _cal_finish: {e}")
            import traceback
            traceback.print_exc()
            
            # Re-enable controls even on error
            self.start_btn.setEnabled(True)
            self.calibrate_btn.setEnabled(True)
            self.connect_lsl_btn.setEnabled(True)
            self.status_label.setText("Calibration error - see console")
    
    def _load_calibration(self) -> None:
        """Load calibration from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            "",
            "Calibration Files (*.json)"
        )
        
        if not filepath:
            return
        
        try:
            self._calibration_data = CalibrationData.load(filepath)
            
            if self.classifier.load_calibration(self._calibration_data):
                self.cal_status.setText("Calibrated!")
                self.cal_status.setStyleSheet("color: #00ff00;")
                
                stats = self._calibration_data.get_statistics()
                self.status_label.setText(
                    f"Loaded calibration from {stats['created_at']}"
                )
            else:
                QMessageBox.warning(self, "Load Failed", "Could not load calibration data.")
                
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Error loading calibration: {e}")
    
    def _start_data_validation(self) -> None:
        """Run data validation test: flicker for 10 seconds and plot data."""
        if not self._use_lsl or not self._lsl_connected:
            QMessageBox.warning(
                self,
                "LSL Required",
                "Please connect to LSL first.\n\n"
                "Data validation requires real EEG data from your headset."
            )
            return
        
        # Note: CCA works with synthetic references even without calibration
        # Calibration enhances accuracy but is not required
        if not self.classifier.is_calibrated:
            print("[VALIDATION] No calibration loaded - using CCA with synthetic references only")
        
        # Disable button during validation
        self.validate_btn.setEnabled(False)
        self.status_label.setText("Data validation starting... Get ready!")
        QApplication.processEvents()
        
        # Start validation in background
        QTimer.singleShot(1000, self._run_data_validation)
    
    def _run_data_validation(self) -> None:
        """Execute the structured data validation test with classification."""
        try:
            # Clear validation_plots folder at start (ensures clean state for each run)
            validation_dir = Path("validation_plots")
            if validation_dir.exists():
                # Remove all files (including subdirectories recursively)
                for file in validation_dir.rglob("*"):
                    try:
                        if file.is_file():
                            file.unlink()
                            print(f"[VALIDATION] Cleared old file: {file.name}")
                        elif file.is_dir():
                            file.rmdir()  # Remove empty directories
                    except Exception as e:
                        print(f"[VALIDATION] Warning: Could not delete {file}: {e}")
                print(f"[VALIDATION] Cleared validation_plots folder")
            else:
                # Create folder if it doesn't exist
                validation_dir.mkdir(exist_ok=True)
                print(f"[VALIDATION] Created validation_plots folder")
            
            # Protocol timings
            baseline_duration = 2.0  # 2 seconds baseline (flickering active, no indicator)
            target_duration = 10.0   # 10 seconds per target
            transition_duration = 1.0  # 1 second transition (indicator moves, trigger sent)
            
            # Initialize data storage
            self._validation_baseline_data = []
            self._validation_top_target_data = {'raw': [], 'processed': [], 'timestamps': []}
            self._validation_bottom_target_data = {'raw': [], 'processed': [], 'timestamps': []}
            self._validation_baseline_mean = None
            self._validation_phase_markers = []  # Store phase transitions with timestamps
            
            self._validation_start_time = time.perf_counter()
            self._validation_stopped = False
            self._validation_current_phase = 'baseline'
            self._validation_phase_start_time = time.perf_counter()
            
            # START FLICKERING FIRST (both targets)
            self.top_target.set_active(True)
            self.bottom_target.set_active(True)
            print(f"[VALIDATION] Flickering started - both targets active")
            
            # Send validation start marker
            if self.marker_sender:
                self.marker_sender.send("Validation:Start")
            
            # Phase 1: Baseline (flickering active, no indicators, no data collection yet)
            self.status_label.setText(f"Flickering active - Baseline ({baseline_duration}s)... Stay still")
            self.top_indicator.set_active(False)
            self.bottom_indicator.set_active(False)
            QApplication.processEvents()
            
            # Start data collection timer
            update_interval = 50  # ms
            self._validation_timer = QTimer()
            self._validation_timer.timeout.connect(self._collect_validation_sample)
            self._validation_timer.start(update_interval)
            
            # Schedule phase transitions
            total_time = 0
            # After baseline: indicator lights on top, trigger sent, start data capture
            QTimer.singleShot(int(baseline_duration * 1000) + 100, 
                             lambda: self._validation_transition_phase('top_target', target_duration))
            total_time += baseline_duration * 1000 + 100
            
            # After top target: indicator moves to bottom, trigger sent, 1s transition
            QTimer.singleShot(int(total_time + target_duration * 1000) + 100,
                             lambda: self._validation_transition_phase('transition', transition_duration))
            total_time += target_duration * 1000 + 100
            
            # After transition: indicator stays on bottom, start data capture
            QTimer.singleShot(int(total_time + transition_duration * 1000) + 100,
                             lambda: self._validation_transition_phase('bottom_target', target_duration))
            total_time += transition_duration * 1000 + 100
            
            # Stop after all phases complete
            QTimer.singleShot(int(total_time + target_duration * 1000) + 200, self._stop_validation)
            
        except Exception as e:
            print(f"[VALIDATION] Error: {e}")
            import traceback
            traceback.print_exc()
            self.validate_btn.setEnabled(True)
            self.status_label.setText("Validation error - see console")
            QMessageBox.warning(self, "Validation Error", f"Error during validation: {e}")
    
    def _validation_transition_phase(self, new_phase: str, duration: float) -> None:
        """Transition to a new validation phase."""
        try:
            if getattr(self, '_validation_stopped', False):
                return
            
            elapsed = time.perf_counter() - self._validation_start_time
            old_phase = self._validation_current_phase
            
            # Record phase transition
            self._validation_phase_markers.append({
                'phase': old_phase,
                'end_time': elapsed,
                'duration': elapsed - self._validation_phase_start_time
            })
            
            self._validation_current_phase = new_phase
            self._validation_phase_start_time = time.perf_counter()
            
            # Handle phase-specific setup
            if new_phase == 'top_target':
                # Compute baseline mean
                if self._validation_baseline_data:
                    baseline_array = np.vstack(self._validation_baseline_data)
                    self._validation_baseline_mean = np.mean(baseline_array, axis=0)
                    print(f"[VALIDATION] Baseline: {len(baseline_array)} samples")
                
                # Indicator lights on TOP target (send trigger)
                self.top_indicator.set_active(True)
                self.bottom_indicator.set_active(False)
                if self.marker_sender:
                    self.marker_sender.send("Validation:TopTarget:Start")
                self.status_label.setText(f"👆 Look at TOP target ({duration}s)")
                print(f"[VALIDATION] Top target phase: Indicator ON, trigger sent, data capture starting")
                
            elif new_phase == 'transition':
                # Indicator moves to BOTTOM target (send trigger)
                self.top_indicator.set_active(False)
                self.bottom_indicator.set_active(True)
                if self.marker_sender:
                    self.marker_sender.send("Validation:TopTarget:End")
                    self.marker_sender.send("Validation:BottomTarget:Transition")
                self.status_label.setText(f"Indicator moved to BOTTOM ({duration}s)... Get ready")
                print(f"[VALIDATION] Transition: Indicator moved to bottom, trigger sent")
                
            elif new_phase == 'bottom_target':
                # Indicator stays on bottom, start data capture
                self.top_indicator.set_active(False)
                self.bottom_indicator.set_active(True)
                if self.marker_sender:
                    self.marker_sender.send("Validation:BottomTarget:Start")
                self.status_label.setText(f"👇 Look at BOTTOM target ({duration}s)")
                print(f"[VALIDATION] Bottom target phase: Indicator ON, trigger sent, data capture starting")
            
            QApplication.processEvents()
            
        except Exception as e:
            print(f"[VALIDATION] Error transitioning phase: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_flicker_phase(self) -> None:
        """Start the flicker phase after baseline collection."""
        try:
            if getattr(self, '_validation_stopped', False):
                return
            
            # Compute baseline mean for each channel
            if self._validation_baseline_data:
                baseline_array = np.vstack(self._validation_baseline_data)
                self._validation_baseline_mean = np.mean(baseline_array, axis=0)
                print(f"[VALIDATION] Baseline collected: {len(baseline_array)} samples, mean per channel: {self._validation_baseline_mean}")
            else:
                print("[VALIDATION] Warning: No baseline data collected!")
                self._validation_baseline_mean = None
            
            # Fixed frequencies: 15Hz (UP) and 12Hz (DOWN)
            higher_freq, lower_freq = 15.0, 12.0
            print(f"[VALIDATION] Target frequencies: Top={higher_freq:.2f} Hz, Bottom={lower_freq:.2f} Hz")
            
            # Reset flicker rate tracking
            self.top_target._flicker_state_changes.clear()
            self.bottom_target._flicker_state_changes.clear()
            self.top_target._last_flicker_state = None
            self.bottom_target._last_flicker_state = None
            self.top_target._paint_count = 0
            self.bottom_target._paint_count = 0
            
            # Start flickering
            self._validation_phase = 'flicker'
            self._validation_flicker_start_time = time.perf_counter()
            self.top_target.set_active(True)
            self.bottom_target.set_active(True)
            self.status_label.setText(f"Flickering active ({self._validation_flicker_duration}s)... Look at targets!")
            QApplication.processEvents()
            
            print(f"[VALIDATION] Flicker phase started - monitoring flicker rates...")
            
        except Exception as e:
            print(f"[VALIDATION] Error starting flicker phase: {e}")
            import traceback
            traceback.print_exc()
    
    def _collect_validation_sample(self) -> None:
        """Collect a single sample during validation (called by timer)."""
        try:
            if getattr(self, '_validation_stopped', False):
                return
            
            elapsed = time.perf_counter() - self._validation_start_time
            phase = getattr(self, '_validation_current_phase', 'baseline')
            phase_elapsed = elapsed - self._validation_phase_start_time
            
            # Pull and process using EXACT same method as main experiment
            if self._use_lsl and self._lsl_connected:
                n_samples_per_update = int(0.050 * self.preprocessor.sample_rate)  # 50ms worth
                
                # Get raw data from buffer
                raw_buffer = self.preprocessor._lsl_receiver.get_recent_data(0.1)
                if len(raw_buffer) > 0 and raw_buffer.shape[1] >= 8:
                    chunk = raw_buffer[-n_samples_per_update:] if len(raw_buffer) >= n_samples_per_update else raw_buffer
                    
                    if phase == 'baseline':
                        # Collect baseline data (flickering active, no indicators)
                        self._validation_baseline_data.append(chunk.copy())
                    elif phase == 'top_target':
                        # Collect data while looking at top target (indicator on top)
                        self._validation_top_target_data['raw'].append(chunk.copy())
                        # Process data
                        processed = self.preprocessor.pull_and_process(n_samples=n_samples_per_update)
                        if len(processed) > 0:
                            self._validation_top_target_data['processed'].append(processed.copy())
                            self._validation_top_target_data['timestamps'].append(phase_elapsed)
                    elif phase == 'bottom_target':
                        # Collect data while looking at bottom target (indicator on bottom)
                        self._validation_bottom_target_data['raw'].append(chunk.copy())
                        # Process data
                        processed = self.preprocessor.pull_and_process(n_samples=n_samples_per_update)
                        if len(processed) > 0:
                            self._validation_bottom_target_data['processed'].append(processed.copy())
                            self._validation_bottom_target_data['timestamps'].append(phase_elapsed)
                    # Transition phase: don't collect data (just indicator movement)
        except Exception as e:
            print(f"[VALIDATION] Error collecting sample: {e}")
    
    def _stop_validation(self) -> None:
        """Stop validation and process data."""
        try:
            if getattr(self, '_validation_stopped', False):
                return
            
            self._validation_stopped = True
            
            if hasattr(self, '_validation_timer'):
                self._validation_timer.stop()
            
            # Stop flickering and indicators
            self.top_target.set_active(False)
            self.bottom_target.set_active(False)
            self.top_indicator.set_active(False)
            self.bottom_indicator.set_active(False)
            
            # Send end marker
            if self.marker_sender:
                if self._validation_current_phase == 'bottom_target':
                    self.marker_sender.send("Validation:BottomTarget:End")
                self.marker_sender.send("Validation:Complete")
            
            # Process validation data
            self._process_validation_data()
            
        except Exception as e:
            print(f"[VALIDATION] Error stopping: {e}")
            import traceback
            traceback.print_exc()
            self.validate_btn.setEnabled(True)
            self.status_label.setText("Validation error - see console")
            
        except Exception as e:
            print(f"[VALIDATION] Error stopping: {e}")
            import traceback
            traceback.print_exc()
            self.validate_btn.setEnabled(True)
            self.status_label.setText("Validation error - see console")
    
    def _process_validation_data(self) -> None:
        """Process validation data: create templates, classify chunks, compare ground truth."""
        try:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Reset classifier state to ensure clean validation run
            # This clears any history and resets debug flags
            self.classifier.reset()
            # Reset debug counter if it exists
            if hasattr(self.classifier, '_down_call_count'):
                self.classifier._down_call_count = 0
            
            # Verify calibration is loaded
            print(f"[VALIDATION] Classifier calibrated: {self.classifier.is_calibrated}")
            print(f"[VALIDATION] Using calibration: {self.classifier._using_calibration}")
            if hasattr(self.classifier, '_ref_signals_down'):
                print(f"[VALIDATION] DOWN ref shape: {self.classifier._ref_signals_down.shape}")
            
            self.status_label.setText("Processing validation data...")
            QApplication.processEvents()
            
            sample_rate = self.preprocessor.sample_rate
            
            # Concatenate data from each phase
            top_processed = np.vstack(self._validation_top_target_data['processed']) if self._validation_top_target_data['processed'] else None
            bottom_processed = np.vstack(self._validation_bottom_target_data['processed']) if self._validation_bottom_target_data['processed'] else None
            
            if top_processed is None or bottom_processed is None:
                QMessageBox.warning(self, "No Data", "Insufficient data collected!")
                self.validate_btn.setEnabled(True)
                return
            
            # Create templates using same method as calibration: epoch averaging
            # Calibration: skip 0.5s, extract multiple epochs of 0.5s each, average them
            window_seconds_template = 0.5  # Match calibration default
            window_samples_template = int(window_seconds_template * sample_rate)
            skip_samples = int(0.5 * sample_rate)  # Skip 0.5s to allow SSVEP to stabilize (match calibration)
            
            top_template = None
            bottom_template = None
            
            # Top template: extract multiple epochs and average (matching calibration method)
            if len(top_processed) > skip_samples + window_samples_template:
                all_epochs_top = []
                # Extract multiple epochs starting after skip period
                n_epochs = (len(top_processed) - skip_samples) // window_samples_template
                for i in range(n_epochs):
                    epoch_start = skip_samples + i * window_samples_template
                    epoch_end = epoch_start + window_samples_template
                    if epoch_end <= len(top_processed):
                        epoch = top_processed[epoch_start:epoch_end]
                        all_epochs_top.append(epoch)
                
                if all_epochs_top:
                    # Average all epochs (matching calibration method)
                    top_template = np.mean(all_epochs_top, axis=0)
                    # Normalize (matching calibration method)
                    top_template = top_template - np.mean(top_template, axis=0)
                    std = np.std(top_template, axis=0)
                    std[std < 1e-6] = 1
                    top_template = top_template / std
                    print(f"[VALIDATION] Created top template: {len(all_epochs_top)} epochs averaged, {len(top_template)} samples per epoch")
            
            # Bottom template: extract multiple epochs and average (matching calibration method)
            if len(bottom_processed) > skip_samples + window_samples_template:
                all_epochs_bottom = []
                # Extract multiple epochs starting after skip period
                n_epochs = (len(bottom_processed) - skip_samples) // window_samples_template
                for i in range(n_epochs):
                    epoch_start = skip_samples + i * window_samples_template
                    epoch_end = epoch_start + window_samples_template
                    if epoch_end <= len(bottom_processed):
                        epoch = bottom_processed[epoch_start:epoch_end]
                        all_epochs_bottom.append(epoch)
                
                if all_epochs_bottom:
                    # Average all epochs (matching calibration method)
                    bottom_template = np.mean(all_epochs_bottom, axis=0)
                    # Normalize (matching calibration method)
                    bottom_template = bottom_template - np.mean(bottom_template, axis=0)
                    std = np.std(bottom_template, axis=0)
                    std[std < 1e-6] = 1
                    bottom_template = bottom_template / std
                    print(f"[VALIDATION] Created bottom template: {len(all_epochs_bottom)} epochs averaged, {len(bottom_template)} samples per epoch")
            
            # Parse remaining data (after template extraction) into chunks matching main experiment EXACTLY
            # Main composition uses: get_recent_data(0.3) called every 50ms
            # So we need: 0.3s window, stepping by 50ms (0.05s)
            window_seconds = 0.3  # Match main composition (not classifier.window_seconds which is 0.5s)
            update_interval_seconds = 0.050  # 50ms - matches composition timer interval
            
            chunk_window_samples = int(window_seconds * sample_rate)
            step_samples = int(update_interval_seconds * sample_rate)
            
            # Set up debug logging to file BEFORE chunking
            import logging
            from datetime import datetime
            log_file = Path("validation_plots") / f"validation_debug_{timestamp_str}.log"
            log_file.parent.mkdir(exist_ok=True)
            
            # Create logger
            val_logger = logging.getLogger('validation_debug')
            val_logger.setLevel(logging.DEBUG)
            val_logger.handlers.clear()
            
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            val_logger.addHandler(fh)
            
            # Calculate expected number of chunks
            target_duration = 10.0  # seconds per target
            classification_start = 2.0  # seconds (after template extraction)
            available_data_duration = target_duration - classification_start  # 8 seconds
            expected_chunks = int((available_data_duration - window_seconds) / update_interval_seconds) + 1
            
            val_logger.info("=" * 60)
            val_logger.info("VALIDATION CHUNKING ANALYSIS")
            val_logger.info("=" * 60)
            val_logger.info(f"Target duration: {target_duration}s")
            val_logger.info(f"Classification starts at: {classification_start}s (after template extraction)")
            val_logger.info(f"Available data duration: {available_data_duration}s")
            val_logger.info(f"Window size: {window_seconds}s ({chunk_window_samples} samples)")
            val_logger.info(f"Step size: {update_interval_seconds}s ({step_samples} samples)")
            val_logger.info(f"Expected chunks per target: {expected_chunks}")
            val_logger.info(f"Top processed data length: {len(top_processed)} samples ({len(top_processed)/sample_rate:.2f}s)")
            val_logger.info(f"Bottom processed data length: {len(bottom_processed)} samples ({len(bottom_processed)/sample_rate:.2f}s)")
            
            # Start classification after template extraction period
            # Templates use epochs from 0.5s onwards, so classify from 2s onwards to exclude template data
            classification_start_samples = int(2.0 * sample_rate)
            
            val_logger.info(f"Classification start: {classification_start_samples} samples ({classification_start}s)")
            val_logger.info(f"Top remaining after {classification_start}s: {len(top_processed) - classification_start_samples} samples "
                          f"({(len(top_processed) - classification_start_samples)/sample_rate:.2f}s)")
            val_logger.info(f"Bottom remaining after {classification_start}s: {len(bottom_processed) - classification_start_samples} samples "
                          f"({(len(bottom_processed) - classification_start_samples)/sample_rate:.2f}s)")
            
            # Top target: classify chunks from 2s onwards
            top_chunks = []
            top_ground_truth = AttentionTarget.UP
            if len(top_processed) > classification_start_samples + chunk_window_samples:
                remaining_top = top_processed[classification_start_samples:]
                # Step by 50ms to match main composition update rate
                for i in range(0, len(remaining_top) - chunk_window_samples + 1, step_samples):
                    chunk = remaining_top[i:i+chunk_window_samples]
                    if len(chunk) == chunk_window_samples:
                        top_chunks.append(chunk)
            
            # Bottom target: classify chunks from 2s onwards
            bottom_chunks = []
            bottom_ground_truth = AttentionTarget.DOWN
            if len(bottom_processed) > classification_start_samples + chunk_window_samples:
                remaining_bottom = bottom_processed[classification_start_samples:]
                # Step by 50ms to match main composition update rate
                for i in range(0, len(remaining_bottom) - chunk_window_samples + 1, step_samples):
                    chunk = remaining_bottom[i:i+chunk_window_samples]
                    if len(chunk) == chunk_window_samples:
                        bottom_chunks.append(chunk)
            
            val_logger.info(f"Actual top chunks created: {len(top_chunks)}")
            val_logger.info(f"Actual bottom chunks created: {len(bottom_chunks)}")
            val_logger.info("=" * 60)
            
            print(f"[VALIDATION] Created {len(top_chunks)} top chunks, {len(bottom_chunks)} bottom chunks")
            print(f"[VALIDATION] Expected: ~{expected_chunks} chunks per target (8s data / 0.05s step)")
            print(f"[VALIDATION] Chunk parameters: window={window_seconds}s, step={update_interval_seconds}s ({step_samples} samples)")
            print(f"[VALIDATION] This matches main composition: get_recent_data({window_seconds}) called every {update_interval_seconds*1000:.0f}ms")
            print(f"[VALIDATION] Debug log: {log_file}")
            
            # Run classifier on chunks - use same method as main composition
            top_predictions = []
            bottom_predictions = []
            
            val_logger.info(f"Classifying {len(top_chunks)} top chunks...")
            for i, chunk in enumerate(top_chunks):
                result = self.classifier.classify(chunk, method="cca")  # Match main composition
                top_predictions.append({
                    'prediction': result.target.value,
                    'confidence': result.confidence,
                    'raw_score': result.raw_score,
                    'corr_up': result.power_higher_freq,
                    'corr_down': result.power_lower_freq
                })
                if i < 5 or i % 20 == 0:  # Log first 5 and every 20th
                    val_logger.debug(f"Top chunk {i+1}/{len(top_chunks)}: corr_up={result.power_higher_freq:.6f}, "
                                   f"corr_down={result.power_lower_freq:.6f}, pred={result.target.value}, "
                                   f"confidence={result.confidence:.3f}")
            
            val_logger.info(f"Classifying {len(bottom_chunks)} bottom chunks...")
            for i, chunk in enumerate(bottom_chunks):
                result = self.classifier.classify(chunk, method="cca")  # Match main composition
                bottom_predictions.append({
                    'prediction': result.target.value,
                    'confidence': result.confidence,
                    'raw_score': result.raw_score,
                    'corr_up': result.power_higher_freq,
                    'corr_down': result.power_lower_freq
                })
                if i < 5 or i % 20 == 0:  # Log first 5 and every 20th
                    val_logger.debug(f"Bottom chunk {i+1}/{len(bottom_chunks)}: corr_up={result.power_higher_freq:.6f}, "
                                   f"corr_down={result.power_lower_freq:.6f}, pred={result.target.value}, "
                                   f"confidence={result.confidence:.3f}")
            
            # Compare against ground truth (calculate BEFORE logging)
            top_correct = sum(1 for p in top_predictions if p['prediction'] == top_ground_truth.value)
            top_accuracy = top_correct / len(top_predictions) if top_predictions else 0.0
            top_mean_confidence = np.mean([p['confidence'] for p in top_predictions]) if top_predictions else 0.0
            
            bottom_correct = sum(1 for p in bottom_predictions if p['prediction'] == bottom_ground_truth.value)
            bottom_accuracy = bottom_correct / len(bottom_predictions) if bottom_predictions else 0.0
            bottom_mean_confidence = np.mean([p['confidence'] for p in bottom_predictions]) if bottom_predictions else 0.0
            
            overall_accuracy = (top_correct + bottom_correct) / (len(top_predictions) + len(bottom_predictions)) if (top_predictions or bottom_predictions) else 0.0
            
            val_logger.info("=" * 60)
            val_logger.info("VALIDATION SUMMARY")
            val_logger.info("=" * 60)
            val_logger.info(f"Top target: {len(top_predictions)} predictions, {top_correct} correct, accuracy={top_accuracy*100:.1f}%")
            val_logger.info(f"Bottom target: {len(bottom_predictions)} predictions, {bottom_correct} correct, accuracy={bottom_accuracy*100:.1f}%")
            val_logger.info(f"Overall: {len(top_predictions) + len(bottom_predictions)} total predictions, "
                          f"{top_correct + bottom_correct} correct, accuracy={overall_accuracy*100:.1f}%")
            
            print(f"[VALIDATION] Debug log saved to {log_file}")
            
            print(f"[VALIDATION] Top target accuracy: {top_accuracy*100:.1f}% ({top_correct}/{len(top_predictions)})")
            print(f"[VALIDATION] Bottom target accuracy: {bottom_accuracy*100:.1f}% ({bottom_correct}/{len(bottom_predictions)})")
            print(f"[VALIDATION] Overall accuracy: {overall_accuracy*100:.1f}%")
            
            # Generate comprehensive report
            self._generate_validation_report(
                top_processed, bottom_processed,
                top_template, bottom_template,
                top_predictions, bottom_predictions,
                top_ground_truth, bottom_ground_truth,
                top_accuracy, bottom_accuracy, overall_accuracy,
                top_mean_confidence, bottom_mean_confidence,
                timestamp_str=timestamp_str
            )
            
        except Exception as e:
            print(f"[VALIDATION] Error processing data: {e}")
            import traceback
            traceback.print_exc()
            self.validate_btn.setEnabled(True)
            self.status_label.setText("Validation error - see console")
            QMessageBox.warning(self, "Validation Error", f"Error processing validation data: {e}")
    
    def _generate_validation_report(self, top_processed, bottom_processed, top_template, bottom_template,
                                    top_predictions, bottom_predictions, top_ground_truth, bottom_ground_truth,
                                    top_accuracy, bottom_accuracy, overall_accuracy,
                                    top_mean_confidence, bottom_mean_confidence, timestamp_str: Optional[str] = None) -> None:
        """Generate comprehensive validation report with plots and JSON."""
        try:
            if timestamp_str is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            validation_dir = Path("validation_plots")
            validation_dir.mkdir(exist_ok=True)
            
            # Save data files
            np.save(validation_dir / f"top_processed_{timestamp_str}.npy", top_processed)
            np.save(validation_dir / f"bottom_processed_{timestamp_str}.npy", bottom_processed)
            if top_template is not None:
                np.save(validation_dir / f"top_template_{timestamp_str}.npy", top_template)
            if bottom_template is not None:
                np.save(validation_dir / f"bottom_template_{timestamp_str}.npy", bottom_template)
            
            # Save RAW EEG data for offline testing of different preprocessing/classification approaches
            # This allows testing without running the GUI every time
            top_raw_shape = None
            bottom_raw_shape = None
            baseline_raw_shape = None
            
            if hasattr(self, '_validation_top_target_data') and self._validation_top_target_data.get('raw'):
                top_raw = np.vstack(self._validation_top_target_data['raw'])
                np.save(validation_dir / f"top_raw_{timestamp_str}.npy", top_raw)
                top_raw_shape = list(top_raw.shape)
                print(f"[VALIDATION] Saved raw top target data: {top_raw.shape} samples")
            
            if hasattr(self, '_validation_bottom_target_data') and self._validation_bottom_target_data.get('raw'):
                bottom_raw = np.vstack(self._validation_bottom_target_data['raw'])
                np.save(validation_dir / f"bottom_raw_{timestamp_str}.npy", bottom_raw)
                bottom_raw_shape = list(bottom_raw.shape)
                print(f"[VALIDATION] Saved raw bottom target data: {bottom_raw.shape} samples")
            
            if hasattr(self, '_validation_baseline_data') and self._validation_baseline_data:
                baseline_raw = np.vstack(self._validation_baseline_data)
                np.save(validation_dir / f"baseline_raw_{timestamp_str}.npy", baseline_raw)
                baseline_raw_shape = list(baseline_raw.shape)
                print(f"[VALIDATION] Saved raw baseline data: {baseline_raw.shape} samples")
            
            # Save metadata for easy loading
            metadata = {
                'timestamp': timestamp_str,
                'sample_rate': float(self.preprocessor.sample_rate),
                'window_seconds': 0.3,
                'update_interval_seconds': 0.050,
                'target_frequencies': {
                    'top': float(self.top_target.frequency),
                    'bottom': float(self.bottom_target.frequency)
                },
                'target_phases': {
                    'top': float(self.top_target.phase_offset),
                    'bottom': float(self.bottom_target.phase_offset)
                },
                'top_raw_shape': top_raw_shape,
                'bottom_raw_shape': bottom_raw_shape,
                'baseline_raw_shape': baseline_raw_shape,
                'top_processed_shape': list(top_processed.shape),
                'bottom_processed_shape': list(bottom_processed.shape),
                'n_channels': top_processed.shape[1] if len(top_processed) > 0 else 0,
                'occipital_channels': self.classifier.occipital_channels,
                'preprocessing': {
                    'bandpass_low': float(self.preprocessor.bandpass_low),
                    'bandpass_high': float(self.preprocessor.bandpass_high),
                    'notch_freq': float(self.preprocessor.notch_freq),
                    'use_car': self.preprocessor.use_car,
                },
                'classifier': {
                    'n_harmonics': self.classifier.n_harmonics,
                    'target_frequencies': list(self.classifier.target_frequencies),
                    'target_phases': list(self.classifier.target_phases),
                }
            }
            
            metadata_file = validation_dir / f"validation_metadata_{timestamp_str}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[VALIDATION] Saved metadata: {metadata_file}")
            
            # Create comprehensive report
            report = {
                'timestamp': timestamp_str,
                'sample_rate': float(self.preprocessor.sample_rate),
                'window_seconds': 0.3,  # Matches main composition (not classifier.window_seconds)
                'update_interval_seconds': 0.050,  # 50ms - matches composition timer
                'target_frequencies': {
                    'top': float(self.top_target.frequency),
                    'bottom': float(self.bottom_target.frequency)
                },
                'top_target': {
                    'ground_truth': top_ground_truth.value,
                    'n_chunks': len(top_predictions),
                    'n_correct': sum(1 for p in top_predictions if p['prediction'] == top_ground_truth.value),
                    'accuracy': float(top_accuracy),
                    'mean_confidence': float(top_mean_confidence),
                    'mean_corr_up': float(np.mean([p.get('corr_up', 0) for p in top_predictions])),
                    'mean_corr_down': float(np.mean([p.get('corr_down', 0) for p in top_predictions])),
                    'predictions': top_predictions
                },
                'bottom_target': {
                    'ground_truth': bottom_ground_truth.value,
                    'n_chunks': len(bottom_predictions),
                    'n_correct': sum(1 for p in bottom_predictions if p['prediction'] == bottom_ground_truth.value),
                    'accuracy': float(bottom_accuracy),
                    'mean_confidence': float(bottom_mean_confidence),
                    'mean_corr_up': float(np.mean([p.get('corr_up', 0) for p in bottom_predictions])),
                    'mean_corr_down': float(np.mean([p.get('corr_down', 0) for p in bottom_predictions])),
                    'predictions': bottom_predictions
                },
                'overall': {
                    'accuracy': float(overall_accuracy),
                    'total_chunks': len(top_predictions) + len(bottom_predictions),
                    'total_correct': sum(1 for p in top_predictions if p['prediction'] == top_ground_truth.value) + 
                                   sum(1 for p in bottom_predictions if p['prediction'] == bottom_ground_truth.value)
                },
                'phase_markers': getattr(self, '_validation_phase_markers', [])
            }
            
            # Save report
            report_file = validation_dir / f"validation_report_{timestamp_str}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[VALIDATION] Report saved to {report_file}")
            
            # Prepare data for plotting
            baseline_mean = getattr(self, '_validation_baseline_mean', None)
            baseline_raw = np.vstack(self._validation_baseline_data) if self._validation_baseline_data else None
            top_raw = np.vstack(self._validation_top_target_data['raw']) if self._validation_top_target_data['raw'] else None
            bottom_raw = np.vstack(self._validation_bottom_target_data['raw']) if self._validation_bottom_target_data['raw'] else None
            
            # Generate three plots: grand plot (all data), top target plot, bottom target plot
            plot_paths = self._plot_validation_data(
                baseline_raw=baseline_raw,
                top_raw=top_raw,
                bottom_raw=bottom_raw,
                top_processed=top_processed,
                bottom_processed=bottom_processed,
                baseline_mean=baseline_mean,
                timestamp_str=timestamp_str
            )
            
            # Show summary
            plot_paths_str = "\n".join([f"{key.capitalize()}: {path}" for key, path in plot_paths.items()])
            summary = (
                f"Validation Complete!\n\n"
                f"Top Target Accuracy: {top_accuracy*100:.1f}% ({report['top_target']['n_correct']}/{report['top_target']['n_chunks']})\n"
                f"Bottom Target Accuracy: {bottom_accuracy*100:.1f}% ({report['bottom_target']['n_correct']}/{report['bottom_target']['n_chunks']})\n"
                f"Overall Accuracy: {overall_accuracy*100:.1f}%\n\n"
                f"Report saved to:\n{report_file}\n\n"
                f"Plots saved to:\n{plot_paths_str}"
            )
            
            QMessageBox.information(self, "Validation Complete", summary)
            self.status_label.setText(f"Validation complete! Accuracy: {overall_accuracy*100:.1f}%")
            self.validate_btn.setEnabled(True)
            
        except Exception as e:
            print(f"[VALIDATION] Error generating report: {e}")
            import traceback
            traceback.print_exc()
            self.validate_btn.setEnabled(True)
            self.status_label.setText("Validation error - see console")
    
    def _finish_data_validation(self, raw_data_list: List, processed_data_list: List, timestamps: List, flicker_stats: dict = None) -> None:
        """Finish validation and generate plots (legacy method - kept for compatibility)."""
        # This method is now deprecated - _process_validation_data handles everything
        pass
    
    def _plot_validation_data(self, baseline_raw: Optional[np.ndarray], top_raw: Optional[np.ndarray], 
                              bottom_raw: Optional[np.ndarray], top_processed: Optional[np.ndarray],
                              bottom_processed: Optional[np.ndarray], baseline_mean: Optional[np.ndarray] = None,
                              timestamp_str: Optional[str] = None) -> dict:
        """Generate diagnostic plots for validation data: grand plot (all data) + top/bottom epoched plots."""
        from .lsl_stream import UNICORN_EEG_CHANNELS, OCCIPITAL_INDICES
        
        # Create output directory
        output_dir = Path("validation_plots")
        output_dir.mkdir(exist_ok=True)
        
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plot_paths = {}
        sample_rate = self.preprocessor.sample_rate
        
        # Fixed frequencies: 15Hz (UP) and 12Hz (DOWN)
        higher_freq, lower_freq = 15.0, 12.0
        
        # Helper function to extract occipital channels and apply baseline removal
        def prepare_raw_data(raw_data, baseline_mean):
            if raw_data is None or len(raw_data) == 0:
                return None, None
            if raw_data.shape[1] >= 8:
                occ_data = raw_data[:, OCCIPITAL_INDICES]
                occ_names = [UNICORN_EEG_CHANNELS[i] for i in OCCIPITAL_INDICES]
            else:
                occ_data = raw_data
                occ_names = [f"Ch{i}" for i in range(raw_data.shape[1])]
            
            # Apply baseline removal
            occ_data_baseline_removed = occ_data.copy()
            if baseline_mean is not None and len(baseline_mean) >= 8:
                occ_baseline_means = baseline_mean[OCCIPITAL_INDICES]
                for ch_idx in range(occ_data.shape[1]):
                    occ_data_baseline_removed[:, ch_idx] = occ_data[:, ch_idx] - occ_baseline_means[ch_idx]
            else:
                for ch_idx in range(occ_data.shape[1]):
                    occ_data_baseline_removed[:, ch_idx] = occ_data[:, ch_idx] - np.mean(occ_data[:, ch_idx])
            
            return occ_data_baseline_removed, occ_names
        
        # Helper function to plot raw EEG subplot
        def plot_raw_eeg_subplot(ax, raw_data, time_axis, title, baseline_mean=None):
            occ_data, occ_names = prepare_raw_data(raw_data, baseline_mean)
            if occ_data is None:
                return
            
            data_range = np.ptp(occ_data, axis=0)
            max_range = np.max(data_range)
            offset_spacing = max(max_range * 1.5, 100)
            
            for ch_idx, ch_name in enumerate(occ_names):
                offset = ch_idx * offset_spacing
                ax.plot(time_axis, occ_data[:, ch_idx] + offset, 
                       label=ch_name, linewidth=0.8, alpha=0.8)
            
            y_min = np.min(occ_data) - offset_spacing * 0.2
            y_max = np.max(occ_data) + offset_spacing * (len(occ_names) - 1) + offset_spacing * 0.2
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Time (seconds)", fontsize=11)
            ax.set_ylabel("Raw EEG (µV, baseline removed)", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Helper function to plot processed EEG subplot
        def plot_processed_eeg_subplot(ax, processed_data, time_axis, title):
            if processed_data is None or len(processed_data) == 0:
                return
            
            if processed_data.shape[1] == 3:
                ch_names = ['PO7', 'Oz', 'PO8']
            else:
                ch_names = [f"Ch{i}" for i in range(processed_data.shape[1])]
            
            data_range = np.ptp(processed_data, axis=0)
            max_range = np.max(data_range)
            offset_spacing = max(max_range * 1.5, 2.0)
            
            for ch_idx, ch_name in enumerate(ch_names):
                offset = ch_idx * offset_spacing
                ax.plot(time_axis, processed_data[:, ch_idx] + offset, 
                       label=ch_name, linewidth=0.8, alpha=0.8)
            
            y_min = np.min(processed_data) - offset_spacing * 0.2
            y_max = np.max(processed_data) + offset_spacing * (len(ch_names) - 1) + offset_spacing * 0.2
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Time (seconds)", fontsize=11)
            ax.set_ylabel("Processed EEG (normalized)", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Helper function to plot power spectrum subplot
        def plot_power_spectrum_subplot(ax, processed_data, title, target_freq):
            if processed_data is None or len(processed_data) == 0:
                return
            
            if processed_data.shape[1] > 1:
                avg_signal = np.mean(processed_data, axis=1)
            else:
                avg_signal = processed_data[:, 0]
            
            n_samples = len(avg_signal)
            freqs = np.fft.rfftfreq(n_samples, 1/sample_rate)
            fft_vals = np.abs(np.fft.rfft(avg_signal))
            power = fft_vals ** 2
            power_log = np.log10(power + 1e-10)
            
            mask = (freqs >= 5) & (freqs <= 25)
            ax.plot(freqs[mask], power_log[mask], linewidth=1.5, color='#1f77b4', label='Power Spectrum')
            
            # Mark target frequencies
            ax.axvline(higher_freq, color='r', linestyle='--', linewidth=2, alpha=0.7, 
                      label=f'Target: {higher_freq:.2f} Hz')
            ax.axvline(lower_freq, color='b', linestyle='--', linewidth=2, alpha=0.7, 
                      label=f'Target: {lower_freq:.2f} Hz')
            
            ax.axvspan(higher_freq - 0.5, higher_freq + 0.5, alpha=0.1, color='red', label='_nolegend_')
            ax.axvspan(lower_freq - 0.5, lower_freq + 0.5, alpha=0.1, color='blue', label='_nolegend_')
            
            ax.set_xlabel("Frequency (Hz)", fontsize=11)
            ax.set_ylabel("Log Power (log₁₀)", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlim(5, 25)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # ===== PLOT 1: Grand Plot (All Data) =====
        # Combine all raw data: baseline + top + bottom
        all_raw_parts = []
        if baseline_raw is not None and len(baseline_raw) > 0:
            all_raw_parts.append(baseline_raw)
        if top_raw is not None and len(top_raw) > 0:
            all_raw_parts.append(top_raw)
        if bottom_raw is not None and len(bottom_raw) > 0:
            all_raw_parts.append(bottom_raw)
        
        all_raw = np.vstack(all_raw_parts) if all_raw_parts else None
        all_processed = None
        if top_processed is not None and bottom_processed is not None:
            all_processed = np.vstack([top_processed, bottom_processed])
        elif top_processed is not None:
            all_processed = top_processed
        elif bottom_processed is not None:
            all_processed = bottom_processed
        
        if all_raw is not None or all_processed is not None:
            n_plots = 0
            if all_raw is not None:
                n_plots += 1
            if all_processed is not None:
                n_plots += 2  # Processed + Power spectrum
            
            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5*n_plots))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            if all_raw is not None:
                all_time_axis = np.arange(len(all_raw)) / sample_rate
                plot_raw_eeg_subplot(axes[plot_idx], all_raw, all_time_axis, 
                                    "Raw EEG Data - All Phases (Baseline + Top + Bottom)", baseline_mean)
                plot_idx += 1
            
            if all_processed is not None:
                all_proc_time_axis = np.arange(len(all_processed)) / sample_rate
                plot_processed_eeg_subplot(axes[plot_idx], all_processed, all_proc_time_axis,
                                         "Processed EEG Data - All Phases (After CAR + Filtering)")
                plot_idx += 1
                plot_power_spectrum_subplot(axes[plot_idx], all_processed,
                                           "Power Spectrum - All Phases (Averaged Occipital Channels)", None)
            
            plt.tight_layout()
            grand_plot_path = output_dir / f"data_validation_grand_{timestamp_str}.png"
            plt.savefig(grand_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['grand'] = grand_plot_path
            print(f"[VALIDATION] Grand plot saved to {grand_plot_path}")
        
        # ===== PLOT 2: Top Target Plot =====
        if top_raw is not None or top_processed is not None:
            n_plots = 0
            if top_raw is not None:
                n_plots += 1
            if top_processed is not None:
                n_plots += 2
            
            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5*n_plots))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            if top_raw is not None:
                top_time_axis = np.arange(len(top_raw)) / sample_rate
                plot_raw_eeg_subplot(axes[plot_idx], top_raw, top_time_axis,
                                    "Raw EEG Data - Top Target Phase", baseline_mean)
                plot_idx += 1
            
            if top_processed is not None:
                top_proc_time_axis = np.arange(len(top_processed)) / sample_rate
                plot_processed_eeg_subplot(axes[plot_idx], top_processed, top_proc_time_axis,
                                         "Processed EEG Data - Top Target Phase")
                plot_idx += 1
                plot_power_spectrum_subplot(axes[plot_idx], top_processed,
                                           f"Power Spectrum - Top Target Phase (Target: {higher_freq:.2f} Hz)", higher_freq)
            
            plt.tight_layout()
            top_plot_path = output_dir / f"data_validation_top_{timestamp_str}.png"
            plt.savefig(top_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['top'] = top_plot_path
            print(f"[VALIDATION] Top target plot saved to {top_plot_path}")
        
        # ===== PLOT 3: Bottom Target Plot =====
        if bottom_raw is not None or bottom_processed is not None:
            n_plots = 0
            if bottom_raw is not None:
                n_plots += 1
            if bottom_processed is not None:
                n_plots += 2
            
            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5*n_plots))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            if bottom_raw is not None:
                bottom_time_axis = np.arange(len(bottom_raw)) / sample_rate
                plot_raw_eeg_subplot(axes[plot_idx], bottom_raw, bottom_time_axis,
                                    "Raw EEG Data - Bottom Target Phase", baseline_mean)
                plot_idx += 1
            
            if bottom_processed is not None:
                bottom_proc_time_axis = np.arange(len(bottom_processed)) / sample_rate
                plot_processed_eeg_subplot(axes[plot_idx], bottom_processed, bottom_proc_time_axis,
                                          "Processed EEG Data - Bottom Target Phase")
                plot_idx += 1
                plot_power_spectrum_subplot(axes[plot_idx], bottom_processed,
                                           f"Power Spectrum - Bottom Target Phase (Target: {lower_freq:.2f} Hz)", lower_freq)
            
            plt.tight_layout()
            bottom_plot_path = output_dir / f"data_validation_bottom_{timestamp_str}.png"
            plt.savefig(bottom_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['bottom'] = bottom_plot_path
            print(f"[VALIDATION] Bottom target plot saved to {bottom_plot_path}")
        
        return plot_paths


def run_bci_app():
    """Run the BCI composition application."""
    app = QApplication(sys.argv)
    window = BCICompositionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_bci_app()
