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
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QComboBox, QSlider, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

from .stimulus import SSVEPStimulus, FlickerState
from .preprocessing import EEGPreprocessor, SimulatedEEGSource, LSLPreprocessor
from .classifier import SSVEPClassifier, AttentionTarget, ClassificationResult
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
        
        # Flicker rate accuracy tracking
        self._paint_intervals = []  # Store intervals between paints for frequency calculation
        self._flicker_rate_samples = 250  # Number of samples to use for rate calculation
        
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
            
            # Calculate actual phase from intensity (inverse of intensity calculation)
            # intensity = (sin(phase) + 1) / 2, so phase = arcsin(2*intensity - 1)
            if 0 < intensity < 1:
                self._actual_phase = np.arcsin(2 * intensity - 1)
            else:
                self._actual_phase = 0.0 if intensity <= 0 else np.pi
            
            # Monitor paint rate and phase drift
            self._paint_count += 1
            if self._last_paint_time is not None:
                dt = current_time - self._last_paint_time
                phase_drift = abs(self._expected_phase - self._actual_phase)
                
                # Track paint intervals for flicker rate calculation
                self._paint_intervals.append(dt)
                if len(self._paint_intervals) > self._flicker_rate_samples:
                    self._paint_intervals.pop(0)
                
                # Log if we detect significant drift or slow updates
                if self._paint_count % 125 == 0:  # Log every ~1 second at 125Hz
                    if dt > 0.020:  # More than 20ms between paints (should be ~8ms)
                        print(f"[FLICKER DEBUG] {self.position}: Slow paint! dt={dt*1000:.1f}ms (expected ~8ms)")
                    if phase_drift > 0.5:  # More than 0.5 radians drift
                        print(f"[FLICKER DEBUG] {self.position}: Phase drift! {phase_drift:.3f} rad, expected={self._expected_phase:.3f}, actual={self._actual_phase:.3f}")
                    
                    # Calculate and log actual flicker rate
                    if len(self._paint_intervals) >= 50:  # Need enough samples
                        mean_interval = np.mean(self._paint_intervals)
                        actual_rate = 1.0 / mean_interval if mean_interval > 0 else 0
                        target_rate = self.frequency
                        rate_error = abs(actual_rate - target_rate)
                        rate_error_pct = (rate_error / target_rate * 100) if target_rate > 0 else 0
                        print(f"[FLICKER RATE] {self.position}: Target={target_rate:.2f}Hz, Actual={actual_rate:.2f}Hz, Error={rate_error:.3f}Hz ({rate_error_pct:.1f}%)")
            
            self._last_paint_time = current_time
        else:
            intensity = 0.0
            self._last_paint_time = None
            self._paint_count = 0
            self._paint_intervals.clear()
        
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
        
        # BCI components
        self.stimulus = SSVEPStimulus(duration=10.0)
        self.classifier = SSVEPClassifier()
        self.controller = BCICursorController(duration=10.0)
        
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
        
        # Get frequencies and phases from screen calibration for dynamic widget creation
        # EXACTLY match screen calibration: each widget creates its own FlickerTarget
        try:
            from .screen_config import get_screen_calibration
            screen_cal = get_screen_calibration()
            higher_freq, lower_freq = screen_cal.frequencies
            phase_higher, phase_lower = screen_cal.phases
        except ImportError:
            higher_freq, lower_freq = 15.0, 12.0
            phase_higher, phase_lower = 0.0, np.pi
        
        # Top target (higher frequency - UP) with separate indicator
        top_container = QHBoxLayout()
        top_container.setSpacing(10)
        self.top_indicator = IndicatorLight()
        self.top_target = FlickerWidget(higher_freq, phase_higher, "top")
        top_container.addWidget(self.top_indicator)
        top_container.addWidget(self.top_target)
        top_container.addStretch()
        top_widget = QWidget()
        top_widget.setLayout(top_container)
        layout.addWidget(top_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Composition canvas
        self.canvas = CompositionCanvas()
        layout.addWidget(self.canvas, stretch=1)
        
        # Bottom target (lower frequency - DOWN) with separate indicator
        bottom_container = QHBoxLayout()
        bottom_container.setSpacing(10)
        self.bottom_indicator = IndicatorLight()
        self.bottom_target = FlickerWidget(lower_freq, phase_lower, "bottom")
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
        
        # Start
        self.controller.start()
        # NOTE: No stimulus.start() - widgets are independent (same as screen calibration)
        
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
        """Check screen calibration compatibility and warn if frequencies aren't factors of refresh rate."""
        from .screen_config import get_screen_calibration
        
        screen_cal = get_screen_calibration()
        
        if screen_cal.is_calibrated and screen_cal.refresh_rate_hz:
            is_compatible, warnings = screen_cal.check_frequency_compatibility()
            
            if warnings:
                warning_msg = "Screen Calibration Warning:\n\n" + "\n".join(warnings)
                warning_msg += "\n\nThis may cause flickering inconsistencies."
                warning_msg += "\nConsider adjusting monitor refresh rate or target frequencies."
                
                QMessageBox.warning(
                    self,
                    "Screen Compatibility Warning",
                    warning_msg
                )
    
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
        """Update composition state."""
        if self.controller.state != ControllerState.RUNNING:
            if self.controller.state == ControllerState.COMPLETED:
                self._stop_composition()
            return
        
        # Get EEG data and classify
        if self._use_lsl and self._lsl_connected:
            # Pull smaller chunks more efficiently (25ms = ~6-7 samples at 250Hz)
            # Smaller chunks = faster preprocessing, less CPU blocking
            n_samples_per_update = int(0.025 * self.preprocessor.sample_rate)  # 25ms worth
            processed = self.preprocessor.pull_and_process(n_samples=n_samples_per_update)
            
            if len(processed) > 0:
                # Use shorter window for faster classification (0.3s instead of 0.5s)
                eeg_buffer = self.preprocessor.get_recent_data(0.3)
                # Use CCA - it's more robust with proper phase-matched references
                result = self.classifier.classify(eeg_buffer, method="cca")
                
                # Track classification metrics
                classify_time = time.perf_counter()
                self._classification_results.append(result)
                self._classification_times.append(classify_time - self._composition_start_time)
                
                # Debug: Log classification results periodically
                if not hasattr(self, '_last_classify_log_time'):
                    self._last_classify_log_time = 0
                current_time = time.time()
                if current_time - self._last_classify_log_time > 1.0:  # Log every second
                    print(f"[CLASSIFY] Target={result.target.name}, Confidence={result.confidence:.2f}, "
                          f"Higher={result.power_higher_freq:.3f}, Lower={result.power_lower_freq:.3f}, "
                          f"Score={result.raw_score:.3f}")
                    self._last_classify_log_time = current_time
            else:
                # No data available - hold position
                result = ClassificationResult(
                    target=AttentionTarget.NONE,
                    confidence=0.0,
                    power_higher_freq=0.0,
                    power_lower_freq=0.0,
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
                # Get frequencies from screen calibration
                try:
                    from .screen_config import get_screen_calibration
                    screen_cal = get_screen_calibration()
                    higher_freq, lower_freq = screen_cal.frequencies
                except ImportError:
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
        trail = self.controller.get_trail_as_tuples()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        self.current_score = BCIScore(
            trail=trail,
            duration=self.controller.duration,
            waveform_name=self.waveform_combo.currentText(),
            metadata={
                'simulated': not self._use_lsl,
                'top_frequency': self.stimulus.top_frequency,
                'bottom_frequency': self.stimulus.bottom_frequency,
                'performance_metrics': performance_metrics
            }
        )
        
        # Enable playback controls
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Show statistics with performance summary
        stats = self.current_score.get_statistics()
        perf_summary = self._format_performance_summary(performance_metrics)
        self.status_label.setText(
            f"Score complete: {stats['num_points']} points, "
            f"pitch range: {stats['pitch_range']:.2f}, "
            f"total movement: {stats['total_movement']:.2f}\n{perf_summary}"
        )
        
        # Print detailed performance report
        self._print_performance_report(performance_metrics)
        
        # Also save to file for easy access
        self._save_performance_report(performance_metrics)
    
    def _calculate_performance_metrics(self) -> dict:
        """Calculate performance metrics from composition session."""
        metrics = {}
        
        # Flicker rate accuracy
        if hasattr(self.top_target, '_paint_intervals') and len(self.top_target._paint_intervals) > 50:
            top_intervals = self.top_target._paint_intervals
            top_mean_interval = np.mean(top_intervals)
            top_actual_rate = 1.0 / top_mean_interval if top_mean_interval > 0 else 0
            top_target_rate = self.top_target.frequency
            top_error = abs(top_actual_rate - top_target_rate)
            top_error_pct = (top_error / top_target_rate * 100) if top_target_rate > 0 else 0
            metrics['top_flicker'] = {
                'target_hz': top_target_rate,
                'actual_hz': top_actual_rate,
                'error_hz': top_error,
                'error_pct': top_error_pct
            }
        
        if hasattr(self.bottom_target, '_paint_intervals') and len(self.bottom_target._paint_intervals) > 50:
            bottom_intervals = self.bottom_target._paint_intervals
            bottom_mean_interval = np.mean(bottom_intervals)
            bottom_actual_rate = 1.0 / bottom_mean_interval if bottom_mean_interval > 0 else 0
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
            print(f"  Mean power (higher): {cf['mean_power_higher']:.3f}")
            print(f"  Mean power (lower):  {cf['mean_power_lower']:.3f}")
        
        print("=" * 60 + "\n")
        
        # Also save to file for easy access
        self._save_performance_report(metrics)
    
    def _save_performance_report(self, metrics: dict) -> None:
        """Save performance report to file for easy access."""
        from datetime import datetime
        report_file = Path("performance_report.txt")
        
        with open(report_file, 'w') as f:
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
        
        # Get frequencies for display/comparison
        try:
            from .screen_config import get_screen_calibration
            screen_cal = get_screen_calibration()
            self._cal_higher_freq, self._cal_lower_freq = screen_cal.frequencies
        except ImportError:
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


def run_bci_app():
    """Run the BCI composition application."""
    app = QApplication(sys.argv)
    window = BCICompositionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_bci_app()
