"""
BCI Composition Interface for BCI-UPIC.

Main real-time interface for BCI-controlled music composition.
Features:
- Two flickering SSVEP targets (15Hz top, 10Hz bottom)
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

try:
    from .lsl_stream import LSLReceiver, LSLMarkerSender, LSL_AVAILABLE
except ImportError:
    LSL_AVAILABLE = False


class SessionMode(Enum):
    """Mode of the BCI session."""
    IDLE = 0
    COMPOSING = 1
    PLAYBACK = 2


class FlickerWidget(QWidget):
    """
    Widget displaying a flickering SSVEP target.
    
    Renders a rectangle that flickers at the specified frequency.
    """
    
    def __init__(
        self,
        frequency: float,
        position: str = "top",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.frequency = frequency
        self.position = position
        self._intensity = 0.0
        self._is_active = False
        
        # Colors
        self.color_on = QColor(255, 255, 255)
        self.color_off = QColor(30, 30, 30)
        self.border_color = QColor(100, 100, 100)
        
        # Size
        self.setMinimumSize(300, 80)
        self.setMaximumHeight(100)
    
    def set_intensity(self, intensity: float) -> None:
        """Set the current intensity (0-1)."""
        self._intensity = intensity
        self.update()
    
    def set_active(self, active: bool) -> None:
        """Set whether the target is active (flickering)."""
        self._is_active = active
        if not active:
            self._intensity = 0.0
        self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the flickering target."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate current color based on intensity
        if self._is_active:
            r = int(self.color_off.red() + self._intensity * (self.color_on.red() - self.color_off.red()))
            g = int(self.color_off.green() + self._intensity * (self.color_on.green() - self.color_off.green()))
            b = int(self.color_off.blue() + self._intensity * (self.color_on.blue() - self.color_off.blue()))
            color = QColor(r, g, b)
        else:
            color = self.color_off
        
        # Draw rounded rectangle
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.setPen(QPen(self.border_color, 2))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(rect, 10, 10)
        
        # Draw frequency label
        painter.setPen(QPen(QColor(150, 150, 150)))
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        
        label = f"{self.frequency:.0f} Hz"
        if self.position == "top":
            label += " ▲ UP"
        else:
            label += " ▼ DOWN"
        
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)


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
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI-UPIC Composer")
        self.setMinimumSize(1000, 700)
        
        # BCI components
        self.stimulus = SSVEPStimulus(duration=10.0)
        self.classifier = SSVEPClassifier()
        self.controller = BCICursorController(duration=10.0)
        
        # LSL or simulated EEG
        self._use_lsl = False
        self._lsl_connected = False
        
        # LSL preprocessor (handles both LSL and simulated)
        self.preprocessor = LSLPreprocessor(sample_rate=250, n_channels=8)
        
        # Simulated EEG source (for testing without hardware)
        self.eeg_source = SimulatedEEGSource(sample_rate=250, n_channels=8)
        
        # Marker sender for LSL
        self.marker_sender = None
        if LSL_AVAILABLE:
            try:
                self.marker_sender = LSLMarkerSender()
            except:
                pass
        
        # Current score
        self.current_score: Optional[BCIScore] = None
        
        # Session state
        self._mode = SessionMode.IDLE
        
        # Timers
        self._stimulus_timer = QTimer()
        self._stimulus_timer.timeout.connect(self._update_stimulus)
        self._stimulus_timer.setInterval(16)  # ~60 FPS
        
        self._composition_timer = QTimer()
        self._composition_timer.timeout.connect(self._update_composition)
        self._composition_timer.setInterval(16)
        
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
        
        # Top target (15Hz - UP)
        self.top_target = FlickerWidget(15.0, "top")
        layout.addWidget(self.top_target)
        
        # Composition canvas
        self.canvas = CompositionCanvas()
        layout.addWidget(self.canvas, stretch=1)
        
        # Bottom target (10Hz - DOWN)
        self.bottom_target = FlickerWidget(10.0, "bottom")
        layout.addWidget(self.bottom_target)
        
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
        
        # Test mode
        test_group = QGroupBox("Test Mode")
        test_layout = QHBoxLayout(test_group)
        
        self.random_btn = QPushButton("Random Test")
        self.random_btn.clicked.connect(self._run_random_test)
        test_layout.addWidget(self.random_btn)
        
        layout.addWidget(test_group)
        
        # LSL Connection
        lsl_group = QGroupBox("EEG Source")
        lsl_layout = QHBoxLayout(lsl_group)
        
        self.lsl_status = QLabel("Simulated")
        self.lsl_status.setStyleSheet("color: #ffa500;")  # Orange for simulated
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
        self._mode = SessionMode.COMPOSING
        
        # Send marker if LSL available
        if self.marker_sender:
            self.marker_sender.send("Composition Start")
        
        # Reset components
        self.controller.reset()
        self.classifier.reset()
        self.preprocessor.reset()
        self.canvas.clear()
        
        # Configure
        duration = self.duration_spin.value()
        self.controller.duration = float(duration)
        self.stimulus.duration = float(duration)
        self.canvas.set_duration(float(duration))
        
        # Start
        self.controller.start()
        self.stimulus.start()
        
        # Start timers
        self._stimulus_timer.start()
        self._composition_timer.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
        self.duration_spin.setEnabled(False)
        self.waveform_combo.setEnabled(False)
        
        self.top_target.set_active(True)
        self.bottom_target.set_active(True)
        self.canvas.set_composing(True)
        
        self.status_label.setText("Composing... Focus on TOP target to move UP, BOTTOM target to move DOWN")
    
    def _stop_composition(self) -> None:
        """Stop composition and finalize score."""
        # Send marker if LSL available
        if self.marker_sender:
            self.marker_sender.send("Composition End")
        
        self._stimulus_timer.stop()
        self._composition_timer.stop()
        
        self.controller.stop()
        self.stimulus.stop()
        
        self.top_target.set_active(False)
        self.bottom_target.set_active(False)
        self.canvas.set_composing(False)
        
        self._finalize_score()
        
        self._mode = SessionMode.IDLE
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.random_btn.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.waveform_combo.setEnabled(True)
        
        self.status_label.setText("Composition complete! Play or save your score.")
    
    def _update_stimulus(self) -> None:
        """Update flickering stimulus display."""
        if not self.stimulus.is_running:
            return
        
        # Get current intensities
        top_intensity, bottom_intensity = self.stimulus.get_intensities()
        
        # Update target widgets
        self.top_target.set_intensity(top_intensity)
        self.bottom_target.set_intensity(bottom_intensity)
    
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
                self.status_label.setText("No LSL streams found - using simulated EEG")
                QMessageBox.information(
                    self,
                    "LSL Connection",
                    "No EEG streams found.\n\n"
                    "Make sure your g.tec Unicorn Black is:\n"
                    "1. Powered on\n"
                    "2. Connected via Bluetooth\n"
                    "3. Streaming via LSL\n\n"
                    "Using simulated EEG for now."
                )
    
    def _update_composition(self) -> None:
        """Update composition state."""
        if self.controller.state != ControllerState.RUNNING:
            if self.controller.state == ControllerState.COMPLETED:
                self._stop_composition()
            return
        
        # Get EEG data and classify
        if self._use_lsl and self._lsl_connected:
            # Pull and process from LSL stream
            processed = self.preprocessor.pull_and_process(n_samples=16)
            
            if len(processed) > 0:
                # Get buffer for classification
                eeg_buffer = self.preprocessor.get_recent_data(1.0)
                result = self.classifier.classify(eeg_buffer, method="fft")
            else:
                # No data available
                result = ClassificationResult(
                    target=AttentionTarget.NONE,
                    confidence=0.0,
                    power_15hz=0.0,
                    power_10hz=0.0,
                    raw_score=0.0
                )
        else:
            # Simulated mode
            # Simulate user attention based on cursor position
            # (for testing - user "wants" to go to center)
            current_pitch = self.controller.position.pitch
            if current_pitch < 0.4:
                self.eeg_source.set_target(15.0)  # Attend to UP
            elif current_pitch > 0.6:
                self.eeg_source.set_target(10.0)  # Attend to DOWN
            else:
                self.eeg_source.set_target(None)  # No strong attention
            
            # Generate and process EEG
            eeg_chunk = self.eeg_source.generate_chunk(16)
            processed = self.preprocessor.process_chunk(eeg_chunk)
            
            # Classify
            eeg_buffer = self.preprocessor.get_recent_data(1.0)
            result = self.classifier.classify(eeg_buffer, method="fft")
        
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
        
        self.current_score = BCIScore(
            trail=trail,
            duration=self.controller.duration,
            waveform_name=self.waveform_combo.currentText(),
            metadata={
                'simulated': self._use_simulated,
                'top_frequency': self.stimulus.top_frequency,
                'bottom_frequency': self.stimulus.bottom_frequency
            }
        )
        
        # Enable playback controls
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Show statistics
        stats = self.current_score.get_statistics()
        self.status_label.setText(
            f"Score complete: {stats['num_points']} points, "
            f"pitch range: {stats['pitch_range']:.2f}, "
            f"total movement: {stats['total_movement']:.2f}"
        )
    
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
        self.stimulus.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
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
                
                # Update flicker
                top_int, bottom_int = self.stimulus.get_intensities()
                self.top_target.set_intensity(top_int)
                self.bottom_target.set_intensity(bottom_int)
                
                # Update progress
                progress = random_ctrl._controller.progress
                self.progress_bar.setValue(int(progress * 100))
            else:
                # Test complete
                random_timer.stop()
                self.stimulus.stop()
                
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
                self.random_btn.setEnabled(True)
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
        self._stimulus_timer.stop()
        self._composition_timer.stop()
        self.controller.stop()
        self.stimulus.stop()
        event.accept()


def run_bci_app():
    """Run the BCI composition application."""
    app = QApplication(sys.argv)
    window = BCICompositionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_bci_app()
