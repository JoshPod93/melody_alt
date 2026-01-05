"""
Envelope Editor for UPIC.

A dialog for creating and editing amplitude envelopes.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QLabel, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QMessageBox, QSlider
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QMouseEvent

from ..core.envelope import Envelope, EnvelopeLibrary, EnvelopeType
from ..core.synthesizer import preview_envelope, AUDIO_AVAILABLE


class EnvelopeCanvas(QWidget):
    """Canvas for drawing/displaying envelopes."""
    
    envelope_changed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.envelope: Optional[Envelope] = None
        self.is_drawing = False
        self.drawing_points: List[Tuple[float, float]] = []
        
        # Breakpoint editing
        self.breakpoints: List[Tuple[float, float]] = []
        self.selected_breakpoint: int = -1
        self.dragging_breakpoint: bool = False
        
        # Colors
        self.bg_color = QColor(25, 25, 30)
        self.grid_color = QColor(50, 50, 60)
        self.envelope_color = QColor(100, 255, 150)
        self.breakpoint_color = QColor(255, 200, 100)
        self.selected_bp_color = QColor(255, 100, 100)
        
        self.setMinimumSize(400, 200)
        self.setMouseTracking(True)
    
    def set_envelope(self, envelope: Optional[Envelope]) -> None:
        """Set the envelope to display."""
        self.envelope = envelope
        self.breakpoints.clear()
        self.selected_breakpoint = -1
        self.update()
    
    def set_breakpoints(self, breakpoints: List[Tuple[float, float]]) -> None:
        """Set breakpoints for editing."""
        self.breakpoints = list(breakpoints)
        self.update()
    
    def get_breakpoints(self) -> List[Tuple[float, float]]:
        """Get current breakpoints."""
        return list(self.breakpoints)
    
    def get_drawn_envelope(self) -> Optional[Envelope]:
        """Get an envelope from drawn points."""
        if len(self.drawing_points) < 2:
            return None
        return Envelope.from_points(self.drawing_points, "Drawn")
    
    def clear_drawing(self) -> None:
        """Clear drawn points."""
        self.drawing_points.clear()
        self.update()
    
    def _pixel_to_data(self, pos: QPointF) -> Tuple[float, float]:
        """Convert pixel position to (x, y) in [0,1] x [0,1]."""
        margin = 10
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        x = (pos.x() - margin) / w
        y = 1 - (pos.y() - margin) / h  # Flip y
        
        return (max(0, min(1, x)), max(0, min(1, y)))
    
    def _data_to_pixel(self, x: float, y: float) -> QPointF:
        """Convert (x, y) in [0,1] x [0,1] to pixel position."""
        margin = 10
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        px = margin + x * w
        py = margin + (1 - y) * h
        
        return QPointF(px, py)
    
    def _find_breakpoint_at(self, pos: QPointF, threshold: float = 10) -> int:
        """Find breakpoint near a position."""
        for i, (x, y) in enumerate(self.breakpoints):
            bp_pos = self._data_to_pixel(x, y)
            dx = pos.x() - bp_pos.x()
            dy = pos.y() - bp_pos.y()
            if (dx * dx + dy * dy) < threshold * threshold:
                return i
        return -1
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        pos = event.position()
        
        if event.button() == Qt.MouseButton.LeftButton:
            # Check for breakpoint
            bp_idx = self._find_breakpoint_at(pos)
            
            if bp_idx >= 0:
                self.selected_breakpoint = bp_idx
                self.dragging_breakpoint = True
            elif self.breakpoints:
                # Add new breakpoint
                x, y = self._pixel_to_data(pos)
                self.breakpoints.append((x, y))
                self.breakpoints.sort(key=lambda p: p[0])
                self.selected_breakpoint = next(
                    i for i, (bx, by) in enumerate(self.breakpoints)
                    if abs(bx - x) < 0.001
                )
                self.envelope_changed.emit()
            else:
                # Start freehand drawing
                self.is_drawing = True
                self.drawing_points.clear()
                x, y = self._pixel_to_data(pos)
                self.drawing_points.append((x, y))
        
        elif event.button() == Qt.MouseButton.RightButton:
            # Delete breakpoint
            bp_idx = self._find_breakpoint_at(pos)
            if bp_idx >= 0 and len(self.breakpoints) > 2:
                del self.breakpoints[bp_idx]
                self.selected_breakpoint = -1
                self.envelope_changed.emit()
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        pos = event.position()
        
        if self.dragging_breakpoint and self.selected_breakpoint >= 0:
            x, y = self._pixel_to_data(pos)
            
            # Constrain x to be between neighbors
            if self.selected_breakpoint > 0:
                x = max(x, self.breakpoints[self.selected_breakpoint - 1][0] + 0.01)
            if self.selected_breakpoint < len(self.breakpoints) - 1:
                x = min(x, self.breakpoints[self.selected_breakpoint + 1][0] - 0.01)
            
            self.breakpoints[self.selected_breakpoint] = (x, y)
            self.envelope_changed.emit()
        
        elif self.is_drawing:
            x, y = self._pixel_to_data(pos)
            self.drawing_points.append((x, y))
        
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_breakpoint = False
            
            if self.is_drawing:
                self.is_drawing = False
                if len(self.drawing_points) >= 2:
                    self.envelope_changed.emit()
        
        self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        margin = 10
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        # Grid
        painter.setPen(QPen(self.grid_color, 1))
        for i in range(5):
            x = margin + i * w / 4
            painter.drawLine(int(x), margin, int(x), self.height() - margin)
        for i in range(5):
            y = margin + i * h / 4
            painter.drawLine(margin, int(y), self.width() - margin, int(y))
        
        # Envelope
        if self.envelope is not None:
            self._draw_envelope(painter, self.envelope.samples)
        
        # Breakpoints
        if self.breakpoints:
            self._draw_breakpoints(painter)
        
        # Drawing preview
        if self.drawing_points:
            self._draw_points(painter, self.drawing_points)
        
        painter.end()
    
    def _draw_envelope(self, painter: QPainter, samples: np.ndarray) -> None:
        """Draw an envelope from samples."""
        path = QPainterPath()
        
        num_points = min(len(samples), self.width())
        step = len(samples) / num_points
        
        for i in range(num_points):
            idx = int(i * step)
            x = i / num_points
            y = samples[idx]
            
            pos = self._data_to_pixel(x, y)
            if i == 0:
                path.moveTo(pos)
            else:
                path.lineTo(pos)
        
        painter.setPen(QPen(self.envelope_color, 2))
        painter.drawPath(path)
    
    def _draw_breakpoints(self, painter: QPainter) -> None:
        """Draw breakpoints and connecting lines."""
        if len(self.breakpoints) < 2:
            return
        
        # Draw lines
        path = QPainterPath()
        pos = self._data_to_pixel(self.breakpoints[0][0], self.breakpoints[0][1])
        path.moveTo(pos)
        
        for x, y in self.breakpoints[1:]:
            pos = self._data_to_pixel(x, y)
            path.lineTo(pos)
        
        painter.setPen(QPen(self.breakpoint_color, 2))
        painter.drawPath(path)
        
        # Draw points
        for i, (x, y) in enumerate(self.breakpoints):
            pos = self._data_to_pixel(x, y)
            
            if i == self.selected_breakpoint:
                painter.setBrush(QBrush(self.selected_bp_color))
                painter.setPen(QPen(self.selected_bp_color.darker(), 2))
            else:
                painter.setBrush(QBrush(self.breakpoint_color))
                painter.setPen(QPen(self.breakpoint_color.darker(), 2))
            
            painter.drawEllipse(pos, 6, 6)
    
    def _draw_points(self, painter: QPainter, points: List[Tuple[float, float]]) -> None:
        """Draw freehand points."""
        if len(points) < 2:
            return
        
        path = QPainterPath()
        pos = self._data_to_pixel(points[0][0], points[0][1])
        path.moveTo(pos)
        
        for x, y in points[1:]:
            pos = self._data_to_pixel(x, y)
            path.lineTo(pos)
        
        painter.setPen(QPen(QColor(255, 200, 100), 2))
        painter.drawPath(path)


class EnvelopeEditorDialog(QDialog):
    """Dialog for editing envelopes."""
    
    def __init__(
        self,
        library: EnvelopeLibrary,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.library = library
        self.current_envelope: Optional[Envelope] = None
        
        self.setWindowTitle("Envelope Editor")
        self.setMinimumSize(700, 500)
        
        self._init_ui()
        self._refresh_list()
    
    def _init_ui(self) -> None:
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        
        # Left panel - envelope list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Envelopes:"))
        
        self.envelope_list = QListWidget()
        self.envelope_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.envelope_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self._new_envelope)
        btn_layout.addWidget(self.new_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_envelope)
        btn_layout.addWidget(self.delete_btn)
        
        left_layout.addLayout(btn_layout)
        
        layout.addWidget(left_panel, stretch=1)
        
        # Right panel - editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_layout.addWidget(self.name_edit)
        right_layout.addLayout(name_layout)
        
        # Canvas
        self.canvas = EnvelopeCanvas()
        self.canvas.envelope_changed.connect(self._on_canvas_changed)
        right_layout.addWidget(self.canvas, stretch=1)
        
        # ADSR controls
        adsr_group = QGroupBox("ADSR")
        adsr_layout = QFormLayout(adsr_group)
        
        self.attack_spin = QDoubleSpinBox()
        self.attack_spin.setRange(0.0, 1.0)
        self.attack_spin.setSingleStep(0.05)
        self.attack_spin.setValue(0.1)
        adsr_layout.addRow("Attack:", self.attack_spin)
        
        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.0, 1.0)
        self.decay_spin.setSingleStep(0.05)
        self.decay_spin.setValue(0.1)
        adsr_layout.addRow("Decay:", self.decay_spin)
        
        self.sustain_spin = QDoubleSpinBox()
        self.sustain_spin.setRange(0.0, 1.0)
        self.sustain_spin.setSingleStep(0.05)
        self.sustain_spin.setValue(0.7)
        adsr_layout.addRow("Sustain Level:", self.sustain_spin)
        
        self.release_spin = QDoubleSpinBox()
        self.release_spin.setRange(0.0, 1.0)
        self.release_spin.setSingleStep(0.05)
        self.release_spin.setValue(0.2)
        adsr_layout.addRow("Release:", self.release_spin)
        
        apply_adsr_btn = QPushButton("Apply ADSR")
        apply_adsr_btn.clicked.connect(self._apply_adsr)
        adsr_layout.addRow(apply_adsr_btn)
        
        right_layout.addWidget(adsr_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        for name, params in [
            ("Constant", {}),
            ("Fade In/Out", {"fade": True}),
            ("Pluck", {"exp": True}),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, n=name, p=params: self._apply_preset(n, p))
            preset_layout.addWidget(btn)
        
        right_layout.addLayout(preset_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._preview)
        self.preview_btn.setEnabled(AUDIO_AVAILABLE)
        action_layout.addWidget(self.preview_btn)
        
        self.apply_btn = QPushButton("Apply Drawing")
        self.apply_btn.clicked.connect(self._apply_drawing)
        action_layout.addWidget(self.apply_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear_drawing)
        action_layout.addWidget(self.clear_btn)
        
        self.edit_bp_btn = QPushButton("Edit Breakpoints")
        self.edit_bp_btn.setCheckable(True)
        self.edit_bp_btn.clicked.connect(self._toggle_breakpoint_mode)
        action_layout.addWidget(self.edit_bp_btn)
        
        right_layout.addLayout(action_layout)
        
        layout.addWidget(right_panel, stretch=2)
    
    def _refresh_list(self) -> None:
        """Refresh the envelope list."""
        self.envelope_list.clear()
        for name in self.library.list_names():
            self.envelope_list.addItem(name)
        
        if self.envelope_list.count() > 0:
            self.envelope_list.setCurrentRow(0)
    
    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        """Handle envelope selection change."""
        if current:
            name = current.text()
            self.current_envelope = self.library.get(name)
            if self.current_envelope:
                self.name_edit.setText(self.current_envelope.name)
                self.canvas.set_envelope(self.current_envelope)
                self.canvas.breakpoints.clear()
                self.edit_bp_btn.setChecked(False)
    
    def _on_name_changed(self, text: str) -> None:
        """Handle name change."""
        if self.current_envelope and text:
            old_name = self.current_envelope.name
            if old_name != text and text not in self.library.envelopes:
                self.library.remove(old_name)
                self.current_envelope.name = text
                self.library.add(self.current_envelope)
                self._refresh_list()
                items = self.envelope_list.findItems(text, Qt.MatchFlag.MatchExactly)
                if items:
                    self.envelope_list.setCurrentItem(items[0])
    
    def _on_canvas_changed(self) -> None:
        """Handle canvas change (breakpoints moved)."""
        if self.edit_bp_btn.isChecked() and self.current_envelope:
            bp = self.canvas.get_breakpoints()
            if len(bp) >= 2:
                new_env = Envelope.from_breakpoints(bp)
                self.current_envelope.samples = new_env.samples.copy()
                self.current_envelope.envelope_type = EnvelopeType.CUSTOM
    
    def _new_envelope(self) -> None:
        """Create a new envelope."""
        envelope = Envelope.adsr(f"Envelope {len(self.library) + 1}")
        if self.library.add(envelope):
            self._refresh_list()
            items = self.envelope_list.findItems(envelope.name, Qt.MatchFlag.MatchExactly)
            if items:
                self.envelope_list.setCurrentItem(items[0])
        else:
            QMessageBox.warning(self, "Warning", "Envelope library is full.")
    
    def _delete_envelope(self) -> None:
        """Delete the selected envelope."""
        if self.current_envelope:
            if len(self.library) <= 1:
                QMessageBox.warning(self, "Warning", "Cannot delete the last envelope.")
                return
            
            self.library.remove(self.current_envelope.name)
            self.current_envelope = None
            self._refresh_list()
    
    def _apply_adsr(self) -> None:
        """Apply ADSR settings."""
        if not self.current_envelope:
            return
        
        new_env = Envelope.adsr(
            attack=self.attack_spin.value(),
            decay=self.decay_spin.value(),
            sustain=0.5,  # Time, not level
            release=self.release_spin.value(),
            sustain_level=self.sustain_spin.value()
        )
        
        self.current_envelope.samples = new_env.samples.copy()
        self.current_envelope.envelope_type = EnvelopeType.ADSR
        self.canvas.set_envelope(self.current_envelope)
    
    def _apply_preset(self, name: str, params: dict) -> None:
        """Apply a preset envelope."""
        if not self.current_envelope:
            return
        
        if name == "Constant":
            new_env = Envelope.constant()
        elif name == "Fade In/Out":
            new_env = Envelope.linear_fade()
        elif name == "Pluck":
            new_env = Envelope.exponential()
        else:
            return
        
        self.current_envelope.samples = new_env.samples.copy()
        self.current_envelope.envelope_type = new_env.envelope_type
        self.canvas.set_envelope(self.current_envelope)
    
    def _apply_drawing(self) -> None:
        """Apply the drawn envelope."""
        if not self.current_envelope:
            return
        
        drawn = self.canvas.get_drawn_envelope()
        if drawn:
            self.current_envelope.samples = drawn.samples.copy()
            self.current_envelope.envelope_type = EnvelopeType.CUSTOM
            self.canvas.clear_drawing()
            self.canvas.set_envelope(self.current_envelope)
    
    def _toggle_breakpoint_mode(self, checked: bool) -> None:
        """Toggle breakpoint editing mode."""
        if checked and self.current_envelope:
            # Initialize breakpoints from current envelope
            # Simple: just use ADSR-like breakpoints
            self.canvas.set_breakpoints([
                (0.0, 0.0),
                (0.1, 1.0),
                (0.3, 0.7),
                (0.8, 0.7),
                (1.0, 0.0)
            ])
        else:
            self.canvas.breakpoints.clear()
            self.canvas.update()
    
    def _preview(self) -> None:
        """Preview the current envelope."""
        if self.current_envelope and AUDIO_AVAILABLE:
            preview_envelope(self.current_envelope, duration=1.0)

