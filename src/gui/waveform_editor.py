"""
Waveform Editor for UPIC.

A dialog for creating and editing waveforms by drawing or
using mathematical functions.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QLabel, QComboBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox,
    QSplitter
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QMouseEvent

from ..core.waveform import Waveform, WaveformLibrary, WaveformType
from ..core.synthesizer import preview_waveform, AUDIO_AVAILABLE


class WaveformCanvas(QWidget):
    """Canvas for drawing/displaying waveforms."""
    
    waveform_changed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.waveform: Optional[Waveform] = None
        self.is_drawing = False
        self.drawing_points: List[Tuple[float, float]] = []
        
        # Colors
        self.bg_color = QColor(25, 25, 30)
        self.grid_color = QColor(50, 50, 60)
        self.wave_color = QColor(100, 200, 255)
        self.zero_line_color = QColor(80, 80, 90)
        
        self.setMinimumSize(400, 200)
        self.setMouseTracking(True)
    
    def set_waveform(self, waveform: Optional[Waveform]) -> None:
        """Set the waveform to display."""
        self.waveform = waveform
        self.update()
    
    def get_drawn_waveform(self) -> Optional[Waveform]:
        """Get a waveform from the drawn points."""
        if len(self.drawing_points) < 2:
            return None
        return Waveform.from_points(self.drawing_points, "Drawn")
    
    def clear_drawing(self) -> None:
        """Clear drawn points."""
        self.drawing_points.clear()
        self.update()
    
    def _pixel_to_data(self, pos: QPointF) -> Tuple[float, float]:
        """Convert pixel position to (x, y) in [0,1] x [-1,1]."""
        margin = 10
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        x = (pos.x() - margin) / w
        y = 1 - 2 * (pos.y() - margin) / h  # Flip y, map to [-1, 1]
        
        return (max(0, min(1, x)), max(-1, min(1, y)))
    
    def _data_to_pixel(self, x: float, y: float) -> QPointF:
        """Convert (x, y) in [0,1] x [-1,1] to pixel position."""
        margin = 10
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        px = margin + x * w
        py = margin + (1 - y) / 2 * h  # Map [-1, 1] to [h, 0]
        
        return QPointF(px, py)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.drawing_points.clear()
            x, y = self._pixel_to_data(event.position())
            self.drawing_points.append((x, y))
            self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        if self.is_drawing:
            x, y = self._pixel_to_data(event.position())
            self.drawing_points.append((x, y))
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            if len(self.drawing_points) >= 2:
                self.waveform_changed.emit()
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
        
        # Zero line
        painter.setPen(QPen(self.zero_line_color, 2))
        y_zero = margin + h / 2
        painter.drawLine(margin, int(y_zero), self.width() - margin, int(y_zero))
        
        # Waveform
        if self.waveform is not None:
            self._draw_waveform(painter, self.waveform.samples)
        
        # Drawing preview
        if self.drawing_points:
            self._draw_points(painter, self.drawing_points)
        
        painter.end()
    
    def _draw_waveform(self, painter: QPainter, samples: np.ndarray) -> None:
        """Draw a waveform from samples."""
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
        
        painter.setPen(QPen(self.wave_color, 2))
        painter.drawPath(path)
    
    def _draw_points(self, painter: QPainter, points: List[Tuple[float, float]]) -> None:
        """Draw points as a path."""
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


class WaveformEditorDialog(QDialog):
    """Dialog for editing waveforms."""
    
    def __init__(
        self,
        library: WaveformLibrary,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.library = library
        self.current_waveform: Optional[Waveform] = None
        
        self.setWindowTitle("Waveform Editor")
        self.setMinimumSize(700, 500)
        
        self._init_ui()
        self._refresh_list()
    
    def _init_ui(self) -> None:
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        
        # Left panel - waveform list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Waveforms:"))
        
        self.waveform_list = QListWidget()
        self.waveform_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.waveform_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self._new_waveform)
        btn_layout.addWidget(self.new_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_waveform)
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
        self.canvas = WaveformCanvas()
        self.canvas.waveform_changed.connect(self._on_canvas_changed)
        right_layout.addWidget(self.canvas, stretch=1)
        
        # Preset buttons
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        
        for name in ["Sine", "Triangle", "Sawtooth", "Square"]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, n=name: self._apply_preset(n))
            preset_layout.addWidget(btn)
        
        right_layout.addWidget(preset_group)
        
        # Harmonics
        harmonics_group = QGroupBox("Harmonics")
        harmonics_layout = QFormLayout(harmonics_group)
        
        self.num_harmonics = QSpinBox()
        self.num_harmonics.setRange(1, 32)
        self.num_harmonics.setValue(8)
        harmonics_layout.addRow("Harmonics:", self.num_harmonics)
        
        self.harmonic_falloff = QDoubleSpinBox()
        self.harmonic_falloff.setRange(0.1, 3.0)
        self.harmonic_falloff.setValue(1.0)
        self.harmonic_falloff.setSingleStep(0.1)
        harmonics_layout.addRow("Falloff:", self.harmonic_falloff)
        
        gen_btn = QPushButton("Generate")
        gen_btn.clicked.connect(self._generate_harmonics)
        harmonics_layout.addRow(gen_btn)
        
        right_layout.addWidget(harmonics_group)
        
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
        
        right_layout.addLayout(action_layout)
        
        layout.addWidget(right_panel, stretch=2)
    
    def _refresh_list(self) -> None:
        """Refresh the waveform list."""
        self.waveform_list.clear()
        for name in self.library.list_names():
            self.waveform_list.addItem(name)
        
        if self.waveform_list.count() > 0:
            self.waveform_list.setCurrentRow(0)
    
    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        """Handle waveform selection change."""
        if current:
            name = current.text()
            self.current_waveform = self.library.get(name)
            if self.current_waveform:
                self.name_edit.setText(self.current_waveform.name)
                self.canvas.set_waveform(self.current_waveform)
    
    def _on_name_changed(self, text: str) -> None:
        """Handle name change."""
        if self.current_waveform and text:
            old_name = self.current_waveform.name
            if old_name != text and text not in self.library.waveforms:
                self.library.remove(old_name)
                self.current_waveform.name = text
                self.library.add(self.current_waveform)
                self._refresh_list()
                # Reselect
                items = self.waveform_list.findItems(text, Qt.MatchFlag.MatchExactly)
                if items:
                    self.waveform_list.setCurrentItem(items[0])
    
    def _on_canvas_changed(self) -> None:
        """Handle canvas drawing change."""
        pass  # Drawing is applied manually
    
    def _new_waveform(self) -> None:
        """Create a new waveform."""
        waveform = Waveform.sine(f"Waveform {len(self.library) + 1}")
        if self.library.add(waveform):
            self._refresh_list()
            items = self.waveform_list.findItems(waveform.name, Qt.MatchFlag.MatchExactly)
            if items:
                self.waveform_list.setCurrentItem(items[0])
        else:
            QMessageBox.warning(self, "Warning", "Waveform library is full.")
    
    def _delete_waveform(self) -> None:
        """Delete the selected waveform."""
        if self.current_waveform:
            if len(self.library) <= 1:
                QMessageBox.warning(self, "Warning", "Cannot delete the last waveform.")
                return
            
            self.library.remove(self.current_waveform.name)
            self.current_waveform = None
            self._refresh_list()
    
    def _apply_preset(self, preset: str) -> None:
        """Apply a preset waveform."""
        if not self.current_waveform:
            return
        
        if preset == "Sine":
            new_wf = Waveform.sine()
        elif preset == "Triangle":
            new_wf = Waveform.triangle()
        elif preset == "Sawtooth":
            new_wf = Waveform.sawtooth()
        elif preset == "Square":
            new_wf = Waveform.square()
        else:
            return
        
        self.current_waveform.samples = new_wf.samples.copy()
        self.current_waveform.waveform_type = new_wf.waveform_type
        self.canvas.set_waveform(self.current_waveform)
    
    def _generate_harmonics(self) -> None:
        """Generate waveform from harmonics."""
        if not self.current_waveform:
            return
        
        num = self.num_harmonics.value()
        falloff = self.harmonic_falloff.value()
        
        harmonics = []
        for i in range(1, num + 1):
            amp = 1.0 / (i ** falloff)
            harmonics.append((i, amp, 0.0))
        
        new_wf = Waveform.from_harmonics(harmonics)
        self.current_waveform.samples = new_wf.samples.copy()
        self.current_waveform.waveform_type = WaveformType.CUSTOM
        self.canvas.set_waveform(self.current_waveform)
    
    def _apply_drawing(self) -> None:
        """Apply the drawn waveform."""
        if not self.current_waveform:
            return
        
        drawn = self.canvas.get_drawn_waveform()
        if drawn:
            self.current_waveform.samples = drawn.samples.copy()
            self.current_waveform.waveform_type = WaveformType.CUSTOM
            self.canvas.clear_drawing()
            self.canvas.set_waveform(self.current_waveform)
    
    def _preview(self) -> None:
        """Preview the current waveform."""
        if self.current_waveform and AUDIO_AVAILABLE:
            preview_waveform(self.current_waveform, duration=0.5)

