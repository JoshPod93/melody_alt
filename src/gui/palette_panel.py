"""
Palette Panel for UPIC.

A panel displaying available waveforms and envelopes that can be
selected for use when drawing new arcs.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QTabWidget, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QPixmap, QIcon

from ..core.waveform import Waveform, WaveformLibrary
from ..core.envelope import Envelope, EnvelopeLibrary


class WaveformPreview(QWidget):
    """Small preview widget for a waveform."""
    
    def __init__(self, waveform: Waveform, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.waveform = waveform
        self.setFixedSize(60, 30)
    
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 45))
        
        # Draw waveform
        path = QPainterPath()
        w = self.width()
        h = self.height()
        
        num_points = min(len(self.waveform.samples), w)
        step = len(self.waveform.samples) / num_points
        
        for i in range(num_points):
            idx = int(i * step)
            x = i
            y = h / 2 - self.waveform.samples[idx] * h / 2 * 0.8
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        painter.setPen(QPen(QColor(100, 200, 255), 1))
        painter.drawPath(path)
        
        painter.end()


class EnvelopePreview(QWidget):
    """Small preview widget for an envelope."""
    
    def __init__(self, envelope: Envelope, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.envelope = envelope
        self.setFixedSize(60, 30)
    
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 45))
        
        # Draw envelope
        path = QPainterPath()
        w = self.width()
        h = self.height()
        
        num_points = min(len(self.envelope.samples), w)
        step = len(self.envelope.samples) / num_points
        
        for i in range(num_points):
            idx = int(i * step)
            x = i
            y = h - self.envelope.samples[idx] * h * 0.9
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        painter.setPen(QPen(QColor(100, 255, 150), 1))
        painter.drawPath(path)
        
        painter.end()


def create_waveform_icon(waveform: Waveform, size: int = 32) -> QIcon:
    """Create an icon from a waveform."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(40, 40, 45))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    path = QPainterPath()
    num_points = min(len(waveform.samples), size)
    step = len(waveform.samples) / num_points
    
    for i in range(num_points):
        idx = int(i * step)
        x = i
        y = size / 2 - waveform.samples[idx] * size / 2 * 0.7
        
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    
    painter.setPen(QPen(QColor(100, 200, 255), 1))
    painter.drawPath(path)
    painter.end()
    
    return QIcon(pixmap)


def create_envelope_icon(envelope: Envelope, size: int = 32) -> QIcon:
    """Create an icon from an envelope."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(40, 40, 45))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    path = QPainterPath()
    num_points = min(len(envelope.samples), size)
    step = len(envelope.samples) / num_points
    
    for i in range(num_points):
        idx = int(i * step)
        x = i
        y = size - envelope.samples[idx] * size * 0.8
        
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    
    painter.setPen(QPen(QColor(100, 255, 150), 1))
    painter.drawPath(path)
    painter.end()
    
    return QIcon(pixmap)


class PalettePanel(QWidget):
    """
    Panel for selecting waveforms and envelopes.
    
    This is the "palette" that users select from when drawing new arcs.
    """
    
    # Signals
    waveform_selected = pyqtSignal(str)  # Waveform name
    envelope_selected = pyqtSignal(str)  # Envelope name
    
    def __init__(
        self,
        waveforms: WaveformLibrary,
        envelopes: EnvelopeLibrary,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        
        self.waveforms = waveforms
        self.envelopes = envelopes
        
        self._init_ui()
        self.refresh()
    
    def _init_ui(self) -> None:
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tab widget for waveforms and envelopes
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Waveforms tab
        waveform_widget = QWidget()
        waveform_layout = QVBoxLayout(waveform_widget)
        
        waveform_layout.addWidget(QLabel("Select Waveform:"))
        
        self.waveform_list = QListWidget()
        self.waveform_list.setIconSize(QSize(32, 32))
        self.waveform_list.currentItemChanged.connect(self._on_waveform_selected)
        waveform_layout.addWidget(self.waveform_list)
        
        # Preview
        self.waveform_preview_label = QLabel("Preview:")
        waveform_layout.addWidget(self.waveform_preview_label)
        
        self.waveform_preview = QFrame()
        self.waveform_preview.setFixedHeight(60)
        self.waveform_preview.setStyleSheet("background-color: #282830;")
        waveform_layout.addWidget(self.waveform_preview)
        
        self.tabs.addTab(waveform_widget, "Waveforms")
        
        # Envelopes tab
        envelope_widget = QWidget()
        envelope_layout = QVBoxLayout(envelope_widget)
        
        envelope_layout.addWidget(QLabel("Select Envelope:"))
        
        self.envelope_list = QListWidget()
        self.envelope_list.setIconSize(QSize(32, 32))
        self.envelope_list.currentItemChanged.connect(self._on_envelope_selected)
        envelope_layout.addWidget(self.envelope_list)
        
        # Preview
        self.envelope_preview_label = QLabel("Preview:")
        envelope_layout.addWidget(self.envelope_preview_label)
        
        self.envelope_preview = QFrame()
        self.envelope_preview.setFixedHeight(60)
        self.envelope_preview.setStyleSheet("background-color: #282830;")
        envelope_layout.addWidget(self.envelope_preview)
        
        self.tabs.addTab(envelope_widget, "Envelopes")
        
        # Current selection display
        self.selection_label = QLabel("Current: Sine / ADSR")
        self.selection_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.selection_label)
    
    def refresh(self) -> None:
        """Refresh the lists from libraries."""
        # Waveforms
        self.waveform_list.clear()
        for waveform in self.waveforms:
            item = QListWidgetItem(waveform.name)
            item.setIcon(create_waveform_icon(waveform))
            self.waveform_list.addItem(item)
        
        if self.waveform_list.count() > 0:
            self.waveform_list.setCurrentRow(0)
        
        # Envelopes
        self.envelope_list.clear()
        for envelope in self.envelopes:
            item = QListWidgetItem(envelope.name)
            item.setIcon(create_envelope_icon(envelope))
            self.envelope_list.addItem(item)
        
        if self.envelope_list.count() > 0:
            self.envelope_list.setCurrentRow(0)
        
        self._update_selection_label()
    
    def _on_waveform_selected(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        """Handle waveform selection."""
        if current:
            name = current.text()
            self.waveform_selected.emit(name)
            self._update_waveform_preview(name)
            self._update_selection_label()
    
    def _on_envelope_selected(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        """Handle envelope selection."""
        if current:
            name = current.text()
            self.envelope_selected.emit(name)
            self._update_envelope_preview(name)
            self._update_selection_label()
    
    def _update_waveform_preview(self, name: str) -> None:
        """Update the waveform preview."""
        waveform = self.waveforms.get(name)
        if waveform:
            # Clear old preview
            if self.waveform_preview.layout():
                while self.waveform_preview.layout().count():
                    item = self.waveform_preview.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            else:
                layout = QVBoxLayout(self.waveform_preview)
                layout.setContentsMargins(5, 5, 5, 5)
            
            preview = WaveformPreviewLarge(waveform)
            self.waveform_preview.layout().addWidget(preview)
    
    def _update_envelope_preview(self, name: str) -> None:
        """Update the envelope preview."""
        envelope = self.envelopes.get(name)
        if envelope:
            # Clear old preview
            if self.envelope_preview.layout():
                while self.envelope_preview.layout().count():
                    item = self.envelope_preview.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            else:
                layout = QVBoxLayout(self.envelope_preview)
                layout.setContentsMargins(5, 5, 5, 5)
            
            preview = EnvelopePreviewLarge(envelope)
            self.envelope_preview.layout().addWidget(preview)
    
    def _update_selection_label(self) -> None:
        """Update the current selection label."""
        wf_item = self.waveform_list.currentItem()
        env_item = self.envelope_list.currentItem()
        
        wf_name = wf_item.text() if wf_item else "None"
        env_name = env_item.text() if env_item else "None"
        
        self.selection_label.setText(f"Current: {wf_name} / {env_name}")
    
    def get_current_waveform(self) -> Optional[str]:
        """Get the currently selected waveform name."""
        item = self.waveform_list.currentItem()
        return item.text() if item else None
    
    def get_current_envelope(self) -> Optional[str]:
        """Get the currently selected envelope name."""
        item = self.envelope_list.currentItem()
        return item.text() if item else None


class WaveformPreviewLarge(QWidget):
    """Larger preview widget for waveforms."""
    
    def __init__(self, waveform: Waveform, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.waveform = waveform
    
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Zero line
        painter.setPen(QPen(QColor(60, 60, 70), 1))
        painter.drawLine(0, h // 2, w, h // 2)
        
        # Waveform
        path = QPainterPath()
        num_points = min(len(self.waveform.samples), w)
        step = len(self.waveform.samples) / num_points
        
        for i in range(num_points):
            idx = int(i * step)
            x = i * w / num_points
            y = h / 2 - self.waveform.samples[idx] * h / 2 * 0.8
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        painter.setPen(QPen(QColor(100, 200, 255), 2))
        painter.drawPath(path)
        
        painter.end()


class EnvelopePreviewLarge(QWidget):
    """Larger preview widget for envelopes."""
    
    def __init__(self, envelope: Envelope, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.envelope = envelope
    
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Envelope
        path = QPainterPath()
        num_points = min(len(self.envelope.samples), w)
        step = len(self.envelope.samples) / num_points
        
        for i in range(num_points):
            idx = int(i * step)
            x = i * w / num_points
            y = h - self.envelope.samples[idx] * h * 0.9
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        painter.setPen(QPen(QColor(100, 255, 150), 2))
        painter.drawPath(path)
        
        painter.end()

