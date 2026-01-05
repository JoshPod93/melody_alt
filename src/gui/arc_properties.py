"""
Arc Properties Panel for UPIC.

Shows and edits properties of the currently selected arc,
including FM modulation coupling.
"""

from __future__ import annotations

from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QGroupBox, QFrame, QPushButton,
    QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.arc import Arc
from ..core.page import Page
from ..core.waveform import WaveformLibrary
from ..core.envelope import EnvelopeLibrary


class ArcPropertiesPanel(QWidget):
    """
    Panel for viewing and editing arc properties.
    
    Shown when an arc is selected. Allows editing amplitude, pan,
    waveform, envelope, and FM modulation settings.
    """
    
    # Signals
    arc_changed = pyqtSignal(str)  # Arc ID that was modified
    modulator_link_requested = pyqtSignal()  # User wants to link a modulator
    
    def __init__(
        self,
        waveforms: WaveformLibrary,
        envelopes: EnvelopeLibrary,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        
        self.waveforms = waveforms
        self.envelopes = envelopes
        self.arc: Optional[Arc] = None
        self.page: Optional[Page] = None
        self._updating = False  # Prevent signal loops
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        self.header = QLabel("Arc Properties")
        self.header.setStyleSheet("font-size: 14px; font-weight: bold; color: #88aaff;")
        layout.addWidget(self.header)
        
        # Arc name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name_changed)
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Sound settings group
        sound_group = QGroupBox("Sound")
        sound_layout = QVBoxLayout(sound_group)
        
        # Waveform
        wf_layout = QHBoxLayout()
        wf_layout.addWidget(QLabel("Waveform:"))
        self.waveform_combo = QComboBox()
        self.waveform_combo.currentTextChanged.connect(self._on_waveform_changed)
        wf_layout.addWidget(self.waveform_combo, stretch=1)
        sound_layout.addLayout(wf_layout)
        
        # Envelope
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("Envelope:"))
        self.envelope_combo = QComboBox()
        self.envelope_combo.currentTextChanged.connect(self._on_envelope_changed)
        env_layout.addWidget(self.envelope_combo, stretch=1)
        sound_layout.addLayout(env_layout)
        
        # Amplitude
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude:"))
        self.amplitude_slider = QSlider(Qt.Orientation.Horizontal)
        self.amplitude_slider.setMinimum(0)
        self.amplitude_slider.setMaximum(100)
        self.amplitude_slider.valueChanged.connect(self._on_amplitude_changed)
        amp_layout.addWidget(self.amplitude_slider, stretch=1)
        self.amplitude_label = QLabel("50%")
        self.amplitude_label.setMinimumWidth(40)
        amp_layout.addWidget(self.amplitude_label)
        sound_layout.addLayout(amp_layout)
        
        # Pan
        pan_layout = QHBoxLayout()
        pan_layout.addWidget(QLabel("Pan:"))
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setMinimum(-100)
        self.pan_slider.setMaximum(100)
        self.pan_slider.valueChanged.connect(self._on_pan_changed)
        pan_layout.addWidget(self.pan_slider, stretch=1)
        self.pan_label = QLabel("C")
        self.pan_label.setMinimumWidth(40)
        pan_layout.addWidget(self.pan_label)
        sound_layout.addLayout(pan_layout)
        
        # Mute checkbox
        self.mute_check = QCheckBox("Mute (silent - use as modulator only)")
        self.mute_check.stateChanged.connect(self._on_mute_changed)
        sound_layout.addWidget(self.mute_check)
        
        layout.addWidget(sound_group)
        
        # FM Modulation group
        fm_group = QGroupBox("FM Modulation")
        fm_layout = QVBoxLayout(fm_group)
        
        # Modulator info
        self.modulator_label = QLabel("Modulator: None")
        self.modulator_label.setStyleSheet("color: #aaaaaa;")
        fm_layout.addWidget(self.modulator_label)
        
        # Link instructions
        self.link_hint = QLabel("Ctrl+Click another arc to link as modulator")
        self.link_hint.setStyleSheet("color: #888888; font-size: 11px; font-style: italic;")
        fm_layout.addWidget(self.link_hint)
        
        # Mod index
        mod_layout = QHBoxLayout()
        mod_layout.addWidget(QLabel("Mod Index:"))
        self.mod_index_slider = QSlider(Qt.Orientation.Horizontal)
        self.mod_index_slider.setMinimum(0)
        self.mod_index_slider.setMaximum(100)  # 0-10 scaled by 10
        self.mod_index_slider.valueChanged.connect(self._on_mod_index_changed)
        mod_layout.addWidget(self.mod_index_slider, stretch=1)
        self.mod_index_label = QLabel("0.0")
        self.mod_index_label.setMinimumWidth(40)
        mod_layout.addWidget(self.mod_index_label)
        fm_layout.addLayout(mod_layout)
        
        # Clear modulator button
        self.clear_mod_btn = QPushButton("Clear Modulator")
        self.clear_mod_btn.clicked.connect(self._on_clear_modulator)
        fm_layout.addWidget(self.clear_mod_btn)
        
        layout.addWidget(fm_group)
        
        # Carriers info (if this arc modulates others)
        self.carriers_group = QGroupBox("Modulating")
        carriers_layout = QVBoxLayout(self.carriers_group)
        self.carriers_label = QLabel("This arc modulates: None")
        self.carriers_label.setStyleSheet("color: #aaaaaa;")
        carriers_layout.addWidget(self.carriers_label)
        layout.addWidget(self.carriers_group)
        
        layout.addStretch()
        
        # Initially hidden
        self.setVisible(False)
    
    def _create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444444;")
        return line
    
    def set_arc(self, arc: Optional[Arc], page: Optional[Page] = None) -> None:
        """Set the arc to display/edit."""
        self.arc = arc
        self.page = page
        
        if arc is None:
            self.setVisible(False)
            return
        
        self._updating = True
        
        # Update all fields
        self.header.setText(f"Arc: {arc.name}")
        self.name_edit.setText(arc.name)
        
        # Refresh combo boxes
        self._refresh_combos()
        
        # Set current values
        self.waveform_combo.setCurrentText(arc.waveform_name)
        self.envelope_combo.setCurrentText(arc.envelope_name)
        self.amplitude_slider.setValue(int(arc.amplitude * 100))
        self.pan_slider.setValue(int(arc.pan * 100))
        self.mute_check.setChecked(arc.muted)
        self.mod_index_slider.setValue(int(arc.modulation_index * 10))
        
        # Update modulator info
        self._update_modulator_info()
        self._update_carriers_info()
        
        self._updating = False
        self.setVisible(True)
    
    def _refresh_combos(self) -> None:
        """Refresh waveform and envelope combo boxes."""
        # Block signals during refresh
        self.waveform_combo.blockSignals(True)
        self.envelope_combo.blockSignals(True)
        
        self.waveform_combo.clear()
        for name in self.waveforms.list_names():
            self.waveform_combo.addItem(name)
        
        self.envelope_combo.clear()
        for name in self.envelopes.list_names():
            self.envelope_combo.addItem(name)
        
        self.waveform_combo.blockSignals(False)
        self.envelope_combo.blockSignals(False)
    
    def _update_modulator_info(self) -> None:
        """Update the modulator display."""
        if self.arc is None or self.page is None:
            self.modulator_label.setText("Modulator: None")
            return
        
        if self.arc.modulator_id:
            mod_arc = self.page.get_arc(self.arc.modulator_id)
            if mod_arc:
                self.modulator_label.setText(f"Modulator: {mod_arc.name}")
                self.modulator_label.setStyleSheet("color: #ffaa00;")
            else:
                self.modulator_label.setText("Modulator: (deleted)")
                self.modulator_label.setStyleSheet("color: #ff6666;")
        else:
            self.modulator_label.setText("Modulator: None")
            self.modulator_label.setStyleSheet("color: #aaaaaa;")
    
    def _update_carriers_info(self) -> None:
        """Update info about what this arc modulates."""
        if self.arc is None or self.page is None:
            self.carriers_label.setText("This arc modulates: None")
            self.carriers_group.setVisible(False)
            return
        
        # Find arcs that use this arc as modulator
        carriers = []
        for other_arc in self.page.arcs.values():
            if other_arc.modulator_id == self.arc.id:
                carriers.append(other_arc.name)
        
        if carriers:
            self.carriers_label.setText(f"This arc modulates: {', '.join(carriers)}")
            self.carriers_label.setStyleSheet("color: #00aaff;")
            self.carriers_group.setVisible(True)
        else:
            self.carriers_label.setText("This arc modulates: None")
            self.carriers_group.setVisible(False)
    
    def link_modulator(self, modulator_arc: Arc) -> None:
        """Link a modulator arc to the current arc."""
        if self.arc is None or modulator_arc is None:
            return
        
        # Don't allow self-modulation
        if modulator_arc.id == self.arc.id:
            return
        
        self.arc.modulator_id = modulator_arc.id
        self._update_modulator_info()
        self.arc_changed.emit(self.arc.id)
    
    def _on_name_changed(self) -> None:
        """Handle name change."""
        if self._updating or self.arc is None:
            return
        self.arc.name = self.name_edit.text()
        self.header.setText(f"Arc: {self.arc.name}")
        self.arc_changed.emit(self.arc.id)
    
    def _on_waveform_changed(self, name: str) -> None:
        """Handle waveform change."""
        if self._updating or self.arc is None:
            return
        self.arc.waveform_name = name
        self.arc_changed.emit(self.arc.id)
    
    def _on_envelope_changed(self, name: str) -> None:
        """Handle envelope change."""
        if self._updating or self.arc is None:
            return
        self.arc.envelope_name = name
        self.arc_changed.emit(self.arc.id)
    
    def _on_amplitude_changed(self, value: int) -> None:
        """Handle amplitude change."""
        if self._updating or self.arc is None:
            return
        self.arc.amplitude = value / 100.0
        self.amplitude_label.setText(f"{value}%")
        self.arc_changed.emit(self.arc.id)
    
    def _on_pan_changed(self, value: int) -> None:
        """Handle pan change."""
        if self._updating or self.arc is None:
            return
        self.arc.pan = value / 100.0
        if value < -5:
            self.pan_label.setText(f"L{abs(value)}")
        elif value > 5:
            self.pan_label.setText(f"R{value}")
        else:
            self.pan_label.setText("C")
        self.arc_changed.emit(self.arc.id)
    
    def _on_mute_changed(self, state: int) -> None:
        """Handle mute change."""
        if self._updating or self.arc is None:
            return
        self.arc.muted = state == Qt.CheckState.Checked.value
        self.arc_changed.emit(self.arc.id)
    
    def _on_mod_index_changed(self, value: int) -> None:
        """Handle modulation index change."""
        if self._updating or self.arc is None:
            return
        self.arc.modulation_index = value / 10.0
        self.mod_index_label.setText(f"{value / 10:.1f}")
        self.arc_changed.emit(self.arc.id)
    
    def _on_clear_modulator(self) -> None:
        """Clear the modulator link."""
        if self.arc is None:
            return
        self.arc.modulator_id = None
        self._update_modulator_info()
        self.arc_changed.emit(self.arc.id)
    
    def refresh(self) -> None:
        """Refresh combo boxes (call after waveforms/envelopes change)."""
        if self.arc is not None:
            self._refresh_combos()
            self.waveform_combo.setCurrentText(self.arc.waveform_name)
            self.envelope_combo.setCurrentText(self.arc.envelope_name)

