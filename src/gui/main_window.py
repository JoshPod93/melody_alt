"""
Main Window for UPIC.

The main application window containing all GUI components:
- Page canvas (drawing area)
- Waveform/envelope palette
- Transport controls
- Tool selection
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QStatusBar, QDockWidget, QFileDialog, QMessageBox,
    QSlider, QLabel, QPushButton, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from ..core.page import Page, Project
from ..core.synthesizer import Synthesizer
from ..core.waveform import WaveformLibrary
from ..core.envelope import EnvelopeLibrary
from ..core.frequency_table import FrequencyTableLibrary

from .page_canvas import PageCanvas
from .palette_panel import PalettePanel
from .waveform_editor import WaveformEditorDialog
from .envelope_editor import EnvelopeEditorDialog


class TransportBar(QWidget):
    """Transport controls for playback."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Play button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.setMinimumWidth(80)
        layout.addWidget(self.play_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setMinimumWidth(80)
        layout.addWidget(self.stop_btn)
        
        layout.addSpacing(20)
        
        # Clear All button
        self.clear_btn = QPushButton("🗑 Clear All")
        self.clear_btn.setMinimumWidth(90)
        layout.addWidget(self.clear_btn)
        
        layout.addSpacing(20)
        
        # Time display
        self.time_label = QLabel("0:00.000")
        self.time_label.setMinimumWidth(80)
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        layout.addWidget(self.time_label)
        
        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(60000)  # Milliseconds
        layout.addWidget(self.time_slider, stretch=1)
        
        # Duration label
        self.duration_label = QLabel("/ 1:00.000")
        self.duration_label.setMinimumWidth(80)
        self.duration_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        layout.addWidget(self.duration_label)
        
        layout.addSpacing(20)
        
        # Loop controls
        self.loop_btn = QPushButton("🔁 Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setMinimumWidth(70)
        self.loop_btn.setToolTip("Enable looping playback")
        layout.addWidget(self.loop_btn)
        
        layout.addStretch()
        
        # Volume
        layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setMaximumWidth(100)
        layout.addWidget(self.volume_slider)
    
    def set_time(self, time_seconds: float) -> None:
        """Update time display."""
        mins = int(time_seconds // 60)
        secs = time_seconds % 60
        self.time_label.setText(f"{mins}:{secs:06.3f}")
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(time_seconds * 1000))
        self.time_slider.blockSignals(False)
    
    def set_duration(self, duration_seconds: float) -> None:
        """Update duration display."""
        mins = int(duration_seconds // 60)
        secs = duration_seconds % 60
        self.duration_label.setText(f"/ {mins}:{secs:06.3f}")
        self.time_slider.setMaximum(int(duration_seconds * 1000))


class ToolBar(QToolBar):
    """Tool selection toolbar."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Tools", parent)
        self._init_tools()
    
    def _init_tools(self) -> None:
        # Selection tool
        self.select_action = QAction("Select", self)
        self.select_action.setCheckable(True)
        self.select_action.setChecked(False)
        self.select_action.setShortcut(QKeySequence("V"))
        self.select_action.setToolTip("Select and move arcs (V)")
        self.addAction(self.select_action)
        
        # Draw tool
        self.draw_action = QAction("Draw", self)
        self.draw_action.setCheckable(True)
        self.draw_action.setChecked(True)  # Start in draw mode
        self.draw_action.setShortcut(QKeySequence("D"))
        self.draw_action.setToolTip("Draw new arcs (D)")
        self.addAction(self.draw_action)
        
        # Line tool
        self.line_action = QAction("Line", self)
        self.line_action.setCheckable(True)
        self.line_action.setShortcut(QKeySequence("L"))
        self.line_action.setToolTip("Draw straight lines (L)")
        self.addAction(self.line_action)
        
        # Erase tool
        self.erase_action = QAction("Erase", self)
        self.erase_action.setCheckable(True)
        self.erase_action.setShortcut(QKeySequence("E"))
        self.erase_action.setToolTip("Erase arcs (E)")
        self.addAction(self.erase_action)
        
        self.addSeparator()
        
        # Pan tool
        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setShortcut(QKeySequence("H"))
        self.pan_action.setToolTip("Pan view (H)")
        self.addAction(self.pan_action)
        
        # Zoom tool
        self.zoom_action = QAction("Zoom", self)
        self.zoom_action.setCheckable(True)
        self.zoom_action.setShortcut(QKeySequence("Z"))
        self.zoom_action.setToolTip("Zoom view (Z)")
        self.addAction(self.zoom_action)
        
        # Make tools mutually exclusive
        self._tools = [
            self.select_action, self.draw_action, self.line_action,
            self.erase_action, self.pan_action, self.zoom_action
        ]
        for tool in self._tools:
            tool.triggered.connect(lambda checked, t=tool: self._on_tool_selected(t))
    
    def _on_tool_selected(self, selected_tool: QAction) -> None:
        """Handle tool selection (make mutually exclusive)."""
        for tool in self._tools:
            if tool != selected_tool:
                tool.setChecked(False)
        selected_tool.setChecked(True)
    
    def get_current_tool(self) -> str:
        """Get the name of the currently selected tool."""
        for tool in self._tools:
            if tool.isChecked():
                return tool.text().lower()
        return "select"


class MainWindow(QMainWindow):
    """Main UPIC application window."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize data
        self.project = Project()
        self.synthesizer = Synthesizer()
        
        # Share libraries between project and synthesizer
        self.synthesizer.waveforms = self.project.waveforms
        self.synthesizer.envelopes = self.project.envelopes
        self.synthesizer.frequency_tables = self.project.frequency_tables
        
        # Set up UI
        self._init_ui()
        self._init_menu()
        self._connect_signals()
        
        # Set initial page
        self._update_page()
        
        # Update timer for playback position
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_playback_position)
        self.update_timer.start(50)  # 20 Hz update rate
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("UPIC - Python Clone")
        self.setMinimumSize(1200, 800)
        
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Transport bar at top
        self.transport = TransportBar()
        main_layout.addWidget(self.transport)
        
        # Splitter for canvas and palette
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)
        
        # Page canvas (main drawing area)
        self.canvas = PageCanvas()
        splitter.addWidget(self.canvas)
        
        # Palette panel (waveforms, envelopes)
        self.palette = PalettePanel(
            self.project.waveforms,
            self.project.envelopes
        )
        splitter.addWidget(self.palette)
        
        # Set splitter sizes (80% canvas, 20% palette)
        splitter.setSizes([800, 200])
        
        # Toolbar
        self.toolbar = ToolBar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolbar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _init_menu(self) -> None:
        """Initialize menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Audio...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_audio)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        delete_action = QAction("&Delete Selected", self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self._delete_selected)
        edit_menu.addAction(delete_action)
        
        clear_action = QAction("&Clear All", self)
        clear_action.setShortcut(QKeySequence("Ctrl+Shift+Delete"))
        clear_action.triggered.connect(self._clear_all)
        edit_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(lambda: self.canvas.zoom(1.2))
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(lambda: self.canvas.zoom(0.8))
        view_menu.addAction(zoom_out_action)
        
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_action)
        
        view_menu.addSeparator()
        
        grid_action = QAction("Show &Grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(True)
        grid_action.triggered.connect(self.canvas.set_show_grid)
        view_menu.addAction(grid_action)
        
        # Sound menu
        sound_menu = menubar.addMenu("&Sound")
        
        waveform_action = QAction("Edit &Waveforms...", self)
        waveform_action.triggered.connect(self._edit_waveforms)
        sound_menu.addAction(waveform_action)
        
        envelope_action = QAction("Edit &Envelopes...", self)
        envelope_action.triggered.connect(self._edit_envelopes)
        sound_menu.addAction(envelope_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About UPIC", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self) -> None:
        """Connect widget signals."""
        # Transport
        self.transport.play_btn.clicked.connect(self._toggle_playback)
        self.transport.stop_btn.clicked.connect(self._stop_playback)
        self.transport.clear_btn.clicked.connect(self._clear_all)
        self.transport.loop_btn.clicked.connect(self._toggle_loop)
        self.transport.time_slider.valueChanged.connect(self._seek)
        self.transport.volume_slider.valueChanged.connect(self._set_volume)
        
        # Toolbar
        self.toolbar.select_action.triggered.connect(
            lambda: self.canvas.set_tool("select"))
        self.toolbar.draw_action.triggered.connect(
            lambda: self.canvas.set_tool("draw"))
        self.toolbar.line_action.triggered.connect(
            lambda: self.canvas.set_tool("line"))
        self.toolbar.erase_action.triggered.connect(
            lambda: self.canvas.set_tool("erase"))
        self.toolbar.pan_action.triggered.connect(
            lambda: self.canvas.set_tool("pan"))
        self.toolbar.zoom_action.triggered.connect(
            lambda: self.canvas.set_tool("zoom"))
        
        # Palette
        self.palette.waveform_selected.connect(self.canvas.set_current_waveform)
        self.palette.envelope_selected.connect(self.canvas.set_current_envelope)
        
        # Canvas
        self.canvas.arc_created.connect(self._on_arc_created)
        self.canvas.selection_changed.connect(self._on_selection_changed)
    
    def _update_page(self) -> None:
        """Update canvas with current page."""
        page = self.project.get_active_page()
        if page:
            self.canvas.set_page(page)
            self.synthesizer.set_page(page)
            self.transport.set_duration(page.settings.duration)
    
    def _toggle_playback(self) -> None:
        """Toggle play/pause."""
        if self.synthesizer.playing:
            self.synthesizer.pause()
            self.transport.play_btn.setText("▶ Play")
            self.transport.play_btn.setChecked(False)
        else:
            # Always start fresh playback from current position
            print(f"Starting playback from {self.synthesizer.current_time:.2f}s")
            self.synthesizer.play(self.synthesizer.current_time)
            self.transport.play_btn.setText("⏸ Pause")
            self.transport.play_btn.setChecked(True)
    
    def _stop_playback(self) -> None:
        """Stop playback and reset to beginning."""
        self.synthesizer.stop()
        self.synthesizer.current_time = 0.0
        self.transport.play_btn.setText("▶ Play")
        self.transport.play_btn.setChecked(False)
        self.transport.set_time(0.0)
        self.canvas.set_playhead_position(0.0)
    
    def _seek(self, value: int) -> None:
        """Seek to position (value in milliseconds)."""
        time_seconds = value / 1000.0
        self.synthesizer.seek(time_seconds)
        self.transport.set_time(time_seconds)
        self.canvas.set_playhead_position(time_seconds)
    
    def _set_volume(self, value: int) -> None:
        """Set master volume."""
        self.synthesizer.master_volume = value / 100.0
    
    def _toggle_loop(self) -> None:
        """Toggle loop playback."""
        page = self.project.get_active_page()
        if page:
            page.settings.loop_enabled = self.transport.loop_btn.isChecked()
            # Set loop to cover all arcs (or full duration if no arcs)
            if page.total_duration > 0:
                page.settings.loop_start = 0.0
                page.settings.loop_end = page.total_duration
            else:
                page.settings.loop_end = page.settings.duration
            
            status = "enabled" if page.settings.loop_enabled else "disabled"
            self.status_bar.showMessage(f"Loop {status}")
    
    def _update_playback_position(self) -> None:
        """Update UI with current playback position."""
        if self.synthesizer.playing:
            self.transport.set_time(self.synthesizer.current_time)
            self.canvas.set_playhead_position(self.synthesizer.current_time)
    
    def _new_project(self) -> None:
        """Create a new project."""
        reply = QMessageBox.question(
            self, "New Project",
            "Create a new project? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.project = Project()
            self.synthesizer.waveforms = self.project.waveforms
            self.synthesizer.envelopes = self.project.envelopes
            self.synthesizer.frequency_tables = self.project.frequency_tables
            self._update_page()
            self.palette.refresh()
    
    def _open_project(self) -> None:
        """Open a project file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            "", "UPIC Project (*.upic);;All Files (*)"
        )
        if filepath:
            try:
                self.project = Project.load(filepath)
                self.synthesizer.waveforms = self.project.waveforms
                self.synthesizer.envelopes = self.project.envelopes
                self.synthesizer.frequency_tables = self.project.frequency_tables
                self._update_page()
                self.palette.refresh()
                self.status_bar.showMessage(f"Opened: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project: {e}")
    
    def _save_project(self) -> None:
        """Save the current project."""
        # For now, always use Save As
        self._save_project_as()
    
    def _save_project_as(self) -> None:
        """Save project with a new filename."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            f"{self.project.name}.upic",
            "UPIC Project (*.upic);;All Files (*)"
        )
        if filepath:
            try:
                self.project.save(filepath)
                self.status_bar.showMessage(f"Saved: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
    
    def _export_audio(self) -> None:
        """Export audio to WAV file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Audio",
            "output.wav",
            "WAV Audio (*.wav);;All Files (*)"
        )
        if filepath:
            try:
                self.synthesizer.render_to_file(filepath)
                self.status_bar.showMessage(f"Exported: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export audio: {e}")
    
    def _undo(self) -> None:
        """Undo last action."""
        page = self.project.get_active_page()
        if page and page.undo():
            self.canvas.update()
    
    def _redo(self) -> None:
        """Redo last undone action."""
        page = self.project.get_active_page()
        if page and page.redo():
            self.canvas.update()
    
    def _delete_selected(self) -> None:
        """Delete selected arcs."""
        self.canvas.delete_selected()
    
    def _clear_all(self) -> None:
        """Clear all arcs from the page immediately."""
        page = self.project.get_active_page()
        if page:
            page.clear()
            self.canvas.selected_arc_ids.clear()
            self.canvas.update()
            self.status_bar.showMessage("Cleared all arcs")
    
    def _edit_waveforms(self) -> None:
        """Open waveform editor dialog."""
        dialog = WaveformEditorDialog(self.project.waveforms, self)
        dialog.exec()
        self.palette.refresh()
    
    def _edit_envelopes(self) -> None:
        """Open envelope editor dialog."""
        dialog = EnvelopeEditorDialog(self.project.envelopes, self)
        dialog.exec()
        self.palette.refresh()
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self, "About UPIC",
            "<h2>UPIC - Python Clone</h2>"
            "<p>A Python implementation of the UPIC system, "
            "originally created by Iannis Xenakis at CEMAMu.</p>"
            "<p>UPIC allows composers to draw waveforms and musical "
            "compositions graphically, converting visual representations "
            "into sound.</p>"
            "<p>Version 0.1.0</p>"
        )
    
    def _on_arc_created(self, arc_id: str) -> None:
        """Handle arc creation."""
        self.status_bar.showMessage(f"Created arc: {arc_id}")
    
    def _on_selection_changed(self, selected_ids: list) -> None:
        """Handle selection change."""
        count = len(selected_ids)
        if count == 0:
            self.status_bar.showMessage("No selection")
        elif count == 1:
            self.status_bar.showMessage(f"Selected 1 arc")
        else:
            self.status_bar.showMessage(f"Selected {count} arcs")
    
    def closeEvent(self, event) -> None:
        """Handle window close."""
        self.synthesizer.stop()
        event.accept()


def run_app() -> None:
    """Run the UPIC application."""
    app = QApplication(sys.argv)
    app.setApplicationName("UPIC")
    app.setApplicationVersion("0.1.0")
    
    # Set dark theme
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

