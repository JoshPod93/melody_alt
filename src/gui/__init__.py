# GUI components for the UPIC interface

from .main_window import MainWindow, run_app
from .page_canvas import PageCanvas
from .waveform_editor import WaveformEditorDialog
from .envelope_editor import EnvelopeEditorDialog
from .palette_panel import PalettePanel
from .arc_properties import ArcPropertiesPanel

__all__ = [
    'MainWindow', 'run_app',
    'PageCanvas',
    'WaveformEditorDialog',
    'EnvelopeEditorDialog',
    'PalettePanel',
    'ArcPropertiesPanel',
]
