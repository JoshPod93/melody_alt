"""
BCI (Brain-Computer Interface) module for UPIC.

This module provides SSVEP-based control for the graphical synthesizer,
replacing mouse input with brain signal decoding.

Components:
- stimulus: SSVEP flickering targets at precise frequencies
- preprocessing: EEG signal filtering and artifact rejection
- classifier: Real-time SSVEP frequency detection
- controller: Cursor control based on BCI output
- interface: Main BCI composition interface
- lsl_stream: LSL integration for g.tec Unicorn Black
"""

from .stimulus import SSVEPStimulus, FlickerTarget
from .preprocessing import EEGPreprocessor, LSLPreprocessor, SimulatedEEGSource
from .classifier import SSVEPClassifier, AttentionTarget, ClassificationResult
from .controller import BCICursorController, ControllerState
from .score import BCIScore
from .calibration import CalibrationData, CalibrationSession
from .screen_config import ScreenCalibration, get_screen_calibration, reload_screen_calibration

# LSL components (optional)
try:
    from .lsl_stream import LSLReceiver, LSLMarkerSender, UnicornInterface, LSL_AVAILABLE
except ImportError:
    LSL_AVAILABLE = False
    LSLReceiver = None
    LSLMarkerSender = None
    UnicornInterface = None

# Direct Unicorn streaming (optional)
try:
    from .unicorn_streamer import UnicornLSLStreamer, UNICORN_AVAILABLE
except ImportError:
    UNICORN_AVAILABLE = False
    UnicornLSLStreamer = None

__all__ = [
    # Stimulus
    'SSVEPStimulus',
    'FlickerTarget',
    # Preprocessing
    'EEGPreprocessor',
    'LSLPreprocessor',
    'SimulatedEEGSource',
    # Classification
    'SSVEPClassifier',
    'AttentionTarget',
    'ClassificationResult',
    # Control
    'BCICursorController',
    'ControllerState',
    # Score
    'BCIScore',
    # Calibration
    'CalibrationData',
    'CalibrationSession',
    # Screen Configuration
    'ScreenCalibration',
    'get_screen_calibration',
    'reload_screen_calibration',
    # LSL
    'LSL_AVAILABLE',
    'LSLReceiver',
    'LSLMarkerSender',
    'UnicornInterface',
    # Unicorn Direct
    'UNICORN_AVAILABLE',
    'UnicornLSLStreamer',
]
