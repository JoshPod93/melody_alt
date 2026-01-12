"""
SSVEP Calibration module for BCI-UPIC.

Captures real SSVEP responses from the user to refine CCA reference signals.
Uses standard CCA structure (sin/cos at fundamental + harmonics) with
template-informed phase/frequency parameters. This accounts for:
- Screen refresh rate/timing inaccuracies
- Individual neural response patterns
- Electrode placement variations
- Hardware-specific signal characteristics

Best Practice: References are always in standard sin/cos format (shape: n_samples, 2*n_harmonics)
for optimal CCA performance. Templates are used to extract refined phase/frequency parameters,
not as direct reference signals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json
import time


@dataclass
class CalibrationTrial:
    """A single calibration trial for one frequency."""
    frequency: float
    duration: float
    eeg_data: NDArray[np.float64]  # Shape: (n_samples, n_channels)
    timestamps: NDArray[np.float64]
    
    def get_occipital_data(self, channels: List[int] = [5, 6, 7]) -> NDArray[np.float64]:
        """Extract only occipital channels."""
        return self.eeg_data[:, channels]


@dataclass
class CalibrationData:
    """
    Complete calibration data for SSVEP BCI.
    
    Contains recorded SSVEP responses for each target frequency,
    which are used to create personalized CCA reference signals.
    """
    trials_15hz: List[CalibrationTrial] = field(default_factory=list)
    trials_12hz: List[CalibrationTrial] = field(default_factory=list)
    sample_rate: float = 250.0
    n_channels: int = 8
    occipital_channels: List[int] = field(default_factory=lambda: [5, 6, 7])
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Computed reference templates
    _template_15hz: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _template_12hz: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    
    def add_trial(self, frequency: float, eeg_data: NDArray[np.float64], 
                  timestamps: NDArray[np.float64], duration: float) -> None:
        """Add a calibration trial."""
        trial = CalibrationTrial(
            frequency=frequency,
            duration=duration,
            eeg_data=eeg_data,
            timestamps=timestamps
        )
        
        # Get target frequencies from screen calibration to determine which list to use
        try:
            from .screen_config import get_screen_calibration
            screen_cal = get_screen_calibration()
            higher_freq, lower_freq = screen_cal.frequencies
            
            # Use tolerance to match frequencies (accounting for calibration differences)
            if abs(frequency - higher_freq) < 0.5:  # Within 0.5 Hz
                self.trials_15hz.append(trial)
            elif abs(frequency - lower_freq) < 0.5:  # Within 0.5 Hz
                self.trials_12hz.append(trial)
            else:
                # Fallback: determine by magnitude if frequencies don't match
                if frequency > (higher_freq + lower_freq) / 2:
                    self.trials_15hz.append(trial)
                else:
                    self.trials_12hz.append(trial)
        except ImportError:
            # Fallback to hard-coded check if screen_config unavailable
            if abs(frequency - 15.0) < 0.5:
                self.trials_15hz.append(trial)
            elif abs(frequency - 12.0) < 0.5:
                self.trials_12hz.append(trial)
        
        # Invalidate templates
        self._template_15hz = None
        self._template_12hz = None
    
    def compute_templates(self, window_seconds: float = 0.5) -> None:
        """
        Compute averaged SSVEP templates from calibration trials.
        
        Uses epoch averaging to extract the consistent SSVEP response
        while canceling out random noise.
        """
        # Get target frequencies from screen calibration
        try:
            from .screen_config import get_screen_calibration
            screen_cal = get_screen_calibration()
            higher_freq, lower_freq = screen_cal.frequencies
        except ImportError:
            # Fallback to defaults
            higher_freq, lower_freq = 15.0, 12.0
        
        self._template_15hz = self._compute_template_for_frequency(
            self.trials_15hz, higher_freq, window_seconds
        )
        self._template_12hz = self._compute_template_for_frequency(
            self.trials_12hz, lower_freq, window_seconds
        )
    
    def _compute_template_for_frequency(
        self, 
        trials: List[CalibrationTrial],
        frequency: float,
        window_seconds: float
    ) -> Optional[NDArray[np.float64]]:
        """
        Compute averaged template for a specific frequency.
        
        Extracts multiple epochs from each trial and averages them
        to get a clean SSVEP template.
        """
        if not trials:
            return None
        
        window_samples = int(window_seconds * self.sample_rate)
        period_samples = int(self.sample_rate / frequency)
        
        all_epochs = []
        
        for trial in trials:
            occ_data = trial.get_occipital_data(self.occipital_channels)
            
            # Skip first 0.5s to allow SSVEP to stabilize
            start_sample = int(0.5 * self.sample_rate)
            
            # Extract multiple epochs aligned to frequency period
            n_epochs = (len(occ_data) - start_sample) // window_samples
            
            for i in range(n_epochs):
                epoch_start = start_sample + i * window_samples
                epoch_end = epoch_start + window_samples
                
                if epoch_end <= len(occ_data):
                    epoch = occ_data[epoch_start:epoch_end]
                    all_epochs.append(epoch)
        
        if not all_epochs:
            return None
        
        # Average all epochs
        template = np.mean(all_epochs, axis=0)
        
        # Normalize
        template = template - np.mean(template, axis=0)
        std = np.std(template, axis=0)
        std[std < 1e-6] = 1  # Avoid division by zero
        template = template / std
        
        return template
    
    def get_cca_references(self, window_seconds: float = 0.5, n_harmonics: int = 2) -> Tuple[NDArray, NDArray]:
        """
        Get CCA reference signals from calibration data.
        
        Uses standard CCA structure: sin/cos components at fundamental + harmonics.
        If templates are available, extracts phase/frequency from them to refine
        the synthetic references.
        
        Returns:
            Tuple of (ref_higher_freq, ref_lower_freq) arrays
            Each array has shape (n_samples, 2*n_harmonics) for standard CCA
        """
        if self._template_15hz is None or self._template_12hz is None:
            self.compute_templates(window_seconds)
        
        # Get target frequencies and phases from screen calibration
        try:
            from .screen_config import get_screen_calibration
            screen_cal = get_screen_calibration()
            higher_freq, lower_freq = screen_cal.frequencies
            higher_phase, lower_phase = screen_cal.phases
        except ImportError:
            # Fallback to defaults
            higher_freq, lower_freq = 15.0, 12.0
            higher_phase, lower_phase = 0.0, np.pi
        
        window_samples = int(window_seconds * self.sample_rate)
        
        # Generate references with template-informed parameters if available
        if self._template_15hz is not None:
            # Extract phase/frequency from template to refine synthetic reference
            refined_freq, refined_phase = self._extract_frequency_phase(
                self._template_15hz, higher_freq, higher_phase
            )
            ref_higher = self._generate_synthetic_reference(
                refined_freq, window_samples, n_harmonics, refined_phase
            )
        else:
            ref_higher = self._generate_synthetic_reference(
                higher_freq, window_samples, n_harmonics, higher_phase
            )
        
        if self._template_12hz is not None:
            # Extract phase/frequency from template to refine synthetic reference
            refined_freq, refined_phase = self._extract_frequency_phase(
                self._template_12hz, lower_freq, lower_phase
            )
            ref_lower = self._generate_synthetic_reference(
                refined_freq, window_samples, n_harmonics, refined_phase
            )
        else:
            ref_lower = self._generate_synthetic_reference(
                lower_freq, window_samples, n_harmonics, lower_phase
            )
        
        return ref_higher, ref_lower
    
    def _extract_frequency_phase(
        self,
        template: NDArray[np.float64],
        expected_freq: float,
        expected_phase: float
    ) -> Tuple[float, float]:
        """
        Extract frequency and phase from template using FFT.
        
        Args:
            template: Template array shape (n_samples, n_channels)
            expected_freq: Expected frequency (Hz)
            expected_phase: Expected phase (radians)
            
        Returns:
            Tuple of (refined_frequency, refined_phase)
        """
        # Average across channels to get single-channel template
        if template.ndim > 1:
            template_1d = np.mean(template, axis=1)
        else:
            template_1d = template
        
        # Remove DC component
        template_1d = template_1d - np.mean(template_1d)
        
        # FFT to find dominant frequency
        fft = np.fft.rfft(template_1d)
        freqs = np.fft.rfftfreq(len(template_1d), 1.0 / self.sample_rate)
        
        # Find peak near expected frequency
        freq_range = (expected_freq * 0.8, expected_freq * 1.2)
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        if np.any(mask):
            peak_idx = np.argmax(np.abs(fft[mask]))
            refined_freq = freqs[mask][peak_idx]
        else:
            refined_freq = expected_freq
        
        # Extract phase from FFT at the peak frequency
        peak_idx = np.argmin(np.abs(freqs - refined_freq))
        phase_complex = fft[peak_idx]
        refined_phase = np.angle(phase_complex)
        
        # Clamp frequency to reasonable range
        refined_freq = np.clip(refined_freq, expected_freq * 0.9, expected_freq * 1.1)
        
        return float(refined_freq), float(refined_phase)
    
    def _generate_synthetic_reference(
        self, 
        frequency: float, 
        n_samples: int,
        n_harmonics: int = 2,
        phase: Optional[float] = None
    ) -> NDArray[np.float64]:
        """
        Generate standard CCA reference signals: sin/cos at fundamental + harmonics.
        
        This is the standard CCA structure used in SSVEP classification.
        Shape: (n_samples, 2*n_harmonics)
        
        Args:
            frequency: Target frequency (Hz)
            n_samples: Number of samples
            n_harmonics: Number of harmonics to include
            phase: Phase offset (radians). If None, uses default based on frequency.
            
        Returns:
            Reference array shape (n_samples, 2*n_harmonics)
        """
        if phase is None:
            # Default phase: 0 for higher freq, π for lower freq
            phase = 0.0 if frequency >= 14.0 else np.pi
        
        t = np.arange(n_samples) / self.sample_rate
        
        refs = []
        for h in range(1, n_harmonics + 1):
            # Phase propagates to harmonics: h * phase
            refs.append(np.sin(2 * np.pi * h * frequency * t + h * phase))
            refs.append(np.cos(2 * np.pi * h * frequency * t + h * phase))
        
        return np.array(refs).T  # Shape: (n_samples, 2*n_harmonics)
    
    def get_statistics(self) -> Dict:
        """Get calibration statistics."""
        return {
            'n_trials_15hz': len(self.trials_15hz),
            'n_trials_12hz': len(self.trials_12hz),
            'total_samples_15hz': sum(len(t.eeg_data) for t in self.trials_15hz),
            'total_samples_12hz': sum(len(t.eeg_data) for t in self.trials_12hz),
            'has_template_15hz': self._template_15hz is not None,
            'has_template_12hz': self._template_12hz is not None,
            'sample_rate': self.sample_rate,
            'created_at': self.created_at
        }
    
    def save(self, filepath: Path | str) -> None:
        """Save calibration data to file."""
        filepath = Path(filepath)
        
        data = {
            'sample_rate': self.sample_rate,
            'n_channels': self.n_channels,
            'occipital_channels': self.occipital_channels,
            'created_at': self.created_at,
            'trials_15hz': [
                {
                    'frequency': t.frequency,
                    'duration': t.duration,
                    'eeg_data': t.eeg_data.tolist(),
                    'timestamps': t.timestamps.tolist()
                }
                for t in self.trials_15hz
            ],
            'trials_12hz': [
                {
                    'frequency': t.frequency,
                    'duration': t.duration,
                    'eeg_data': t.eeg_data.tolist(),
                    'timestamps': t.timestamps.tolist()
                }
                for t in self.trials_12hz
            ]
        }
        
        # Also save computed templates if available
        if self._template_15hz is not None:
            data['template_15hz'] = self._template_15hz.tolist()
        if self._template_12hz is not None:
            data['template_12hz'] = self._template_12hz.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Calibration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path | str) -> 'CalibrationData':
        """Load calibration data from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        cal = cls(
            sample_rate=data['sample_rate'],
            n_channels=data['n_channels'],
            occipital_channels=data['occipital_channels'],
            created_at=data.get('created_at', 'unknown')
        )
        
        # Load trials
        for t in data.get('trials_15hz', []):
            cal.trials_15hz.append(CalibrationTrial(
                frequency=t['frequency'],
                duration=t['duration'],
                eeg_data=np.array(t['eeg_data']),
                timestamps=np.array(t['timestamps'])
            ))
        
        # Support both old (trials_10hz) and new (trials_12hz) format for backward compatibility
        for t in data.get('trials_12hz', data.get('trials_10hz', [])):
            cal.trials_12hz.append(CalibrationTrial(
                frequency=t['frequency'],
                duration=t['duration'],
                eeg_data=np.array(t['eeg_data']),
                timestamps=np.array(t['timestamps'])
            ))
        
        # Load pre-computed templates if available
        if 'template_15hz' in data:
            cal._template_15hz = np.array(data['template_15hz'])
        # Support both old and new format
        if 'template_12hz' in data:
            cal._template_12hz = np.array(data['template_12hz'])
        elif 'template_10hz' in data:
            cal._template_12hz = np.array(data['template_10hz'])
        
        return cal


@dataclass
class CalibrationSession:
    """
    Manages a calibration session.
    
    Presents stimuli and records EEG responses for each frequency.
    """
    n_trials_per_frequency: int = 3
    trial_duration: float = 5.0  # seconds per trial
    rest_duration: float = 2.0   # rest between trials
    
    # Callbacks
    on_trial_start: Optional[callable] = None
    on_trial_end: Optional[callable] = None
    on_rest_start: Optional[callable] = None
    on_calibration_complete: Optional[callable] = None
    
    # State
    _calibration_data: CalibrationData = field(init=False)
    _current_trial: int = field(default=0, repr=False)
    _current_frequency: float = field(default=15.0, repr=False)
    _is_running: bool = field(default=False, repr=False)
    _trial_start_time: float = field(default=0.0, repr=False)
    _eeg_buffer: List[NDArray] = field(default_factory=list, repr=False)
    _timestamp_buffer: List[float] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        self._calibration_data = CalibrationData()
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def current_frequency(self) -> float:
        return self._current_frequency
    
    @property
    def current_trial(self) -> int:
        return self._current_trial
    
    @property
    def total_trials(self) -> int:
        return self.n_trials_per_frequency * 2  # 2 frequencies
    
    @property
    def progress(self) -> float:
        """Get overall progress (0-1)."""
        completed = len(self._calibration_data.trials_15hz) + len(self._calibration_data.trials_12hz)
        return completed / self.total_trials
    
    def get_trial_sequence(self) -> List[float]:
        """
        Get the sequence of frequencies for calibration.
        
        Uses actual screen calibration frequencies and alternates between them
        to reduce order effects.
        """
        # Get actual frequencies from screen calibration
        try:
            from .screen_config import get_screen_calibration
            screen_cal = get_screen_calibration()
            higher_freq, lower_freq = screen_cal.frequencies
        except ImportError:
            # Fallback to defaults if screen_config unavailable
            higher_freq, lower_freq = 15.0, 12.0
        
        sequence = []
        for i in range(self.n_trials_per_frequency):
            # Alternate: higher, lower, higher, lower, ...
            if i % 2 == 0:
                sequence.extend([higher_freq, lower_freq])
            else:
                sequence.extend([lower_freq, higher_freq])
        return sequence[:self.total_trials]
    
    def start(self, sample_rate: float = 250.0, n_channels: int = 8) -> None:
        """Start calibration session."""
        self._calibration_data = CalibrationData(
            sample_rate=sample_rate,
            n_channels=n_channels
        )
        self._current_trial = 0
        self._is_running = True
        self._eeg_buffer = []
        self._timestamp_buffer = []
        
        # Start first trial
        sequence = self.get_trial_sequence()
        if sequence:
            self._current_frequency = sequence[0]
    
    def start_trial(self) -> None:
        """Start recording for current trial."""
        self._trial_start_time = time.perf_counter()
        self._eeg_buffer = []
        self._timestamp_buffer = []
        
        if self.on_trial_start:
            self.on_trial_start(self._current_frequency, self._current_trial)
    
    def add_sample(self, sample: NDArray[np.float64], timestamp: float) -> None:
        """Add an EEG sample during trial."""
        self._eeg_buffer.append(sample)
        self._timestamp_buffer.append(timestamp)
    
    def end_trial(self) -> None:
        """End current trial and save data."""
        if not self._eeg_buffer:
            return
        
        eeg_data = np.array(self._eeg_buffer)
        timestamps = np.array(self._timestamp_buffer)
        
        self._calibration_data.add_trial(
            frequency=self._current_frequency,
            eeg_data=eeg_data,
            timestamps=timestamps,
            duration=self.trial_duration
        )
        
        if self.on_trial_end:
            self.on_trial_end(self._current_frequency, self._current_trial)
        
        # Move to next trial
        self._current_trial += 1
        sequence = self.get_trial_sequence()
        
        if self._current_trial < len(sequence):
            self._current_frequency = sequence[self._current_trial]
        else:
            self._finish_calibration()
    
    def _finish_calibration(self) -> None:
        """Finish calibration and compute templates."""
        self._is_running = False
        
        # Compute templates from recorded data
        self._calibration_data.compute_templates()
        
        if self.on_calibration_complete:
            self.on_calibration_complete(self._calibration_data)
    
    def get_calibration_data(self) -> CalibrationData:
        """Get the calibration data."""
        return self._calibration_data
    
    def cancel(self) -> None:
        """Cancel calibration."""
        self._is_running = False


if __name__ == "__main__":
    # Test calibration data structures
    print("Testing calibration system...")
    
    # Create fake calibration data
    cal = CalibrationData(sample_rate=250, n_channels=8)
    
    # Simulate some trials
    for freq in [15.0, 15.0, 12.0, 12.0]:
        n_samples = int(5 * 250)  # 5 seconds
        t = np.arange(n_samples) / 250
        
        # Fake EEG with SSVEP
        eeg = np.random.randn(n_samples, 8) * 0.5
        ssvep = np.sin(2 * np.pi * freq * t)
        eeg[:, 5] += ssvep  # PO7
        eeg[:, 6] += ssvep * 1.2  # Oz (strongest)
        eeg[:, 7] += ssvep  # PO8
        
        cal.add_trial(freq, eeg, t, 5.0)
    
    # Compute templates
    cal.compute_templates()
    
    # Get references
    ref_higher, ref_lower = cal.get_cca_references()
    
    print(f"Stats: {cal.get_statistics()}")
    print(f"Higher freq template shape: {ref_higher.shape}")
    print(f"Lower freq template shape: {ref_lower.shape}")
    
    # Test save/load
    cal.save("test_calibration.json")
    cal_loaded = CalibrationData.load("test_calibration.json")
    print(f"Loaded stats: {cal_loaded.get_statistics()}")
    
    # Cleanup
    import os
    os.remove("test_calibration.json")
    print("Test complete!")
