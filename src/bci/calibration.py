"""
SSVEP Calibration module for BCI-UPIC.

Captures real SSVEP responses from the user to create personalized
CCA reference signals. This accounts for:
- Screen refresh rate/timing inaccuracies
- Individual neural response patterns
- Electrode placement variations
- Hardware-specific signal characteristics
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
    trials_10hz: List[CalibrationTrial] = field(default_factory=list)
    sample_rate: float = 250.0
    n_channels: int = 8
    occipital_channels: List[int] = field(default_factory=lambda: [5, 6, 7])
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Computed reference templates
    _template_15hz: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _template_10hz: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    
    def add_trial(self, frequency: float, eeg_data: NDArray[np.float64], 
                  timestamps: NDArray[np.float64], duration: float) -> None:
        """Add a calibration trial."""
        trial = CalibrationTrial(
            frequency=frequency,
            duration=duration,
            eeg_data=eeg_data,
            timestamps=timestamps
        )
        
        if frequency == 15.0:
            self.trials_15hz.append(trial)
        elif frequency == 10.0:
            self.trials_10hz.append(trial)
        
        # Invalidate templates
        self._template_15hz = None
        self._template_10hz = None
    
    def compute_templates(self, window_seconds: float = 0.5) -> None:
        """
        Compute averaged SSVEP templates from calibration trials.
        
        Uses epoch averaging to extract the consistent SSVEP response
        while canceling out random noise.
        """
        self._template_15hz = self._compute_template_for_frequency(
            self.trials_15hz, 15.0, window_seconds
        )
        self._template_10hz = self._compute_template_for_frequency(
            self.trials_10hz, 10.0, window_seconds
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
    
    def get_cca_references(self, window_seconds: float = 0.5) -> Tuple[NDArray, NDArray]:
        """
        Get CCA reference signals from calibration data.
        
        Returns templates that can be used directly in CCA classification.
        
        Returns:
            Tuple of (ref_15hz, ref_10hz) arrays
        """
        if self._template_15hz is None or self._template_10hz is None:
            self.compute_templates(window_seconds)
        
        # If we have real templates, use them
        # Otherwise fall back to synthetic
        window_samples = int(window_seconds * self.sample_rate)
        
        if self._template_15hz is not None:
            ref_15hz = self._template_15hz
        else:
            ref_15hz = self._generate_synthetic_reference(15.0, window_samples)
        
        if self._template_10hz is not None:
            ref_10hz = self._template_10hz
        else:
            ref_10hz = self._generate_synthetic_reference(10.0, window_samples)
        
        return ref_15hz, ref_10hz
    
    def _generate_synthetic_reference(
        self, 
        frequency: float, 
        n_samples: int,
        n_harmonics: int = 2
    ) -> NDArray[np.float64]:
        """Generate synthetic reference if no calibration data available."""
        t = np.arange(n_samples) / self.sample_rate
        
        refs = []
        phase = 0.0 if frequency == 15.0 else np.pi
        
        for h in range(1, n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * frequency * t + h * phase))
            refs.append(np.cos(2 * np.pi * h * frequency * t + h * phase))
        
        return np.array(refs).T
    
    def get_statistics(self) -> Dict:
        """Get calibration statistics."""
        return {
            'n_trials_15hz': len(self.trials_15hz),
            'n_trials_10hz': len(self.trials_10hz),
            'total_samples_15hz': sum(len(t.eeg_data) for t in self.trials_15hz),
            'total_samples_10hz': sum(len(t.eeg_data) for t in self.trials_10hz),
            'has_template_15hz': self._template_15hz is not None,
            'has_template_10hz': self._template_10hz is not None,
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
            'trials_10hz': [
                {
                    'frequency': t.frequency,
                    'duration': t.duration,
                    'eeg_data': t.eeg_data.tolist(),
                    'timestamps': t.timestamps.tolist()
                }
                for t in self.trials_10hz
            ]
        }
        
        # Also save computed templates if available
        if self._template_15hz is not None:
            data['template_15hz'] = self._template_15hz.tolist()
        if self._template_10hz is not None:
            data['template_10hz'] = self._template_10hz.tolist()
        
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
        
        for t in data.get('trials_10hz', []):
            cal.trials_10hz.append(CalibrationTrial(
                frequency=t['frequency'],
                duration=t['duration'],
                eeg_data=np.array(t['eeg_data']),
                timestamps=np.array(t['timestamps'])
            ))
        
        # Load pre-computed templates if available
        if 'template_15hz' in data:
            cal._template_15hz = np.array(data['template_15hz'])
        if 'template_10hz' in data:
            cal._template_10hz = np.array(data['template_10hz'])
        
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
        completed = len(self._calibration_data.trials_15hz) + len(self._calibration_data.trials_10hz)
        return completed / self.total_trials
    
    def get_trial_sequence(self) -> List[float]:
        """
        Get the sequence of frequencies for calibration.
        
        Alternates between frequencies to reduce order effects.
        """
        sequence = []
        for i in range(self.n_trials_per_frequency):
            # Alternate: 15, 10, 15, 10, ...
            if i % 2 == 0:
                sequence.extend([15.0, 10.0])
            else:
                sequence.extend([10.0, 15.0])
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
    for freq in [15.0, 15.0, 10.0, 10.0]:
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
    ref_15, ref_10 = cal.get_cca_references()
    
    print(f"Stats: {cal.get_statistics()}")
    print(f"15Hz template shape: {ref_15.shape}")
    print(f"10Hz template shape: {ref_10.shape}")
    
    # Test save/load
    cal.save("test_calibration.json")
    cal_loaded = CalibrationData.load("test_calibration.json")
    print(f"Loaded stats: {cal_loaded.get_statistics()}")
    
    # Cleanup
    import os
    os.remove("test_calibration.json")
    print("Test complete!")
