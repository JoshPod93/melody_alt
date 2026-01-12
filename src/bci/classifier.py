"""
SSVEP Classifier module for BCI-UPIC.

        Provides lightweight, real-time classification of SSVEP responses.
Determines whether the user is attending to the higher frequency (up) or lower frequency (down) target.

Methods implemented:
1. FFT-based power ratio (simple, fast) - includes harmonics
2. Canonical Correlation Analysis (CCA) - gold standard for SSVEP - includes harmonics

Harmonics: Both methods use n_harmonics=2 (fundamental + 1st harmonic) to improve detection.
"""

from __future__ import annotations

import json
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, TYPE_CHECKING
from pathlib import Path
from scipy import signal
from scipy.linalg import eig
from enum import Enum

if TYPE_CHECKING:
    from .calibration import CalibrationData


class AttentionTarget(Enum):
    """Which target the user is attending to."""
    NONE = 0      # No clear attention detected
    UP = 1        # Attending to higher frequency target (top) - move cursor up
    DOWN = 2      # Attending to lower frequency target (bottom) - move cursor down


@dataclass
class ClassificationResult:
    """Result of SSVEP classification."""
    target: AttentionTarget
    confidence: float  # 0-1, how confident the classification is
    power_higher_freq: float  # Power at higher frequency target (including harmonics)
    power_lower_freq: float  # Power at lower frequency target (including harmonics)
    raw_score: float   # Raw classification score (positive = up, negative = down)


@dataclass
class SSVEPClassifier:
    """
    Real-time SSVEP classifier for BCI control.
    
    Classifies EEG data to determine which flickering target
    the user is attending to.
    
    Attributes:
        sample_rate: EEG sampling rate in Hz
        target_frequencies: Tuple of (up_freq, down_freq) in Hz
        window_seconds: Analysis window size in seconds
        n_harmonics: Number of harmonics to include in analysis
        threshold: Confidence threshold for making a decision
        occipital_channels: Indices of occipital channels (best for SSVEP)
    """
    sample_rate: float = 250.0  # Unicorn Black sample rate
    target_frequencies: Tuple[float, float] = (15.0, 12.0)  # (higher_freq, lower_freq) - will be overridden by screen calibration if available
    target_phases: Tuple[float, float] = (0.0, np.pi)  # Phase offsets: higher freq at 0°, lower freq at 180°
    window_seconds: float = 0.5  # Shorter window for faster response
    n_harmonics: int = 2  # Include fundamental + 1 harmonic
    threshold: float = 0.05  # Very low threshold - almost always move
    occipital_channels: List[int] = field(default_factory=lambda: [5, 6, 7])  # PO7, Oz, PO8
    _screen_calibration: Optional[Dict] = field(default=None, repr=False)  # Screen calibration data
    
    # CCA reference signals - can be from calibration or synthetic
    _ref_signals_up: NDArray[np.float64] = field(init=False, repr=False)
    _ref_signals_down: NDArray[np.float64] = field(init=False, repr=False)
    _using_calibration: bool = field(default=False, repr=False)
    
    # Smoothing for temporal stability
    _history: List[ClassificationResult] = field(default_factory=list, repr=False)
    _history_size: int = 3  # Reduced for faster response
    
    def __post_init__(self) -> None:
        """Initialize reference signals for CCA."""
        self._load_screen_calibration()  # Load screen calibration if available
        self._generate_reference_signals()
    
    def _load_screen_calibration(self) -> None:
        """Load screen calibration data using the centralized configuration."""
        from .screen_config import get_screen_calibration
        
        screen_cal = get_screen_calibration()
        
        # Use actual measured frequencies from calibration
        self.target_frequencies = screen_cal.frequencies
        self.target_phases = screen_cal.phases
        
        # Store calibration data for reference (using generic field names)
        self._screen_calibration = {
            'refresh_rate_hz': screen_cal.refresh_rate_hz,
            'actual_higher_freq': screen_cal.actual_higher_freq,
            'actual_lower_freq': screen_cal.actual_lower_freq,
            'calibrated_at': screen_cal.calibrated_at
        }
        
        if screen_cal.is_calibrated:
            print(f"[SCREEN CONFIG] Using calibrated frequencies: "
                  f"{self.target_frequencies[0]:.3f}Hz, {self.target_frequencies[1]:.3f}Hz")
        else:
            print(f"[SCREEN CONFIG] Using default frequencies: "
                  f"{self.target_frequencies[0]:.1f}Hz, {self.target_frequencies[1]:.1f}Hz")
    
    def load_calibration(self, calibration_data: 'CalibrationData') -> bool:
        """
        Load personalized reference signals from calibration data.
        
        This replaces synthetic references with real SSVEP templates
        captured from the user's brain responses.
        
        Args:
            calibration_data: CalibrationData with recorded SSVEP responses
            
        Returns:
            True if calibration was loaded successfully
        """
        try:
            ref_up, ref_down = calibration_data.get_cca_references(self.window_seconds)
            
            if ref_up is not None and ref_down is not None:
                # Ensure correct shape
                n_samples = int(self.window_seconds * self.sample_rate)
                
                # Resize if needed
                if len(ref_up) != n_samples:
                    ref_up = self._resize_reference(ref_up, n_samples)
                if len(ref_down) != n_samples:
                    ref_down = self._resize_reference(ref_down, n_samples)
                
                self._ref_signals_up = ref_up
                self._ref_signals_down = ref_down
                self._using_calibration = True
                
                print(f"Loaded calibration: UP={ref_up.shape}, DOWN={ref_down.shape}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def _resize_reference(self, ref: NDArray, target_samples: int) -> NDArray:
        """Resize reference signal to match target sample count."""
        if len(ref) == target_samples:
            return ref
        
        # Use interpolation to resize
        old_indices = np.linspace(0, 1, len(ref))
        new_indices = np.linspace(0, 1, target_samples)
        
        if ref.ndim == 1:
            return np.interp(new_indices, old_indices, ref)
        else:
            # Multi-channel
            result = np.zeros((target_samples, ref.shape[1]))
            for ch in range(ref.shape[1]):
                result[:, ch] = np.interp(new_indices, old_indices, ref[:, ch])
            return result
    
    @property
    def is_calibrated(self) -> bool:
        """Check if using calibration data."""
        return self._using_calibration
    
    def _generate_reference_signals(self) -> None:
        """
        Generate reference signals for CCA that MATCH the visual stimulus.
        
        The reference signals must have the same frequency AND phase as
        the flickering targets to maximize correlation with the SSVEP response.
        
        Uses actual measured frequencies from screen calibration if available,
        otherwise uses target frequencies.
        
        Visual stimulus uses: sin(2π * f * t + phase)
        - Higher frequency (up): phase = 0
        - Lower frequency (down): phase = π (180°)
        """
        n_samples = int(self.window_seconds * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate
        
        # Reference signals for each target frequency WITH MATCHING PHASE
        for freq_idx, (freq, phase) in enumerate(zip(self.target_frequencies, self.target_phases)):
            ref_signals = []
            for h in range(1, self.n_harmonics + 1):
                # Sine and cosine at each harmonic WITH PHASE OFFSET
                # The phase offset propagates to harmonics as h * phase
                ref_signals.append(np.sin(2 * np.pi * h * freq * t + h * phase))
                ref_signals.append(np.cos(2 * np.pi * h * freq * t + h * phase))
            
            ref_array = np.array(ref_signals).T  # Shape: (n_samples, 2*n_harmonics)
            
            if freq_idx == 0:
                self._ref_signals_up = ref_array
            else:
                self._ref_signals_down = ref_array
    
    def classify_fft(self, eeg_data: NDArray[np.float64]) -> ClassificationResult:
        """
        Classify using FFT-based power ratio method.
        
        Simple and fast, good for real-time applications.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            ClassificationResult
        """
        # Use occipital channels only
        if eeg_data.shape[1] > max(self.occipital_channels):
            data = eeg_data[:, self.occipital_channels]
        else:
            data = eeg_data
        
        # Average across selected channels
        data_avg = np.mean(data, axis=1)
        
        # Compute FFT
        n_samples = len(data_avg)
        freqs = np.fft.rfftfreq(n_samples, 1/self.sample_rate)
        fft_vals = np.abs(np.fft.rfft(data_avg))
        
        # Get power at target frequencies (with small bandwidth)
        bandwidth = 1.0  # Hz
        
        def get_band_power(target_freq: float) -> float:
            idx = np.where((freqs >= target_freq - bandwidth) & 
                          (freqs <= target_freq + bandwidth))[0]
            if len(idx) == 0:
                return 0.0
            return np.sum(fft_vals[idx] ** 2)
        
        # Calculate power at each target frequency (including harmonics)
        power_up = sum(get_band_power(self.target_frequencies[0] * h) 
                       for h in range(1, self.n_harmonics + 1))
        power_down = sum(get_band_power(self.target_frequencies[1] * h) 
                         for h in range(1, self.n_harmonics + 1))
        
        # Calculate ratio and confidence
        total_power = power_up + power_down + 1e-10
        ratio = (power_up - power_down) / total_power
        
        # Map ratio to confidence (0-1)
        confidence = min(abs(ratio) * 2, 1.0)
        
        # Determine target
        if confidence < self.threshold:
            target = AttentionTarget.NONE
        elif ratio > 0:
            target = AttentionTarget.UP
        else:
            target = AttentionTarget.DOWN
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_higher_freq=power_up,
            power_lower_freq=power_down,
            raw_score=ratio
        )
    
    def classify_cca(self, eeg_data: NDArray[np.float64]) -> ClassificationResult:
        """
        Classify using Canonical Correlation Analysis (CCA).
        
        More robust than FFT, better for noisy data.
        Uses ONLY occipital channels (PO7, Oz, PO8) for SSVEP detection.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            ClassificationResult
        """
        # CRITICAL: Use ONLY occipital channels for SSVEP
        if eeg_data.shape[1] > max(self.occipital_channels):
            data = eeg_data[:, self.occipital_channels]
        else:
            data = eeg_data
        
        # Ensure data length matches reference signals
        n_ref = self._ref_signals_up.shape[0]
        if data.shape[0] > n_ref:
            data = data[-n_ref:]
        elif data.shape[0] < n_ref:
            # Pad with zeros (not ideal, but handles edge case)
            pad_size = n_ref - data.shape[0]
            data = np.vstack([np.zeros((pad_size, data.shape[1])), data])
        
        # Compute CCA correlation for each target
        corr_up = self._cca_correlation(data, self._ref_signals_up)
        corr_down = self._cca_correlation(data, self._ref_signals_down)
        
        # Also compute FFT-based power for comparison
        fft_result = self.classify_fft(eeg_data)
        
        # Combine CCA and FFT evidence
        # CCA correlation difference
        cca_diff = corr_up - corr_down
        
        # FFT power difference (normalized)
        fft_total = fft_result.power_higher_freq + fft_result.power_lower_freq + 1e-10
        fft_diff = (fft_result.power_higher_freq - fft_result.power_lower_freq) / fft_total
        
        # Combined score: weight CCA more if calibrated, otherwise equal
        if self._using_calibration:
            combined_score = 0.7 * np.sign(cca_diff) * min(abs(cca_diff) * 3, 1) + 0.3 * fft_diff
        else:
            combined_score = 0.5 * np.sign(cca_diff) * min(abs(cca_diff) * 3, 1) + 0.5 * fft_diff
        
        # Confidence based on agreement and magnitude
        confidence = min(abs(combined_score) * 2, 1.0)
        confidence = max(confidence, 0.2)  # Minimum confidence
        
        # Direction based on combined score
        if combined_score >= 0:
            target = AttentionTarget.UP
        else:
            target = AttentionTarget.DOWN
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_higher_freq=corr_up,
            power_lower_freq=corr_down,
            raw_score=combined_score
        )
    
    def _cca_correlation(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64]
    ) -> float:
        """
        Compute canonical correlation between EEG data and reference signals.
        
        Args:
            X: EEG data (n_samples, n_channels)
            Y: Reference signals (n_samples, n_refs)
            
        Returns:
            Maximum canonical correlation
        """
        # Center the data
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
        
        # Compute covariance matrices
        n = X.shape[0]
        Cxx = X.T @ X / n
        Cyy = Y.T @ Y / n
        Cxy = X.T @ Y / n
        
        # Regularization for numerical stability
        reg = 1e-6
        Cxx += reg * np.eye(Cxx.shape[0])
        Cyy += reg * np.eye(Cyy.shape[0])
        
        try:
            # Solve generalized eigenvalue problem
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
            
            M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
            eigenvalues, _ = eig(M)
            
            # Return maximum correlation (square root of max eigenvalue)
            max_eigenvalue = np.max(np.real(eigenvalues))
            return np.sqrt(max(0, max_eigenvalue))
        except np.linalg.LinAlgError:
            return 0.0
    
    def classify(
        self,
        eeg_data: NDArray[np.float64],
        method: str = "fft"
    ) -> ClassificationResult:
        """
        Classify EEG data using specified method.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            method: "fft" or "cca"
            
        Returns:
            ClassificationResult with temporal smoothing applied
        """
        if method == "cca":
            result = self.classify_cca(eeg_data)
        else:
            result = self.classify_fft(eeg_data)
        
        # Add to history for smoothing
        self._history.append(result)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        
        # Apply temporal smoothing
        return self._smooth_result(result)
    
    def _smooth_result(self, current: ClassificationResult) -> ClassificationResult:
        """
        Apply minimal smoothing - mostly just pass through current result.
        
        For SSVEP we want fast response, not heavy smoothing.
        """
        # Just return current result with slight confidence boost from history
        if len(self._history) < 2:
            return current
        
        # Light smoothing of raw_score only
        avg_score = np.mean([r.raw_score for r in self._history])
        
        # If history agrees with current, boost confidence
        same_direction = sum(1 for r in self._history if r.target == current.target)
        agreement_ratio = same_direction / len(self._history)
        
        boosted_confidence = current.confidence * (0.7 + 0.3 * agreement_ratio)
        
        return ClassificationResult(
            target=current.target,
            confidence=min(boosted_confidence, 1.0),
            power_higher_freq=current.power_higher_freq,
            power_lower_freq=current.power_lower_freq,
            raw_score=avg_score
        )
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._history.clear()
    
    def get_cursor_velocity(self, result: ClassificationResult) -> float:
        """
        Convert classification result to cursor velocity.
        
        Args:
            result: Classification result
            
        Returns:
            Velocity value: positive = up, negative = down, 0 = no movement
        """
        if result.target == AttentionTarget.NONE:
            return 0.0
        
        # Scale velocity by confidence
        base_velocity = 1.0
        velocity = base_velocity * result.confidence
        
        if result.target == AttentionTarget.DOWN:
            velocity = -velocity
        
        return velocity


if __name__ == "__main__":
    # Test classifier with simulated data
    print("Testing SSVEP classifier...")
    
    # Create classifier
    classifier = SSVEPClassifier(sample_rate=256)
    
    # Generate test data with 15Hz SSVEP
    n_samples = 256  # 1 second
    t = np.arange(n_samples) / 256
    
    # Simulated SSVEP at 15Hz in occipital channels
    eeg_data = np.random.randn(n_samples, 8) * 0.5
    ssvep_signal = np.sin(2 * np.pi * 15 * t)
    eeg_data[:, 6] += ssvep_signal  # O1
    eeg_data[:, 7] += ssvep_signal  # O2
    
    # Classify
    result_fft = classifier.classify(eeg_data, method="fft")
    print(f"FFT Result: {result_fft.target.name}, confidence: {result_fft.confidence:.2f}")
    
    result_cca = classifier.classify(eeg_data, method="cca")
    print(f"CCA Result: {result_cca.target.name}, confidence: {result_cca.confidence:.2f}")
