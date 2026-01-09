"""
SSVEP Classifier module for BCI-UPIC.

Provides lightweight, real-time classification of SSVEP responses.
Determines whether the user is attending to the 15Hz (up) or 10Hz (down) target.

Methods implemented:
1. FFT-based power ratio (simple, fast)
2. Canonical Correlation Analysis (CCA) - gold standard for SSVEP
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from scipy import signal
from scipy.linalg import eig
from enum import Enum


class AttentionTarget(Enum):
    """Which target the user is attending to."""
    NONE = 0      # No clear attention detected
    UP = 1        # Attending to top target (15Hz) - move cursor up
    DOWN = 2      # Attending to bottom target (10Hz) - move cursor down


@dataclass
class ClassificationResult:
    """Result of SSVEP classification."""
    target: AttentionTarget
    confidence: float  # 0-1, how confident the classification is
    power_15hz: float  # Power at 15Hz
    power_10hz: float  # Power at 10Hz
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
    sample_rate: float = 256.0
    target_frequencies: Tuple[float, float] = (15.0, 10.0)  # (up, down)
    window_seconds: float = 1.0
    n_harmonics: int = 2  # Include fundamental + 1 harmonic
    threshold: float = 0.3  # Minimum confidence to make a decision
    occipital_channels: List[int] = field(default_factory=lambda: [6, 7])  # O1, O2
    
    # CCA reference signals (precomputed)
    _ref_signals_up: NDArray[np.float64] = field(init=False, repr=False)
    _ref_signals_down: NDArray[np.float64] = field(init=False, repr=False)
    
    # Smoothing for temporal stability
    _history: List[ClassificationResult] = field(default_factory=list, repr=False)
    _history_size: int = 5
    
    def __post_init__(self) -> None:
        """Initialize reference signals for CCA."""
        self._generate_reference_signals()
    
    def _generate_reference_signals(self) -> None:
        """
        Generate reference signals for CCA.
        
        Reference signals are sine and cosine waves at the target
        frequencies and their harmonics.
        """
        n_samples = int(self.window_seconds * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate
        
        # Reference signals for each target frequency
        for freq_idx, freq in enumerate(self.target_frequencies):
            ref_signals = []
            for h in range(1, self.n_harmonics + 1):
                # Sine and cosine at each harmonic
                ref_signals.append(np.sin(2 * np.pi * h * freq * t))
                ref_signals.append(np.cos(2 * np.pi * h * freq * t))
            
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
            power_15hz=power_up,
            power_10hz=power_down,
            raw_score=ratio
        )
    
    def classify_cca(self, eeg_data: NDArray[np.float64]) -> ClassificationResult:
        """
        Classify using Canonical Correlation Analysis (CCA).
        
        More robust than FFT, better for noisy data.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            ClassificationResult
        """
        # Use occipital channels
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
        
        # Calculate confidence and determine target
        max_corr = max(corr_up, corr_down)
        confidence = max_corr
        
        ratio = corr_up - corr_down
        
        if confidence < self.threshold:
            target = AttentionTarget.NONE
        elif corr_up > corr_down:
            target = AttentionTarget.UP
        else:
            target = AttentionTarget.DOWN
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_15hz=corr_up,
            power_10hz=corr_down,
            raw_score=ratio
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
        Apply temporal smoothing to reduce jitter.
        
        Uses majority voting over recent classifications.
        """
        if len(self._history) < 3:
            return current
        
        # Count votes for each target
        votes = {AttentionTarget.NONE: 0, AttentionTarget.UP: 0, AttentionTarget.DOWN: 0}
        total_confidence = 0
        
        for result in self._history:
            votes[result.target] += result.confidence
            total_confidence += result.confidence
        
        # Find winner
        winner = max(votes, key=votes.get)
        smoothed_confidence = votes[winner] / total_confidence if total_confidence > 0 else 0
        
        # Average the power values
        avg_15hz = np.mean([r.power_15hz for r in self._history])
        avg_10hz = np.mean([r.power_10hz for r in self._history])
        avg_score = np.mean([r.raw_score for r in self._history])
        
        return ClassificationResult(
            target=winner,
            confidence=smoothed_confidence,
            power_15hz=avg_15hz,
            power_10hz=avg_10hz,
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
