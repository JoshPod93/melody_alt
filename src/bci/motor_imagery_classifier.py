"""
Motor Imagery Classifier module for BCI-UPIC.

Provides real-time classification of motor imagery (left vs right hand) using
Common Spatial Patterns (CSP) for feature extraction.

Motor Imagery paradigm:
- Left hand imagery -> UP movement
- Right hand imagery -> DOWN movement
- Uses sensorimotor channels from Unicorn system
- Frequency bands: mu (8-13 Hz) and beta (13-30 Hz)
- CSP extracts spatial patterns that maximize variance difference between classes
- LDA classifier on CSP features for final decision

Unicorn Hybrid Black channel mapping (0-indexed):
- Index 0: Fz (Frontal midline)
- Index 1: C3 (Left motor cortex) <- Primary for left hand imagery
- Index 2: Cz (Central midline) <- Reference/ground
- Index 3: C4 (Right motor cortex) <- Primary for right hand imagery
- Index 4: Pz (Parietal midline)
- Index 5: PO7 (Left parieto-occipital)
- Index 6: Oz (Occipital midline)
- Index 7: PO8 (Right parieto-occipital)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, TYPE_CHECKING
from enum import Enum
from scipy import signal
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from .calibration import CalibrationData

# Unicorn Hybrid Black sensorimotor channel mapping
# These are the indices in the 8-channel EEG array from Unicorn
# Channel layout: Fz=0, C3=1, Cz=2, C4=3, Pz=4, PO7=5, Oz=6, PO8=7
UNICORN_C3_INDEX = 1  # Left motor cortex (left hand imagery)
UNICORN_CZ_INDEX = 2  # Central midline (reference)
UNICORN_C4_INDEX = 3  # Right motor cortex (right hand imagery)

# Channel name mapping for Unicorn
UNICORN_CHANNEL_NAMES = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

# Default sensorimotor channels for motor imagery (C3, Cz, C4)
DEFAULT_SENSORIMOTOR_CHANNELS = [UNICORN_C3_INDEX, UNICORN_CZ_INDEX, UNICORN_C4_INDEX]


def get_channel_names(channel_indices: List[int]) -> List[str]:
    """
    Get channel names for given indices (Unicorn system).
    
    Args:
        channel_indices: List of channel indices (0-7)
        
    Returns:
        List of channel names
    """
    return [UNICORN_CHANNEL_NAMES[idx] if 0 <= idx < len(UNICORN_CHANNEL_NAMES) else f"Ch{idx}" 
            for idx in channel_indices]


class AttentionTarget(Enum):
    """Which target the user is attending to."""
    NONE = 0      # No clear attention detected
    UP = 1        # Left hand imagery - move cursor up
    DOWN = 2      # Right hand imagery - move cursor down


@dataclass
class ClassificationResult:
    """Result of motor imagery classification."""
    target: AttentionTarget
    confidence: float  # 0-1, how confident the classification is
    left_score: float  # Score for left hand imagery
    right_score: float  # Score for right hand imagery
    raw_score: float   # Raw classification score (positive = left/UP, negative = right/DOWN)


@dataclass
class MotorImageryClassifier:
    """
    Real-time motor imagery classifier for BCI control.
    
    Classifies EEG data to determine left vs right hand motor imagery
    using Common Spatial Patterns (CSP) and LDA.
    
    Attributes:
        sample_rate: EEG sampling rate in Hz
        window_seconds: Analysis window size in seconds
        mu_band: Mu rhythm frequency band (8-13 Hz) - default for motor imagery
        beta_band: Beta rhythm frequency band (13-30 Hz) - default for motor imagery
        n_csp_components: Number of CSP components to use (typically 2-6)
        sensorimotor_channels: Indices of sensorimotor channels (default: C3=1, Cz=2, C4=3 for Unicorn)
        threshold: Confidence threshold for making a decision
    """
    sample_rate: float = 250.0  # Unicorn Black sample rate
    window_seconds: float = 1.0  # Optimal window size for motor imagery (1 second = 250 samples at 250Hz)
    # Note: Literature (MI-PLVGAT, BCI Competition IV-2a) shows 1s windows perform best
    # 2s windows reduce responsiveness and add unnecessary delay
    mu_band: Tuple[float, float] = (8.0, 13.0)  # Mu rhythm band
    beta_band: Tuple[float, float] = (13.0, 30.0)  # Beta rhythm band
    n_csp_components: int = 4  # Use 4 CSP components (2 per class)
    sensorimotor_channels: List[int] = field(default_factory=lambda: DEFAULT_SENSORIMOTOR_CHANNELS.copy())  # C3, Cz, C4 (Unicorn indices: 1, 2, 3)
    threshold: float = 0.3  # Minimum confidence threshold
    baseline_duration: float = 10.0  # Baseline capture duration in seconds (configurable)
    
    # CSP filters (learned from calibration or default)
    _csp_filters: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _csp_using_calibration: bool = field(default=False, repr=False)
    
    # LDA classifier
    _lda: Optional[LinearDiscriminantAnalysis] = field(default=None, repr=False)
    _scaler: Optional[StandardScaler] = field(default=None, repr=False)
    
    # Smoothing for temporal stability
    _history: List[ClassificationResult] = field(default_factory=list, repr=False)
    _history_size: int = 5
    
    # Default CSP filters (identity if no calibration)
    _default_csp_filters: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    
    # Baseline statistics (from 10-second baseline capture)
    _baseline_data: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _baseline_mu_power: Optional[float] = field(default=None, repr=False)
    _baseline_beta_power: Optional[float] = field(default=None, repr=False)
    _baseline_mu_mean: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _baseline_beta_mean: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _baseline_mu_std: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _baseline_beta_std: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _has_baseline: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Initialize default CSP filters."""
        # Validate sensorimotor channels are within valid range (0-7 for Unicorn)
        for ch_idx in self.sensorimotor_channels:
            if ch_idx < 0 or ch_idx >= 8:
                raise ValueError(
                    f"Invalid sensorimotor channel index {ch_idx}. "
                    f"Unicorn has 8 EEG channels (indices 0-7): "
                    f"Fz=0, C3=1, Cz=2, C4=3, Pz=4, PO7=5, Oz=6, PO8=7"
                )
        
        # Initialize with identity filters (no spatial filtering) until calibration
        n_channels = len(self.sensorimotor_channels)
        self._default_csp_filters = np.eye(n_channels)
        self._csp_filters = self._default_csp_filters.copy()
        
        # Initialize LDA and scaler
        self._lda = LinearDiscriminantAnalysis()
        self._scaler = StandardScaler()
        
        # Log channel mapping
        used_channels = get_channel_names(self.sensorimotor_channels)
        print(f"[MI CLASSIFIER] Using sensorimotor channels: {used_channels} (indices: {self.sensorimotor_channels})")
        print(f"[MI CLASSIFIER] Channel mapping: C3 (index {UNICORN_C3_INDEX}) = Left hand, C4 (index {UNICORN_C4_INDEX}) = Right hand")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if classifier is calibrated."""
        return self._csp_using_calibration and self._csp_filters is not None
    
    @property
    def has_baseline(self) -> bool:
        """Check if baseline data has been captured."""
        return self._has_baseline
    
    def _bandpass_filter(
        self,
        data: NDArray[np.float64],
        low_freq: float,
        high_freq: float
    ) -> NDArray[np.float64]:
        """
        Apply bandpass filter to data.
        
        Args:
            data: EEG data (n_samples, n_channels)
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)
            
        Returns:
            Filtered data
        """
        nyquist = self.sample_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter along time axis (axis 0)
        filtered = signal.filtfilt(b, a, data, axis=0)
        
        return filtered
    
    def _compute_csp(
        self,
        left_data: NDArray[np.float64],
        right_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute Common Spatial Patterns (CSP) filters.
        
        CSP finds spatial filters that maximize variance for one class
        while minimizing it for the other.
        
        Args:
            left_data: Left hand imagery data (n_trials, n_channels, n_samples) or (n_samples, n_channels)
            right_data: Right hand imagery data (n_trials, n_channels, n_samples) or (n_samples, n_channels)
            
        Returns:
            CSP filters (n_channels, n_csp_components)
        """
        # Handle different input shapes
        # CSP expects (n_trials, n_channels, n_samples) or (n_samples, n_channels) for single trial
        if left_data.ndim == 2:
            # Single trial: (n_samples, n_channels) -> reshape to (1, n_channels, n_samples)
            left_data = left_data.T[np.newaxis, :, :]  # (1, n_channels, n_samples)
        elif left_data.ndim == 3 and left_data.shape[1] != left_data.shape[2]:
            # If shape is (n_trials, n_samples, n_channels), transpose to (n_trials, n_channels, n_samples)
            if left_data.shape[2] < left_data.shape[1]:  # Likely (n_trials, n_samples, n_channels)
                left_data = np.transpose(left_data, (0, 2, 1))  # (n_trials, n_channels, n_samples)
        
        if right_data.ndim == 2:
            right_data = right_data.T[np.newaxis, :, :]
        elif right_data.ndim == 3 and right_data.shape[1] != right_data.shape[2]:
            if right_data.shape[2] < right_data.shape[1]:
                right_data = np.transpose(right_data, (0, 2, 1))
        
        # Compute normalized covariance matrices for each trial
        def compute_covariance(trial_data: NDArray) -> NDArray:
            """Compute normalized covariance matrix for a trial."""
            # trial_data shape: (n_samples, n_channels)
            cov = np.cov(trial_data.T)
            # Normalize by trace
            trace = np.trace(cov)
            if trace > 0:
                cov = cov / trace
            return cov
        
        # Compute average covariance for each class
        left_covs = np.array([compute_covariance(trial) for trial in left_data])
        right_covs = np.array([compute_covariance(trial) for trial in right_data])
        
        left_cov_avg = np.mean(left_covs, axis=0)
        right_cov_avg = np.mean(right_covs, axis=0)
        
        # Solve generalized eigenvalue problem: left_cov * w = lambda * right_cov * w
        # This finds filters that maximize variance for left while minimizing for right
        eigenvalues, eigenvectors = eigh(left_cov_avg, left_cov_avg + right_cov_avg)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        
        # Select CSP components (first and last n_csp_components/2 each)
        n_select = self.n_csp_components // 2
        selected_indices = list(range(n_select)) + list(range(len(eigenvalues) - n_select, len(eigenvalues)))
        csp_filters = eigenvectors[:, selected_indices]
        
        return csp_filters
    
    def _extract_csp_features(
        self,
        data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Extract CSP features from data.
        
        Args:
            data: EEG data (n_samples, n_channels)
            
        Returns:
            CSP features (n_features,) - log variance of CSP-filtered signals
        """
        # Apply CSP filters
        filtered = data @ self._csp_filters  # (n_samples, n_csp_components)
        
        # Compute log variance of each CSP component
        variances = np.var(filtered, axis=0)
        # Add small epsilon to avoid log(0)
        variances = variances + 1e-10
        features = np.log(variances)
        
        return features
    
    def classify(
        self,
        eeg_data: NDArray[np.float64],
        method: str = "default"
    ) -> ClassificationResult:
        """
        Classify motor imagery from EEG data.
        
        Args:
            eeg_data: EEG data array (n_samples, n_channels)
            method: Classification method (ignored, always uses CSP+LDA)
            
        Returns:
            ClassificationResult
        """
        # DEBUG: Log input data shape
        if not hasattr(self, '_classify_call_count'):
            self._classify_call_count = 0
        self._classify_call_count += 1
        
        if self._classify_call_count % 20 == 1:  # Log every 20th call
            print(f"[MI CLASSIFY DEBUG] Call #{self._classify_call_count}: Input shape={eeg_data.shape}, "
                  f"Has baseline={self._has_baseline}, Is calibrated={self.is_calibrated}")
        
        if eeg_data.shape[0] < 10:
            # Not enough data
            if self._classify_call_count % 20 == 1:
                print(f"[MI CLASSIFY DEBUG] Insufficient data: {eeg_data.shape[0]} samples (need >= 10)")
            return ClassificationResult(
                target=AttentionTarget.NONE,
                confidence=0.0,
                left_score=0.0,
                right_score=0.0,
                raw_score=0.0
            )
        
        # Extract sensorimotor channels
        sensorimotor_data = eeg_data[:, self.sensorimotor_channels]
        
        if self._classify_call_count % 20 == 1:
            print(f"[MI CLASSIFY DEBUG] Sensorimotor data shape={sensorimotor_data.shape}, "
                  f"Channels={self.sensorimotor_channels}")
        
        # Apply bandpass filters (mu and beta bands)
        mu_data = self._bandpass_filter(sensorimotor_data, self.mu_band[0], self.mu_band[1])
        beta_data = self._bandpass_filter(sensorimotor_data, self.beta_band[0], self.beta_band[1])
        
        # Normalize using baseline if available
        if self._has_baseline:
            # Z-score normalization: (x - mean) / std
            if self._classify_call_count % 20 == 1:
                print(f"[MI CLASSIFY DEBUG] Applying baseline normalization")
                print(f"  Baseline mu_mean shape={self._baseline_mu_mean.shape}, "
                      f"mu_std shape={self._baseline_mu_std.shape}")
                print(f"  Mu data before norm: mean={np.mean(mu_data):.4f}, std={np.std(mu_data):.4f}")
            
            mu_data = (mu_data - self._baseline_mu_mean) / self._baseline_mu_std
            beta_data = (beta_data - self._baseline_beta_mean) / self._baseline_beta_std
            
            if self._classify_call_count % 20 == 1:
                print(f"  Mu data after norm: mean={np.mean(mu_data):.4f}, std={np.std(mu_data):.4f}")
        else:
            if self._classify_call_count % 20 == 1:
                print(f"[MI CLASSIFY DEBUG] No baseline - skipping normalization")
        
        # Extract CSP features from both bands
        mu_features = self._extract_csp_features(mu_data)
        beta_features = self._extract_csp_features(beta_data)
        
        # Combine features
        features = np.concatenate([mu_features, beta_features])
        features = features.reshape(1, -1)  # (1, n_features)
        
        # Scale features
        if self._scaler is not None and hasattr(self._scaler, 'mean_'):
            features_scaled = self._scaler.transform(features)
        else:
            features_scaled = features
        
        # Classify using LDA
        if self._lda is not None and hasattr(self._lda, 'classes_'):
            # Trained LDA
            prediction = self._lda.predict(features_scaled)[0]
            decision_score = self._lda.decision_function(features_scaled)[0]
            
            # Get probabilities
            try:
                probabilities = self._lda.predict_proba(features_scaled)[0]
                left_prob = probabilities[0] if len(probabilities) > 0 else 0.5
                right_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            except:
                # Fallback if predict_proba fails
                left_prob = 0.5 + decision_score * 0.1
                right_prob = 0.5 - decision_score * 0.1
                left_prob = np.clip(left_prob, 0.0, 1.0)
                right_prob = np.clip(right_prob, 0.0, 1.0)
        else:
            # No trained model - use power asymmetry between C3 and C4 channels
            # This is a direct motor imagery indicator: left hand imagery -> C3 desynchronization
            # Right hand imagery -> C4 desynchronization
            
            # mu_data and beta_data are already filtered and available
            # Shape: (n_samples, n_channels) where channels are [C3, Cz, C4] (indices 0, 1, 2)
            c3_idx = 0  # C3 is first in sensorimotor_channels
            c4_idx = 2  # C4 is third in sensorimotor_channels (Cz is in between)
            
            if mu_data.shape[1] > c4_idx:
                # Compute power (variance) for C3 and C4 in mu band
                c3_mu_power = np.var(mu_data[:, c3_idx])
                c4_mu_power = np.var(mu_data[:, c4_idx])
                
                # Also check beta band
                c3_beta_power = np.var(beta_data[:, c3_idx])
                c4_beta_power = np.var(beta_data[:, c4_idx])
                
                if self._classify_call_count % 20 == 1:
                    print(f"[MI CLASSIFY DEBUG] Power analysis (untrained heuristic):")
                    print(f"  C3 mu power={c3_mu_power:.6f}, C4 mu power={c4_mu_power:.6f}")
                    print(f"  C3 beta power={c3_beta_power:.6f}, C4 beta power={c4_beta_power:.6f}")
                
                # Combined power difference (mu + beta)
                # When imagining left hand: C3 power decreases (desynchronization)
                # When imagining right hand: C4 power decreases
                # So: C3 < C4 means left imagery, C3 > C4 means right imagery
                mu_power_diff = c3_mu_power - c4_mu_power
                beta_power_diff = c3_beta_power - c4_beta_power
                combined_diff = mu_power_diff + 0.5 * beta_power_diff
                
                # Normalize by total power to get relative difference
                total_power = c3_mu_power + c4_mu_power + c3_beta_power + c4_beta_power
                if total_power > 1e-10:
                    normalized_diff = combined_diff / total_power
                else:
                    normalized_diff = 0.0
                
                if self._classify_call_count % 20 == 1:
                    print(f"  Power diff: mu={mu_power_diff:.6f}, beta={beta_power_diff:.6f}, "
                          f"combined={combined_diff:.6f}, normalized={normalized_diff:.6f}")
                
                # Scale to decision score
                # Positive = C3 > C4 (right hand imagery, C4 desynchronizes) -> DOWN
                # Negative = C3 < C4 (left hand imagery, C3 desynchronizes) -> UP
                decision_score = normalized_diff * 10.0  # Amplify for sensitivity
                
                # Convert to probabilities
                scaled_score = np.clip(decision_score, -3.0, 3.0)  # Clip to avoid extreme values
                # Negative score = left (UP), positive = right (DOWN)
                left_prob = 0.5 - np.tanh(scaled_score) * 0.4  # Range: 0.1 to 0.9
                right_prob = 1.0 - left_prob
                
                if self._classify_call_count % 20 == 1:
                    print(f"  Decision score={decision_score:.4f}, scaled={scaled_score:.4f}, "
                          f"left_prob={left_prob:.3f}, right_prob={right_prob:.3f}")
            else:
                # Fallback to CSP features if channel structure is unexpected
                if len(mu_features) >= 2:
                    decision_score = mu_features[0] - mu_features[-1]
                else:
                    decision_score = 0.0
                
                if abs(decision_score) > 0.01:
                    scaled_score = decision_score * 10.0
                    left_prob = 0.5 + np.tanh(scaled_score) * 0.4
                    right_prob = 1.0 - left_prob
                else:
                    left_prob = 0.5
                    right_prob = 0.5
            
            # prediction 0 = left hand = UP, 1 = right hand = DOWN
            prediction = 0 if decision_score < 0 else 1
        
        # Map prediction to AttentionTarget
        # prediction 0 = left hand = UP
        # prediction 1 = right hand = DOWN
        if prediction == 0:
            target = AttentionTarget.UP
            confidence = left_prob
        else:
            target = AttentionTarget.DOWN
            confidence = right_prob
        
        # Ensure minimum confidence threshold
        # Lower threshold for untrained classifier to allow predictions
        effective_threshold = self.threshold if self._lda is not None and hasattr(self._lda, 'classes_') else 0.1
        
        # For untrained classifier, be more permissive - allow predictions even with lower confidence
        # but scale confidence to ensure some movement
        if self._lda is None or not hasattr(self._lda, 'classes_'):
            # Untrained: boost low confidences to ensure predictions are made
            if confidence < effective_threshold and confidence > 0.05:  # If there's any signal
                confidence = effective_threshold  # Boost to threshold to allow prediction
            elif confidence < effective_threshold:
                target = AttentionTarget.NONE
                confidence = 0.0
        else:
            # Trained: use normal threshold
            if confidence < effective_threshold:
                target = AttentionTarget.NONE
                confidence = 0.0
        
        result = ClassificationResult(
            target=target,
            confidence=confidence,
            left_score=left_prob,
            right_score=right_prob,
            raw_score=decision_score
        )
        
        # Add to history for smoothing
        self._history.append(result)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        
        # Apply temporal smoothing
        if len(self._history) >= 2:
            # Weighted average of recent results
            weights = np.linspace(0.5, 1.0, len(self._history))
            weights = weights / np.sum(weights)
            
            # Average raw scores
            avg_raw_score = np.average([r.raw_score for r in self._history], weights=weights)
            
            # Average confidence
            avg_confidence = np.average([r.confidence for r in self._history], weights=weights)
            
            # Re-determine target based on smoothed score
            # Lower threshold for untrained classifier
            effective_threshold = self.threshold if self._lda is not None and hasattr(self._lda, 'classes_') else 0.1
            if avg_confidence < effective_threshold:
                smoothed_target = AttentionTarget.NONE
            elif avg_raw_score < 0:
                # Negative score = C3 < C4 = left hand imagery = UP
                smoothed_target = AttentionTarget.UP
            else:
                # Positive score = C3 > C4 = right hand imagery = DOWN
                smoothed_target = AttentionTarget.DOWN
            
            # Average left/right scores
            avg_left = np.average([r.left_score for r in self._history], weights=weights)
            avg_right = np.average([r.right_score for r in self._history], weights=weights)
            
            result = ClassificationResult(
                target=smoothed_target,
                confidence=min(avg_confidence, 1.0),
                left_score=avg_left,
                right_score=avg_right,
                raw_score=avg_raw_score
            )
        
        return result
    
    def capture_baseline(self, eeg_data: NDArray[np.float64]) -> bool:
        """
        Capture baseline data for normalization (10 seconds of rest).
        
        This computes baseline statistics from rest data to normalize
        motor imagery signals during classification.
        
        Args:
            eeg_data: Baseline EEG data (n_samples, n_channels) - should be ~10 seconds
            
        Returns:
            True if baseline captured successfully
        """
        print(f"[MI BASELINE DEBUG] capture_baseline called with shape={eeg_data.shape}")
        
        if eeg_data.shape[0] < 100:  # Need at least some data
            print(f"[MI BASELINE DEBUG] ERROR: Insufficient data: {eeg_data.shape[0]} samples (need >= 100)")
            return False
        
        # Store baseline data
        self._baseline_data = eeg_data.copy()
        print(f"[MI BASELINE DEBUG] Stored baseline data: shape={self._baseline_data.shape}")
        
        # Extract sensorimotor channels
        sensorimotor_data = eeg_data[:, self.sensorimotor_channels]
        print(f"[MI BASELINE DEBUG] Sensorimotor data shape={sensorimotor_data.shape}, "
              f"channels={self.sensorimotor_channels}")
        
        # Apply bandpass filters
        mu_data = self._bandpass_filter(sensorimotor_data, self.mu_band[0], self.mu_band[1])
        beta_data = self._bandpass_filter(sensorimotor_data, self.beta_band[0], self.beta_band[1])
        print(f"[MI BASELINE DEBUG] Filtered data: mu shape={mu_data.shape}, beta shape={beta_data.shape}")
        print(f"[MI BASELINE DEBUG] Mu data stats: mean={np.mean(mu_data):.4f}, std={np.std(mu_data):.4f}")
        print(f"[MI BASELINE DEBUG] Beta data stats: mean={np.mean(beta_data):.4f}, std={np.std(beta_data):.4f}")
        
        # Compute baseline statistics for each channel
        # Mean and std for normalization
        self._baseline_mu_mean = np.mean(mu_data, axis=0)
        self._baseline_mu_std = np.std(mu_data, axis=0) + 1e-10  # Add epsilon to avoid division by zero
        self._baseline_beta_mean = np.mean(beta_data, axis=0)
        self._baseline_beta_std = np.std(beta_data, axis=0) + 1e-10
        
        print(f"[MI BASELINE DEBUG] Baseline mu_mean shape={self._baseline_mu_mean.shape}, "
              f"values={self._baseline_mu_mean}")
        print(f"[MI BASELINE DEBUG] Baseline mu_std shape={self._baseline_mu_std.shape}, "
              f"values={self._baseline_mu_std}")
        print(f"[MI BASELINE DEBUG] Baseline beta_mean shape={self._baseline_beta_mean.shape}, "
              f"values={self._baseline_beta_mean}")
        print(f"[MI BASELINE DEBUG] Baseline beta_std shape={self._baseline_beta_std.shape}, "
              f"values={self._baseline_beta_std}")
        
        # Compute overall baseline power (for thresholding)
        self._baseline_mu_power = np.mean(np.var(mu_data, axis=0))
        self._baseline_beta_power = np.mean(np.var(beta_data, axis=0))
        
        self._has_baseline = True
        print(f"[MI BASELINE] Captured {eeg_data.shape[0] / self.sample_rate:.1f}s baseline data")
        print(f"[MI BASELINE] Mu power: {self._baseline_mu_power:.4f}, Beta power: {self._baseline_beta_power:.4f}")
        print(f"[MI BASELINE DEBUG] Baseline capture complete - _has_baseline={self._has_baseline}")
        
        return True
    
    def load_calibration(self, calibration_data: 'CalibrationData') -> bool:
        """
        Load calibration data to train CSP filters and LDA classifier.
        
        Args:
            calibration_data: CalibrationData object with motor imagery trials
            
        Returns:
            True if calibration loaded successfully
        """
        # For now, return False (calibration not yet implemented)
        # This would require calibration data structure for motor imagery
        return False
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._history.clear()
        # Keep CSP filters and LDA (they're learned from calibration)
