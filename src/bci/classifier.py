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
    target_frequencies: Tuple[float, float] = (15.0, 12.0)  # Fixed: 15Hz (UP), 12Hz (DOWN)
    target_phases: Tuple[float, float] = (0.0, np.pi)  # Phase offsets: 15Hz at 0°, 12Hz at 180°
    window_seconds: float = 0.5  # Shorter window for faster response
    n_harmonics: int = 2  # Include fundamental + 1 harmonic
    threshold: float = 0.05  # Very low threshold - almost always move
    occipital_channels: List[int] = field(default_factory=lambda: [5, 6, 7])  # PO7, Oz, PO8
    
    # CCA reference signals - synthetic (screen-calibrated) + calibrated (subject-specific)
    _ref_signals_up_synthetic: NDArray[np.float64] = field(init=False, repr=False)
    _ref_signals_down_synthetic: NDArray[np.float64] = field(init=False, repr=False)
    _ref_signals_up_calibrated: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _ref_signals_down_calibrated: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    # Combined references (synthetic + calibrated concatenated)
    _ref_signals_up: NDArray[np.float64] = field(init=False, repr=False)
    _ref_signals_down: NDArray[np.float64] = field(init=False, repr=False)
    _using_calibration: bool = field(default=False, repr=False)
    
    # Smoothing for temporal stability
    _history: List[ClassificationResult] = field(default_factory=list, repr=False)
    _history_size: int = 5  # Increased for better stability (was 3)
    min_correlation_diff: float = 0.05  # Minimum difference between correlations to make a decision
    
    def __post_init__(self) -> None:
        """Initialize reference signals for CCA."""
        self._generate_reference_signals()
    
    def load_calibration(self, calibration_data: 'CalibrationData') -> bool:
        """
        Load personalized reference signals from calibration data.
        
        Uses standard CCA structure (sin/cos at fundamental + harmonics).
        Templates are used to refine phase/frequency parameters, but references
        remain in standard sin/cos format for optimal CCA performance.
        
        Args:
            calibration_data: CalibrationData with recorded SSVEP responses
            
        Returns:
            True if calibration was loaded successfully
        """
        try:
            ref_up, ref_down = calibration_data.get_cca_references(
                self.window_seconds, n_harmonics=self.n_harmonics
            )
            
            if ref_up is not None and ref_down is not None:
                # Ensure correct shape: (n_samples, 2*n_harmonics)
                n_samples = int(self.window_seconds * self.sample_rate)
                expected_cols = 2 * self.n_harmonics
                
                # Resize if needed
                if ref_up.shape[0] != n_samples:
                    ref_up = self._resize_reference(ref_up, n_samples)
                if ref_down.shape[0] != n_samples:
                    ref_down = self._resize_reference(ref_down, n_samples)
                
                # Ensure correct number of columns (should be 2*n_harmonics)
                if ref_up.shape[1] != expected_cols:
                    print(f"[WARNING] Reference shape mismatch: got {ref_up.shape[1]} columns, "
                          f"expected {expected_cols}. Truncating or padding.")
                    if ref_up.shape[1] > expected_cols:
                        ref_up = ref_up[:, :expected_cols]
                        ref_down = ref_down[:, :expected_cols]
                    else:
                        # Pad with zeros (shouldn't happen with new implementation)
                        pad = np.zeros((ref_up.shape[0], expected_cols - ref_up.shape[1]))
                        ref_up = np.hstack([ref_up, pad])
                        ref_down = np.hstack([ref_down, pad])
                
                # Store calibrated references separately
                self._ref_signals_up_calibrated = ref_up
                self._ref_signals_down_calibrated = ref_down
                
                # Validate reference signals before combining
                if np.all(ref_up == 0) or np.all(np.abs(ref_up) < 1e-10):
                    print(f"[CLASSIFIER ERROR] UP reference is all zeros!")
                if np.all(ref_down == 0) or np.all(np.abs(ref_down) < 1e-10):
                    print(f"[CLASSIFIER ERROR] DOWN reference is all zeros!")
                
                # Check shapes match for hstack
                if self._ref_signals_up_synthetic.shape[0] != ref_up.shape[0]:
                    print(f"[CLASSIFIER ERROR] Shape mismatch: synthetic_up={self._ref_signals_up_synthetic.shape}, "
                          f"calibrated_up={ref_up.shape}")
                if self._ref_signals_down_synthetic.shape[0] != ref_down.shape[0]:
                    print(f"[CLASSIFIER ERROR] Shape mismatch: synthetic_down={self._ref_signals_down_synthetic.shape}, "
                          f"calibrated_down={ref_down.shape}")
                
                # Combine synthetic + calibrated references (additive approach)
                # This gives CCA more information: screen-calibrated + subject-specific
                # Shape: (n_samples, 2*n_harmonics + 2*n_harmonics) = (n_samples, 4*n_harmonics)
                self._ref_signals_up = np.hstack([self._ref_signals_up_synthetic, ref_up])
                self._ref_signals_down = np.hstack([self._ref_signals_down_synthetic, ref_down])
                
                # Validate combined references
                up_var = np.var(self._ref_signals_up, axis=0)
                down_var = np.var(self._ref_signals_down, axis=0)
                print(f"[CLASSIFIER] Combined UP reference variance: min={np.min(up_var):.6f}, max={np.max(up_var):.6f}")
                print(f"[CLASSIFIER] Combined DOWN reference variance: min={np.min(down_var):.6f}, max={np.max(down_var):.6f}")
                
                self._using_calibration = True
                
                # Debug: Verify which frequency is mapped to which
                print(f"[CLASSIFIER] Loaded calibration: UP={ref_up.shape}, DOWN={ref_down.shape}")
                print(f"[CLASSIFIER] Reference structure: {self.n_harmonics} harmonics = "
                      f"{2*self.n_harmonics} components (sin/cos pairs) per reference set")
                print(f"[CLASSIFIER] Combined references: {self._ref_signals_up.shape[1]} total components "
                      f"(synthetic {2*self.n_harmonics} + calibrated {2*self.n_harmonics})")
                print(f"[CLASSIFIER] Reference mapping: UP=higher_freq ({self.target_frequencies[0]:.2f}Hz), "
                      f"DOWN=lower_freq ({self.target_frequencies[1]:.2f}Hz)")
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            import traceback
            traceback.print_exc()
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
        
        Uses fixed frequencies: 15Hz (UP, 0° phase) and 12Hz (DOWN, 180° phase).
        
        Visual stimulus uses: sin(2π * f * t + phase)
        - Higher frequency (up): 15Hz, phase = 0
        - Lower frequency (down): 12Hz, phase = π (180°)
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
                self._ref_signals_up_synthetic = ref_array
            else:
                self._ref_signals_down_synthetic = ref_array
        
        # Initialize combined references (synthetic only initially)
        self._ref_signals_up = self._ref_signals_up_synthetic.copy()
        self._ref_signals_down = self._ref_signals_down_synthetic.copy()
    
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
        
        # OPTIMIZED: Use only fundamental frequencies (skip harmonics for speed)
        # Harmonics can be added back if needed, but fundamentals are usually enough
        bandwidth = 1.0  # Hz
        
        # Vectorized band power calculation (faster than loop)
        def get_band_power_fast(target_freq: float) -> float:
            mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
            if not np.any(mask):
                return 0.0
            return np.sum(fft_vals[mask] ** 2)
        
        # Only use fundamental frequencies for speed
        power_up = get_band_power_fast(self.target_frequencies[0])
        power_down = get_band_power_fast(self.target_frequencies[1])
        
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
    
    def classify_fbcca(self, eeg_data: NDArray[np.float64]) -> ClassificationResult:
        """
        Filter Bank CCA (FBCCA) - applies multiple bandpass filters and votes.
        
        Uses 3 filter banks (applied directly to input data - no pre-filtering):
        - Fundamental: 11-16 Hz (covers 12Hz and 15Hz)
        - 1st harmonic: 23-31 Hz (covers 24Hz and 30Hz)  
        - 2nd harmonic: 35-46 Hz (covers 36Hz and 45Hz)
        
        Each filter bank runs CCA independently, then votes are combined.
        This is more accurate than single-band CCA and still lightweight.
        
        NOTE: Input data should only have CAR and notch filter applied (no main bandpass).
        The filter banks handle all frequency selection.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels) - should be raw or CAR+notch only
            
        Returns:
            ClassificationResult
        """
        # CRITICAL: Use ONLY occipital channels for SSVEP
        if eeg_data.shape[1] > max(self.occipital_channels):
            data = eeg_data[:, self.occipital_channels]
        else:
            data = eeg_data
        
        # Filter bank definitions (fundamental, 1st harmonic, 2nd harmonic)
        filter_banks = [
            (11.0, 16.0),   # Fundamental: 12Hz, 15Hz
            (23.0, 31.0),   # 1st harmonic: 24Hz, 30Hz
            (35.0, 46.0)    # 2nd harmonic: 36Hz, 45Hz
        ]
        
        # Votes for each target
        up_votes = 0
        down_votes = 0
        correlations_up = []
        correlations_down = []
        
        # Apply each filter bank and run CCA
        for fb_low, fb_high in filter_banks:
            # Apply bandpass filter to data
            filtered_data = self._apply_bandpass(data, fb_low, fb_high)
            
            # Resize references to match filtered data length
            n_data = filtered_data.shape[0]
            if n_data != self._ref_signals_up.shape[0]:
                ref_up_resized = self._resize_reference(self._ref_signals_up, n_data)
                ref_down_resized = self._resize_reference(self._ref_signals_down, n_data)
            else:
                ref_up_resized = self._ref_signals_up
                ref_down_resized = self._ref_signals_down
            
            # Run CCA on filtered data
            try:
                corr_up = self._cca_correlation(filtered_data, ref_up_resized)
                corr_down = self._cca_correlation(filtered_data, ref_down_resized)
                
                correlations_up.append(corr_up)
                correlations_down.append(corr_down)
                
                # Vote: which target has higher correlation in this filter bank?
                if corr_up > corr_down:
                    up_votes += 1
                elif corr_down > corr_up:
                    down_votes += 1
                # If equal, no vote
            except Exception as e:
                # Skip this filter bank if CCA fails
                continue
        
        # Combine results: majority vote
        if up_votes > down_votes:
            target = AttentionTarget.UP
            avg_corr_up = np.mean(correlations_up) if correlations_up else 0.0
            avg_corr_down = np.mean(correlations_down) if correlations_down else 0.0
        elif down_votes > up_votes:
            target = AttentionTarget.DOWN
            avg_corr_up = np.mean(correlations_up) if correlations_up else 0.0
            avg_corr_down = np.mean(correlations_down) if correlations_down else 0.0
        else:
            # Tie or no votes - return NONE
            target = AttentionTarget.NONE
            avg_corr_up = np.mean(correlations_up) if correlations_up else 0.0
            avg_corr_down = np.mean(correlations_down) if correlations_down else 0.0
        
        # Calculate confidence based on vote margin and average correlations
        total_votes = up_votes + down_votes
        if total_votes == 0:
            confidence = 0.1
        else:
            vote_margin = abs(up_votes - down_votes) / total_votes
            avg_corr = (avg_corr_up + avg_corr_down) / 2
            confidence = vote_margin * min(avg_corr * 1.5, 1.0)
            confidence = max(confidence, 0.2)
        
        # Score based on vote difference
        score = (up_votes - down_votes) / max(total_votes, 1)
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_higher_freq=avg_corr_up,
            power_lower_freq=avg_corr_down,
            raw_score=score
        )
    
    def _apply_bandpass(self, data: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
        """
        Apply bandpass filter to data (lightweight, no state tracking needed).
        
        Args:
            data: EEG data (n_samples, n_channels)
            low: Lower cutoff frequency (Hz)
            high: Upper cutoff frequency (Hz)
            
        Returns:
            Filtered data of same shape
        """
        nyquist = self.sample_rate / 2
        low_norm = max(0.001, min(low / nyquist, 0.99))
        high_norm = max(low_norm + 0.01, min(high / nyquist, 0.99))
        
        # Design filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.lfilter(b, a, data[:, ch])
        
        return filtered
    
    def classify_cca(self, eeg_data: NDArray[np.float64]) -> ClassificationResult:
        """
        Classify using Canonical Correlation Analysis (CCA) with 2 harmonics.
        
        CCA works with synthetic references (screen-calibrated) even without
        subject-specific calibration. If calibration is loaded, it enhances
        the references with subject-specific templates.
        
        This method is used by both:
        - Main live composition (0.3s chunks every 50ms)
        - Data validation/checking (0.3s chunks for testing)
        
        References are automatically resized to match data chunk size to preserve
        phase alignment (especially important for DOWN target with π phase offset).
        
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
        
        # Resize reference signals to match data length (preserves phase profile)
        # This is better than padding data with zeros, which breaks phase alignment
        n_data = data.shape[0]
        n_ref = self._ref_signals_up.shape[0]
        
        if n_data != n_ref:
            # Resize references to match data chunk size (preserves phase)
            ref_up_resized = self._resize_reference(self._ref_signals_up, n_data)
            ref_down_resized = self._resize_reference(self._ref_signals_down, n_data)
        else:
            ref_up_resized = self._ref_signals_up
            ref_down_resized = self._ref_signals_down
        
        # Compute CCA correlation for each target using resized references
        # References include synthetic (screen-calibrated) + calibrated (subject-specific) if available
        try:
            corr_up = self._cca_correlation(data, ref_up_resized)
        except Exception as e:
            print(f"[CCA ERROR] Failed UP correlation: {e}")
            print(f"  data.shape={data.shape}, ref_up.shape={ref_up_resized.shape}")
            corr_up = 0.0
        
        # ALWAYS log first few calls to diagnose issue
        if not hasattr(self, '_down_call_count'):
            self._down_call_count = 0
        
        self._down_call_count += 1
        
        # Log first 3 calls ALWAYS
        if self._down_call_count <= 3:
            import sys
            from pathlib import Path
            
            log_msg = (f"[CCA DEBUG] Call #{self._down_call_count} - DOWN correlation attempt\n"
                      f"  data.shape={data.shape}, ref_down.shape={ref_down_resized.shape}\n"
                      f"  ref_down stats: min={np.min(ref_down_resized):.6f}, max={np.max(ref_down_resized):.6f}, "
                      f"mean={np.mean(ref_down_resized):.6f}, var={np.var(ref_down_resized, axis=0)[:3]}\n"
                      f"  data stats: min={np.min(data):.6f}, max={np.max(data):.6f}, mean={np.mean(data):.6f}")
            
            print(log_msg, file=sys.stderr, flush=True)
            
            # Also log to file
            log_file = Path("validation_plots") / "cca_diagnostic.log"
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f"{log_msg}\n")
        
        try:
            corr_down = self._cca_correlation(data, ref_down_resized)
            
            # Debug: ALWAYS log if DOWN correlation is zero (should happen for every chunk)
            if corr_down < 1e-6:
                import sys
                from pathlib import Path
                
                log_msg = (f"[CCA WARNING] Call #{self._down_call_count} - DOWN correlation is near zero: {corr_down:.6f}\n"
                          f"  data.shape={data.shape}, ref_down.shape={ref_down_resized.shape}\n"
                          f"  ref_down stats: min={np.min(ref_down_resized):.6f}, max={np.max(ref_down_resized):.6f}, "
                          f"mean={np.mean(ref_down_resized):.6f}, var={np.var(ref_down_resized, axis=0)[:3]}\n"
                          f"  data stats: min={np.min(data):.6f}, max={np.max(data):.6f}, mean={np.mean(data):.6f}")
                
                print(log_msg, file=sys.stderr, flush=True)
                
                # Also log to file
                log_file = Path("validation_plots") / "cca_diagnostic.log"
                log_file.parent.mkdir(exist_ok=True)
                with open(log_file, 'a') as f:
                    f.write(f"{log_msg}\n")
        except Exception as e:
            import sys
            import traceback
            from pathlib import Path
            
            error_msg = (f"[CCA ERROR] Call #{self._down_call_count} - Failed DOWN correlation: {e}\n"
                        f"  data.shape={data.shape}, ref_down.shape={ref_down_resized.shape}\n")
            error_trace = traceback.format_exc()
            
            print(error_msg, file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            
            # Also log to file
            log_file = Path("validation_plots") / "cca_diagnostic.log"
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f"{error_msg}\n{error_trace}\n")
            
            corr_down = 0.0
        
        # DEBUG: Log all CCA correlations to file for validation analysis
        import logging
        import time
        from pathlib import Path
        
        # Set up logger for CCA debug info
        if not hasattr(self, '_cca_logger_initialized'):
            log_file = Path("validation_plots") / "cca_debug.log"
            log_file.parent.mkdir(exist_ok=True)
            
            # Create logger
            self._cca_logger = logging.getLogger('cca_debug')
            self._cca_logger.setLevel(logging.DEBUG)
            self._cca_logger.handlers.clear()  # Clear any existing handlers
            
            # File handler
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self._cca_logger.addHandler(fh)
            
            self._cca_logger_initialized = True
            self._cca_log_count = 0
        
        # Log every correlation (useful for validation)
        self._cca_log_count += 1
        self._cca_logger.debug(
            f"CCA#{self._cca_log_count}: corr_up={corr_up:.6f}, corr_down={corr_down:.6f}, "
            f"diff={corr_up - corr_down:.6f}, "
            f"UP_freq={self.target_frequencies[0]:.2f}Hz, DOWN_freq={self.target_frequencies[1]:.2f}Hz, "
            f"prediction={'UP' if corr_up > corr_down else 'DOWN'}, "
            f"data_shape={data.shape}, ref_up_shape={ref_up_resized.shape}, ref_down_shape={ref_down_resized.shape}, "
            f"using_calibration={self._using_calibration}"
        )
        
        # CCA correlation difference with improved scoring
        cca_diff = corr_up - corr_down
        
        # Only make a decision if difference is significant enough
        if abs(cca_diff) < self.min_correlation_diff:
            # Difference too small - return NONE with low confidence
            return ClassificationResult(
                target=AttentionTarget.NONE,
                confidence=0.1,
                power_higher_freq=corr_up,
                power_lower_freq=corr_down,
                raw_score=0.0
            )
        
        # Improved scoring: use normalized difference weighted by average correlation
        # This gives more weight when both correlations are high (strong SSVEP)
        avg_corr = (corr_up + corr_down) / 2
        normalized_diff = cca_diff / (avg_corr + 0.1)  # Normalize by average correlation
        score = np.sign(cca_diff) * min(abs(normalized_diff) * 2, 1)
        
        # Confidence based on magnitude and average correlation strength
        confidence = min(abs(score) * 1.2, 1.0)
        confidence *= min(avg_corr * 1.5, 1.0)  # Boost confidence when correlations are high
        confidence = max(confidence, 0.2)  # Minimum confidence
        
        # Direction based on CCA score
        if score >= 0:
            target = AttentionTarget.UP
        else:
            target = AttentionTarget.DOWN
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_higher_freq=corr_up,
            power_lower_freq=corr_down,
            raw_score=score
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
            # Check for invalid inputs
            if Y.shape[1] == 0:
                import sys
                print(f"[CCA WARNING] Empty reference signals! Y.shape={Y.shape}", file=sys.stderr, flush=True)
                return 0.0
            
            # Check if Y has zero variance (all same values)
            Y_var = np.var(Y, axis=0)
            if np.all(Y_var < 1e-10):
                import sys
                print(f"[CCA WARNING] Reference signals have zero variance! Y_var={Y_var}", file=sys.stderr, flush=True)
                return 0.0
            
            # Solve generalized eigenvalue problem
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
            
            M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
            eigenvalues, _ = eig(M)
            
            # Return maximum correlation (square root of max eigenvalue)
            max_eigenvalue = np.max(np.real(eigenvalues))
            corr = np.sqrt(max(0, max_eigenvalue))
            
            # Debug: Log if correlation is suspiciously low
            if corr < 1e-6:
                import sys
                print(f"[CCA WARNING] Very low correlation: {corr:.6f}, Y.shape={Y.shape}, Y_var={Y_var[:3]}", file=sys.stderr, flush=True)
            
            return corr
        except np.linalg.LinAlgError as e:
            import sys
            print(f"[CCA ERROR] Linear algebra error: {e}, X.shape={X.shape}, Y.shape={Y.shape}", file=sys.stderr, flush=True)
            print(f"  Y stats: mean={np.mean(Y, axis=0)[:3]}, var={np.var(Y, axis=0)[:3]}", file=sys.stderr, flush=True)
            return 0.0
        except Exception as e:
            import sys
            import traceback
            print(f"[CCA ERROR] Unexpected error in _cca_correlation: {e}", file=sys.stderr, flush=True)
            print(f"  X.shape={X.shape}, Y.shape={Y.shape}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
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
        if method == "fbcca":
            result = self.classify_fbcca(eeg_data)
        elif method == "cca":
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
        Apply temporal smoothing for better stability.
        
        Uses weighted average of recent results, with more weight on recent samples.
        """
        # Add current to history
        self._history.append(current)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        
        # If not enough history, return current
        if len(self._history) < 2:
            return current
        
        # Weighted average: more recent samples get higher weight
        weights = np.linspace(0.5, 1.0, len(self._history))
        weights = weights / np.sum(weights)
        
        # Average raw scores with weights
        avg_score = np.average([r.raw_score for r in self._history], weights=weights)
        
        # Determine target based on weighted average score
        if abs(avg_score) < 0.05:  # Too close to zero
            smoothed_target = AttentionTarget.NONE
        else:
            smoothed_target = AttentionTarget.UP if avg_score > 0 else AttentionTarget.DOWN
        
        # Count agreement with smoothed target
        same_direction = sum(1 for r in self._history if r.target == smoothed_target)
        agreement_ratio = same_direction / len(self._history)
        
        # Boost confidence based on agreement and average correlation strength
        avg_corr_up = np.mean([r.power_higher_freq for r in self._history])
        avg_corr_down = np.mean([r.power_lower_freq for r in self._history])
        avg_corr = (avg_corr_up + avg_corr_down) / 2
        
        boosted_confidence = current.confidence * (0.6 + 0.4 * agreement_ratio)
        boosted_confidence *= min(avg_corr * 1.2, 1.0)  # Boost when correlations are strong
        boosted_confidence = min(boosted_confidence, 1.0)
        
        return ClassificationResult(
            target=smoothed_target,
            confidence=max(boosted_confidence, 0.2),
            power_higher_freq=avg_corr_up,
            power_lower_freq=avg_corr_down,
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
