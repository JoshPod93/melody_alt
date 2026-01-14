"""
P300 Oddball Classifier module for BCI-UPIC.

Provides ERP-based classification for P300 oddball paradigm.
Determines which target the user is attending to based on ERP responses.

P300 Oddball paradigm:
- Red flashes are targets (oddball, rare, attended)
- Other colors are non-targets (frequent, ignored)
- P300 ERP is larger for attended (red) targets
- Peak latency: ~300ms post-stimulus
- Epoch window: -100ms to +800ms
- Baseline correction: -100ms to 0ms
- Uses only Cz electrode (optimal for P300)
- Template matching and peak detection for robust classification
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum
from scipy import signal


class AttentionTarget(Enum):
    """Which target the user is attending to."""
    NONE = 0      # No clear attention detected
    UP = 1        # Attending to top target - move cursor up
    DOWN = 2      # Attending to bottom target - move cursor down


@dataclass
class ClassificationResult:
    """Result of P300 classification."""
    target: AttentionTarget
    confidence: float  # 0-1, how confident the classification is
    p300_amplitude_top: float  # P300 amplitude at top target (μV)
    p300_amplitude_bottom: float  # P300 amplitude at bottom target (μV)
    raw_score: float   # Raw classification score (positive = up, negative = down)


@dataclass
class P300Classifier:
    """
    Real-time P300 classifier for BCI control.
    
    Classifies EEG data to determine which flash target
    the user is attending to based on ERP responses.
    
    Attributes:
        sample_rate: EEG sampling rate in Hz
        epoch_window_ms: Epoch window in milliseconds (pre, post) relative to stimulus
        baseline_window_ms: Baseline window in milliseconds (pre-stimulus)
        p300_window_ms: P300 detection window in milliseconds (post-stimulus)
        n_epochs_to_average: Number of epochs to average for better SNR
        threshold: Confidence threshold for making a decision
    """
    sample_rate: float = 250.0  # Unicorn Black sample rate
    epoch_window_ms: Tuple[int, int] = (-100, 800)  # -100ms to +800ms
    baseline_window_ms: Tuple[int, int] = (-100, 0)  # -100ms to 0ms (pre-stimulus)
    p300_window_ms: Tuple[int, int] = (250, 450)  # 250ms to 450ms (P300 peak window)
    n_epochs_to_average: int = 5  # Average 5 epochs for better SNR
    threshold: float = 0.3  # Minimum confidence threshold
    
    # ERP templates (averaged ERPs for each target)
    _top_template: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _bottom_template: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    
    # Epoch buffer for averaging
    _top_epochs: List[NDArray[np.float64]] = field(default_factory=list, repr=False)
    _bottom_epochs: List[NDArray[np.float64]] = field(default_factory=list, repr=False)
    
    # Smoothing for temporal stability
    _history: List[ClassificationResult] = field(default_factory=list, repr=False)
    _history_size: int = 3
    
    @property
    def is_calibrated(self) -> bool:
        """
        Check if classifier is calibrated.
        
        For P300, calibration is optional (templates can be learned online).
        Returns False by default (no subject-specific calibration required).
        """
        return False  # P300 doesn't require calibration like SSVEP
    
    def epoch_data(
        self,
        eeg_data: NDArray[np.float64],
        stimulus_time: float,
        data_times: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """
        Extract epoch around stimulus onset.
        
        Args:
            eeg_data: EEG data array (n_samples, n_channels)
            stimulus_time: Absolute time of stimulus onset (seconds)
            data_times: Timestamps for each sample in eeg_data (seconds)
            
        Returns:
            Epoched data (n_epoch_samples, n_channels) or None if insufficient data
        """
        # Convert epoch window to samples
        pre_samples = int(self.epoch_window_ms[0] * self.sample_rate / 1000)
        post_samples = int(self.epoch_window_ms[1] * self.sample_rate / 1000)
        epoch_length = post_samples - pre_samples
        
        # Find stimulus index in data_times
        stimulus_idx = np.searchsorted(data_times, stimulus_time)
        
        # Diagnostic: Log timestamp alignment issues (first few epochs only)
        if not hasattr(self, '_epoch_diagnostic_count'):
            self._epoch_diagnostic_count = 0
        if self._epoch_diagnostic_count < 5:
            time_diff = abs(data_times[stimulus_idx] - stimulus_time) if stimulus_idx < len(data_times) else float('inf')
            if time_diff > 0.1:  # More than 100ms difference
                print(f"[P300 TIMESTAMP WARNING] Flash time {stimulus_time:.3f} vs EEG time {data_times[stimulus_idx] if stimulus_idx < len(data_times) else 'N/A':.3f}, diff={time_diff*1000:.1f}ms")
            self._epoch_diagnostic_count += 1
        
        # Check if we have enough data
        if stimulus_idx + pre_samples < 0 or stimulus_idx + post_samples > len(eeg_data):
            return None
        
        # Extract epoch
        start_idx = stimulus_idx + pre_samples
        end_idx = stimulus_idx + post_samples
        epoch = eeg_data[start_idx:end_idx]
        
        return epoch
    
    def baseline_correct(self, epoch: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply baseline correction to epoch.
        
        Args:
            epoch: Epoch data (n_samples, n_channels)
            
        Returns:
            Baseline-corrected epoch
        """
        # Convert baseline window to samples
        baseline_start = int(self.baseline_window_ms[0] * self.sample_rate / 1000)
        baseline_end = int(self.baseline_window_ms[1] * self.sample_rate / 1000)
        
        # Convert to indices relative to epoch start
        epoch_start_idx = -int(self.epoch_window_ms[0] * self.sample_rate / 1000)
        baseline_start_idx = epoch_start_idx + baseline_start
        baseline_end_idx = epoch_start_idx + baseline_end
        
        # Extract baseline
        baseline = epoch[baseline_start_idx:baseline_end_idx]
        baseline_mean = np.mean(baseline, axis=0, keepdims=True)
        
        # Subtract baseline
        corrected = epoch - baseline_mean
        
        return corrected
    
    def extract_p300_amplitude(self, epoch: NDArray[np.float64]) -> float:
        """
        Extract P300 amplitude from epoch.
        
        Uses peak detection in the P300 window (250-450ms).
        
        Args:
            epoch: Baseline-corrected epoch (n_samples, n_channels)
            
        Returns:
            P300 amplitude (average across channels, peak in window)
        """
        # Convert P300 window to samples
        epoch_start_idx = -int(self.epoch_window_ms[0] * self.sample_rate / 1000)
        p300_start = epoch_start_idx + int(self.p300_window_ms[0] * self.sample_rate / 1000)
        p300_end = epoch_start_idx + int(self.p300_window_ms[1] * self.sample_rate / 1000)
        
        # Extract P300 window
        p300_window = epoch[p300_start:p300_end]
        
        # Average across channels
        p300_channel_avg = np.mean(p300_window, axis=1)
        
        # Find peak amplitude (absolute value)
        p300_amplitude = np.max(np.abs(p300_channel_avg))
        
        return float(p300_amplitude)
    
    def classify_epoch(
        self,
        epoch: NDArray[np.float64],
        target_position: str
    ) -> Optional[float]:
        """
        Classify a single epoch.
        
        Args:
            epoch: Epoch data (n_samples, n_channels)
            target_position: 'top' or 'bottom'
            
        Returns:
            P300 amplitude or None if epoch is invalid
        """
        if epoch is None or epoch.shape[0] == 0:
            return None
        
        # Baseline correct
        corrected = self.baseline_correct(epoch)
        
        # Extract P300 amplitude
        amplitude = self.extract_p300_amplitude(corrected)
        
        # Store epoch for averaging
        if target_position == "top":
            self._top_epochs.append(corrected)
            if len(self._top_epochs) > self.n_epochs_to_average:
                self._top_epochs.pop(0)
        elif target_position == "bottom":
            self._bottom_epochs.append(corrected)
            if len(self._bottom_epochs) > self.n_epochs_to_average:
                self._bottom_epochs.pop(0)
        
        return amplitude
    
    def classify_averaged(
        self,
        eeg_data: NDArray[np.float64],
        flash_onsets: List[Tuple[str, str, float]],
        data_times: NDArray[np.float64]
    ) -> ClassificationResult:
        """
        Classify based on averaged ERPs from multiple epochs (oddball paradigm).
        
        Args:
            eeg_data: EEG data array (n_samples, 1) - Cz channel only
            flash_onsets: List of (position, color, timestamp) tuples
            data_times: Timestamps for each sample in eeg_data
            
        Returns:
            ClassificationResult
        """
        # Separate target (red) and non-target epochs
        top_target_epochs = []
        bottom_target_epochs = []
        top_nontarget_epochs = []
        bottom_nontarget_epochs = []
        
        # Process recent flash onsets
        # Get more flashes to ensure we have enough target (red) flashes
        n_flashes_to_check = self.n_epochs_to_average * 4  # Check more to get enough targets
        for position, color, flash_time in flash_onsets[-n_flashes_to_check:]:
            epoch = self.epoch_data(eeg_data, flash_time, data_times)
            if epoch is not None:
                amplitude = self.classify_epoch(epoch, position)
                if amplitude is not None:
                    # Ensure color is a string for comparison
                    color_str = str(color).lower() if isinstance(color, str) else str(color)
                    is_target = (color_str == 'red')
                    if position == "top":
                        if is_target:
                            top_target_epochs.append(amplitude)
                        else:
                            top_nontarget_epochs.append(amplitude)
                    else:  # bottom
                        if is_target:
                            bottom_target_epochs.append(amplitude)
                        else:
                            bottom_nontarget_epochs.append(amplitude)
        
        # Calculate average P300 amplitudes
        # For oddball: compare target (red) vs non-target amplitudes
        avg_top_target = np.mean(top_target_epochs) if top_target_epochs else 0.0
        avg_bottom_target = np.mean(bottom_target_epochs) if bottom_target_epochs else 0.0
        avg_top_nontarget = np.mean(top_nontarget_epochs) if top_nontarget_epochs else 0.0
        avg_bottom_nontarget = np.mean(bottom_nontarget_epochs) if bottom_nontarget_epochs else 0.0
        
        # Debug: Log epoch counts periodically
        if not hasattr(self, '_last_debug_log_time'):
            self._last_debug_log_time = 0
        import time as time_module
        current_debug_time = time_module.time()
        if current_debug_time - self._last_debug_log_time > 2.0:  # Log every 2 seconds
            print(f"[P300 DEBUG] Epochs - Top target: {len(top_target_epochs)}, Top non-target: {len(top_nontarget_epochs)}, "
                  f"Bottom target: {len(bottom_target_epochs)}, Bottom non-target: {len(bottom_nontarget_epochs)}, "
                  f"Total flashes processed: {len(flash_onsets)}")
            self._last_debug_log_time = current_debug_time
        
        # P300 oddball effect: target (red) should have larger P300 than non-targets
        # Calculate difference: target amplitude - non-target amplitude
        # If no non-targets, just use target amplitude
        if avg_top_nontarget > 0:
            top_diff = avg_top_target - avg_top_nontarget
        else:
            top_diff = avg_top_target if avg_top_target > 0 else 0.0
        
        if avg_bottom_nontarget > 0:
            bottom_diff = avg_bottom_target - avg_bottom_nontarget
        else:
            bottom_diff = avg_bottom_target if avg_bottom_target > 0 else 0.0
        
        # Classification: larger difference indicates attended target
        diff = top_diff - bottom_diff
        raw_score = diff
        
        # Normalize confidence based on absolute differences
        total_diff = abs(top_diff) + abs(bottom_diff)
        if total_diff > 0:
            confidence = abs(diff) / total_diff
        else:
            # No clear difference - use raw amplitudes
            total_amp = avg_top_target + avg_bottom_target
            if total_amp > 0:
                confidence = abs(diff) / total_amp if diff != 0 else 0.1
            else:
                confidence = 0.1  # Minimum confidence
        
        # Boost confidence if we have clear target responses
        if avg_top_target > 0 or avg_bottom_target > 0:
            # At least one target has response
            max_target = max(avg_top_target, avg_bottom_target)
            min_target = min(avg_top_target, avg_bottom_target)
            if max_target > 0:
                # Scale confidence by how much larger the max is
                if min_target > 0:
                    confidence *= (max_target / (max_target + min_target))
                else:
                    confidence = min(confidence * 1.5, 1.0)  # Boost if only one has response
        
        # Determine target - BINARY: always UP or DOWN, never NONE
        # Use a very low threshold or just use the sign of diff
        if abs(diff) < 0.01 and total_diff < 0.1:
            # Truly no signal - use very low confidence but still pick a direction
            target = AttentionTarget.UP if diff >= 0 else AttentionTarget.DOWN
            confidence = 0.1  # Very low confidence
        elif diff > 0:
            target = AttentionTarget.UP
        else:
            target = AttentionTarget.DOWN
        
        # Apply temporal smoothing
        result = ClassificationResult(
            target=target,
            confidence=min(confidence, 1.0),
            p300_amplitude_top=avg_top_target,
            p300_amplitude_bottom=avg_bottom_target,
            raw_score=raw_score
        )
        
        # Add to history
        self._history.append(result)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        
        # Smooth result
        if len(self._history) >= 2:
            # Weighted average of recent results
            weights = np.linspace(0.5, 1.0, len(self._history))
            weights = weights / np.sum(weights)
            
            # Average raw scores
            avg_raw_score = np.average([r.raw_score for r in self._history], weights=weights)
            
            # Re-determine target based on smoothed score - BINARY: always UP or DOWN
            if abs(avg_raw_score) < 0.01:
                # No clear signal, but still pick a direction (use last direction or default to UP)
                smoothed_target = AttentionTarget.UP if avg_raw_score >= 0 else AttentionTarget.DOWN
            elif avg_raw_score > 0:
                smoothed_target = AttentionTarget.UP
            else:
                smoothed_target = AttentionTarget.DOWN
            
            # Average confidence
            avg_confidence = np.average([r.confidence for r in self._history], weights=weights)
            
            result = ClassificationResult(
                target=smoothed_target,
                confidence=min(avg_confidence, 1.0),
                p300_amplitude_top=avg_top_target,
                p300_amplitude_bottom=avg_bottom_target,
                raw_score=avg_raw_score
            )
        
        return result
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._top_epochs.clear()
        self._bottom_epochs.clear()
        self._history.clear()
        self._top_template = None
        self._bottom_template = None
