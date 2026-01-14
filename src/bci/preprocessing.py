"""
EEG Preprocessing module for BCI-UPIC.

Provides real-time signal processing for SSVEP detection:
- Common Average Reference (CAR) - removes common noise, CRITICAL for SSVEP
- Bandpass filtering (10-32Hz: focused on SSVEP range 11-15Hz + harmonics up to 30Hz)
- Notch filtering for powerline noise (50/60Hz)
- Artifact rejection
- Signal normalization
- LSL stream integration for g.tec Unicorn Black

Designed to be lightweight for real-time processing.

NOTE: The g.tec Unicorn Black has built-in reference/ground electrodes.
The device also has separate left/right mastoid sensors that can be placed
on the mastoid bones behind the ears. If these are placed and appear in the
LSL stream, you can use them as a reference instead of CAR.

For SSVEP, CAR is typically preferred because:
- Mastoid sensors can pick up muscle artifacts
- CAR reduces common noise more effectively with 8 channels
- CAR is standard practice for SSVEP studies

However, if mastoid sensors are properly placed and you prefer mastoid reference,
you can enable it by setting use_mastoid_ref=True and providing the channel indices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, TYPE_CHECKING
from scipy import signal
from collections import deque

if TYPE_CHECKING:
    from .lsl_stream import LSLReceiver


@dataclass
class FilterCoefficients:
    """Store filter coefficients for real-time filtering."""
    b: NDArray[np.float64]
    a: NDArray[np.float64]
    zi: Optional[NDArray[np.float64]] = None


@dataclass
class EEGPreprocessor:
    """
    Real-time EEG preprocessing for SSVEP detection.
    
    Optimized for detecting SSVEP responses at 11-15 Hz with harmonics up to 30 Hz.
    Uses focused bandpass filtering (10-32 Hz) to improve signal-to-noise ratio.
    
    Attributes:
        sample_rate: EEG sampling rate in Hz (default 256)
        n_channels: Number of EEG channels
        bandpass_low: Lower cutoff frequency (Hz) - removes low-freq noise/drift
        bandpass_high: Upper cutoff frequency (Hz) - includes harmonics up to 30 Hz
        notch_freq: Powerline frequency for notch filter (50 or 60 Hz)
        buffer_seconds: Size of the rolling buffer in seconds
    """
    sample_rate: float = 256.0
    n_channels: int = 8
    bandpass_low: float = 11.0   # Lower cutoff: focus on SSVEP range (12-15 Hz), reduce noise
    bandpass_high: float = 30.0  # Upper cutoff: include harmonics up to 30 Hz (15Hz*2, 12Hz*2)
    notch_freq: float = 50.0     # 50Hz for EU, 60Hz for US
    buffer_seconds: float = 2.0   # 2 second rolling buffer
    use_car: bool = True  # Common Average Reference - CRITICAL for SSVEP signal quality
    use_mastoid_ref: bool = False  # Use mastoid reference if available (overrides CAR)
    mastoid_channel_indices: Optional[List[int]] = None  # Indices of mastoid channels if available
    
    # Filter states
    _bandpass_filter: FilterCoefficients = field(init=False, repr=False)
    _notch_filter: FilterCoefficients = field(init=False, repr=False)
    
    # Rolling buffer for each channel
    _buffer: deque = field(init=False, repr=False)
    _buffer_size: int = field(init=False, repr=False)
    
    # Raw buffer (CAR + notch only, no bandpass) for FBCCA
    _raw_buffer: deque = field(init=False, repr=False)
    use_fbcca: bool = False  # If True, skip main bandpass (filter banks handle it)
    
    # Filter states for real-time filtering (per channel)
    _bp_zi: List[NDArray] = field(init=False, repr=False)
    _notch_zi: List[NDArray] = field(init=False, repr=False)
    
    # Artifact detection thresholds
    artifact_threshold: float = 5.0  # Amplitude-based artifact threshold (std devs)
    muscle_artifact_threshold: float = 0.3  # High-freq power threshold (normalized)
    blink_threshold: float = 3.0  # Low-freq transient threshold (std devs)
    
    # Physics-based denoising filters
    _muscle_filter: Optional[FilterCoefficients] = field(init=False, repr=False, default=None)
    _blink_filter: Optional[FilterCoefficients] = field(init=False, repr=False, default=None)
    _muscle_filter_zi: List[NDArray] = field(init=False, repr=False)
    _blink_filter_zi: List[NDArray] = field(init=False, repr=False)
    
    # Enable/disable physics-based denoising
    enable_muscle_artifact_detection: bool = True
    enable_blink_detection: bool = True
    
    # Running statistics for normalization
    _running_mean: NDArray[np.float64] = field(init=False, repr=False)
    _running_var: NDArray[np.float64] = field(init=False, repr=False)
    _n_samples_seen: int = field(default=0, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize filters and buffers."""
        self._init_filters()
        self._init_buffer()
        self._init_statistics()
    
    def _init_filters(self) -> None:
        """Initialize bandpass and notch filters."""
        # Bandpass filter (Butterworth, order 4)
        # Using second-order sections for numerical stability
        nyquist = self.sample_rate / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        
        # Ensure frequencies are valid
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = signal.butter(4, [low, high], btype='band')
        self._bandpass_filter = FilterCoefficients(b=b, a=a)
        
        # Notch filter for powerline noise
        notch_low = (self.notch_freq - 2) / nyquist
        notch_high = (self.notch_freq + 2) / nyquist
        notch_low = max(0.001, min(notch_low, 0.99))
        notch_high = max(notch_low + 0.01, min(notch_high, 0.99))
        
        b_notch, a_notch = signal.butter(2, [notch_low, notch_high], btype='bandstop')
        self._notch_filter = FilterCoefficients(b=b_notch, a=a_notch)
        
        # Initialize filter states for each channel
        self._bp_zi = [
            signal.lfilter_zi(self._bandpass_filter.b, self._bandpass_filter.a)
            for _ in range(self.n_channels)
        ]
        self._notch_zi = [
            signal.lfilter_zi(self._notch_filter.b, self._notch_filter.a)
            for _ in range(self.n_channels)
        ]
        
        # Initialize physics-based denoising filters
        if self.enable_muscle_artifact_detection:
            # High-frequency filter for muscle artifacts (30-100 Hz)
            muscle_low = 30.0 / nyquist
            muscle_high = min(100.0 / nyquist, 0.99)
            muscle_low = max(0.001, min(muscle_low, 0.99))
            muscle_high = max(muscle_low + 0.01, min(muscle_high, 0.99))
            b_muscle, a_muscle = signal.butter(4, [muscle_low, muscle_high], btype='band')
            self._muscle_filter = FilterCoefficients(b=b_muscle, a=a_muscle)
            self._muscle_filter_zi = [
                signal.lfilter_zi(b_muscle, a_muscle)
                for _ in range(self.n_channels)
            ]
        else:
            self._muscle_filter = None
            self._muscle_filter_zi = []
        
        if self.enable_blink_detection:
            # Low-frequency filter for eye blinks (0.5-4 Hz)
            blink_low = 0.5 / nyquist
            blink_high = 4.0 / nyquist
            blink_low = max(0.001, min(blink_low, 0.99))
            blink_high = max(blink_low + 0.01, min(blink_high, 0.99))
            b_blink, a_blink = signal.butter(4, [blink_low, blink_high], btype='band')
            self._blink_filter = FilterCoefficients(b=b_blink, a=a_blink)
            self._blink_filter_zi = [
                signal.lfilter_zi(b_blink, a_blink)
                for _ in range(self.n_channels)
            ]
        else:
            self._blink_filter = None
            self._blink_filter_zi = []
    
    def _init_buffer(self) -> None:
        """Initialize rolling buffer."""
        self._buffer_size = int(self.buffer_seconds * self.sample_rate)
        self._buffer = deque(maxlen=self._buffer_size)
        
        # Pre-fill with zeros
        for _ in range(self._buffer_size):
            self._buffer.append(np.zeros(self.n_channels))
    
    def _init_statistics(self) -> None:
        """Initialize running statistics for normalization."""
        self._running_mean = np.zeros(self.n_channels)
        self._running_var = np.ones(self.n_channels)
        self._n_samples_seen = 0
    
    def reset(self) -> None:
        """Reset all filter states and buffers."""
        self._init_filters()
        self._init_buffer()
        self._init_statistics()
        # Reset physics-based filter states
        if self.enable_muscle_artifact_detection and self._muscle_filter is not None:
            self._muscle_filter_zi = [
                signal.lfilter_zi(self._muscle_filter.b, self._muscle_filter.a)
                for _ in range(self.n_channels)
            ]
        if self.enable_blink_detection and self._blink_filter is not None:
            self._blink_filter_zi = [
                signal.lfilter_zi(self._blink_filter.b, self._blink_filter.a)
                for _ in range(self.n_channels)
            ]
    
    def process_sample(self, sample: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Process a single EEG sample (all channels).
        
        Args:
            sample: Array of shape (n_channels,) containing one sample per channel
            
        Returns:
            Filtered and normalized sample
        """
        if len(sample) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")
        
        # STEP 1: Reference (CAR or Mastoid)
        # Option A: Mastoid reference (if available and enabled)
        if self.use_mastoid_ref and self.mastoid_channel_indices is not None:
            mastoid_ref = np.mean(sample[self.mastoid_channel_indices])
            sample = sample - mastoid_ref
        # Option B: Common Average Reference (CAR) - default for SSVEP
        elif self.use_car:
            car_reference = np.mean(sample)
            sample = sample - car_reference
        
        filtered = np.zeros(self.n_channels)
        
        for ch in range(self.n_channels):
            # Apply bandpass filter
            bp_out, self._bp_zi[ch] = signal.lfilter(
                self._bandpass_filter.b,
                self._bandpass_filter.a,
                [sample[ch]],
                zi=self._bp_zi[ch]
            )
            
            # Apply notch filter
            notch_out, self._notch_zi[ch] = signal.lfilter(
                self._notch_filter.b,
                self._notch_filter.a,
                bp_out,
                zi=self._notch_zi[ch]
            )
            
            filtered[ch] = notch_out[0]
        
        # Update running statistics
        self._update_statistics(filtered)
        
        # Normalize
        normalized = self._normalize(filtered)
        
        # Add to buffer
        self._buffer.append(normalized)
        
        return normalized
    
    def process_chunk(self, chunk: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Process a chunk of EEG data (multiple samples) - OPTIMIZED VERSION.
        
        Uses vectorized operations for much faster processing.
        
        Args:
            chunk: Array of shape (n_samples, n_channels)
            
        Returns:
            Filtered and normalized chunk of same shape
        """
        if chunk.shape[0] == 0:
            return chunk
        
        # VECTORIZED: Process all channels at once (much faster than sample-by-sample)
        
        # STEP 1: Reference (CAR or Mastoid)
        # Option A: Mastoid reference (if available and enabled)
        if self.use_mastoid_ref and self.mastoid_channel_indices is not None:
            # Use average of mastoid channels as reference
            mastoid_ref = np.mean(chunk[:, self.mastoid_channel_indices], axis=1, keepdims=True)
            chunk = chunk - mastoid_ref
        # Option B: Common Average Reference (CAR) - default for SSVEP
        elif self.use_car:
            # Compute mean across all channels for each sample
            car_reference = np.mean(chunk, axis=1, keepdims=True)  # Shape: (n_samples, 1)
            chunk = chunk - car_reference  # Subtract CAR from each channel
        
        # STEP 1.5: Physics-based artifact detection (on raw referenced data)
        # Detect artifacts before filtering to avoid corrupting filter states
        muscle_artifacts = self.detect_muscle_artifact(chunk) if self.enable_muscle_artifact_detection else np.zeros(chunk.shape[0], dtype=bool)
        blink_artifacts = self.detect_blink_artifact(chunk) if self.enable_blink_detection else np.zeros(chunk.shape[0], dtype=bool)
        artifact_mask = muscle_artifacts | blink_artifacts
        
        filtered = np.zeros_like(chunk)
        raw_filtered = np.zeros_like(chunk)  # For FBCCA (CAR + notch only)
        
        # STEP 2: Apply filters channel by channel (but vectorized per channel)
        for ch in range(self.n_channels):
            # Notch filter first (needed for both paths)
            notch_out, self._notch_zi[ch] = signal.lfilter(
                self._notch_filter.b,
                self._notch_filter.a,
                chunk[:, ch],
                zi=self._notch_zi[ch]
            )
            
            # Store CAR + notch only for FBCCA
            raw_filtered[:, ch] = notch_out
            
            # Bandpass filter (only if not using FBCCA)
            if not self.use_fbcca:
                bp_out, self._bp_zi[ch] = signal.lfilter(
                    self._bandpass_filter.b,
                    self._bandpass_filter.a,
                    notch_out,
                    zi=self._bp_zi[ch]
                )
                filtered[:, ch] = bp_out
            else:
                # For FBCCA, skip bandpass - filter banks handle it
                filtered[:, ch] = notch_out
        
        # STEP 3: Artifact rejection - replace artifact samples with interpolated values
        if np.any(artifact_mask):
            # Interpolate artifact samples using neighboring clean samples
            for ch in range(self.n_channels):
                artifact_indices = np.where(artifact_mask)[0]
                if len(artifact_indices) > 0:
                    # Simple forward-fill for artifacts (use last clean value)
                    for idx in artifact_indices:
                        if idx > 0:
                            filtered[idx, ch] = filtered[idx - 1, ch]
                        elif idx < len(filtered) - 1:
                            filtered[idx, ch] = filtered[idx + 1, ch]
                        else:
                            filtered[idx, ch] = 0.0
        
        # Update statistics in batch (more efficient) - skip artifact samples
        clean_samples = filtered[~artifact_mask] if np.any(artifact_mask) else filtered
        for sample in clean_samples:
            self._update_statistics(sample)
        
        # Normalize in batch
        std = np.sqrt(self._running_var + 1e-8)
        normalized = (filtered - self._running_mean) / std
        normalized = np.clip(normalized, -10, 10)
        
        # Add to buffer
        for sample in normalized:
            self._buffer.append(sample)
        
        # Also store raw (CAR + notch only, normalized) for FBCCA
        # Normalize raw_filtered using same statistics
        if hasattr(self, '_raw_buffer'):
            std = np.sqrt(self._running_var + 1e-8)
            raw_normalized = (raw_filtered - self._running_mean) / std
            raw_normalized = np.clip(raw_normalized, -10, 10)
            for sample in raw_normalized:
                self._raw_buffer.append(sample)
        
        return normalized
    
    def _update_statistics(self, sample: NDArray[np.float64]) -> None:
        """Update running mean and variance using Welford's algorithm."""
        # Check for NaN/Inf in sample - skip if corrupted
        if not np.all(np.isfinite(sample)):
            return
        
        self._n_samples_seen += 1
        n = self._n_samples_seen
        
        delta = sample - self._running_mean
        self._running_mean += delta / n
        delta2 = sample - self._running_mean
        self._running_var += (delta * delta2 - self._running_var) / n
        
        # Clamp variance to prevent overflow
        self._running_var = np.clip(self._running_var, 1e-10, 1e10)
    
    def _normalize(self, sample: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score normalize using running statistics."""
        # Check for NaN/Inf - return zeros if corrupted
        if not np.all(np.isfinite(sample)):
            return np.zeros_like(sample)
        
        std = np.sqrt(self._running_var + 1e-8)  # Add epsilon for stability
        normalized = (sample - self._running_mean) / std
        
        # Clamp output to prevent extreme values
        return np.clip(normalized, -10, 10)
    
    def detect_artifact(self, sample: NDArray[np.float64]) -> bool:
        """
        Detect if a sample contains artifacts using amplitude threshold.
        
        Args:
            sample: Normalized sample
            
        Returns:
            True if artifact detected
        """
        return np.any(np.abs(sample) > self.artifact_threshold)
    
    def detect_muscle_artifact(self, raw_chunk: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Detect muscle artifacts using high-frequency power (30-100 Hz).
        
        Muscle artifacts manifest as high-frequency noise (>30 Hz) that's
        not part of SSVEP harmonics.
        
        Args:
            raw_chunk: Raw EEG chunk (n_samples, n_channels) - before bandpass
            
        Returns:
            Boolean array (n_samples,) indicating muscle artifacts
        """
        if not self.enable_muscle_artifact_detection or self._muscle_filter is None:
            return np.zeros(raw_chunk.shape[0], dtype=bool)
        
        muscle_power = np.zeros(raw_chunk.shape[0])
        
        for ch in range(self.n_channels):
            # Apply high-frequency filter
            muscle_signal, self._muscle_filter_zi[ch] = signal.lfilter(
                self._muscle_filter.b,
                self._muscle_filter.a,
                raw_chunk[:, ch],
                zi=self._muscle_filter_zi[ch]
            )
            # Compute power (RMS) in sliding window
            window_size = int(0.1 * self.sample_rate)  # 100ms window
            if window_size < raw_chunk.shape[0]:
                for i in range(raw_chunk.shape[0] - window_size):
                    window_power = np.mean(muscle_signal[i:i+window_size]**2)
                    muscle_power[i + window_size // 2] = max(muscle_power[i + window_size // 2], window_power)
            else:
                muscle_power = np.mean(muscle_signal**2)
        
        # Normalize by channel variance
        channel_vars = np.var(raw_chunk, axis=0, ddof=1)
        avg_var = np.mean(channel_vars) if len(channel_vars) > 0 else 1.0
        normalized_power = muscle_power / (avg_var + 1e-8)
        
        # Threshold: muscle artifacts have high normalized power
        return normalized_power > self.muscle_artifact_threshold
    
    def detect_blink_artifact(self, raw_chunk: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Detect eye blink artifacts using low-frequency transients (0.5-4 Hz).
        
        Eye blinks manifest as large low-frequency deflections, especially
        in frontal channels. For SSVEP, we focus on occipital channels but
        can still detect blinks that affect all channels.
        
        Args:
            raw_chunk: Raw EEG chunk (n_samples, n_channels) - before bandpass
            
        Returns:
            Boolean array (n_samples,) indicating blink artifacts
        """
        if not self.enable_blink_detection or self._blink_filter is None:
            return np.zeros(raw_chunk.shape[0], dtype=bool)
        
        blink_signal = np.zeros(raw_chunk.shape[0])
        
        for ch in range(self.n_channels):
            # Apply low-frequency filter
            blink_ch, self._blink_filter_zi[ch] = signal.lfilter(
                self._blink_filter.b,
                self._blink_filter.a,
                raw_chunk[:, ch],
                zi=self._blink_filter_zi[ch]
            )
            # Use maximum absolute value across channels
            blink_signal = np.maximum(blink_signal, np.abs(blink_ch))
        
        # Normalize by channel std dev
        channel_stds = np.std(raw_chunk, axis=0, ddof=1)
        avg_std = np.mean(channel_stds) if len(channel_stds) > 0 else 1.0
        normalized_blink = blink_signal / (avg_std + 1e-8)
        
        # Threshold: blinks have large normalized deflections
        return normalized_blink > self.blink_threshold
    
    def get_buffer(self) -> NDArray[np.float64]:
        """
        Get the current buffer contents.
        
        Returns:
            Array of shape (buffer_size, n_channels)
        """
        return np.array(self._buffer)
    
    def get_buffer_for_channel(self, channel: int) -> NDArray[np.float64]:
        """
        Get buffer contents for a single channel.
        
        Args:
            channel: Channel index
            
        Returns:
            Array of shape (buffer_size,)
        """
        return np.array([s[channel] for s in self._buffer])
    
    def get_recent_data_minimal(self, seconds: float) -> NDArray[np.float64]:
        """
        Get recent data with minimal preprocessing (CAR + notch only, NO bandpass).
        
        Used for Filter Bank CCA where filter banks handle frequency selection.
        Gets data from _raw_buffer which stores CAR + notch only.
        
        Args:
            seconds: How many seconds of data to retrieve
            
        Returns:
            Data array of shape (n_samples, 3) - occipital channels only (PO7, Oz, PO8)
        """
        n_samples = min(int(seconds * self.sample_rate), self._buffer_size)
        buffer_array = np.array(list(self._raw_buffer))
        data = buffer_array[-n_samples:]
        
        # Extract only occipital channels (PO7, Oz, PO8 = indices 5, 6, 7)
        if len(data) > 0 and data.shape[1] >= 8:
            occipital_indices = [5, 6, 7]
            return data[:, occipital_indices]
        elif len(data) > 0 and data.shape[1] == 3:
            # Already occipital channels
            return data
        else:
            return data
    
    def get_recent_data(self, seconds: float) -> NDArray[np.float64]:
        """
        Get the most recent N seconds of data (occipital channels only).
        
        Args:
            seconds: Number of seconds of data to retrieve
            
        Returns:
            Array of shape (n_samples, 3) - occipital channels only (PO7, Oz, PO8)
        """
        n_samples = min(int(seconds * self.sample_rate), self._buffer_size)
        buffer_array = self.get_buffer()
        data = buffer_array[-n_samples:]
        
        # Extract only occipital channels (PO7, Oz, PO8 = indices 5, 6, 7)
        if len(data) > 0 and data.shape[1] >= 8:
            occipital_indices = [5, 6, 7]
            return data[:, occipital_indices]
        elif len(data) > 0 and data.shape[1] == 3:
            # Already occipital channels (from previous processing)
            return data
        else:
            # Fallback: return as-is if structure is unexpected
            return data
    
    def compute_psd(
        self,
        channel: int,
        window_seconds: float = 1.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute power spectral density for a channel.
        
        Args:
            channel: Channel index
            window_seconds: Window size in seconds
            
        Returns:
            Tuple of (frequencies, psd_values)
        """
        data = self.get_buffer_for_channel(channel)
        n_samples = int(window_seconds * self.sample_rate)
        
        if len(data) < n_samples:
            n_samples = len(data)
        
        # Use last n_samples
        data = data[-n_samples:]
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(
            data,
            fs=self.sample_rate,
            nperseg=min(256, n_samples),
            noverlap=min(128, n_samples // 2)
        )
        
        return freqs, psd
    
    def get_band_power(
        self,
        channel: int,
        freq_low: float,
        freq_high: float,
        window_seconds: float = 1.0
    ) -> float:
        """
        Get power in a specific frequency band.
        
        Args:
            channel: Channel index
            freq_low: Lower frequency bound
            freq_high: Upper frequency bound
            window_seconds: Window size
            
        Returns:
            Band power value
        """
        freqs, psd = self.compute_psd(channel, window_seconds)
        
        # Find indices for frequency range
        idx = np.where((freqs >= freq_low) & (freqs <= freq_high))[0]
        
        if len(idx) == 0:
            return 0.0
        
        # Return mean power in band
        return np.mean(psd[idx])


@dataclass
class SimulatedEEGSource:
    """
    Simulated EEG source for testing without real hardware.
    
    Generates synthetic EEG with embedded SSVEP responses.
    
    Attributes:
        sample_rate: Sampling rate in Hz
        n_channels: Number of channels
        target_frequency: The frequency the "user" is attending to
        snr: Signal-to-noise ratio for SSVEP signal
    """
    sample_rate: float = 256.0
    n_channels: int = 8
    target_frequency: Optional[float] = None  # None = no attention
    snr: float = 0.5  # SSVEP signal strength relative to noise
    
    # Internal state
    _time: float = field(default=0.0, repr=False)
    _phase: float = field(default=0.0, repr=False)
    
    def set_target(self, frequency: Optional[float]) -> None:
        """Set the frequency the simulated user is attending to."""
        self.target_frequency = frequency
    
    def generate_sample(self) -> NDArray[np.float64]:
        """
        Generate one sample of simulated EEG data.
        
        Returns:
            Array of shape (n_channels,)
        """
        # Base noise (pink noise approximation)
        noise = np.random.randn(self.n_channels) * 0.5
        
        # Add 1/f characteristic
        noise *= (1 + 0.5 * np.sin(2 * np.pi * 10 * self._time))
        
        # Add SSVEP response if attending to a target
        if self.target_frequency is not None:
            ssvep = self.snr * np.sin(2 * np.pi * self.target_frequency * self._time)
            # SSVEP is strongest in occipital channels (last few channels)
            ssvep_weights = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])[:self.n_channels]
            noise += ssvep * ssvep_weights
        
        # Update time
        self._time += 1.0 / self.sample_rate
        
        return noise
    
    def generate_chunk(self, n_samples: int) -> NDArray[np.float64]:
        """
        Generate multiple samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, n_channels)
        """
        return np.array([self.generate_sample() for _ in range(n_samples)])
    
    def reset(self) -> None:
        """Reset the time counter."""
        self._time = 0.0


@dataclass
class LSLPreprocessor(EEGPreprocessor):
    """
    EEG Preprocessor with integrated LSL streaming.
    
    Combines LSL data reception with real-time preprocessing
    for the g.tec Unicorn Black.
    
    Attributes:
        All attributes from EEGPreprocessor plus:
        use_lsl: Whether to use LSL streaming (vs simulated)
        stream_name: LSL stream name to connect to
    """
    use_lsl: bool = True
    stream_name: Optional[str] = None
    
    # LSL components
    _lsl_receiver = None
    _lsl_connected: bool = field(default=False, repr=False)
    
    def connect_lsl(self, stream_name: Optional[str] = None) -> bool:
        """
        Connect to an LSL stream.
        
        Args:
            stream_name: Name of stream (None = auto-detect)
            
        Returns:
            True if connected
        """
        try:
            from .lsl_stream import LSLReceiver
            
            self._lsl_receiver = LSLReceiver(
                stream_name=stream_name,
                buffer_seconds=self.buffer_seconds
            )
            
            if self._lsl_receiver.connect(stream_name):
                # Update sample rate from stream
                self.sample_rate = self._lsl_receiver.sample_rate
                
                # Store the actual stream channel count for extraction
                self._stream_n_channels = self._lsl_receiver.n_channels
                
                # ALWAYS use 8 EEG channels for processing (Unicorn sends 17 total)
                # We'll extract only the first 8 (EEG) channels during pull_and_process
                self.n_channels = 8  # Force 8 EEG channels
                
                print(f"[LSL] Stream has {self._stream_n_channels} channels, using first 8 (EEG)")
                print(f"[LSL] Analysis restricted to occipital channels: PO7, Oz, PO8 (indices 5, 6, 7)")
                
                # Reinitialize filters for correct sample rate and 8 channels
                self._init_filters()
                self._init_buffer()
                
                # Start receiving
                self._lsl_receiver.start_receiving()
                self._lsl_connected = True
                
                print(f"LSL connected: {self._lsl_receiver.stream_info}")
                return True
            
            return False
            
        except ImportError:
            print("LSL not available")
            return False
    
    def disconnect_lsl(self) -> None:
        """Disconnect from LSL stream."""
        if self._lsl_receiver:
            self._lsl_receiver.disconnect()
            self._lsl_connected = False
    
    @property
    def is_lsl_connected(self) -> bool:
        """Check if LSL is connected."""
        return self._lsl_connected and self._lsl_receiver is not None
    
    def pull_and_process(self, n_samples: int = 16) -> NDArray[np.float64]:
        """
        Pull data from LSL and process it.
        
        LSL receiver already extracts EEG channels (first 8) immediately after pulling.
        This method processes the EEG data and returns only occipital channels.
        
        Args:
            n_samples: Number of samples to pull
            
        Returns:
            Processed EEG data (occipital channels only: PO7, Oz, PO8 = 3 channels)
        """
        if not self.is_lsl_connected:
            return np.array([])
        
        # Pull from LSL (already contains only EEG channels)
        samples, timestamps = self._lsl_receiver.pull_chunk(n_samples)
        
        if len(samples) == 0:
            return np.array([])
        
        # Process all EEG channels (needed for CAR)
        processed = self.process_chunk(samples)
        
        # Extract only occipital channels (PO7, Oz, PO8 = indices 5, 6, 7)
        # SSVEP is strongest in occipital cortex
        if processed.shape[1] >= 8:
            occipital_indices = [5, 6, 7]  # PO7, Oz, PO8
            return processed[:, occipital_indices]
        else:
            # Fallback: return all channels if structure is unexpected
            return processed
    
    def get_lsl_buffer(self, seconds: float = 1.0) -> NDArray[np.float64]:
        """
        Get processed data from LSL buffer (occipital channels only).
        
        Args:
            seconds: How many seconds of data
        
        Returns:
            Processed buffer data (occipital channels only: PO7, Oz, PO8 = 3 channels)
        """
        if not self.is_lsl_connected:
            # For non-LSL mode, get recent data and extract occipital
            data = self.get_recent_data(seconds)
            if len(data) > 0 and data.shape[1] >= 8:
                occipital_indices = [5, 6, 7]  # PO7, Oz, PO8
                return data[:, occipital_indices]
            return data
        
        # Get raw from LSL (already contains only EEG channels)
        raw_data = self._lsl_receiver.get_recent_data(seconds)
        
        if len(raw_data) == 0:
            return np.array([])
        
        # Process the chunk (all EEG channels needed for CAR)
        processed = self.process_chunk(raw_data)
        
        # Extract only occipital channels (PO7, Oz, PO8 = indices 5, 6, 7)
        if processed.shape[1] >= 8:
            occipital_indices = [5, 6, 7]
            return processed[:, occipital_indices]
        else:
            return processed


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("Testing EEG preprocessing pipeline...")
    
    # Create preprocessor
    preprocessor = EEGPreprocessor(sample_rate=256, n_channels=8)
    
    # Create simulated source attending to 15Hz
    source = SimulatedEEGSource(sample_rate=256, n_channels=8, target_frequency=15.0)
    
    # Process 2 seconds of data
    n_samples = int(2 * 256)
    for _ in range(n_samples):
        sample = source.generate_sample()
        processed = preprocessor.process_sample(sample)
    
    # Check band powers
    power_10hz = preprocessor.get_band_power(7, 9, 11)  # Occipital channel
    power_15hz = preprocessor.get_band_power(7, 14, 16)
    
    print(f"10Hz band power: {power_10hz:.4f}")
    print(f"15Hz band power: {power_15hz:.4f}")
    print(f"15Hz/10Hz ratio: {power_15hz/power_10hz:.2f}")
    
    # Test LSL preprocessor
    print("\n" + "=" * 50)
    print("Testing LSL Preprocessor...")
    
    lsl_prep = LSLPreprocessor(sample_rate=250, n_channels=8)
    
    print("Attempting LSL connection...")
    if lsl_prep.connect_lsl():
        print("Connected! Receiving data...")
        import time
        time.sleep(2)
        
        data = lsl_prep.get_lsl_buffer(1.0)
        print(f"Received {len(data)} samples")
        
        lsl_prep.disconnect_lsl()
    else:
        print("No LSL stream found (this is normal without hardware)")
