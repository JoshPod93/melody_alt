"""
EEG Preprocessing module for BCI-UPIC.

Provides real-time signal processing for SSVEP detection:
- Bandpass filtering (focus on 8-20Hz for our 10Hz and 15Hz targets)
- Notch filtering for powerline noise (50/60Hz)
- Artifact rejection
- Signal normalization
- LSL stream integration for g.tec Unicorn Black

Designed to be lightweight for real-time processing.
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
    
    Optimized for detecting 10Hz and 15Hz SSVEP responses.
    
    Attributes:
        sample_rate: EEG sampling rate in Hz (default 256)
        n_channels: Number of EEG channels
        bandpass_low: Lower cutoff frequency (Hz)
        bandpass_high: Upper cutoff frequency (Hz)
        notch_freq: Powerline frequency for notch filter (50 or 60 Hz)
        buffer_seconds: Size of the rolling buffer in seconds
    """
    sample_rate: float = 256.0
    n_channels: int = 8
    bandpass_low: float = 5.0   # Include harmonics below target frequencies
    bandpass_high: float = 25.0  # Include first harmonic of 15Hz (30Hz)
    notch_freq: float = 50.0     # 50Hz for EU, 60Hz for US
    buffer_seconds: float = 2.0   # 2 second rolling buffer
    
    # Filter states
    _bandpass_filter: FilterCoefficients = field(init=False, repr=False)
    _notch_filter: FilterCoefficients = field(init=False, repr=False)
    
    # Rolling buffer for each channel
    _buffer: deque = field(init=False, repr=False)
    _buffer_size: int = field(init=False, repr=False)
    
    # Filter states for real-time filtering (per channel)
    _bp_zi: List[NDArray] = field(init=False, repr=False)
    _notch_zi: List[NDArray] = field(init=False, repr=False)
    
    # Artifact detection threshold (in standard deviations)
    artifact_threshold: float = 5.0
    
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
        Process a chunk of EEG data (multiple samples).
        
        Args:
            chunk: Array of shape (n_samples, n_channels)
            
        Returns:
            Filtered and normalized chunk of same shape
        """
        n_samples = chunk.shape[0]
        output = np.zeros_like(chunk)
        
        for i in range(n_samples):
            output[i] = self.process_sample(chunk[i])
        
        return output
    
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
        Detect if a sample contains artifacts.
        
        Args:
            sample: Normalized sample
            
        Returns:
            True if artifact detected
        """
        return np.any(np.abs(sample) > self.artifact_threshold)
    
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
    
    def get_recent_data(self, seconds: float) -> NDArray[np.float64]:
        """
        Get the most recent N seconds of data.
        
        Args:
            seconds: Number of seconds of data to retrieve
            
        Returns:
            Array of shape (n_samples, n_channels)
        """
        n_samples = min(int(seconds * self.sample_rate), self._buffer_size)
        buffer_array = self.get_buffer()
        return buffer_array[-n_samples:]
    
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
        
        Handles 17-channel Unicorn streams by extracting only EEG channels.
        
        Args:
            n_samples: Number of samples to pull
            
        Returns:
            Processed EEG data (8 channels)
        """
        if not self.is_lsl_connected:
            return np.array([])
        
        # Pull from LSL
        samples, timestamps = self._lsl_receiver.pull_chunk(n_samples)
        
        if len(samples) == 0:
            return np.array([])
        
        # ALWAYS extract first 8 EEG channels (Unicorn sends 17 total: 8 EEG + 9 aux)
        if samples.shape[1] > 8:
            samples = samples[:, :8]  # Take first 8 channels (EEG only)
        
        # Process
        return self.process_chunk(samples)
    
    def get_lsl_buffer(self, seconds: float = 1.0) -> NDArray[np.float64]:
        """
        Get processed data from LSL buffer.
        
        Args:
            seconds: How many seconds of data
            
        Returns:
            Processed buffer data
        """
        if not self.is_lsl_connected:
            return self.get_recent_data(seconds)
        
        # Get raw from LSL
        raw_data = self._lsl_receiver.get_recent_data(seconds)
        
        if len(raw_data) == 0:
            return np.array([])
        
        # ALWAYS extract first 8 EEG channels (Unicorn sends 17 total)
        if raw_data.shape[1] > 8:
            raw_data = raw_data[:, :8]
        
        # Process the chunk
        return self.process_chunk(raw_data)


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
