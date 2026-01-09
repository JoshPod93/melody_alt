"""
LSL (Lab Streaming Layer) integration for BCI-UPIC.

Provides real-time EEG data streaming from the g.tec Unicorn Black
and other LSL-compatible devices.

g.tec Unicorn Black specifications:
- 8 EEG channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)
- 250 Hz sampling rate
- 24-bit resolution
- Bluetooth connection
"""

from __future__ import annotations

import time
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple
from threading import Thread, Event
from collections import deque
import queue

try:
    from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_streams, resolve_byprop
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


# g.tec Unicorn Black channel configuration
UNICORN_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
UNICORN_SAMPLE_RATE = 250  # Hz
UNICORN_N_CHANNELS = 8

# Occipital channels (best for SSVEP)
OCCIPITAL_CHANNELS = ['PO7', 'Oz', 'PO8']
OCCIPITAL_INDICES = [5, 6, 7]  # Indices in Unicorn channel order


@dataclass
class LSLStreamInfo:
    """Information about an LSL stream."""
    name: str
    type: str
    channel_count: int
    sample_rate: float
    source_id: str
    
    def __str__(self) -> str:
        return f"{self.name} ({self.type}) - {self.channel_count}ch @ {self.sample_rate}Hz"


@dataclass
class LSLReceiver:
    """
    Receives EEG data from an LSL stream.
    
    Designed for the g.tec Unicorn Black but compatible with
    any LSL EEG stream.
    
    Attributes:
        stream_name: Name of the LSL stream to connect to
        stream_type: Type of stream (default 'EEG')
        buffer_seconds: Size of internal buffer in seconds
        timeout: Connection timeout in seconds
    """
    stream_name: Optional[str] = None  # None = auto-detect
    stream_type: str = "EEG"
    buffer_seconds: float = 5.0
    timeout: float = 10.0
    
    # Stream state
    _inlet: Optional[StreamInlet] = field(default=None, repr=False)
    _stream_info: Optional[LSLStreamInfo] = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)
    
    # Threading
    _receive_thread: Optional[Thread] = field(default=None, repr=False)
    _stop_event: Event = field(default_factory=Event, repr=False)
    
    # Data buffer (thread-safe queue)
    _data_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=10000), repr=False)
    _buffer: deque = field(default_factory=deque, repr=False)
    _buffer_size: int = field(default=0, repr=False)
    
    # Callbacks
    on_connect: Optional[Callable[[LSLStreamInfo], None]] = None
    on_disconnect: Optional[Callable[[], None]] = None
    on_data: Optional[Callable[[NDArray, float], None]] = None
    
    # Stream properties (set after connection)
    sample_rate: float = field(default=250.0, repr=False)
    n_channels: int = field(default=8, repr=False)
    channel_names: List[str] = field(default_factory=list, repr=False)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to a stream."""
        return self._connected and self._inlet is not None
    
    @property
    def stream_info(self) -> Optional[LSLStreamInfo]:
        """Get information about the connected stream."""
        return self._stream_info
    
    @staticmethod
    def list_streams(timeout: float = 2.0) -> List[LSLStreamInfo]:
        """
        List all available LSL streams.
        
        Args:
            timeout: How long to wait for streams
            
        Returns:
            List of available stream info
        """
        if not LSL_AVAILABLE:
            print("LSL not available")
            return []
        
        streams = resolve_streams(timeout)
        
        result = []
        for stream in streams:
            info = LSLStreamInfo(
                name=stream.name(),
                type=stream.type(),
                channel_count=stream.channel_count(),
                sample_rate=stream.nominal_srate(),
                source_id=stream.source_id()
            )
            result.append(info)
        
        return result
    
    @staticmethod
    def find_unicorn(timeout: float = 5.0) -> Optional[LSLStreamInfo]:
        """
        Find a g.tec Unicorn Black stream.
        
        Args:
            timeout: How long to search
            
        Returns:
            Stream info if found, None otherwise
        """
        if not LSL_AVAILABLE:
            return None
        
        # Try to find by name pattern
        streams = LSLReceiver.list_streams(timeout)
        
        for stream in streams:
            # Unicorn streams typically have 'Unicorn' in the name
            if 'unicorn' in stream.name.lower() or 'gtec' in stream.name.lower():
                return stream
            # Also check for 8-channel EEG streams at 250Hz
            if (stream.type == 'EEG' and 
                stream.channel_count == 8 and 
                abs(stream.sample_rate - 250) < 10):
                return stream
        
        return None
    
    def connect(self, stream_name: Optional[str] = None) -> bool:
        """
        Connect to an LSL stream.
        
        Args:
            stream_name: Name of stream to connect to (None = auto-detect)
            
        Returns:
            True if connected successfully
        """
        if not LSL_AVAILABLE:
            print("LSL not available - install pylsl")
            return False
        
        if self._connected:
            self.disconnect()
        
        try:
            # Find stream
            if stream_name:
                print(f"Looking for stream: {stream_name}")
                streams = resolve_byprop('name', stream_name, timeout=self.timeout)
            else:
                print("Auto-detecting EEG stream...")
                streams = resolve_byprop('type', self.stream_type, timeout=self.timeout)
            
            if not streams:
                print("No streams found")
                return False
            
            # Use first matching stream
            stream = streams[0]
            
            # Create inlet
            self._inlet = StreamInlet(stream, max_buflen=int(self.buffer_seconds))
            
            # Get stream info
            info = self._inlet.info()
            self._stream_info = LSLStreamInfo(
                name=info.name(),
                type=info.type(),
                channel_count=info.channel_count(),
                sample_rate=info.nominal_srate(),
                source_id=info.source_id()
            )
            
            # Set properties
            self.sample_rate = info.nominal_srate()
            self.n_channels = info.channel_count()
            
            # Get channel names
            self.channel_names = []
            ch = info.desc().child("channels").child("channel")
            for _ in range(self.n_channels):
                name = ch.child_value("label")
                self.channel_names.append(name if name else f"Ch{len(self.channel_names)+1}")
                ch = ch.next_sibling()
            
            # Initialize buffer
            self._buffer_size = int(self.buffer_seconds * self.sample_rate)
            self._buffer = deque(maxlen=self._buffer_size)
            
            self._connected = True
            print(f"Connected to: {self._stream_info}")
            
            if self.on_connect:
                self.on_connect(self._stream_info)
            
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the LSL stream."""
        self.stop_receiving()
        
        if self._inlet:
            self._inlet.close_stream()
            self._inlet = None
        
        self._connected = False
        self._stream_info = None
        
        if self.on_disconnect:
            self.on_disconnect()
    
    def start_receiving(self) -> None:
        """Start receiving data in a background thread."""
        if not self._connected:
            print("Not connected to a stream")
            return
        
        if self._receive_thread and self._receive_thread.is_alive():
            return  # Already receiving
        
        self._stop_event.clear()
        self._receive_thread = Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        print("Started receiving data")
    
    def stop_receiving(self) -> None:
        """Stop receiving data."""
        self._stop_event.set()
        
        if self._receive_thread:
            self._receive_thread.join(timeout=1.0)
            self._receive_thread = None
    
    def _receive_loop(self) -> None:
        """Background thread for receiving data."""
        while not self._stop_event.is_set():
            try:
                # Pull chunk of samples
                samples, timestamps = self._inlet.pull_chunk(timeout=0.1)
                
                if samples:
                    samples = np.array(samples)
                    timestamps = np.array(timestamps)
                    
                    # Add to buffer
                    for i in range(len(samples)):
                        self._buffer.append(samples[i])
                        
                        # Put in queue for external processing
                        try:
                            self._data_queue.put_nowait((samples[i], timestamps[i]))
                        except queue.Full:
                            pass  # Drop oldest if queue full
                    
                    # Callback
                    if self.on_data:
                        self.on_data(samples, timestamps[-1])
                        
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Receive error: {e}")
    
    def pull_sample(self, timeout: float = 0.0) -> Tuple[Optional[NDArray], Optional[float]]:
        """
        Pull a single sample from the queue.
        
        Args:
            timeout: How long to wait (0 = non-blocking)
            
        Returns:
            Tuple of (sample, timestamp) or (None, None) if no data
        """
        try:
            if timeout > 0:
                return self._data_queue.get(timeout=timeout)
            else:
                return self._data_queue.get_nowait()
        except queue.Empty:
            return None, None
    
    def pull_chunk(self, n_samples: int, timeout: float = 0.1) -> Tuple[NDArray, NDArray]:
        """
        Pull multiple samples from the queue.
        
        Args:
            n_samples: Number of samples to pull
            timeout: Total timeout
            
        Returns:
            Tuple of (samples, timestamps) arrays
        """
        samples = []
        timestamps = []
        
        start = time.time()
        while len(samples) < n_samples and (time.time() - start) < timeout:
            sample, ts = self.pull_sample(timeout=0.01)
            if sample is not None:
                samples.append(sample)
                timestamps.append(ts)
        
        if samples:
            return np.array(samples), np.array(timestamps)
        return np.array([]).reshape(0, self.n_channels), np.array([])
    
    def get_buffer(self) -> NDArray:
        """
        Get the current buffer contents.
        
        Returns:
            Array of shape (n_samples, n_channels)
        """
        return np.array(list(self._buffer))
    
    def get_recent_data(self, seconds: float) -> NDArray:
        """
        Get the most recent N seconds of data from buffer.
        
        Args:
            seconds: Number of seconds
            
        Returns:
            Array of shape (n_samples, n_channels)
        """
        n_samples = int(seconds * self.sample_rate)
        buffer = self.get_buffer()
        
        if len(buffer) < n_samples:
            return buffer
        return buffer[-n_samples:]
    
    def get_occipital_data(self, seconds: float = 1.0) -> NDArray:
        """
        Get recent data from occipital channels only (best for SSVEP).
        
        Args:
            seconds: Number of seconds
            
        Returns:
            Array of shape (n_samples, 3) for PO7, Oz, PO8
        """
        data = self.get_recent_data(seconds)
        
        if len(data) == 0:
            return data
        
        # Extract occipital channels
        if self.n_channels >= 8:
            return data[:, OCCIPITAL_INDICES]
        return data


@dataclass
class LSLMarkerSender:
    """
    Sends markers/triggers via LSL.
    
    Useful for marking events in the EEG recording:
    - Trial start/end
    - Stimulus onset
    - Classification results
    """
    stream_name: str = "BCI-UPIC-Markers"
    
    _outlet = None
    
    def __post_init__(self):
        """Initialize the marker outlet."""
        if not LSL_AVAILABLE:
            print("LSL not available for markers")
            return
        
        info = StreamInfo(
            name=self.stream_name,
            type='Markers',
            channel_count=1,
            nominal_srate=0,  # Irregular rate
            channel_format='string',
            source_id='bci-upic-markers'
        )
        
        self._outlet = StreamOutlet(info)
        print(f"Marker stream created: {self.stream_name}")
    
    def send(self, marker: str) -> None:
        """
        Send a marker.
        
        Args:
            marker: Marker string to send
        """
        if self._outlet:
            self._outlet.push_sample([marker])
    
    def send_trial_start(self, trial_id: int) -> None:
        """Send trial start marker."""
        self.send(f"Trial Start:{trial_id}")
    
    def send_trial_end(self, trial_id: int) -> None:
        """Send trial end marker."""
        self.send(f"Trial End:{trial_id}")
    
    def send_stimulus_onset(self, frequency: float) -> None:
        """Send stimulus onset marker."""
        self.send(f"Stimulus:{frequency}Hz")
    
    def send_classification(self, result: str, confidence: float) -> None:
        """Send classification result marker."""
        self.send(f"Classification:{result}:{confidence:.2f}")


class UnicornInterface:
    """
    High-level interface for the g.tec Unicorn Black.
    
    Combines LSL receiving with preprocessing for SSVEP detection.
    """
    
    def __init__(self):
        self.receiver = LSLReceiver()
        self.marker_sender = LSLMarkerSender()
        self._connected = False
    
    def connect(self, auto_detect: bool = True) -> bool:
        """
        Connect to the Unicorn Black.
        
        Args:
            auto_detect: If True, automatically find the Unicorn stream
            
        Returns:
            True if connected
        """
        if auto_detect:
            # Try to find Unicorn
            info = LSLReceiver.find_unicorn()
            if info:
                self._connected = self.receiver.connect(info.name)
            else:
                print("Unicorn not found. Available streams:")
                for s in LSLReceiver.list_streams():
                    print(f"  - {s}")
                return False
        else:
            self._connected = self.receiver.connect()
        
        if self._connected:
            self.receiver.start_receiving()
        
        return self._connected
    
    def disconnect(self) -> None:
        """Disconnect from the Unicorn."""
        self.receiver.disconnect()
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def get_eeg_chunk(self, n_samples: int = 64) -> NDArray:
        """
        Get a chunk of EEG data.
        
        Args:
            n_samples: Number of samples to get
            
        Returns:
            EEG data array
        """
        samples, _ = self.receiver.pull_chunk(n_samples)
        return samples
    
    def get_occipital_buffer(self, seconds: float = 1.0) -> NDArray:
        """
        Get buffered occipital channel data for SSVEP analysis.
        
        Args:
            seconds: How many seconds of data
            
        Returns:
            Occipital channel data
        """
        return self.receiver.get_occipital_data(seconds)


def test_lsl_connection():
    """Test LSL connection and data reception."""
    print("=" * 50)
    print("LSL Connection Test")
    print("=" * 50)
    
    if not LSL_AVAILABLE:
        print("ERROR: pylsl not installed")
        return
    
    # List available streams
    print("\nSearching for LSL streams...")
    streams = LSLReceiver.list_streams(timeout=3.0)
    
    if not streams:
        print("No LSL streams found.")
        print("\nTo test, you can:")
        print("1. Start the Unicorn Black with LSL streaming")
        print("2. Use a simulated stream (e.g., from OpenViBE)")
        return
    
    print(f"\nFound {len(streams)} stream(s):")
    for i, s in enumerate(streams):
        print(f"  {i+1}. {s}")
    
    # Try to connect to first EEG stream
    eeg_streams = [s for s in streams if s.type == 'EEG']
    if eeg_streams:
        print(f"\nConnecting to: {eeg_streams[0].name}")
        
        receiver = LSLReceiver()
        if receiver.connect(eeg_streams[0].name):
            receiver.start_receiving()
            
            print("Receiving data for 3 seconds...")
            time.sleep(3)
            
            data = receiver.get_buffer()
            print(f"Received {len(data)} samples")
            
            if len(data) > 0:
                print(f"Data shape: {data.shape}")
                print(f"Sample rate: {receiver.sample_rate} Hz")
                print(f"Channels: {receiver.channel_names}")
            
            receiver.disconnect()
    else:
        print("No EEG streams found")


if __name__ == "__main__":
    test_lsl_connection()
