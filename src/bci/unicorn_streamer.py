"""
Unicorn Black EEG Streamer for BCI-UPIC.

Streams data from g.tec Unicorn Black directly to LSL using the Unicorn Python API.
This avoids needing the Unicorn Suite GUI running.

SETUP INSTRUCTIONS:
==================
1. Install the Unicorn Suite from g.tec (required for drivers)
   https://www.gtec.at/product/unicorn-suite/

2. The UnicornPy module is included with Unicorn Suite installation.
   Find it at: C:\\Program Files\\gtec\\Unicorn Suite\\Hybrid Black\\Unicorn Python\\
   
3. Add to your Python path or copy the module:
   - Copy 'Lib' folder contents to your site-packages, OR
   - Add the path to sys.path before importing

4. Pair your Unicorn Black via Bluetooth before running.

USAGE:
======
    from src.bci.unicorn_streamer import UnicornLSLStreamer
    
    streamer = UnicornLSLStreamer()
    streamer.start()  # Starts streaming to LSL
    # ... your BCI code ...
    streamer.stop()
"""

from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from threading import Thread, Event

# Try to import UnicornPy from common installation paths
UNICORN_PATHS = [
    r"C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib",
    r"C:\Program Files (x86)\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib",
]

UNICORN_AVAILABLE = False
UnicornPy = None

for path in UNICORN_PATHS:
    if Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)

try:
    import UnicornPy
    UNICORN_AVAILABLE = True
except ImportError:
    pass

# LSL imports
try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


# Unicorn Black specifications
UNICORN_SAMPLE_RATE = 250  # Hz
UNICORN_N_EEG_CHANNELS = 8
UNICORN_N_TOTAL_CHANNELS = 17  # 8 EEG + accelerometer + gyroscope + battery + counter + validation

UNICORN_EEG_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
UNICORN_ALL_CHANNELS = [
    'Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8',  # EEG
    'AccX', 'AccY', 'AccZ',  # Accelerometer
    'GyroX', 'GyroY', 'GyroZ',  # Gyroscope
    'Battery', 'Counter', 'Validation'  # Status
]


@dataclass
class UnicornLSLStreamer:
    """
    Streams Unicorn Black EEG data to LSL.
    
    This allows your BCI application to receive data via LSL
    without running the Unicorn Suite GUI.
    
    Attributes:
        stream_name: Name for the LSL stream
        serial_number: Unicorn serial number (None = auto-detect)
        eeg_only: If True, only stream EEG channels (not accelerometer etc.)
    """
    stream_name: str = "Unicorn"
    serial_number: Optional[str] = None
    eeg_only: bool = True
    
    # Internal state
    _device = None
    _outlet: Optional[StreamOutlet] = field(default=None, repr=False)
    _streaming: bool = field(default=False, repr=False)
    _thread: Optional[Thread] = field(default=None, repr=False)
    _stop_event: Event = field(default_factory=Event, repr=False)
    
    @staticmethod
    def is_available() -> bool:
        """Check if Unicorn streaming is available."""
        return UNICORN_AVAILABLE and LSL_AVAILABLE
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """
        Get list of available Unicorn devices.
        
        Returns:
            List of device serial numbers
        """
        if not UNICORN_AVAILABLE:
            print("UnicornPy not available. Install Unicorn Suite first.")
            return []
        
        try:
            device_list = UnicornPy.GetAvailableDevices(True)
            return list(device_list) if device_list else []
        except Exception as e:
            print(f"Error getting devices: {e}")
            return []
    
    def connect(self) -> bool:
        """
        Connect to the Unicorn device.
        
        Returns:
            True if connected successfully
        """
        if not UNICORN_AVAILABLE:
            print("UnicornPy not available.")
            print("Install Unicorn Suite and ensure UnicornPy is in your Python path.")
            return False
        
        if not LSL_AVAILABLE:
            print("pylsl not available.")
            return False
        
        try:
            # Get available devices
            devices = self.get_available_devices()
            
            if not devices:
                print("No Unicorn devices found. Make sure:")
                print("  1. Unicorn Black is powered on")
                print("  2. Bluetooth is paired")
                return False
            
            # Select device
            if self.serial_number:
                if self.serial_number not in devices:
                    print(f"Device {self.serial_number} not found.")
                    print(f"Available: {devices}")
                    return False
                device_serial = self.serial_number
            else:
                device_serial = devices[0]
                print(f"Auto-selected device: {device_serial}")
            
            # Connect to device
            self._device = UnicornPy.Unicorn(device_serial)
            print(f"Connected to Unicorn: {device_serial}")
            
            # Create LSL outlet
            n_channels = UNICORN_N_EEG_CHANNELS if self.eeg_only else UNICORN_N_TOTAL_CHANNELS
            channel_names = UNICORN_EEG_CHANNELS if self.eeg_only else UNICORN_ALL_CHANNELS
            
            info = StreamInfo(
                name=self.stream_name,
                type='EEG',
                channel_count=n_channels,
                nominal_srate=UNICORN_SAMPLE_RATE,
                channel_format='float32',
                source_id=f'unicorn-{device_serial}'
            )
            
            # Add channel metadata
            channels = info.desc().append_child("channels")
            for ch_name in channel_names:
                ch = channels.append_child("channel")
                ch.append_child_value("label", ch_name)
                ch.append_child_value("unit", "microvolts" if ch_name in UNICORN_EEG_CHANNELS else "")
                ch.append_child_value("type", "EEG" if ch_name in UNICORN_EEG_CHANNELS else "AUX")
            
            self._outlet = StreamOutlet(info)
            print(f"LSL stream created: {self.stream_name}")
            
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start streaming data to LSL.
        
        Returns:
            True if streaming started
        """
        if self._device is None:
            if not self.connect():
                return False
        
        if self._streaming:
            print("Already streaming")
            return True
        
        try:
            # Start acquisition
            self._device.StartAcquisition(False)  # False = no test signal
            self._streaming = True
            
            # Start streaming thread
            self._stop_event.clear()
            self._thread = Thread(target=self._stream_loop, daemon=True)
            self._thread.start()
            
            print("Streaming started")
            return True
            
        except Exception as e:
            print(f"Start error: {e}")
            return False
    
    def stop(self) -> None:
        """Stop streaming and disconnect."""
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._device and self._streaming:
            try:
                self._device.StopAcquisition()
            except:
                pass
            self._streaming = False
        
        if self._device:
            try:
                del self._device
            except:
                pass
            self._device = None
        
        self._outlet = None
        print("Streaming stopped")
    
    def _stream_loop(self) -> None:
        """Background thread for continuous data streaming."""
        # Buffer for one sample (all channels)
        buffer_length = UNICORN_N_TOTAL_CHANNELS * 4  # 4 bytes per float
        
        while not self._stop_event.is_set():
            try:
                # Get data from device
                data = self._device.GetData(buffer_length)
                
                # Convert to numpy array
                sample = np.frombuffer(data, dtype=np.float32)
                
                # Extract EEG channels if eeg_only
                if self.eeg_only:
                    sample = sample[:UNICORN_N_EEG_CHANNELS]
                
                # Push to LSL
                self._outlet.push_sample(sample.tolist())
                
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Stream error: {e}")
                break
    
    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming


def check_unicorn_setup():
    """Check and report Unicorn setup status."""
    print("=" * 50)
    print("Unicorn Black Setup Check")
    print("=" * 50)
    
    # Check UnicornPy
    print(f"\n1. UnicornPy available: {UNICORN_AVAILABLE}")
    if not UNICORN_AVAILABLE:
        print("   -> Install Unicorn Suite from g.tec")
        print("   -> Add UnicornPy to Python path:")
        for path in UNICORN_PATHS:
            print(f"      {path}")
    
    # Check LSL
    print(f"\n2. pylsl available: {LSL_AVAILABLE}")
    if not LSL_AVAILABLE:
        print("   -> pip install pylsl")
    
    # Check devices
    if UNICORN_AVAILABLE:
        print("\n3. Searching for devices...")
        devices = UnicornLSLStreamer.get_available_devices()
        if devices:
            print(f"   Found {len(devices)} device(s):")
            for d in devices:
                print(f"      - {d}")
        else:
            print("   No devices found. Check:")
            print("      - Unicorn is powered on")
            print("      - Bluetooth is paired")
    
    print("\n" + "=" * 50)


def run_streamer():
    """Run the Unicorn LSL streamer from command line."""
    check_unicorn_setup()
    
    if not UnicornLSLStreamer.is_available():
        print("\nCannot start streamer - missing dependencies")
        return
    
    streamer = UnicornLSLStreamer()
    
    if streamer.start():
        print("\nStreaming... Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            streamer.stop()


if __name__ == "__main__":
    run_streamer()
