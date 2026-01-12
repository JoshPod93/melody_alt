"""
High-precision timer for Windows using multimedia timer API.

Provides better precision than QTimer on Windows for flicker protocols.
Uses Windows timeSetEvent API via ctypes.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from typing import Optional, Callable
import time

# Windows multimedia timer constants
TIME_ONESHOT = 0x0000
TIME_PERIODIC = 0x0001
TIME_CALLBACK_FUNCTION = 0x0000

# Load winmm.dll
try:
    winmm = ctypes.windll.winmm
except AttributeError:
    winmm = None


class HighPrecisionTimer:
    """
    High-precision timer using Windows multimedia timer API.
    
    Provides millisecond-level precision (typically 1-5ms) compared to
    QTimer's ~15-20ms precision on Windows.
    """
    
    def __init__(self, interval_ms: int, callback: Callable[[], None]):
        """
        Initialize high-precision timer.
        
        Args:
            interval_ms: Timer interval in milliseconds
            callback: Function to call on each timer fire
        """
        if winmm is None:
            raise RuntimeError("Windows multimedia timer API not available")
        
        self.interval_ms = interval_ms
        self.callback = callback
        self._timer_id: Optional[int] = None
        
        # Create callback function type
        self._callback_type = ctypes.WINFUNCTYPE(
            ctypes.wintypes.UINT,
            ctypes.wintypes.UINT,
            ctypes.wintypes.UINT,
            ctypes.POINTER(ctypes.wintypes.DWORD),
            ctypes.wintypes.UINT,
            ctypes.wintypes.UINT
        )
        
        # Wrap Python callback
        self._wrapped_callback = self._callback_type(self._timer_callback)
    
    def _timer_callback(self, uID, uMsg, dwUser, dw1, dw2):
        """Internal callback wrapper."""
        try:
            self.callback()
        except Exception as e:
            print(f"[HIGH_PRECISION_TIMER] Callback error: {e}")
        return 0
    
    def start(self) -> bool:
        """
        Start the timer.
        
        Returns:
            True if started successfully
        """
        if self._timer_id is not None:
            return False  # Already running
        
        if winmm is None:
            return False
        
        # timeSetEvent(interval, resolution, callback, user, event_type)
        timer_id = winmm.timeSetEvent(
            self.interval_ms,  # uDelay (milliseconds)
            1,  # uResolution (1ms = highest precision)
            self._wrapped_callback,  # lpTimeProc
            0,  # dwUser (user data)
            TIME_PERIODIC | TIME_CALLBACK_FUNCTION  # fuEvent
        )
        
        if timer_id == 0:
            return False
        
        self._timer_id = timer_id
        return True
    
    def stop(self) -> None:
        """Stop the timer."""
        if self._timer_id is not None and winmm is not None:
            winmm.timeKillEvent(self._timer_id)
            self._timer_id = None
    
    @property
    def is_running(self) -> bool:
        """Check if timer is running."""
        return self._timer_id is not None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


def is_available() -> bool:
    """Check if high-precision timer is available."""
    return winmm is not None
