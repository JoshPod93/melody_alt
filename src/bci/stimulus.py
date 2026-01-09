"""
SSVEP Stimulus module for BCI-UPIC.

Provides precise frequency flickering targets for SSVEP paradigm.
Two targets flicker at 15Hz (top) and 10Hz (bottom), perfectly out of phase.

The phase relationship is critical for SSVEP:
- 15Hz target: flickers every 66.67ms (period = 1/15s)
- 10Hz target: flickers every 100ms (period = 1/10s)
- Out of phase means when one is ON, the other is OFF at their respective midpoints
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable
from enum import Enum


class FlickerState(Enum):
    """State of a flickering target."""
    ON = 1
    OFF = 0


@dataclass
class FlickerTarget:
    """
    A single flickering SSVEP target.
    
    Attributes:
        frequency: Flicker frequency in Hz
        phase_offset: Phase offset in radians (0 to 2π)
        position: Screen position ('top' or 'bottom')
        color_on: RGB color when target is ON
        color_off: RGB color when target is OFF (typically black/dark)
        size: Size of the target (width, height) in pixels
    """
    frequency: float
    phase_offset: float = 0.0
    position: str = "top"
    color_on: Tuple[int, int, int] = (255, 255, 255)  # White
    color_off: Tuple[int, int, int] = (30, 30, 30)    # Dark gray
    size: Tuple[int, int] = (200, 80)
    
    # Internal state
    _start_time: float = field(default=0.0, repr=False)
    
    def start(self) -> None:
        """Start the flickering from current time."""
        self._start_time = time.perf_counter()
    
    def get_state(self, current_time: Optional[float] = None) -> FlickerState:
        """
        Get the current flicker state based on time.
        
        Uses a sine wave to determine ON/OFF state:
        - sin(2π * f * t + phase) > 0 → ON
        - sin(2π * f * t + phase) ≤ 0 → OFF
        
        Args:
            current_time: Time in seconds (uses perf_counter if None)
            
        Returns:
            FlickerState.ON or FlickerState.OFF
        """
        if current_time is None:
            current_time = time.perf_counter() - self._start_time
        
        # Calculate sine wave value
        phase = 2 * np.pi * self.frequency * current_time + self.phase_offset
        value = np.sin(phase)
        
        return FlickerState.ON if value > 0 else FlickerState.OFF
    
    def get_color(self, current_time: Optional[float] = None) -> Tuple[int, int, int]:
        """Get the current color based on flicker state."""
        state = self.get_state(current_time)
        return self.color_on if state == FlickerState.ON else self.color_off
    
    def get_intensity(self, current_time: Optional[float] = None) -> float:
        """
        Get continuous intensity value (0-1) for smoother rendering.
        
        Uses (sin + 1) / 2 to map to [0, 1] range.
        """
        if current_time is None:
            current_time = time.perf_counter() - self._start_time
        
        phase = 2 * np.pi * self.frequency * current_time + self.phase_offset
        return (np.sin(phase) + 1) / 2


@dataclass
class SSVEPStimulus:
    """
    Complete SSVEP stimulus system with two flickering targets.
    
    The paradigm uses:
    - Top target: 15Hz (associated with "move up")
    - Bottom target: 10Hz (associated with "move down")
    - Targets are perfectly out of phase (π radians offset)
    
    Attributes:
        top_frequency: Frequency of top target in Hz (default 15)
        bottom_frequency: Frequency of bottom target in Hz (default 10)
        duration: Total stimulus duration in seconds
        on_state_change: Optional callback when flicker state changes
    """
    top_frequency: float = 15.0
    bottom_frequency: float = 10.0
    duration: float = 10.0
    
    # Targets
    top_target: FlickerTarget = field(init=False)
    bottom_target: FlickerTarget = field(init=False)
    
    # State
    _running: bool = field(default=False, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _elapsed_time: float = field(default=0.0, repr=False)
    
    # Callbacks
    on_state_change: Optional[Callable[[str, FlickerState], None]] = None
    on_completion: Optional[Callable[[], None]] = None
    
    def __post_init__(self) -> None:
        """Initialize the flickering targets."""
        # Top target at 15Hz, phase 0 degrees (0 radians)
        self.top_target = FlickerTarget(
            frequency=self.top_frequency,
            phase_offset=0.0,  # 0 degrees
            position="top",
            color_on=(255, 255, 255),  # Bright white
            color_off=(40, 40, 40),     # Dark
            size=(300, 100)
        )
        
        # Bottom target at 10Hz, phase 180 degrees (π radians)
        self.bottom_target = FlickerTarget(
            frequency=self.bottom_frequency,
            phase_offset=np.pi,  # 180 degrees
            position="bottom",
            color_on=(255, 255, 255),
            color_off=(40, 40, 40),
            size=(300, 100)
        )
    
    def start(self) -> None:
        """Start the stimulus presentation."""
        self._start_time = time.perf_counter()
        self._running = True
        self._elapsed_time = 0.0
        self.top_target.start()
        self.bottom_target.start()
    
    def stop(self) -> None:
        """Stop the stimulus presentation."""
        self._running = False
    
    def reset(self) -> None:
        """Reset the stimulus to initial state."""
        self._running = False
        self._elapsed_time = 0.0
        self._start_time = 0.0
    
    @property
    def is_running(self) -> bool:
        """Check if stimulus is currently running."""
        return self._running
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self._running:
            return time.perf_counter() - self._start_time
        return self._elapsed_time
    
    @property
    def progress(self) -> float:
        """Get progress as fraction (0-1)."""
        return min(self.elapsed_time / self.duration, 1.0)
    
    @property
    def is_complete(self) -> bool:
        """Check if the stimulus duration has elapsed."""
        return self.elapsed_time >= self.duration
    
    def update(self) -> Tuple[FlickerState, FlickerState]:
        """
        Update stimulus state and return current target states.
        
        Returns:
            Tuple of (top_state, bottom_state)
        """
        if not self._running:
            return (FlickerState.OFF, FlickerState.OFF)
        
        current_time = self.elapsed_time
        
        # Check if duration has elapsed
        if current_time >= self.duration:
            self._running = False
            self._elapsed_time = self.duration
            if self.on_completion:
                self.on_completion()
            return (FlickerState.OFF, FlickerState.OFF)
        
        top_state = self.top_target.get_state(current_time)
        bottom_state = self.bottom_target.get_state(current_time)
        
        return (top_state, bottom_state)
    
    def get_colors(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get current colors for both targets.
        
        Returns:
            Tuple of (top_color, bottom_color)
        """
        current_time = self.elapsed_time
        return (
            self.top_target.get_color(current_time),
            self.bottom_target.get_color(current_time)
        )
    
    def get_intensities(self) -> Tuple[float, float]:
        """
        Get current intensities (0-1) for both targets.
        
        Returns:
            Tuple of (top_intensity, bottom_intensity)
        """
        current_time = self.elapsed_time
        return (
            self.top_target.get_intensity(current_time),
            self.bottom_target.get_intensity(current_time)
        )


def verify_frequency_timing() -> dict:
    """
    Verify that the flicker frequencies are accurate.
    
    This function measures actual flicker timing over 1 second
    and returns statistics for validation.
    
    Returns:
        Dictionary with timing statistics
    """
    stimulus = SSVEPStimulus()
    stimulus.start()
    
    # Track state changes
    top_changes = []
    bottom_changes = []
    prev_top = None
    prev_bottom = None
    
    start = time.perf_counter()
    while time.perf_counter() - start < 1.0:
        top_state, bottom_state = stimulus.update()
        
        if prev_top is not None and top_state != prev_top:
            top_changes.append(time.perf_counter() - start)
        if prev_bottom is not None and bottom_state != prev_bottom:
            bottom_changes.append(time.perf_counter() - start)
        
        prev_top = top_state
        prev_bottom = bottom_state
        
        time.sleep(0.001)  # 1ms resolution
    
    stimulus.stop()
    
    # Calculate actual frequencies
    top_freq = len(top_changes) / 2  # Divide by 2 because each cycle has 2 transitions
    bottom_freq = len(bottom_changes) / 2
    
    return {
        'target_top_hz': stimulus.top_frequency,
        'actual_top_hz': top_freq,
        'target_bottom_hz': stimulus.bottom_frequency,
        'actual_bottom_hz': bottom_freq,
        'top_transitions': len(top_changes),
        'bottom_transitions': len(bottom_changes)
    }


if __name__ == "__main__":
    # Test the stimulus timing
    print("Testing SSVEP stimulus timing...")
    stats = verify_frequency_timing()
    print(f"Top target: {stats['target_top_hz']}Hz target, {stats['actual_top_hz']}Hz actual")
    print(f"Bottom target: {stats['target_bottom_hz']}Hz target, {stats['actual_bottom_hz']}Hz actual")
