"""
P300 Oddball Stimulus module for BCI-UPIC.

Provides discrete flash stimuli for P300 ERP-based BCI oddball paradigm.
Two targets flash discretely with color cycling - red is the oddball target.

P300 Oddball paradigm:
- Discrete flashes (100-200ms duration)
- Inter-stimulus interval: 500-1000ms
- Color cycling: Red (target/oddball), Blue, Green, Yellow, etc. (non-targets)
- Red never appears on both top and bottom simultaneously
- ERP peaks around 300ms post-stimulus for attended (red) targets
- Epoching: -100ms to +800ms relative to stimulus onset
"""

from __future__ import annotations

import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Dict
from enum import Enum


class FlashState(Enum):
    """State of a P300 flash target."""
    IDLE = 0      # Not flashing, waiting
    FLASHING = 1  # Currently flashing (ON)
    POST_FLASH = 2  # Just finished flash, in post-flash period


# Color definitions for P300 oddball paradigm
COLORS = {
    'red': (255, 0, 0),        # Target/oddball - attended
    'blue': (0, 0, 255),       # Non-target
    'green': (0, 255, 0),      # Non-target
    'yellow': (255, 255, 0),  # Non-target
    'cyan': (0, 255, 255),    # Non-target
    'magenta': (255, 0, 255), # Non-target
    'orange': (255, 165, 0),  # Non-target
    'purple': (128, 0, 128),  # Non-target
}

NON_TARGET_COLORS = ['blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
TARGET_COLOR = 'red'


@dataclass
class P300FlashTarget:
    """
    A single P300 flash target with color cycling.
    
    Attributes:
        position: Screen position ('top' or 'bottom')
        flash_duration_ms: Duration of each flash in milliseconds (default 150ms)
        isi_ms: Inter-stimulus interval in milliseconds (default 750ms)
        current_color: Current color name (for this flash)
        color_off: RGB color when idle
        size: Size of the target (width, height) in pixels
    """
    position: str = "top"
    flash_duration_ms: int = 150  # 150ms flash duration
    isi_ms: int = 750  # 750ms inter-stimulus interval (total cycle = 900ms)
    current_color: str = "blue"  # Current color name
    color_off: Tuple[int, int, int] = (30, 30, 30)    # Dark gray idle
    size: Tuple[int, int] = (100, 100)
    
    # Internal state
    _state: FlashState = field(default=FlashState.IDLE, repr=False)
    _last_flash_time: float = field(default=0.0, repr=False)
    _flash_count: int = field(default=0, repr=False)
    
    def start(self) -> None:
        """Start the flash sequence."""
        self._last_flash_time = time.perf_counter()
        self._state = FlashState.IDLE
        self._flash_count = 0
    
    def update(self, current_time: Optional[float] = None) -> FlashState:
        """
        Update flash state based on timing.
        
        Args:
            current_time: Current time in seconds (uses perf_counter if None)
            
        Returns:
            Current FlashState
        """
        if current_time is None:
            current_time = time.perf_counter()
        
        elapsed_since_last = current_time - self._last_flash_time
        elapsed_ms = elapsed_since_last * 1000
        
        if self._state == FlashState.IDLE:
            # Check if it's time to flash
            if elapsed_ms >= self.isi_ms:
                self._state = FlashState.FLASHING
                self._last_flash_time = current_time
                self._flash_count += 1
        elif self._state == FlashState.FLASHING:
            # Check if flash duration has elapsed
            if elapsed_ms >= self.flash_duration_ms:
                self._state = FlashState.POST_FLASH
        elif self._state == FlashState.POST_FLASH:
            # Check if ISI has elapsed (ready for next flash)
            if elapsed_ms >= self.isi_ms:
                self._state = FlashState.IDLE
                self._last_flash_time = current_time
        
        return self._state
    
    def get_state(self, current_time: Optional[float] = None) -> FlashState:
        """Get current flash state."""
        return self.update(current_time)
    
    def get_color(self, current_time: Optional[float] = None) -> Tuple[int, int, int]:
        """Get current color based on flash state."""
        state = self.get_state(current_time)
        if state == FlashState.FLASHING:
            return COLORS.get(self.current_color, COLORS['blue'])
        return self.color_off
    
    def set_color(self, color_name: str) -> None:
        """Set the color for the next flash."""
        self.current_color = color_name
    
    def is_target(self) -> bool:
        """Check if current color is the target (red)."""
        return self.current_color == TARGET_COLOR
    
    def is_flashing(self, current_time: Optional[float] = None) -> bool:
        """Check if target is currently flashing."""
        return self.get_state(current_time) == FlashState.FLASHING
    
    def get_flash_times(self) -> List[float]:
        """Get list of flash onset times (for epoching)."""
        # This will be populated by the stimulus system
        return []


@dataclass
class P300Stimulus:
    """
    Complete P300 oddball stimulus system with color cycling.
    
    The paradigm uses:
    - Top target: Flashes discretely with color cycling
    - Bottom target: Flashes discretely with color cycling
    - Red is the oddball target (rare, attended)
    - Other colors are non-targets (frequent, ignored)
    - Red NEVER appears on both top and bottom simultaneously
    - ERP extraction: -100ms to +800ms around each flash
    
    Attributes:
        flash_duration_ms: Duration of each flash in milliseconds
        isi_ms: Inter-stimulus interval in milliseconds
        duration: Total stimulus duration in seconds
        target_probability: Probability of red (target) appearing (default 0.2 = 20%)
    """
    flash_duration_ms: int = 150
    isi_ms: int = 750
    duration: float = 10.0
    target_probability: float = 0.2  # 20% chance of red (oddball)
    
    # Targets
    top_target: P300FlashTarget = field(init=False)
    bottom_target: P300FlashTarget = field(init=False)
    
    # State
    _running: bool = field(default=False, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _elapsed_time: float = field(default=0.0, repr=False)
    
    # Flash timing tracking (for epoching) - now includes color
    _flash_onsets: List[Tuple[str, str, float]] = field(default_factory=list, repr=False)  # (position, color, time)
    _last_flash_position: Optional[str] = field(default=None, repr=False)
    _last_top_color: Optional[str] = field(default=None, repr=False)
    _last_bottom_color: Optional[str] = field(default=None, repr=False)
    
    # Callbacks
    on_flash: Optional[Callable[[str, float], None]] = None  # (position, timestamp)
    on_completion: Optional[Callable[[], None]] = None
    
    def __post_init__(self) -> None:
        """Initialize the flash targets."""
        self.top_target = P300FlashTarget(
            position="top",
            flash_duration_ms=self.flash_duration_ms,
            isi_ms=self.isi_ms,
            current_color="blue",
            color_off=(40, 40, 40),
            size=(100, 100)
        )
        
        self.bottom_target = P300FlashTarget(
            position="bottom",
            flash_duration_ms=self.flash_duration_ms,
            isi_ms=self.isi_ms,
            current_color="green",
            color_off=(40, 40, 40),
            size=(100, 100)
        )
    
    def _select_colors(self) -> Tuple[str, str]:
        """
        Select colors for top and bottom targets.
        
        Ensures red never appears on both simultaneously.
        Red appears with target_probability on each position independently.
        
        Returns:
            Tuple of (top_color, bottom_color)
        """
        # Decide if each position should show red
        top_is_red = random.random() < self.target_probability
        bottom_is_red = random.random() < self.target_probability
        
        # CRITICAL: If both would be red, randomly choose one
        if top_is_red and bottom_is_red:
            if random.random() < 0.5:
                top_is_red = True
                bottom_is_red = False
            else:
                top_is_red = False
                bottom_is_red = True
        
        # Select colors
        top_color = TARGET_COLOR if top_is_red else random.choice(NON_TARGET_COLORS)
        bottom_color = TARGET_COLOR if bottom_is_red else random.choice(NON_TARGET_COLORS)
        
        return (top_color, bottom_color)
    
    def start(self) -> None:
        """Start the stimulus presentation."""
        start_timestamp = time.perf_counter()
        self._start_time = start_timestamp
        self._running = True
        self._elapsed_time = 0.0
        self._flash_onsets.clear()
        self._last_flash_position = None
        
        # Start both targets
        self.top_target.start()
        self.bottom_target.start()
        
        # Offset bottom target by half ISI for alternating pattern
        if self.flash_mode == "alternating":
            self.bottom_target._last_flash_time = start_timestamp + (self.isi_ms / 2000.0)
    
    def stop(self) -> None:
        """Stop the stimulus presentation."""
        self._running = False
    
    def reset(self) -> None:
        """Reset the stimulus to initial state."""
        self._running = False
        self._elapsed_time = 0.0
        self._start_time = 0.0
        self._flash_onsets.clear()
        self._last_flash_position = None
    
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
    
    def update(self) -> Tuple[FlashState, FlashState]:
        """
        Update stimulus state and return current target states.
        
        Returns:
            Tuple of (top_state, bottom_state)
        """
        if not self._running:
            return (FlashState.IDLE, FlashState.IDLE)
        
        current_time = self.elapsed_time
        
        # Check if duration has elapsed
        if current_time >= self.duration:
            self._running = False
            self._elapsed_time = self.duration
            if self.on_completion:
                self.on_completion()
            return (FlashState.IDLE, FlashState.IDLE)
        
        # Update targets
        top_state = self.top_target.update(current_time)
        bottom_state = self.bottom_target.update(current_time)
        
        # Select new colors when targets enter IDLE state (ready for next flash)
        if top_state == FlashState.IDLE and self._last_top_color != self.top_target.current_color:
            top_color, bottom_color = self._select_colors()
            self.top_target.set_color(top_color)
            self.bottom_target.set_color(bottom_color)
            self._last_top_color = top_color
            self._last_bottom_color = bottom_color
        
        # Track flash onsets (for epoching) - now includes color
        if top_state == FlashState.FLASHING and self._last_flash_position != "top":
            absolute_time = time.perf_counter()
            color = self.top_target.current_color
            self._flash_onsets.append(("top", color, absolute_time))
            self._last_flash_position = "top"
            if self.on_flash:
                self.on_flash("top", absolute_time)
        
        if bottom_state == FlashState.FLASHING and self._last_flash_position != "bottom":
            absolute_time = time.perf_counter()
            color = self.bottom_target.current_color
            self._flash_onsets.append(("bottom", color, absolute_time))
            self._last_flash_position = "bottom"
            if self.on_flash:
                self.on_flash("bottom", absolute_time)
        
        return (top_state, bottom_state)
    
    def get_colors(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get current colors for both targets.
        
        Returns:
            Tuple of (top_color_rgb, bottom_color_rgb)
        """
        current_time = self.elapsed_time
        return (
            self.top_target.get_color(current_time),
            self.bottom_target.get_color(current_time)
        )
    
    def get_flash_onsets(self) -> List[Tuple[str, str, float]]:
        """
        Get list of flash onset times (position, color, absolute_time).
        
        Used for epoching EEG data around each flash.
        """
        return self._flash_onsets.copy()
    
    def get_target_flashes(self) -> List[Tuple[str, float]]:
        """
        Get only target (red) flash onsets.
        
        Returns:
            List of (position, absolute_time) for red flashes only
        """
        return [(pos, time) for pos, color, time in self._flash_onsets if color == TARGET_COLOR]
