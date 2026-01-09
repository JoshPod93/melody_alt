"""
BCI Cursor Controller for BCI-UPIC.

Controls the cursor/playhead based on BCI classification results.
- Horizontal movement: Automatic (playhead moves left to right over duration)
- Vertical movement: BCI-controlled (up/down based on SSVEP attention)

The pen never lifts - it continuously draws as the playhead moves.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum

from .classifier import ClassificationResult, AttentionTarget


class ControllerState(Enum):
    """State of the BCI controller."""
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    COMPLETED = 3


@dataclass
class CursorPosition:
    """Current position of the cursor."""
    time: float      # Horizontal position (0 to duration)
    pitch: float     # Vertical position (0 to 1, 0.5 is center)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.time, self.pitch)


@dataclass
class BCICursorController:
    """
    Controls cursor movement based on BCI input.
    
    The cursor automatically moves horizontally (playhead), while
    vertical movement is controlled by BCI classification results.
    
    Attributes:
        duration: Total composition duration in seconds
        vertical_speed: Speed of vertical movement (units per second)
        pitch_min: Minimum pitch value (default 0)
        pitch_max: Maximum pitch value (default 1)
        start_pitch: Starting pitch position (default 0.5 = center)
        smoothing: Amount of smoothing for vertical movement (0-1)
    """
    duration: float = 10.0
    vertical_speed: float = 0.15  # Pitch units per second at full confidence
    pitch_min: float = 0.0
    pitch_max: float = 1.0
    start_pitch: float = 0.5
    smoothing: float = 0.3  # Exponential smoothing factor
    
    # Current state
    _state: ControllerState = field(default=ControllerState.IDLE, repr=False)
    _current_time: float = field(default=0.0, repr=False)
    _current_pitch: float = field(default=0.5, repr=False)
    _velocity: float = field(default=0.0, repr=False)
    
    # Timing
    _start_timestamp: float = field(default=0.0, repr=False)
    _last_update: float = field(default=0.0, repr=False)
    
    # Trail of positions (the "score" being drawn)
    _trail: List[CursorPosition] = field(default_factory=list, repr=False)
    _trail_sample_rate: float = 60.0  # Samples per second for trail
    _last_trail_time: float = field(default=0.0, repr=False)
    
    # Callbacks
    on_position_update: Optional[Callable[[CursorPosition], None]] = None
    on_completion: Optional[Callable[[List[CursorPosition]], None]] = None
    on_state_change: Optional[Callable[[ControllerState], None]] = None
    
    def __post_init__(self) -> None:
        """Initialize cursor position."""
        self._current_pitch = self.start_pitch
    
    @property
    def state(self) -> ControllerState:
        """Get current controller state."""
        return self._state
    
    @property
    def position(self) -> CursorPosition:
        """Get current cursor position."""
        return CursorPosition(time=self._current_time, pitch=self._current_pitch)
    
    @property
    def progress(self) -> float:
        """Get progress as fraction (0-1)."""
        return self._current_time / self.duration
    
    @property
    def trail(self) -> List[CursorPosition]:
        """Get the recorded trail of positions."""
        return self._trail.copy()
    
    @property
    def is_running(self) -> bool:
        """Check if controller is currently running."""
        return self._state == ControllerState.RUNNING
    
    def start(self) -> None:
        """Start the cursor movement."""
        self._state = ControllerState.RUNNING
        self._start_timestamp = time.perf_counter()
        self._last_update = self._start_timestamp
        self._current_time = 0.0
        self._current_pitch = self.start_pitch
        self._velocity = 0.0
        self._trail.clear()
        self._last_trail_time = 0.0
        
        # Record initial position
        self._trail.append(self.position)
        
        if self.on_state_change:
            self.on_state_change(self._state)
    
    def stop(self) -> None:
        """Stop the cursor movement."""
        self._state = ControllerState.IDLE
        
        if self.on_state_change:
            self.on_state_change(self._state)
    
    def pause(self) -> None:
        """Pause the cursor movement."""
        if self._state == ControllerState.RUNNING:
            self._state = ControllerState.PAUSED
            
            if self.on_state_change:
                self.on_state_change(self._state)
    
    def resume(self) -> None:
        """Resume paused cursor movement."""
        if self._state == ControllerState.PAUSED:
            self._state = ControllerState.RUNNING
            self._last_update = time.perf_counter()
            
            if self.on_state_change:
                self.on_state_change(self._state)
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = ControllerState.IDLE
        self._current_time = 0.0
        self._current_pitch = self.start_pitch
        self._velocity = 0.0
        self._trail.clear()
        
        if self.on_state_change:
            self.on_state_change(self._state)
    
    def update(self, classification: Optional[ClassificationResult] = None) -> CursorPosition:
        """
        Update cursor position based on elapsed time and BCI input.
        
        Args:
            classification: Optional classification result from SSVEP classifier
            
        Returns:
            Current cursor position
        """
        if self._state != ControllerState.RUNNING:
            return self.position
        
        # Calculate time delta
        current_timestamp = time.perf_counter()
        dt = current_timestamp - self._last_update
        self._last_update = current_timestamp
        
        # Update horizontal position (automatic playhead)
        self._current_time = current_timestamp - self._start_timestamp
        
        # Check if completed
        if self._current_time >= self.duration:
            self._current_time = self.duration
            self._state = ControllerState.COMPLETED
            
            # Record final position
            self._trail.append(self.position)
            
            if self.on_completion:
                self.on_completion(self._trail)
            
            if self.on_state_change:
                self.on_state_change(self._state)
            
            return self.position
        
        # Update vertical position based on BCI input
        if classification is not None:
            target_velocity = self._calculate_velocity(classification)
            
            # Apply smoothing
            self._velocity = (self.smoothing * target_velocity + 
                            (1 - self.smoothing) * self._velocity)
        
        # Apply velocity to pitch
        self._current_pitch += self._velocity * dt
        
        # Clamp pitch to valid range
        self._current_pitch = max(self.pitch_min, min(self.pitch_max, self._current_pitch))
        
        # Record position to trail at fixed sample rate
        if self._current_time - self._last_trail_time >= 1.0 / self._trail_sample_rate:
            self._trail.append(self.position)
            self._last_trail_time = self._current_time
        
        # Notify position update
        if self.on_position_update:
            self.on_position_update(self.position)
        
        return self.position
    
    def _calculate_velocity(self, classification: ClassificationResult) -> float:
        """
        Calculate target velocity from classification result.
        
        Args:
            classification: SSVEP classification result
            
        Returns:
            Target velocity (positive = up, negative = down)
        """
        if classification.target == AttentionTarget.NONE:
            return 0.0
        
        # Base velocity scaled by confidence
        velocity = self.vertical_speed * classification.confidence
        
        # Direction based on target
        if classification.target == AttentionTarget.DOWN:
            velocity = -velocity
        
        return velocity
    
    def simulate_input(self, direction: str, confidence: float = 0.8) -> None:
        """
        Simulate BCI input for testing.
        
        Args:
            direction: "up", "down", or "none"
            confidence: Confidence level (0-1)
        """
        if direction == "up":
            target = AttentionTarget.UP
        elif direction == "down":
            target = AttentionTarget.DOWN
        else:
            target = AttentionTarget.NONE
        
        result = ClassificationResult(
            target=target,
            confidence=confidence,
            power_15hz=0.5 if target == AttentionTarget.UP else 0.3,
            power_10hz=0.5 if target == AttentionTarget.DOWN else 0.3,
            raw_score=confidence if target == AttentionTarget.UP else -confidence
        )
        
        self.update(result)
    
    def get_trail_as_tuples(self) -> List[Tuple[float, float]]:
        """Get trail as list of (time, pitch) tuples."""
        return [pos.to_tuple() for pos in self._trail]
    
    def get_trail_interpolated(self, num_points: int) -> List[Tuple[float, float]]:
        """
        Get trail interpolated to a fixed number of points.
        
        Useful for converting to arc points.
        
        Args:
            num_points: Number of points in output
            
        Returns:
            List of (time, pitch) tuples
        """
        if len(self._trail) < 2:
            return [(0.0, self.start_pitch), (self.duration, self.start_pitch)]
        
        # Extract times and pitches
        times = np.array([p.time for p in self._trail])
        pitches = np.array([p.pitch for p in self._trail])
        
        # Create interpolation points
        new_times = np.linspace(0, self.duration, num_points)
        new_pitches = np.interp(new_times, times, pitches)
        
        return list(zip(new_times.tolist(), new_pitches.tolist()))


@dataclass
class RandomController:
    """
    Random cursor controller for testing/validation.
    
    Generates random cursor movements to test the system
    without real BCI input.
    
    Attributes:
        duration: Total duration in seconds
        change_interval: How often to change direction (seconds)
        vertical_speed: Speed of movement
    """
    duration: float = 10.0
    change_interval: float = 1.0
    vertical_speed: float = 0.15
    
    # Internal state
    _controller: BCICursorController = field(init=False, repr=False)
    _current_direction: AttentionTarget = field(default=AttentionTarget.NONE, repr=False)
    _last_change: float = field(default=0.0, repr=False)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the underlying controller."""
        self._controller = BCICursorController(
            duration=self.duration,
            vertical_speed=self.vertical_speed
        )
    
    def start(self) -> None:
        """Start random movement."""
        self._controller.start()
        self._last_change = 0.0
        self._pick_new_direction()
    
    def _pick_new_direction(self) -> None:
        """Randomly select a new direction."""
        choices = [AttentionTarget.UP, AttentionTarget.DOWN, AttentionTarget.NONE]
        weights = [0.4, 0.4, 0.2]  # Slight bias towards movement
        self._current_direction = self._rng.choice(choices, p=weights)
    
    def update(self) -> CursorPosition:
        """Update with random input."""
        if not self._controller.is_running:
            return self._controller.position
        
        # Check if time to change direction
        if self._controller._current_time - self._last_change >= self.change_interval:
            self._pick_new_direction()
            self._last_change = self._controller._current_time
        
        # Create classification result
        confidence = self._rng.uniform(0.5, 0.9)
        result = ClassificationResult(
            target=self._current_direction,
            confidence=confidence,
            power_15hz=0.5,
            power_10hz=0.5,
            raw_score=0.0
        )
        
        return self._controller.update(result)
    
    @property
    def trail(self) -> List[CursorPosition]:
        """Get recorded trail."""
        return self._controller.trail
    
    @property
    def is_running(self) -> bool:
        """Check if running."""
        return self._controller.is_running
    
    @property
    def state(self) -> ControllerState:
        """Get current state."""
        return self._controller.state


if __name__ == "__main__":
    # Test cursor controller
    print("Testing BCI cursor controller...")
    
    controller = BCICursorController(duration=5.0)
    controller.start()
    
    # Simulate some updates
    import time as time_module
    
    start = time_module.perf_counter()
    while controller.is_running:
        # Simulate alternating up/down attention
        elapsed = time_module.perf_counter() - start
        if int(elapsed) % 2 == 0:
            controller.simulate_input("up", 0.7)
        else:
            controller.simulate_input("down", 0.7)
        
        time_module.sleep(0.016)  # ~60 FPS
    
    print(f"Trail has {len(controller.trail)} points")
    print(f"Final position: time={controller.position.time:.2f}, pitch={controller.position.pitch:.2f}")
