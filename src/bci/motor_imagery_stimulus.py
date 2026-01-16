"""
Motor Imagery Stimulus module for BCI-UPIC.

Provides free-choice motor imagery BCI paradigm.
After baseline capture, user can freely imagine left or right hand movement at any time.

Motor Imagery paradigm:
- After baseline: Shows "Free Choice - Imagine LEFT or RIGHT"
- User can imagine either direction at any time
- No alternating instructions - continuous free choice
- Classification runs continuously on 2-second windows
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum


class InstructionState(Enum):
    """State of motor imagery instruction."""
    IDLE = 0           # No instruction shown
    FREE_CHOICE = 1    # Free choice mode - user can imagine left or right


@dataclass
class MotorImageryStimulus:
    """
    Motor imagery free-choice stimulus system.
    
    After baseline capture, displays free choice prompt allowing user to
    imagine left or right hand movement at any time.
    
    Attributes:
        duration: Total stimulus duration in seconds
        free_choice_text: Text to display during free choice period
    """
    duration: float = 30.0  # Total duration
    free_choice_text: str = "Free Choice - Imagine LEFT or RIGHT"
    
    # State
    _running: bool = field(default=False, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _elapsed_time: float = field(default=0.0, repr=False)
    _current_state: InstructionState = field(default=InstructionState.IDLE, repr=False)
    _current_instruction: Optional[str] = field(default=None, repr=False)
    
    def start(self) -> None:
        """Start the free choice period."""
        self._running = True
        self._start_time = time.perf_counter()
        self._elapsed_time = 0.0
        self._current_state = InstructionState.FREE_CHOICE
        self._current_instruction = self.free_choice_text
    
    def stop(self) -> None:
        """Stop the instruction sequence."""
        self._running = False
        self._current_state = InstructionState.IDLE
    
    def update(self, current_time: Optional[float] = None) -> Tuple[InstructionState, Optional[str]]:
        """
        Update stimulus state (free choice mode).
        
        Args:
            current_time: Current time in seconds (uses perf_counter if None)
            
        Returns:
            Tuple of (current_state, current_instruction_text)
        """
        if not self._running:
            return (InstructionState.IDLE, None)
        
        if current_time is None:
            current_time = time.perf_counter()
        
        self._elapsed_time = current_time - self._start_time
        
        # Check if duration exceeded
        if self._elapsed_time >= self.duration:
            self.stop()
            return (InstructionState.IDLE, None)
        
        # Free choice mode - show prompt continuously
        if self._current_state != InstructionState.FREE_CHOICE:
            self._current_state = InstructionState.FREE_CHOICE
            self._current_instruction = self.free_choice_text
        
        return (self._current_state, self._current_instruction)
    
    def get_current_instruction(self) -> Optional[str]:
        """Get current instruction text."""
        return self._current_instruction
    
    def get_instruction_onsets(self) -> List[Tuple[str, float]]:
        """
        Get list of instruction onsets.
        
        For free choice mode, returns empty list (no discrete instructions).
        
        Returns:
            List of (instruction, timestamp) tuples (empty for free choice)
        """
        return []  # No discrete instructions in free choice mode
    
    def is_running(self) -> bool:
        """Check if stimulus is running."""
        return self._running
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return self._elapsed_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time."""
        return max(0.0, self.duration - self._elapsed_time)
