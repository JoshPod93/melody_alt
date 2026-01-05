"""
Envelope module for UPIC.

An envelope defines the amplitude evolution of a sound over time.
In the original UPIC, envelopes had 4096 samples and 128 envelopes were available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class EnvelopeType(Enum):
    """Standard envelope types."""
    ADSR = "adsr"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


@dataclass
class Envelope:
    """
    An amplitude envelope.
    
    The envelope is stored as a table of amplitude values (0 to 1) that
    gets stretched to match the duration of an arc during playback.
    
    Attributes:
        name: Human-readable identifier
        samples: The envelope data, normalized to [0, 1]
        envelope_type: The type of envelope
    """
    name: str
    samples: NDArray[np.float64] = field(default_factory=lambda: np.ones(4096))
    envelope_type: EnvelopeType = EnvelopeType.CUSTOM
    
    # Default table size matching original UPIC
    TABLE_SIZE: int = 4096
    
    def __post_init__(self) -> None:
        """Ensure samples array is the correct size and normalized."""
        if len(self.samples) != self.TABLE_SIZE:
            # Resample to correct size
            self.samples = np.interp(
                np.linspace(0, 1, self.TABLE_SIZE),
                np.linspace(0, 1, len(self.samples)),
                self.samples
            )
        # Clamp to [0, 1]
        self.samples = np.clip(self.samples, 0, 1)
    
    @classmethod
    def constant(cls, name: str = "Constant", level: float = 1.0) -> Envelope:
        """Create a constant envelope (no amplitude change)."""
        samples = np.full(cls.TABLE_SIZE, level)
        return cls(name=name, samples=samples, envelope_type=EnvelopeType.LINEAR)
    
    @classmethod
    def adsr(
        cls,
        name: str = "ADSR",
        attack: float = 0.1,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.2,
        sustain_level: float = 0.8
    ) -> Envelope:
        """
        Create an ADSR envelope.
        
        Args:
            attack: Attack time as fraction of total duration (0-1)
            decay: Decay time as fraction of total duration (0-1)
            sustain: Sustain time as fraction of total duration (0-1)
            release: Release time as fraction of total duration (0-1)
            sustain_level: Amplitude level during sustain (0-1)
        
        Note: attack + decay + sustain + release should equal 1.0
        """
        # Normalize times
        total = attack + decay + sustain + release
        if total > 0:
            attack /= total
            decay /= total
            sustain /= total
            release /= total
        
        samples = np.zeros(cls.TABLE_SIZE)
        
        # Calculate sample boundaries
        attack_end = int(attack * cls.TABLE_SIZE)
        decay_end = int((attack + decay) * cls.TABLE_SIZE)
        sustain_end = int((attack + decay + sustain) * cls.TABLE_SIZE)
        
        # Attack: 0 to 1
        if attack_end > 0:
            samples[:attack_end] = np.linspace(0, 1, attack_end)
        
        # Decay: 1 to sustain_level
        if decay_end > attack_end:
            samples[attack_end:decay_end] = np.linspace(1, sustain_level, decay_end - attack_end)
        
        # Sustain: constant at sustain_level
        if sustain_end > decay_end:
            samples[decay_end:sustain_end] = sustain_level
        
        # Release: sustain_level to 0
        if cls.TABLE_SIZE > sustain_end:
            samples[sustain_end:] = np.linspace(sustain_level, 0, cls.TABLE_SIZE - sustain_end)
        
        return cls(name=name, samples=samples, envelope_type=EnvelopeType.ADSR)
    
    @classmethod
    def linear_fade(
        cls,
        name: str = "Linear Fade",
        fade_in: float = 0.1,
        fade_out: float = 0.1
    ) -> Envelope:
        """
        Create a simple fade in/out envelope.
        
        Args:
            fade_in: Fade in time as fraction of total (0-1)
            fade_out: Fade out time as fraction of total (0-1)
        """
        samples = np.ones(cls.TABLE_SIZE)
        
        fade_in_samples = int(fade_in * cls.TABLE_SIZE)
        fade_out_samples = int(fade_out * cls.TABLE_SIZE)
        
        if fade_in_samples > 0:
            samples[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
        
        if fade_out_samples > 0:
            samples[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
        
        return cls(name=name, samples=samples, envelope_type=EnvelopeType.LINEAR)
    
    @classmethod
    def exponential(
        cls,
        name: str = "Exponential",
        attack: float = 0.01,
        decay_rate: float = 3.0
    ) -> Envelope:
        """
        Create an exponential decay envelope (like a plucked string).
        
        Args:
            attack: Attack time as fraction of total (0-1)
            decay_rate: Rate of exponential decay (higher = faster decay)
        """
        samples = np.zeros(cls.TABLE_SIZE)
        attack_samples = int(attack * cls.TABLE_SIZE)
        
        # Attack
        if attack_samples > 0:
            samples[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Exponential decay
        decay_length = cls.TABLE_SIZE - attack_samples
        if decay_length > 0:
            t = np.linspace(0, 1, decay_length)
            samples[attack_samples:] = np.exp(-decay_rate * t)
        
        return cls(name=name, samples=samples, envelope_type=EnvelopeType.EXPONENTIAL)
    
    @classmethod
    def from_points(cls, points: List[Tuple[float, float]], name: str = "Drawn") -> Envelope:
        """
        Create an envelope from hand-drawn points.
        
        Args:
            points: List of (x, y) tuples where x is in [0, 1] and y is in [0, 1]
            name: Name for the envelope
        """
        if len(points) < 2:
            return cls(name=name, samples=np.ones(cls.TABLE_SIZE))
        
        # Sort points by x coordinate
        points = sorted(points, key=lambda p: p[0])
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        
        # Ensure we start at 0 and end at 1
        if x_coords[0] > 0:
            x_coords = np.insert(x_coords, 0, 0)
            y_coords = np.insert(y_coords, 0, y_coords[0])
        if x_coords[-1] < 1:
            x_coords = np.append(x_coords, 1)
            y_coords = np.append(y_coords, y_coords[-1])
        
        # Interpolate to full table size
        x_full = np.linspace(0, 1, cls.TABLE_SIZE)
        samples = np.interp(x_full, x_coords, y_coords)
        
        return cls(name=name, samples=samples, envelope_type=EnvelopeType.CUSTOM)
    
    @classmethod
    def from_breakpoints(
        cls,
        breakpoints: List[Tuple[float, float]],
        name: str = "Breakpoint"
    ) -> Envelope:
        """
        Create an envelope from breakpoints with linear interpolation.
        Same as from_points but with clearer semantics for ADSR-style editing.
        
        Args:
            breakpoints: List of (time, level) tuples, time in [0,1], level in [0,1]
            name: Name for the envelope
        """
        return cls.from_points(breakpoints, name)
    
    def get_amplitude(self, position: float) -> float:
        """
        Get interpolated amplitude at a given position.
        
        Args:
            position: Position in range [0, 1] representing progress through envelope
            
        Returns:
            Interpolated amplitude value (0 to 1)
        """
        # Clamp position to [0, 1]
        position = max(0, min(1, position))
        
        # Calculate index and interpolation factor
        index_float = position * (self.TABLE_SIZE - 1)
        index = int(index_float)
        frac = index_float - index
        
        # Handle edge case
        if index >= self.TABLE_SIZE - 1:
            return float(self.samples[-1])
        
        # Linear interpolation between samples
        return float(self.samples[index] * (1 - frac) + self.samples[index + 1] * frac)
    
    def get_amplitudes_at_positions(
        self, 
        positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Get interpolated amplitudes at multiple positions (vectorized).
        
        Args:
            positions: Array of positions in range [0, 1]
            
        Returns:
            Array of interpolated amplitude values
        """
        # Clamp positions to [0, 1]
        positions = np.clip(positions, 0, 1)
        
        # Calculate indices and interpolation factors
        index_float = positions * (self.TABLE_SIZE - 1)
        indices = index_float.astype(int)
        fracs = index_float - indices
        
        # Handle edge cases
        indices = np.minimum(indices, self.TABLE_SIZE - 2)
        
        # Linear interpolation
        return self.samples[indices] * (1 - fracs) + self.samples[indices + 1] * fracs
    
    def copy(self) -> Envelope:
        """Create a copy of this envelope."""
        return Envelope(
            name=f"{self.name} (copy)",
            samples=self.samples.copy(),
            envelope_type=self.envelope_type
        )
    
    def invert(self) -> Envelope:
        """Create an inverted copy (1 - amplitude)."""
        return Envelope(
            name=f"{self.name} (inverted)",
            samples=1.0 - self.samples,
            envelope_type=EnvelopeType.CUSTOM
        )
    
    def reverse(self) -> Envelope:
        """Create a time-reversed copy."""
        return Envelope(
            name=f"{self.name} (reversed)",
            samples=self.samples[::-1].copy(),
            envelope_type=EnvelopeType.CUSTOM
        )


class EnvelopeLibrary:
    """
    A collection of envelopes (the "palette" in UPIC terms).
    Original UPIC supported 128 envelopes.
    """
    
    MAX_ENVELOPES: int = 128
    
    def __init__(self) -> None:
        self.envelopes: dict[str, Envelope] = {}
        self._init_default_envelopes()
    
    def _init_default_envelopes(self) -> None:
        """Initialize with standard envelopes."""
        self.add(Envelope.constant())
        self.add(Envelope.adsr())
        self.add(Envelope.linear_fade(name="Fade In/Out"))
        self.add(Envelope.exponential(name="Pluck"))
        self.add(Envelope.adsr(
            name="Soft Attack",
            attack=0.3, decay=0.1, sustain=0.4, release=0.2
        ))
        self.add(Envelope.adsr(
            name="Percussive",
            attack=0.01, decay=0.2, sustain=0.0, release=0.79
        ))
    
    def add(self, envelope: Envelope) -> bool:
        """
        Add an envelope to the library.
        
        Returns:
            True if added, False if library is full
        """
        if len(self.envelopes) >= self.MAX_ENVELOPES:
            return False
        
        # Ensure unique name
        name = envelope.name
        counter = 1
        while name in self.envelopes:
            name = f"{envelope.name}_{counter}"
            counter += 1
        envelope.name = name
        
        self.envelopes[name] = envelope
        return True
    
    def remove(self, name: str) -> bool:
        """Remove an envelope by name."""
        if name in self.envelopes:
            del self.envelopes[name]
            return True
        return False
    
    def get(self, name: str) -> Envelope | None:
        """Get an envelope by name."""
        return self.envelopes.get(name)
    
    def list_names(self) -> list[str]:
        """Get list of all envelope names."""
        return list(self.envelopes.keys())
    
    def __len__(self) -> int:
        return len(self.envelopes)
    
    def __iter__(self):
        return iter(self.envelopes.values())

