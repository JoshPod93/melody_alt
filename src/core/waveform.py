"""
Waveform (Wave Table) module for UPIC.

A waveform defines one period of a sound - the basic timbre/tone color.
In the original UPIC, waveforms had 4096 samples per period.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable
from dataclasses import dataclass, field
from enum import Enum


class WaveformType(Enum):
    """Standard waveform types."""
    SINE = "sine"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    SQUARE = "square"
    CUSTOM = "custom"


@dataclass
class Waveform:
    """
    A single-period waveform (wave table).
    
    Attributes:
        name: Human-readable identifier
        samples: The waveform data, normalized to [-1, 1]
        waveform_type: The type of waveform
    """
    name: str
    samples: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4096))
    waveform_type: WaveformType = WaveformType.CUSTOM
    
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
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(self.samples))
        if max_val > 0:
            self.samples = self.samples / max_val
    
    @classmethod
    def sine(cls, name: str = "Sine") -> Waveform:
        """Create a sine wave."""
        phase = np.linspace(0, 2 * np.pi, cls.TABLE_SIZE, endpoint=False)
        samples = np.sin(phase)
        return cls(name=name, samples=samples, waveform_type=WaveformType.SINE)
    
    @classmethod
    def triangle(cls, name: str = "Triangle") -> Waveform:
        """Create a triangle wave."""
        phase = np.linspace(0, 1, cls.TABLE_SIZE, endpoint=False)
        samples = 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1
        return cls(name=name, samples=samples, waveform_type=WaveformType.TRIANGLE)
    
    @classmethod
    def sawtooth(cls, name: str = "Sawtooth") -> Waveform:
        """Create a sawtooth wave."""
        phase = np.linspace(0, 1, cls.TABLE_SIZE, endpoint=False)
        samples = 2 * (phase - np.floor(phase + 0.5))
        return cls(name=name, samples=samples, waveform_type=WaveformType.SAWTOOTH)
    
    @classmethod
    def square(cls, name: str = "Square") -> Waveform:
        """Create a square wave."""
        phase = np.linspace(0, 1, cls.TABLE_SIZE, endpoint=False)
        samples = np.sign(np.sin(2 * np.pi * phase))
        return cls(name=name, samples=samples, waveform_type=WaveformType.SQUARE)
    
    @classmethod
    def from_function(cls, func: Callable[[NDArray], NDArray], name: str = "Custom") -> Waveform:
        """
        Create a waveform from a mathematical function.
        
        Args:
            func: Function that takes phase (0 to 2π) and returns amplitude (-1 to 1)
            name: Name for the waveform
        """
        phase = np.linspace(0, 2 * np.pi, cls.TABLE_SIZE, endpoint=False)
        samples = func(phase)
        return cls(name=name, samples=samples, waveform_type=WaveformType.CUSTOM)
    
    @classmethod
    def from_points(cls, points: list[tuple[float, float]], name: str = "Drawn") -> Waveform:
        """
        Create a waveform from hand-drawn points.
        
        Args:
            points: List of (x, y) tuples where x is in [0, 1] and y is in [-1, 1]
            name: Name for the waveform
        """
        if len(points) < 2:
            return cls(name=name, samples=np.zeros(cls.TABLE_SIZE))
        
        # Sort points by x coordinate
        points = sorted(points, key=lambda p: p[0])
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        
        # Interpolate to full table size
        x_full = np.linspace(0, 1, cls.TABLE_SIZE)
        samples = np.interp(x_full, x_coords, y_coords)
        
        return cls(name=name, samples=samples, waveform_type=WaveformType.CUSTOM)
    
    @classmethod
    def from_harmonics(
        cls, 
        harmonics: list[tuple[int, float, float]], 
        name: str = "Harmonic"
    ) -> Waveform:
        """
        Create a waveform from harmonic components (additive synthesis).
        
        Args:
            harmonics: List of (harmonic_number, amplitude, phase) tuples
            name: Name for the waveform
        """
        phase = np.linspace(0, 2 * np.pi, cls.TABLE_SIZE, endpoint=False)
        samples = np.zeros(cls.TABLE_SIZE)
        
        for harmonic_num, amplitude, harmonic_phase in harmonics:
            samples += amplitude * np.sin(harmonic_num * phase + harmonic_phase)
        
        return cls(name=name, samples=samples, waveform_type=WaveformType.CUSTOM)
    
    def get_sample(self, phase: float) -> float:
        """
        Get interpolated sample at a given phase.
        
        Args:
            phase: Phase in range [0, 1) representing position in the cycle
            
        Returns:
            Interpolated amplitude value
        """
        # Wrap phase to [0, 1)
        phase = phase % 1.0
        
        # Calculate index and interpolation factor
        index_float = phase * self.TABLE_SIZE
        index = int(index_float)
        frac = index_float - index
        
        # Linear interpolation between samples
        next_index = (index + 1) % self.TABLE_SIZE
        return self.samples[index] * (1 - frac) + self.samples[next_index] * frac
    
    def get_samples_at_phases(self, phases: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get interpolated samples at multiple phases (vectorized).
        
        Args:
            phases: Array of phases in range [0, 1)
            
        Returns:
            Array of interpolated amplitude values
        """
        # Wrap phases to [0, 1)
        phases = phases % 1.0
        
        # Calculate indices and interpolation factors
        index_float = phases * self.TABLE_SIZE
        indices = index_float.astype(int)
        fracs = index_float - indices
        
        # Linear interpolation
        next_indices = (indices + 1) % self.TABLE_SIZE
        return self.samples[indices] * (1 - fracs) + self.samples[next_indices] * fracs
    
    def copy(self) -> Waveform:
        """Create a copy of this waveform."""
        return Waveform(
            name=f"{self.name} (copy)",
            samples=self.samples.copy(),
            waveform_type=self.waveform_type
        )


class WaveformLibrary:
    """
    A collection of waveforms (the "palette" in UPIC terms).
    Original UPIC supported 64 waveforms.
    """
    
    MAX_WAVEFORMS: int = 64
    
    def __init__(self) -> None:
        self.waveforms: dict[str, Waveform] = {}
        self._init_default_waveforms()
    
    def _init_default_waveforms(self) -> None:
        """Initialize with standard waveforms."""
        self.add(Waveform.sine())
        self.add(Waveform.triangle())
        self.add(Waveform.sawtooth())
        self.add(Waveform.square())
    
    def add(self, waveform: Waveform) -> bool:
        """
        Add a waveform to the library.
        
        Returns:
            True if added, False if library is full
        """
        if len(self.waveforms) >= self.MAX_WAVEFORMS:
            return False
        
        # Ensure unique name
        name = waveform.name
        counter = 1
        while name in self.waveforms:
            name = f"{waveform.name}_{counter}"
            counter += 1
        waveform.name = name
        
        self.waveforms[name] = waveform
        return True
    
    def remove(self, name: str) -> bool:
        """Remove a waveform by name."""
        if name in self.waveforms:
            del self.waveforms[name]
            return True
        return False
    
    def get(self, name: str) -> Waveform | None:
        """Get a waveform by name."""
        return self.waveforms.get(name)
    
    def list_names(self) -> list[str]:
        """Get list of all waveform names."""
        return list(self.waveforms.keys())
    
    def __len__(self) -> int:
        return len(self.waveforms)
    
    def __iter__(self):
        return iter(self.waveforms.values())

