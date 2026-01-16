"""
Frequency Table module for UPIC.

A frequency table maps the vertical axis of the page to actual frequencies.
This defines the "scale" or pitch space available for composition.

In the original UPIC, frequency tables had 16384 entries, allowing for
extremely fine pitch resolution (about 1/100th of a semitone).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class ScaleType(Enum):
    """Standard scale/tuning types."""
    CONTINUOUS = "continuous"  # Any frequency (glissando-friendly)
    EQUAL_TEMPERED = "equal_tempered"  # 12-TET
    MICROTONAL = "microtonal"  # Custom divisions of octave
    CUSTOM = "custom"  # Arbitrary frequency mapping


@dataclass
class FrequencyTable:
    """
    A frequency table that maps normalized positions (0-1) to frequencies (Hz).
    
    The vertical axis of the UPIC page is mapped through this table to
    determine the actual pitch of drawn arcs.
    
    Attributes:
        name: Human-readable identifier
        frequencies: Array of frequency values in Hz
        scale_type: The type of scale/tuning
        min_freq: Minimum frequency (Hz) - typically ~20 Hz
        max_freq: Maximum frequency (Hz) - typically ~20000 Hz
    """
    name: str
    frequencies: NDArray[np.float64] = field(default_factory=lambda: np.zeros(16384))
    scale_type: ScaleType = ScaleType.CONTINUOUS
    min_freq: float = 50.0
    max_freq: float = 2000.0
    
    # Default table size matching original UPIC (16K entries)
    TABLE_SIZE: int = 16384
    
    def __post_init__(self) -> None:
        """Ensure frequencies array is the correct size."""
        if len(self.frequencies) != self.TABLE_SIZE:
            # Resample to correct size
            self.frequencies = np.interp(
                np.linspace(0, 1, self.TABLE_SIZE),
                np.linspace(0, 1, len(self.frequencies)),
                self.frequencies
            )
    
    @classmethod
    def continuous(
        cls,
        name: str = "Continuous",
        min_freq: float = 50.0,
        max_freq: float = 2000.0,
        logarithmic: bool = True
    ) -> FrequencyTable:
        """
        Create a continuous frequency table (any pitch is accessible).
        
        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
            logarithmic: If True, use logarithmic spacing (perceptually linear)
        """
        if logarithmic:
            # Logarithmic spacing - perceptually uniform
            frequencies = np.logspace(
                np.log10(min_freq),
                np.log10(max_freq),
                cls.TABLE_SIZE
            )
        else:
            # Linear spacing
            frequencies = np.linspace(min_freq, max_freq, cls.TABLE_SIZE)
        
        return cls(
            name=name,
            frequencies=frequencies,
            scale_type=ScaleType.CONTINUOUS,
            min_freq=min_freq,
            max_freq=max_freq
        )
    
    @classmethod
    def equal_tempered(
        cls,
        name: str = "12-TET",
        base_freq: float = 440.0,  # A4
        base_midi: int = 69,  # A4 in MIDI
        min_midi: int = 21,  # A0
        max_midi: int = 108  # C8
    ) -> FrequencyTable:
        """
        Create a 12-tone equal tempered frequency table.
        
        Only the 12 chromatic pitches are accessible; intermediate
        positions snap to the nearest semitone.
        
        Args:
            base_freq: Reference frequency (usually A4 = 440 Hz)
            base_midi: MIDI note number of reference frequency
            min_midi: Lowest MIDI note
            max_midi: Highest MIDI note
        """
        # Calculate frequencies for each MIDI note
        midi_notes = np.arange(min_midi, max_midi + 1)
        note_freqs = base_freq * (2 ** ((midi_notes - base_midi) / 12))
        
        # Create stepped frequency table
        frequencies = np.zeros(cls.TABLE_SIZE)
        num_notes = len(midi_notes)
        
        for i in range(cls.TABLE_SIZE):
            # Map table position to note index
            note_idx = int((i / cls.TABLE_SIZE) * num_notes)
            note_idx = min(note_idx, num_notes - 1)
            frequencies[i] = note_freqs[note_idx]
        
        return cls(
            name=name,
            frequencies=frequencies,
            scale_type=ScaleType.EQUAL_TEMPERED,
            min_freq=note_freqs[0],
            max_freq=note_freqs[-1]
        )
    
    @classmethod
    def microtonal(
        cls,
        name: str = "Microtonal",
        divisions_per_octave: int = 24,  # Quarter tones
        base_freq: float = 440.0,
        num_octaves: float = 8.0
    ) -> FrequencyTable:
        """
        Create a microtonal frequency table with custom octave divisions.
        
        Args:
            divisions_per_octave: Number of equal divisions per octave
            base_freq: Reference frequency
            num_octaves: Number of octaves to span
        """
        # Calculate all pitches
        num_pitches = int(divisions_per_octave * num_octaves)
        pitch_indices = np.arange(num_pitches) - (num_pitches // 2)
        pitch_freqs = base_freq * (2 ** (pitch_indices / divisions_per_octave))
        
        # Create stepped frequency table
        frequencies = np.zeros(cls.TABLE_SIZE)
        
        for i in range(cls.TABLE_SIZE):
            pitch_idx = int((i / cls.TABLE_SIZE) * num_pitches)
            pitch_idx = min(pitch_idx, num_pitches - 1)
            frequencies[i] = pitch_freqs[pitch_idx]
        
        return cls(
            name=name,
            frequencies=frequencies,
            scale_type=ScaleType.MICROTONAL,
            min_freq=pitch_freqs[0],
            max_freq=pitch_freqs[-1]
        )
    
    @classmethod
    def from_ratios(
        cls,
        ratios: List[float],
        base_freq: float = 261.63,  # C4
        num_octaves: int = 4,
        name: str = "Just Intonation"
    ) -> FrequencyTable:
        """
        Create a frequency table from frequency ratios (just intonation, etc).
        
        Args:
            ratios: List of frequency ratios within one octave (e.g., [1, 9/8, 5/4, ...])
            base_freq: Base frequency for ratio 1.0
            num_octaves: Number of octaves to generate
            name: Name for the table
        """
        ratios = np.array(ratios)
        all_freqs = []
        
        # Generate frequencies for all octaves
        for octave in range(-num_octaves // 2, num_octaves // 2 + 1):
            octave_mult = 2 ** octave
            for ratio in ratios:
                all_freqs.append(base_freq * ratio * octave_mult)
        
        all_freqs = np.array(sorted(all_freqs))
        
        # Create stepped frequency table
        frequencies = np.zeros(cls.TABLE_SIZE)
        num_pitches = len(all_freqs)
        
        for i in range(cls.TABLE_SIZE):
            pitch_idx = int((i / cls.TABLE_SIZE) * num_pitches)
            pitch_idx = min(pitch_idx, num_pitches - 1)
            frequencies[i] = all_freqs[pitch_idx]
        
        return cls(
            name=name,
            frequencies=frequencies,
            scale_type=ScaleType.CUSTOM,
            min_freq=all_freqs[0],
            max_freq=all_freqs[-1]
        )
    
    @classmethod
    def from_frequencies(
        cls,
        freq_list: List[float],
        name: str = "Custom Scale"
    ) -> FrequencyTable:
        """
        Create a frequency table from an explicit list of frequencies.
        
        Args:
            freq_list: List of frequencies in Hz
            name: Name for the table
        """
        freq_list = np.array(sorted(freq_list))
        
        # Create stepped frequency table
        frequencies = np.zeros(cls.TABLE_SIZE)
        num_pitches = len(freq_list)
        
        for i in range(cls.TABLE_SIZE):
            pitch_idx = int((i / cls.TABLE_SIZE) * num_pitches)
            pitch_idx = min(pitch_idx, num_pitches - 1)
            frequencies[i] = freq_list[pitch_idx]
        
        return cls(
            name=name,
            frequencies=frequencies,
            scale_type=ScaleType.CUSTOM,
            min_freq=freq_list[0],
            max_freq=freq_list[-1]
        )
    
    def get_frequency(self, position: float) -> float:
        """
        Get frequency at a given normalized position.
        
        Args:
            position: Position in range [0, 1] (bottom to top of page)
            
        Returns:
            Frequency in Hz
        """
        # Clamp position to [0, 1]
        position = max(0, min(1, position))
        
        # Calculate index and interpolation factor
        index_float = position * (self.TABLE_SIZE - 1)
        index = int(index_float)
        
        # For stepped scales, just return the value at index
        if self.scale_type in (ScaleType.EQUAL_TEMPERED, ScaleType.MICROTONAL):
            return float(self.frequencies[min(index, self.TABLE_SIZE - 1)])
        
        # For continuous scales, interpolate
        frac = index_float - index
        if index >= self.TABLE_SIZE - 1:
            return float(self.frequencies[-1])
        
        # Logarithmic interpolation for perceptually smooth glissandi
        f1 = self.frequencies[index]
        f2 = self.frequencies[index + 1]
        return float(f1 * ((f2 / f1) ** frac))
    
    def get_frequencies_at_positions(
        self,
        positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Get frequencies at multiple positions (vectorized).
        
        Args:
            positions: Array of positions in range [0, 1]
            
        Returns:
            Array of frequencies in Hz
        """
        # Clamp positions to [0, 1]
        positions = np.clip(positions, 0, 1)
        
        # Calculate indices
        index_float = positions * (self.TABLE_SIZE - 1)
        indices = index_float.astype(int)
        indices = np.minimum(indices, self.TABLE_SIZE - 2)
        
        if self.scale_type in (ScaleType.EQUAL_TEMPERED, ScaleType.MICROTONAL):
            # Stepped - no interpolation
            return self.frequencies[indices]
        
        # Continuous - logarithmic interpolation
        fracs = index_float - indices
        f1 = self.frequencies[indices]
        f2 = self.frequencies[indices + 1]
        return f1 * ((f2 / f1) ** fracs)
    
    def get_position(self, frequency: float) -> float:
        """
        Get the normalized position for a given frequency (inverse lookup).
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Position in range [0, 1]
        """
        # Binary search for closest frequency
        idx = np.searchsorted(self.frequencies, frequency)
        
        if idx == 0:
            return 0.0
        if idx >= self.TABLE_SIZE:
            return 1.0
        
        # Interpolate position
        f1 = self.frequencies[idx - 1]
        f2 = self.frequencies[idx]
        
        if f2 == f1:
            return (idx - 1) / (self.TABLE_SIZE - 1)
        
        # Logarithmic interpolation
        frac = np.log(frequency / f1) / np.log(f2 / f1)
        return (idx - 1 + frac) / (self.TABLE_SIZE - 1)
    
    def copy(self) -> FrequencyTable:
        """Create a copy of this frequency table."""
        return FrequencyTable(
            name=f"{self.name} (copy)",
            frequencies=self.frequencies.copy(),
            scale_type=self.scale_type,
            min_freq=self.min_freq,
            max_freq=self.max_freq
        )


# Standard tuning constants
A4_FREQ = 440.0
C4_FREQ = 261.6255653  # Middle C

# MIDI note utilities
def midi_to_freq(midi_note: float, a4_freq: float = A4_FREQ) -> float:
    """Convert MIDI note number to frequency."""
    return a4_freq * (2 ** ((midi_note - 69) / 12))

def freq_to_midi(frequency: float, a4_freq: float = A4_FREQ) -> float:
    """Convert frequency to MIDI note number."""
    return 69 + 12 * np.log2(frequency / a4_freq)

def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"


class FrequencyTableLibrary:
    """
    A collection of frequency tables.
    Original UPIC supported 4 frequency tables.
    """
    
    MAX_TABLES: int = 16  # We allow more than original
    
    def __init__(self) -> None:
        self.tables: dict[str, FrequencyTable] = {}
        self._init_default_tables()
    
    def _init_default_tables(self) -> None:
        """Initialize with standard frequency tables."""
        self.add(FrequencyTable.continuous())
        self.add(FrequencyTable.equal_tempered())
        self.add(FrequencyTable.microtonal(name="Quarter Tones", divisions_per_octave=24))
        self.add(FrequencyTable.microtonal(name="Eighth Tones", divisions_per_octave=48))
        
        # Just intonation major scale
        major_ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]
        self.add(FrequencyTable.from_ratios(major_ratios, name="Just Major"))
    
    def add(self, table: FrequencyTable) -> bool:
        """Add a frequency table to the library."""
        if len(self.tables) >= self.MAX_TABLES:
            return False
        
        # Ensure unique name
        name = table.name
        counter = 1
        while name in self.tables:
            name = f"{table.name}_{counter}"
            counter += 1
        table.name = name
        
        self.tables[name] = table
        return True
    
    def remove(self, name: str) -> bool:
        """Remove a frequency table by name."""
        if name in self.tables:
            del self.tables[name]
            return True
        return False
    
    def get(self, name: str) -> FrequencyTable | None:
        """Get a frequency table by name."""
        return self.tables.get(name)
    
    def list_names(self) -> list[str]:
        """Get list of all table names."""
        return list(self.tables.keys())
    
    def __len__(self) -> int:
        return len(self.tables)
    
    def __iter__(self):
        return iter(self.tables.values())

