"""
Arc module for UPIC.

An Arc is a single sound gesture drawn on the page - the fundamental
compositional unit in UPIC. It represents a pitch contour over time,
associated with a waveform and envelope.

In the original UPIC, a page could contain up to 4000 arcs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from uuid import uuid4
import json


@dataclass
class ArcPoint:
    """A single point in an arc's pitch contour."""
    time: float  # Time in seconds from start of page
    pitch: float  # Normalized pitch position (0-1, maps through frequency table)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.time, self.pitch)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> ArcPoint:
        return cls(time=t[0], pitch=t[1])


@dataclass
class Arc:
    """
    A single sound gesture (arc) on the UPIC page.
    
    An arc is defined by:
    - A series of (time, pitch) points defining the melodic contour
    - References to a waveform and envelope
    - Optional modulation settings
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        points: List of (time, pitch) points defining the contour
        waveform_name: Name of the waveform to use
        envelope_name: Name of the envelope to use
        amplitude: Overall amplitude multiplier (0-1)
        pan: Stereo pan position (-1 = left, 0 = center, 1 = right)
        modulator_id: ID of arc that modulates this one (FM synthesis)
        modulation_index: Depth of frequency modulation
        muted: Whether this arc is muted
        solo: Whether this arc is soloed
        color: RGB color for display (tuple of 0-255 values)
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Arc"
    points: List[ArcPoint] = field(default_factory=list)
    waveform_name: str = "Sine"
    envelope_name: str = "ADSR"
    amplitude: float = 0.8
    pan: float = 0.0
    modulator_id: Optional[str] = None
    modulation_index: float = 0.0
    muted: bool = False
    solo: bool = False
    color: Tuple[int, int, int] = (66, 135, 245)  # Default blue
    
    # Envelope editing points (for per-arc amplitude control)
    envelope_points: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def start_time(self) -> float:
        """Get the start time of this arc."""
        if not self.points:
            return 0.0
        return min(p.time for p in self.points)
    
    @property
    def end_time(self) -> float:
        """Get the end time of this arc."""
        if not self.points:
            return 0.0
        return max(p.time for p in self.points)
    
    @property
    def duration(self) -> float:
        """Get the duration of this arc."""
        return self.end_time - self.start_time
    
    @property
    def min_pitch(self) -> float:
        """Get the minimum pitch position."""
        if not self.points:
            return 0.5
        return min(p.pitch for p in self.points)
    
    @property
    def max_pitch(self) -> float:
        """Get the maximum pitch position."""
        if not self.points:
            return 0.5
        return max(p.pitch for p in self.points)
    
    def add_point(self, time: float, pitch: float) -> None:
        """Add a point to the arc."""
        self.points.append(ArcPoint(time=time, pitch=pitch))
        # Keep points sorted by time
        self.points.sort(key=lambda p: p.time)
    
    def set_points_from_tuples(self, points: List[Tuple[float, float]]) -> None:
        """Set points from a list of (time, pitch) tuples."""
        self.points = [ArcPoint.from_tuple(p) for p in points]
        self.points.sort(key=lambda p: p.time)
    
    def get_points_as_tuples(self) -> List[Tuple[float, float]]:
        """Get points as a list of (time, pitch) tuples."""
        return [p.to_tuple() for p in self.points]
    
    def get_pitch_at_time(self, time: float) -> float:
        """
        Get interpolated pitch at a given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Interpolated pitch position (0-1)
        """
        if not self.points:
            return 0.5
        
        if len(self.points) == 1:
            return self.points[0].pitch
        
        # Handle times outside the arc
        if time <= self.points[0].time:
            return self.points[0].pitch
        if time >= self.points[-1].time:
            return self.points[-1].pitch
        
        # Find surrounding points and interpolate
        for i in range(len(self.points) - 1):
            if self.points[i].time <= time <= self.points[i + 1].time:
                t1, p1 = self.points[i].time, self.points[i].pitch
                t2, p2 = self.points[i + 1].time, self.points[i + 1].pitch
                
                if t2 == t1:
                    return p1
                
                # Linear interpolation
                frac = (time - t1) / (t2 - t1)
                return p1 + frac * (p2 - p1)
        
        return self.points[-1].pitch
    
    def get_pitches_at_times(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get interpolated pitches at multiple times (vectorized).
        
        Args:
            times: Array of times in seconds
            
        Returns:
            Array of pitch positions (0-1)
        """
        if not self.points:
            return np.full_like(times, 0.5)
        
        if len(self.points) == 1:
            return np.full_like(times, self.points[0].pitch)
        
        # Extract times and pitches as arrays
        arc_times = np.array([p.time for p in self.points])
        arc_pitches = np.array([p.pitch for p in self.points])
        
        # Use numpy interpolation
        return np.interp(times, arc_times, arc_pitches)
    
    def get_envelope_amplitude_at_time(self, time: float) -> float:
        """
        Get the per-arc envelope amplitude at a given time.
        This is the blue line that can be edited on top of arcs.
        
        Args:
            time: Time in seconds
            
        Returns:
            Amplitude multiplier (0-1)
        """
        if not self.envelope_points:
            return 1.0
        
        if len(self.envelope_points) == 1:
            return self.envelope_points[0][1]
        
        # Normalize time to arc duration
        if self.duration == 0:
            return 1.0
        
        normalized_time = (time - self.start_time) / self.duration
        normalized_time = max(0, min(1, normalized_time))
        
        # Sort points and interpolate
        sorted_points = sorted(self.envelope_points, key=lambda p: p[0])
        times = [p[0] for p in sorted_points]
        amplitudes = [p[1] for p in sorted_points]
        
        return float(np.interp(normalized_time, times, amplitudes))
    
    def translate(self, delta_time: float = 0.0, delta_pitch: float = 0.0) -> None:
        """Move the arc in time and/or pitch."""
        for point in self.points:
            point.time += delta_time
            point.pitch += delta_pitch
            # Clamp pitch to valid range
            point.pitch = max(0, min(1, point.pitch))
    
    def scale_time(self, factor: float, anchor: float = None) -> None:
        """
        Scale the arc's duration.
        
        Args:
            factor: Scale factor (>1 stretches, <1 compresses)
            anchor: Time point to anchor (default: start of arc)
        """
        if anchor is None:
            anchor = self.start_time
        
        for point in self.points:
            point.time = anchor + (point.time - anchor) * factor
    
    def transpose(self, delta_pitch: float) -> None:
        """Transpose the arc by a pitch amount."""
        for point in self.points:
            point.pitch += delta_pitch
            point.pitch = max(0, min(1, point.pitch))
    
    def copy(self) -> Arc:
        """Create a copy of this arc with a new ID."""
        new_arc = Arc(
            name=f"{self.name} (copy)",
            points=[ArcPoint(p.time, p.pitch) for p in self.points],
            waveform_name=self.waveform_name,
            envelope_name=self.envelope_name,
            amplitude=self.amplitude,
            pan=self.pan,
            modulator_id=self.modulator_id,
            modulation_index=self.modulation_index,
            muted=self.muted,
            solo=False,
            color=self.color,
            envelope_points=list(self.envelope_points)
        )
        return new_arc
    
    def contains_time(self, time: float) -> bool:
        """Check if a time falls within this arc's duration."""
        return self.start_time <= time <= self.end_time
    
    def overlaps_time_range(self, start: float, end: float) -> bool:
        """Check if this arc overlaps with a time range."""
        return not (self.end_time < start or self.start_time > end)
    
    def to_dict(self) -> dict:
        """Serialize arc to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'points': [(p.time, p.pitch) for p in self.points],
            'waveform_name': self.waveform_name,
            'envelope_name': self.envelope_name,
            'amplitude': self.amplitude,
            'pan': self.pan,
            'modulator_id': self.modulator_id,
            'modulation_index': self.modulation_index,
            'muted': self.muted,
            'solo': self.solo,
            'color': self.color,
            'envelope_points': self.envelope_points
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Arc:
        """Deserialize arc from dictionary."""
        arc = cls(
            id=data.get('id', str(uuid4())),
            name=data.get('name', 'Arc'),
            waveform_name=data.get('waveform_name', 'Sine'),
            envelope_name=data.get('envelope_name', 'ADSR'),
            amplitude=data.get('amplitude', 0.8),
            pan=data.get('pan', 0.0),
            modulator_id=data.get('modulator_id'),
            modulation_index=data.get('modulation_index', 0.0),
            muted=data.get('muted', False),
            solo=data.get('solo', False),
            color=tuple(data.get('color', (66, 135, 245))),
            envelope_points=data.get('envelope_points', [])
        )
        arc.set_points_from_tuples(data.get('points', []))
        return arc


@dataclass
class ArcGroup:
    """
    A group of arcs that can be manipulated together.
    
    Groups allow multiple arcs to be selected, moved, copied,
    and edited as a single unit.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Group"
    arc_ids: List[str] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 165, 0)  # Orange
    
    def add_arc(self, arc_id: str) -> None:
        """Add an arc to this group."""
        if arc_id not in self.arc_ids:
            self.arc_ids.append(arc_id)
    
    def remove_arc(self, arc_id: str) -> None:
        """Remove an arc from this group."""
        if arc_id in self.arc_ids:
            self.arc_ids.remove(arc_id)
    
    def contains(self, arc_id: str) -> bool:
        """Check if an arc is in this group."""
        return arc_id in self.arc_ids
    
    def __len__(self) -> int:
        return len(self.arc_ids)
    
    def to_dict(self) -> dict:
        """Serialize group to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'arc_ids': self.arc_ids,
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ArcGroup:
        """Deserialize group from dictionary."""
        return cls(
            id=data.get('id', str(uuid4())),
            name=data.get('name', 'Group'),
            arc_ids=data.get('arc_ids', []),
            color=tuple(data.get('color', (255, 165, 0)))
        )

