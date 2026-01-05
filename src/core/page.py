"""
Page module for UPIC.

A Page is the main composition workspace - a collection of arcs in a
pitch-versus-time space. It's analogous to a musical score.

In the original UPIC, up to 4 pages could be loaded simultaneously,
each with up to 4000 arcs.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterator, Tuple
from uuid import uuid4

from .arc import Arc, ArcGroup
from .waveform import WaveformLibrary
from .envelope import EnvelopeLibrary
from .frequency_table import FrequencyTableLibrary


@dataclass
class PageSettings:
    """Settings for a UPIC page."""
    # Time range
    duration: float = 60.0  # Total duration in seconds
    
    # Pitch range (for display, actual range is determined by frequency table)
    min_pitch_display: float = 0.0  # Normalized (0-1)
    max_pitch_display: float = 1.0
    
    # Grid settings
    show_grid: bool = True
    time_grid_interval: float = 1.0  # Seconds between vertical lines
    pitch_grid_divisions: int = 12  # Horizontal lines per octave equivalent
    
    # Default frequency table
    frequency_table_name: str = "Continuous"
    
    # Playback settings
    tempo_scale: float = 1.0  # Tempo multiplier
    loop_start: float = 0.0
    loop_end: float = 60.0
    loop_enabled: bool = False


@dataclass
class Page:
    """
    A UPIC composition page (score).
    
    The page is the main workspace where arcs are drawn and arranged.
    It contains all the arcs, groups, and settings for a composition.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        arcs: Dictionary of arcs by ID
        groups: Dictionary of arc groups by ID
        settings: Page settings
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Untitled"
    arcs: Dict[str, Arc] = field(default_factory=dict)
    groups: Dict[str, ArcGroup] = field(default_factory=dict)
    settings: PageSettings = field(default_factory=PageSettings)
    
    # Maximum arcs per page (matching original UPIC)
    MAX_ARCS: int = 4000
    
    # Undo/redo history
    _undo_stack: List[str] = field(default_factory=list, repr=False)
    _redo_stack: List[str] = field(default_factory=list, repr=False)
    _max_history: int = 50
    
    def add_arc(self, arc: Arc) -> bool:
        """
        Add an arc to the page.
        
        Returns:
            True if added, False if page is full
        """
        if len(self.arcs) >= self.MAX_ARCS:
            return False
        
        self.arcs[arc.id] = arc
        return True
    
    def remove_arc(self, arc_id: str) -> Optional[Arc]:
        """
        Remove an arc from the page.
        
        Returns:
            The removed arc, or None if not found
        """
        arc = self.arcs.pop(arc_id, None)
        
        # Also remove from any groups
        if arc:
            for group in self.groups.values():
                group.remove_arc(arc_id)
        
        return arc
    
    def get_arc(self, arc_id: str) -> Optional[Arc]:
        """Get an arc by ID."""
        return self.arcs.get(arc_id)
    
    def get_arcs_at_time(self, time: float) -> List[Arc]:
        """Get all arcs that are active at a given time."""
        return [arc for arc in self.arcs.values() 
                if arc.contains_time(time) and not arc.muted]
    
    def get_arcs_in_range(self, start_time: float, end_time: float) -> List[Arc]:
        """Get all arcs that overlap with a time range."""
        return [arc for arc in self.arcs.values()
                if arc.overlaps_time_range(start_time, end_time)]
    
    def get_arcs_in_region(
        self,
        start_time: float,
        end_time: float,
        min_pitch: float,
        max_pitch: float
    ) -> List[Arc]:
        """Get all arcs that fall within a rectangular region."""
        result = []
        for arc in self.arcs.values():
            if arc.overlaps_time_range(start_time, end_time):
                if arc.min_pitch <= max_pitch and arc.max_pitch >= min_pitch:
                    result.append(arc)
        return result
    
    def get_soloed_arcs(self) -> List[Arc]:
        """Get all arcs that are soloed."""
        soloed = [arc for arc in self.arcs.values() if arc.solo]
        return soloed if soloed else None
    
    def get_active_arcs(self) -> List[Arc]:
        """Get all arcs that should be played (respecting mute/solo)."""
        soloed = self.get_soloed_arcs()
        if soloed:
            return soloed
        return [arc for arc in self.arcs.values() if not arc.muted]
    
    def create_group(self, arc_ids: List[str], name: str = "Group") -> ArcGroup:
        """Create a new group from a list of arc IDs."""
        group = ArcGroup(name=name, arc_ids=arc_ids)
        self.groups[group.id] = group
        return group
    
    def delete_group(self, group_id: str) -> bool:
        """Delete a group (arcs remain on page)."""
        if group_id in self.groups:
            del self.groups[group_id]
            return True
        return False
    
    def get_group_for_arc(self, arc_id: str) -> Optional[ArcGroup]:
        """Get the group containing an arc, if any."""
        for group in self.groups.values():
            if group.contains(arc_id):
                return group
        return None
    
    def duplicate_arc(self, arc_id: str, offset_time: float = 0.5) -> Optional[Arc]:
        """
        Duplicate an arc with a time offset.
        
        Returns:
            The new arc, or None if original not found or page is full
        """
        original = self.get_arc(arc_id)
        if not original:
            return None
        
        new_arc = original.copy()
        new_arc.translate(delta_time=offset_time)
        
        if self.add_arc(new_arc):
            return new_arc
        return None
    
    def duplicate_group(
        self,
        group_id: str,
        offset_time: float = 0.5
    ) -> Optional[ArcGroup]:
        """
        Duplicate all arcs in a group with a time offset.
        
        Returns:
            A new group containing the duplicated arcs
        """
        group = self.groups.get(group_id)
        if not group:
            return None
        
        new_arc_ids = []
        for arc_id in group.arc_ids:
            new_arc = self.duplicate_arc(arc_id, offset_time)
            if new_arc:
                new_arc_ids.append(new_arc.id)
        
        if new_arc_ids:
            return self.create_group(new_arc_ids, f"{group.name} (copy)")
        return None
    
    def clear(self) -> None:
        """Remove all arcs and groups from the page."""
        self.arcs.clear()
        self.groups.clear()
    
    @property
    def total_duration(self) -> float:
        """Get the total duration covered by all arcs."""
        if not self.arcs:
            return 0.0
        return max(arc.end_time for arc in self.arcs.values())
    
    @property
    def arc_count(self) -> int:
        """Get the number of arcs on the page."""
        return len(self.arcs)
    
    def __iter__(self) -> Iterator[Arc]:
        """Iterate over all arcs."""
        return iter(self.arcs.values())
    
    def __len__(self) -> int:
        return len(self.arcs)
    
    def __bool__(self) -> bool:
        """Page is always truthy (even if empty)."""
        return True
    
    # Undo/Redo support
    def save_state(self) -> None:
        """Save current state for undo."""
        state = self._serialize_state()
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
    
    def undo(self) -> bool:
        """Undo the last action."""
        if not self._undo_stack:
            return False
        
        # Save current state to redo stack
        current = self._serialize_state()
        self._redo_stack.append(current)
        
        # Restore previous state
        state = self._undo_stack.pop()
        self._deserialize_state(state)
        return True
    
    def redo(self) -> bool:
        """Redo the last undone action."""
        if not self._redo_stack:
            return False
        
        # Save current state to undo stack
        current = self._serialize_state()
        self._undo_stack.append(current)
        
        # Restore next state
        state = self._redo_stack.pop()
        self._deserialize_state(state)
        return True
    
    def _serialize_state(self) -> str:
        """Serialize current state to JSON string."""
        data = {
            'arcs': {id: arc.to_dict() for id, arc in self.arcs.items()},
            'groups': {id: group.to_dict() for id, group in self.groups.items()}
        }
        return json.dumps(data)
    
    def _deserialize_state(self, state: str) -> None:
        """Restore state from JSON string."""
        data = json.loads(state)
        self.arcs = {id: Arc.from_dict(d) for id, d in data['arcs'].items()}
        self.groups = {id: ArcGroup.from_dict(d) for id, d in data['groups'].items()}
    
    # File I/O
    def to_dict(self) -> dict:
        """Serialize page to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'arcs': {id: arc.to_dict() for id, arc in self.arcs.items()},
            'groups': {id: group.to_dict() for id, group in self.groups.items()},
            'settings': {
                'duration': self.settings.duration,
                'min_pitch_display': self.settings.min_pitch_display,
                'max_pitch_display': self.settings.max_pitch_display,
                'show_grid': self.settings.show_grid,
                'time_grid_interval': self.settings.time_grid_interval,
                'pitch_grid_divisions': self.settings.pitch_grid_divisions,
                'frequency_table_name': self.settings.frequency_table_name,
                'tempo_scale': self.settings.tempo_scale,
                'loop_start': self.settings.loop_start,
                'loop_end': self.settings.loop_end,
                'loop_enabled': self.settings.loop_enabled
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Page:
        """Deserialize page from dictionary."""
        page = cls(
            id=data.get('id', str(uuid4())),
            name=data.get('name', 'Untitled')
        )
        
        # Load arcs
        for id, arc_data in data.get('arcs', {}).items():
            page.arcs[id] = Arc.from_dict(arc_data)
        
        # Load groups
        for id, group_data in data.get('groups', {}).items():
            page.groups[id] = ArcGroup.from_dict(group_data)
        
        # Load settings
        settings_data = data.get('settings', {})
        page.settings = PageSettings(
            duration=settings_data.get('duration', 60.0),
            min_pitch_display=settings_data.get('min_pitch_display', 0.0),
            max_pitch_display=settings_data.get('max_pitch_display', 1.0),
            show_grid=settings_data.get('show_grid', True),
            time_grid_interval=settings_data.get('time_grid_interval', 1.0),
            pitch_grid_divisions=settings_data.get('pitch_grid_divisions', 12),
            frequency_table_name=settings_data.get('frequency_table_name', 'Continuous'),
            tempo_scale=settings_data.get('tempo_scale', 1.0),
            loop_start=settings_data.get('loop_start', 0.0),
            loop_end=settings_data.get('loop_end', 60.0),
            loop_enabled=settings_data.get('loop_enabled', False)
        )
        
        return page
    
    def save(self, filepath: Path | str) -> None:
        """Save page to a JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path | str) -> Page:
        """Load page from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Project:
    """
    A UPIC project containing multiple pages and shared resources.
    
    This is the top-level container for a composition, including:
    - Multiple pages (original UPIC supported 4)
    - Shared waveform library
    - Shared envelope library
    - Shared frequency table library
    """
    name: str = "Untitled Project"
    pages: Dict[str, Page] = field(default_factory=dict)
    waveforms: WaveformLibrary = field(default_factory=WaveformLibrary)
    envelopes: EnvelopeLibrary = field(default_factory=EnvelopeLibrary)
    frequency_tables: FrequencyTableLibrary = field(default_factory=FrequencyTableLibrary)
    
    # Currently active page
    active_page_id: Optional[str] = None
    
    MAX_PAGES: int = 8  # Allow more than original UPIC
    
    def __post_init__(self) -> None:
        """Create a default page if none exist."""
        if not self.pages:
            self.add_page(Page(name="Page 1"))
    
    def add_page(self, page: Page) -> bool:
        """Add a page to the project."""
        if len(self.pages) >= self.MAX_PAGES:
            return False
        
        self.pages[page.id] = page
        if self.active_page_id is None:
            self.active_page_id = page.id
        return True
    
    def remove_page(self, page_id: str) -> bool:
        """Remove a page from the project."""
        if page_id not in self.pages:
            return False
        if len(self.pages) <= 1:
            return False  # Keep at least one page
        
        del self.pages[page_id]
        
        # Update active page if needed
        if self.active_page_id == page_id:
            self.active_page_id = next(iter(self.pages.keys()))
        
        return True
    
    def get_active_page(self) -> Optional[Page]:
        """Get the currently active page."""
        if self.active_page_id:
            return self.pages.get(self.active_page_id)
        return None
    
    def set_active_page(self, page_id: str) -> bool:
        """Set the active page."""
        if page_id in self.pages:
            self.active_page_id = page_id
            return True
        return False
    
    def to_dict(self) -> dict:
        """Serialize project to dictionary."""
        return {
            'name': self.name,
            'pages': {id: page.to_dict() for id, page in self.pages.items()},
            'active_page_id': self.active_page_id,
            'waveforms': [
                {'name': w.name, 'samples': w.samples.tolist(), 'type': w.waveform_type.value}
                for w in self.waveforms
            ],
            'envelopes': [
                {'name': e.name, 'samples': e.samples.tolist(), 'type': e.envelope_type.value}
                for e in self.envelopes
            ]
            # Frequency tables use default initialization for now
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Project:
        """Deserialize project from dictionary."""
        project = cls(name=data.get('name', 'Untitled Project'))
        project.pages.clear()
        
        # Load pages
        for id, page_data in data.get('pages', {}).items():
            project.pages[id] = Page.from_dict(page_data)
        
        project.active_page_id = data.get('active_page_id')
        
        # Waveforms and envelopes would need custom loading
        # For now, use defaults
        
        return project
    
    def save(self, filepath: Path | str) -> None:
        """Save project to a JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path | str) -> Project:
        """Load project from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

