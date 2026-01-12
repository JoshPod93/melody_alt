"""
Screen Calibration Configuration Module.

Provides centralized access to screen calibration data, ensuring all components
use the same measured frequencies for consistent SSVEP processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field


@dataclass
class ScreenCalibration:
    """
    Screen calibration configuration.
    
    This is the "mother variable" that controls all frequency-dependent
    components in the BCI system.
    
    Uses generic labels (higher_frequency, lower_frequency) to allow
    dynamic frequency selection based on monitor refresh rate.
    """
    refresh_rate_hz: Optional[float] = None
    target_higher_freq: float = 15.0  # Higher frequency target (top/up)
    actual_higher_freq: float = 15.0
    target_lower_freq: float = 12.0  # Lower frequency target (bottom/down)
    actual_lower_freq: float = 12.0
    phase_higher_freq: float = 0.0
    phase_lower_freq: float = 3.141592653589793  # π radians
    calibrated_at: Optional[str] = None
    monitor_info: str = "Unknown"
    
    # Legacy field names for backward compatibility (deprecated)
    target_15hz: float = field(init=False, repr=False)
    actual_15hz: float = field(init=False, repr=False)
    target_12hz: float = field(init=False, repr=False)
    actual_12hz: float = field(init=False, repr=False)
    phase_15hz: float = field(init=False, repr=False)
    phase_12hz: float = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize legacy fields for backward compatibility."""
        # Map generic fields to legacy names
        self.target_15hz = self.target_higher_freq
        self.actual_15hz = self.actual_higher_freq
        self.target_12hz = self.target_lower_freq
        self.actual_12hz = self.actual_lower_freq
        self.phase_15hz = self.phase_higher_freq
        self.phase_12hz = self.phase_lower_freq
    
    @property
    def frequencies(self) -> Tuple[float, float]:
        """Get actual frequencies (higher, lower) for use in system."""
        return (self.actual_higher_freq, self.actual_lower_freq)
    
    @property
    def phases(self) -> Tuple[float, float]:
        """Get phase offsets (higher, lower) for use in system."""
        return (self.phase_higher_freq, self.phase_lower_freq)
    
    @property
    def is_calibrated(self) -> bool:
        """Check if calibration data is available."""
        return self.calibrated_at is not None
    
    def check_frequency_compatibility(self) -> Tuple[bool, List[str]]:
        """
        Check if target frequencies are compatible with refresh rate.
        
        For optimal SSVEP, flicker frequencies should be factors of refresh rate
        (or close to factors) to ensure stable, consistent flickering.
        
        Returns:
            Tuple of (is_compatible, warnings)
        """
        warnings = []
        
        if self.refresh_rate_hz is None or self.refresh_rate_hz <= 0:
            warnings.append("Refresh rate not measured - cannot verify compatibility")
            return False, warnings
        
        refresh_rate = self.refresh_rate_hz
        
        # Check if frequencies are close to factors of refresh rate
        def check_frequency(freq: float, name: str) -> bool:
            # Calculate how many frames per cycle
            frames_per_cycle = refresh_rate / freq
            
            # Check if it's close to an integer (within 0.15 frames for tolerance)
            deviation_from_int = abs(frames_per_cycle - round(frames_per_cycle))
            
            if deviation_from_int < 0.15:
                return True
            
            # Check if it's a half-integer (also acceptable for some refresh rates)
            # e.g., 7.5 frames/cycle is acceptable
            half_int = round(frames_per_cycle * 2) / 2
            deviation_from_half = abs(frames_per_cycle - half_int)
            
            if deviation_from_half < 0.15:
                return True
            
            # Check if it's close to a quarter-integer (e.g., 3.25, 3.75)
            quarter_int = round(frames_per_cycle * 4) / 4
            deviation_from_quarter = abs(frames_per_cycle - quarter_int)
            
            if deviation_from_quarter < 0.15:
                return True
            
            warnings.append(
                f"{name} ({freq:.3f} Hz) is not a factor of refresh rate "
                f"({refresh_rate:.2f} Hz). Frames per cycle: {frames_per_cycle:.2f}. "
                f"This may cause flickering inconsistencies."
            )
            return False
        
        freq_higher_ok = check_frequency(self.actual_higher_freq, "Higher frequency target")
        freq_lower_ok = check_frequency(self.actual_lower_freq, "Lower frequency target")
        
        is_compatible = freq_higher_ok and freq_lower_ok
        
        return is_compatible, warnings
    
    @classmethod
    def load(cls, cal_file: Optional[Path] = None) -> 'ScreenCalibration':
        """
        Load screen calibration from file.
        
        Args:
            cal_file: Path to calibration file (default: screen_calibration.json)
            
        Returns:
            ScreenCalibration instance with loaded or default values
        """
        if cal_file is None:
            # Look for calibration file in screen_calibration folder or root
            cal_file = Path("screen_calibration/screen_calibration.json")
            if not cal_file.exists():
                cal_file = Path("screen_calibration.json")
        
        if not cal_file.exists():
            # Return default calibration (uses target frequencies)
            return cls()
        
        try:
            with open(cal_file, 'r') as f:
                data = json.load(f)
            
            # Support both new generic format and old hard-coded format for backward compatibility
            # New format: target_higher_freq, target_lower_freq
            # Old format: target_15hz, target_12hz (or target_10hz)
            target_higher = data.get('target_higher_freq', data.get('target_15hz', 15.0))
            actual_higher = data.get('actual_higher_freq', data.get('actual_15hz', data.get('actual_15hz', 15.0)))
            target_lower = data.get('target_lower_freq', data.get('target_12hz', data.get('target_10hz', 12.0)))
            actual_lower = data.get('actual_lower_freq', data.get('actual_12hz', data.get('actual_10hz', 12.0)))
            phase_higher = data.get('phase_higher_freq', data.get('phase_15hz', 0.0))
            phase_lower = data.get('phase_lower_freq', data.get('phase_12hz', data.get('phase_10hz', 3.141592653589793)))
            
            return cls(
                refresh_rate_hz=data.get('refresh_rate_hz'),
                target_higher_freq=float(target_higher),
                actual_higher_freq=float(actual_higher),
                target_lower_freq=float(target_lower),
                actual_lower_freq=float(actual_lower),
                phase_higher_freq=float(phase_higher),
                phase_lower_freq=float(phase_lower),
                calibrated_at=data.get('calibrated_at'),
                monitor_info=data.get('monitor_info', 'Unknown')
            )
        except Exception as e:
            print(f"[SCREEN CONFIG] Failed to load calibration: {e}")
            print(f"[SCREEN CONFIG] Using default frequencies")
            return cls()
    
    def get_info_string(self) -> str:
        """Get formatted info string for display."""
        if not self.is_calibrated:
            return "Using default frequencies (not calibrated)"
        
        return (
            f"Calibrated: {self.calibrated_at}\n"
            f"Higher freq: {self.actual_higher_freq:.3f} Hz (target: {self.target_higher_freq:.1f} Hz)\n"
            f"Lower freq: {self.actual_lower_freq:.3f} Hz (target: {self.target_lower_freq:.1f} Hz)"
        )


# Global instance - the "mother variable"
_screen_calibration: Optional[ScreenCalibration] = None


def get_screen_calibration(cal_file: Optional[Path] = None) -> ScreenCalibration:
    """
    Get the global screen calibration configuration.
    
    This is the "mother variable" that should be used by all components:
    - Classifier (CCA reference signals)
    - Stimulus (flicker frequencies)
    - Any other frequency-dependent components
    
    Args:
        cal_file: Optional path to calibration file (for testing/reloading)
        
    Returns:
        ScreenCalibration instance
    """
    global _screen_calibration
    
    if _screen_calibration is None or cal_file is not None:
        _screen_calibration = ScreenCalibration.load(cal_file)
    
    return _screen_calibration


def reload_screen_calibration() -> ScreenCalibration:
    """
    Reload screen calibration from file.
    
    Useful when calibration is updated.
    
    Returns:
        Updated ScreenCalibration instance
    """
    global _screen_calibration
    _screen_calibration = ScreenCalibration.load()
    return _screen_calibration
