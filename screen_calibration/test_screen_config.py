"""
Test script to verify screen calibration values are loaded correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bci.screen_config import get_screen_calibration, reload_screen_calibration
from src.bci.classifier import SSVEPClassifier
from src.bci.stimulus import SSVEPStimulus

def test_screen_config():
    """Test that screen calibration is loaded and used correctly."""
    print("=" * 60)
    print("Screen Calibration Configuration Test")
    print("=" * 60)
    print()
    
    # Load calibration
    screen_cal = get_screen_calibration()
    
    print("1. Screen Calibration Status:")
    print("-" * 60)
    print(f"   Calibrated: {screen_cal.is_calibrated}")
    if screen_cal.is_calibrated:
        print(f"   Calibrated at: {screen_cal.calibrated_at}")
        print(f"   Refresh rate: {screen_cal.refresh_rate_hz:.2f} Hz" if screen_cal.refresh_rate_hz else "   Refresh rate: N/A")
    print()
    
    print("2. Frequency Configuration:")
    print("-" * 60)
    freq_15hz, freq_10hz = screen_cal.frequencies
    phase_15hz, phase_10hz = screen_cal.phases
    print(f"   15Hz Target: {screen_cal.target_15hz:.3f} Hz")
    print(f"   15Hz Actual: {freq_15hz:.3f} Hz (offset: {freq_15hz - screen_cal.target_15hz:+.3f} Hz)")
    print(f"   15Hz Phase:  {phase_15hz:.3f} radians ({phase_15hz * 180 / 3.14159:.1f}°)")
    print()
    print(f"   10Hz Target: {screen_cal.target_10hz:.3f} Hz")
    print(f"   10Hz Actual: {freq_10hz:.3f} Hz (offset: {freq_10hz - screen_cal.target_10hz:+.3f} Hz)")
    print(f"   10Hz Phase:  {phase_10hz:.3f} radians ({phase_10hz * 180 / 3.14159:.1f}°)")
    print()
    
    print("3. Classifier Configuration:")
    print("-" * 60)
    classifier = SSVEPClassifier()
    print(f"   Target frequencies: {classifier.target_frequencies}")
    print(f"   Target phases: {classifier.target_phases}")
    print(f"   Match calibration: {classifier.target_frequencies == screen_cal.frequencies}")
    print()
    
    print("4. Stimulus Configuration:")
    print("-" * 60)
    stimulus = SSVEPStimulus()
    print(f"   Top frequency: {stimulus.top_frequency:.3f} Hz")
    print(f"   Bottom frequency: {stimulus.bottom_frequency:.3f} Hz")
    print(f"   Top target frequency: {stimulus.top_target.frequency:.3f} Hz")
    print(f"   Bottom target frequency: {stimulus.bottom_target.frequency:.3f} Hz")
    print(f"   Top target phase: {stimulus.top_target.phase_offset:.3f} radians")
    print(f"   Bottom target phase: {stimulus.bottom_target.phase_offset:.3f} radians")
    print()
    print(f"   Match calibration frequencies: {stimulus.top_target.frequency == freq_15hz and stimulus.bottom_target.frequency == freq_10hz}")
    print(f"   Match calibration phases: {stimulus.top_target.phase_offset == phase_15hz and stimulus.bottom_target.phase_offset == phase_10hz}")
    print()
    
    print("5. Summary:")
    print("-" * 60)
    all_match = (
        classifier.target_frequencies == screen_cal.frequencies and
        classifier.target_phases == screen_cal.phases and
        stimulus.top_target.frequency == freq_15hz and
        stimulus.bottom_target.frequency == freq_10hz and
        stimulus.top_target.phase_offset == phase_15hz and
        stimulus.bottom_target.phase_offset == phase_10hz
    )
    
    if all_match:
        print("   [OK] All components using calibrated values correctly!")
    else:
        print("   [ERROR] Some components not matching calibration")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    test_screen_config()
