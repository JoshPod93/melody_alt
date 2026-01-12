"""
Test script to verify screen calibration compatibility checking and refresh rate detection.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bci.screen_config import ScreenCalibration, get_screen_calibration

def test_compatibility_checking():
    """Test frequency compatibility checking with various refresh rates."""
    print("=" * 70)
    print("Screen Calibration Compatibility Test")
    print("=" * 70)
    print()
    
    # Test cases: (refresh_rate, expected_compatible, description)
    test_cases = [
        (60.0, True, "Standard 60Hz - both frequencies are factors"),
        (120.0, True, "120Hz - both frequencies are factors"),
        (75.0, True, "75Hz - 10Hz is half-integer (7.5 frames/cycle, acceptable)"),
        (144.0, True, "144Hz - both frequencies are factors"),
        (59.94, True, "59.94Hz - close enough to factors"),
        (50.0, True, "50Hz - 10Hz is factor, 15Hz is close (3.33 frames/cycle)"),
        (100.0, False, "100Hz - 15Hz gives 6.67 frames/cycle (not close to integer)"),
    ]
    
    print("1. Testing Frequency Compatibility Checking:")
    print("-" * 70)
    
    for refresh_rate, expected_compatible, description in test_cases:
        cal = ScreenCalibration(
            refresh_rate_hz=refresh_rate,
            actual_15hz=15.0,
            actual_10hz=10.0
        )
        
        is_compatible, warnings = cal.check_frequency_compatibility()
        
        # Calculate frames per cycle for display
        frames_15hz = refresh_rate / 15.0
        frames_10hz = refresh_rate / 10.0
        
        status = "[OK]" if is_compatible == expected_compatible else "[FAIL]"
        print(f"{status} {description}")
        print(f"   Refresh Rate: {refresh_rate:.2f} Hz")
        print(f"   Frames/cycle - 15Hz: {frames_15hz:.2f}, 10Hz: {frames_10hz:.2f}")
        print(f"   Compatible: {is_compatible} (expected: {expected_compatible})")
        if warnings:
            for warning in warnings:
                print(f"   Warning: {warning}")
        print()
    
    print()
    
    print("2. Testing Current Calibration:")
    print("-" * 70)
    
    # Load actual calibration
    screen_cal = get_screen_calibration()
    
    if screen_cal.is_calibrated:
        print(f"Calibrated: {screen_cal.calibrated_at}")
        print(f"Refresh Rate: {screen_cal.refresh_rate_hz:.2f} Hz" if screen_cal.refresh_rate_hz else "Refresh Rate: Not measured")
        print(f"15Hz: {screen_cal.actual_15hz:.3f} Hz")
        print(f"10Hz: {screen_cal.actual_10hz:.3f} Hz")
        print()
        
        if screen_cal.refresh_rate_hz:
            is_compatible, warnings = screen_cal.check_frequency_compatibility()
            
            # Calculate frames per cycle
            frames_15hz = screen_cal.refresh_rate_hz / screen_cal.actual_15hz
            frames_10hz = screen_cal.refresh_rate_hz / screen_cal.actual_10hz
            
            print("Compatibility Analysis:")
            print(f"  Frames per cycle - 15Hz: {frames_15hz:.2f}, 10Hz: {frames_10hz:.2f}")
            print(f"  Compatible: {is_compatible}")
            
            if warnings:
                print("  Warnings:")
                for warning in warnings:
                    print(f"    - {warning}")
            else:
                print("  [OK] No compatibility issues detected")
        else:
            print("  ⚠ Refresh rate not measured - cannot check compatibility")
    else:
        print("No calibration found - using defaults")
        print("Run: python screen_calibration.py")
    
    print()
    
    print("3. Testing Edge Cases:")
    print("-" * 70)
    
    # Test with None refresh rate
    cal_none = ScreenCalibration(refresh_rate_hz=None)
    is_compat, warnings = cal_none.check_frequency_compatibility()
    print(f"None refresh rate: Compatible={is_compat}, Warnings={len(warnings)}")
    assert len(warnings) > 0, "Should warn when refresh rate is None"
    
    # Test with invalid refresh rate
    cal_invalid = ScreenCalibration(refresh_rate_hz=-10.0)
    is_compat, warnings = cal_invalid.check_frequency_compatibility()
    print(f"Invalid refresh rate: Compatible={is_compat}, Warnings={len(warnings)}")
    
    # Test with very high refresh rate
    cal_high = ScreenCalibration(refresh_rate_hz=240.0, actual_15hz=15.0, actual_10hz=10.0)
    is_compat, warnings = cal_high.check_frequency_compatibility()
    frames_15 = 240.0 / 15.0
    frames_10 = 240.0 / 10.0
    print(f"240Hz refresh rate: Compatible={is_compat}")
    print(f"  Frames/cycle - 15Hz: {frames_15:.1f}, 10Hz: {frames_10:.1f}")
    
    print()
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    test_compatibility_checking()
