"""Diagnostic script to check CCA reference configuration."""
import numpy as np
from pathlib import Path
import json

# Try to load calibration data
cal_file = Path("calibration_data.json")
if cal_file.exists():
    print("=" * 60)
    print("CALIBRATION DATA CHECK")
    print("=" * 60)
    with open(cal_file, 'r') as f:
        cal_data = json.load(f)
    
    print(f"\nTrials 15Hz (higher freq): {len(cal_data.get('trials_15hz', []))}")
    print(f"Trials 12Hz (lower freq): {len(cal_data.get('trials_12hz', []))}")
    
    if cal_data.get('trials_15hz'):
        freq_15hz = cal_data['trials_15hz'][0].get('frequency', 'N/A')
        print(f"  First trial frequency: {freq_15hz} Hz")
    
    if cal_data.get('trials_12hz'):
        freq_12hz = cal_data['trials_12hz'][0].get('frequency', 'N/A')
        print(f"  First trial frequency: {freq_12hz} Hz")

# Check screen calibration
print("\n" + "=" * 60)
print("SCREEN CALIBRATION CHECK")
print("=" * 60)
try:
    from src.bci.screen_config import get_screen_calibration
    screen_cal = get_screen_calibration()
    higher_freq, lower_freq = screen_cal.frequencies
    higher_phase, lower_phase = screen_cal.phases
    print(f"\nHigher frequency: {higher_freq:.2f} Hz (phase: {higher_phase:.3f} rad)")
    print(f"Lower frequency: {lower_freq:.2f} Hz (phase: {lower_phase:.3f} rad)")
    print(f"\nExpected mapping:")
    print(f"  UP target = Higher frequency = {higher_freq:.2f} Hz")
    print(f"  DOWN target = Lower frequency = {lower_freq:.2f} Hz")
except Exception as e:
    print(f"Error loading screen calibration: {e}")

# Check classifier configuration
print("\n" + "=" * 60)
print("CLASSIFIER CONFIGURATION CHECK")
print("=" * 60)
try:
    from src.bci.classifier import SSVEPClassifier
    from src.bci.calibration import CalibrationData
    import json
    
    classifier = SSVEPClassifier()
    print(f"\nClassifier target frequencies: {classifier.target_frequencies}")
    print(f"  Index 0 (UP): {classifier.target_frequencies[0]:.2f} Hz")
    print(f"  Index 1 (DOWN): {classifier.target_frequencies[1]:.2f} Hz")
    
    # Check synthetic references
    print(f"\nSynthetic reference shapes:")
    print(f"  UP synthetic: {classifier._ref_signals_up_synthetic.shape}")
    print(f"  DOWN synthetic: {classifier._ref_signals_down_synthetic.shape}")
    
    # Try to load calibration
    if cal_file.exists():
        with open(cal_file, 'r') as f:
            cal_json = json.load(f)
        cal_data_obj = CalibrationData.from_dict(cal_json)
        if classifier.load_calibration(cal_data_obj):
            print(f"\n✓ Calibration loaded successfully")
            print(f"\nCombined reference shapes:")
            print(f"  UP combined: {classifier._ref_signals_up.shape}")
            print(f"  DOWN combined: {classifier._ref_signals_down.shape}")
            print(f"  UP calibrated: {classifier._ref_signals_up_calibrated.shape if classifier._ref_signals_up_calibrated is not None else 'None'}")
            print(f"  DOWN calibrated: {classifier._ref_signals_down_calibrated.shape if classifier._ref_signals_down_calibrated is not None else 'None'}")
            
            # Verify frequency mapping
            print(f"\n✓ Reference mapping verification:")
            print(f"  _ref_signals_up should match frequency {classifier.target_frequencies[0]:.2f} Hz")
            print(f"  _ref_signals_down should match frequency {classifier.target_frequencies[1]:.2f} Hz")
        else:
            print(f"\n✗ Failed to load calibration")
    else:
        print(f"\nNo calibration file found at {cal_file}")
        
except Exception as e:
    print(f"Error checking classifier: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VALIDATION REPORT CHECK")
print("=" * 60)
val_reports = sorted(Path("validation_plots").glob("validation_report_*.json"))
if val_reports:
    latest = val_reports[-1]
    with open(latest, 'r') as f:
        val_data = json.load(f)
    
    print(f"\nLatest validation: {latest.name}")
    print(f"Top target frequency: {val_data['target_frequencies']['top']:.2f} Hz")
    print(f"Bottom target frequency: {val_data['target_frequencies']['bottom']:.2f} Hz")
    print(f"\nTop target results:")
    print(f"  Ground truth: {val_data['top_target']['ground_truth']} (1=UP, 2=DOWN)")
    print(f"  Accuracy: {val_data['top_target']['accuracy']*100:.1f}%")
    print(f"  Mean confidence: {val_data['top_target']['mean_confidence']:.3f}")
    print(f"\nBottom target results:")
    print(f"  Ground truth: {val_data['bottom_target']['ground_truth']} (1=UP, 2=DOWN)")
    print(f"  Accuracy: {val_data['bottom_target']['accuracy']*100:.1f}%")
    print(f"  Mean confidence: {val_data['bottom_target']['mean_confidence']:.3f}")
    
    # Check if predictions are all the same
    top_preds = [p['prediction'] for p in val_data['top_target']['predictions']]
    bottom_preds = [p['prediction'] for p in val_data['bottom_target']['predictions']]
    print(f"\nTop target predictions: {len(set(top_preds))} unique values: {set(top_preds)}")
    print(f"Bottom target predictions: {len(set(bottom_preds))} unique values: {set(bottom_preds)}")
    
    if len(set(bottom_preds)) == 1 and bottom_preds[0] == 1:
        print(f"\nWARNING: Bottom target always predicts UP (1)")
        print(f"  This suggests CCA references may be swapped or classification logic inverted")
else:
    print("No validation reports found")

print("\n" + "=" * 60)
