"""Quick verification script before launching BCI app."""
import sys
from pathlib import Path

print("=" * 60)
print("BCI APP PRE-LAUNCH VERIFICATION")
print("=" * 60)

# Check Python version
print(f"\n[OK] Python: {sys.version.split()[0]}")

# Check imports
try:
    import numpy as np
    print(f"[OK] NumPy: {np.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy import failed: {e}")
    sys.exit(1)

try:
    from src.bci.classifier import SSVEPClassifier
    print("[OK] Classifier import: OK")
except ImportError as e:
    print(f"[FAIL] Classifier import failed: {e}")
    sys.exit(1)

try:
    from src.bci.interface import BCICompositionWindow
    print("[OK] Interface import: OK")
except ImportError as e:
    print(f"[FAIL] Interface import failed: {e}")
    sys.exit(1)

# Check calibration files
cal_file = Path("calibration_data.json")
screen_cal_file = Path("screen_calibration/screen_calibration.json")

if cal_file.exists():
    print(f"[OK] Calibration data found: {cal_file}")
else:
    print(f"[WARN] Calibration data missing: {cal_file}")

if screen_cal_file.exists():
    print(f"[OK] Screen calibration found: {screen_cal_file}")
else:
    print(f"[WARN] Screen calibration missing: {screen_cal_file}")

# Verify fix is in place
try:
    import inspect
    src = inspect.getsource(SSVEPClassifier.classify_cca)
    if "ref_up_resized = self._resize_reference" in src:
        print("[OK] CCA fix verified: Reference resizing is active")
    else:
        print("[FAIL] CCA fix NOT found: Reference resizing missing!")
        sys.exit(1)
except Exception as e:
    print(f"[WARN] Could not verify fix: {e}")

print("\n" + "=" * 60)
print("READY TO LAUNCH")
print("=" * 60)
print("\nLaunch command:")
print("  conda activate hack")
print("  python bci_main.py --mode gui")
print("\nFirst step: Run 'Check Data' validation to verify fix")
