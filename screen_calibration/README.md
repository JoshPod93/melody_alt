# Screen Calibration

This folder contains all screen calibration tools and documentation for the BCI-UPIC system.

## Files

- **`screen_calibration.py`** - Main calibration GUI and CLI tool
- **`screen_calibration.json`** - Saved calibration data (created after calibration)
- **`SCREEN_CALIBRATION.md`** - Detailed documentation
- **`CALIBRATION_QUICKSTART.md`** - Quick reference guide
- **`test_screen_config.py`** - Test script to verify calibration loading
- **`test_screen_compatibility.py`** - Test script for compatibility checking

## Quick Start

**Important**: All commands should be run from the project root directory.

```bash
conda activate hack
python screen_calibration/screen_calibration.py
```

Or use CLI commands:

```bash
# Check if calibration exists
python screen_calibration/screen_calibration.py --check

# View calibration info
python screen_calibration/screen_calibration.py --info

# Delete calibration
python screen_calibration/screen_calibration.py --delete
```

## Running Tests

Test scripts must also be run from the project root:

```bash
python screen_calibration/test_screen_config.py
python screen_calibration/test_screen_compatibility.py
```

## Calibration Data

The calibration file (`screen_calibration.json`) is saved in this folder and automatically loaded by the BCI system components (classifier, stimulus generator, etc.) via `src/bci/screen_config.py`.

## Integration

The calibration data is accessed through the centralized configuration module:
- `src/bci/screen_config.py` - Provides `get_screen_calibration()` function
- All BCI components automatically use calibrated frequencies
- No manual path configuration needed
