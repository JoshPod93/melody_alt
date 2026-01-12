# Screen Calibration Guide

## Overview

The screen calibration system measures your monitor's actual refresh rate and the real flicker frequencies being displayed. This ensures the CCA (Canonical Correlation Analysis) reference signals match what your brain actually sees, improving SSVEP classification accuracy.

## Why Screen Calibration?

Different monitors have different refresh rates (60Hz, 120Hz, 144Hz, etc.), and the actual flicker frequencies displayed may differ slightly from the target frequencies (15Hz and 10Hz) due to:

- Refresh rate limitations
- Frame timing variations
- Display hardware differences

By measuring the actual frequencies, the classifier can use reference signals that perfectly match what your brain responds to.

## Running Calibration

### Quick Start

**Run the calibration GUI:**
```bash
python screen_calibration.py
```

### Command-Line Options

The calibration tool supports several CLI commands:

**Check if calibration exists:**
```bash
python screen_calibration.py --check
```

**View current calibration info:**
```bash
python screen_calibration.py --info
```

**Delete existing calibration:**
```bash
python screen_calibration.py --delete
```

**Show help:**
```bash
python screen_calibration.py --help
```

### GUI Workflow

1. **Run the calibration script:**
   ```bash
   python screen_calibration.py
   ```

2. **Full screen mode (recommended):**
   - Press F11 to enter full screen
   - Ensures accurate refresh rate measurement

3. **Start calibration:**
   - Click "Start Calibration"
   - Wait 10 seconds while measurements are taken
   - The script measures:
     - Monitor refresh rate
     - Actual 15Hz flicker frequency
     - Actual 10Hz flicker frequency

4. **Save calibration:**
   - Click "Save Calibration" when complete
   - Data is saved to `screen_calibration.json`

## Calibration File Format

The calibration file (`screen_calibration.json`) contains:

```json
{
  "refresh_rate_hz": 59.94,
  "target_15hz": 15.0,
  "actual_15hz": 14.985,
  "target_10hz": 10.0,
  "actual_10hz": 9.992,
  "phase_15hz": 0.0,
  "phase_10hz": 3.141592653589793,
  "calibrated_at": "2026-01-12 10:30:00",
  "monitor_info": "Unknown"
}
```

## Using Calibration Data

The BCI system automatically loads `screen_calibration.json` if it exists:

- **Classifier**: Uses measured frequencies for CCA reference signals
- **Stimulus**: Can optionally use measured frequencies (future enhancement)

## Porting to Different Monitors

When moving to a different monitor:

1. **Delete old calibration:**
   ```bash
   rm screen_calibration.json
   ```

2. **Run calibration on new monitor:**
   ```bash
   python screen_calibration.py
   ```

3. **Save new calibration:**
   - The new monitor's characteristics will be saved
   - System will automatically use the new calibration

## Troubleshooting

**Calibration incomplete:**
- Ensure window is visible (not minimized)
- Try full screen mode (F11)
- Close other applications that might interfere

**Frequencies seem wrong:**
- Check your monitor's refresh rate settings
- Ensure you're using the monitor's native refresh rate
- Some monitors have variable refresh rate (VRR) - disable if possible

**No calibration file found:**
- System will use default frequencies (15Hz, 10Hz)
- Calibration is optional but recommended for best accuracy

## Technical Details

- **Refresh Rate Detection**: Measures `paintEvent` call frequency
- **Frequency Measurement**: Tracks intensity transitions over time
- **Measurement Duration**: 10 seconds for stable averages
- **Accuracy**: Typically ±0.01Hz for frequencies, ±0.1Hz for refresh rate

## Integration

The calibration system integrates with:

- `src/bci/classifier.py`: Loads calibration and uses measured frequencies
- `src/bci/stimulus.py`: (Future) Can use measured frequencies for display
- `screen_calibration.py`: Standalone calibration tool

---

**Note**: Screen calibration is separate from EEG calibration (`calibration_data.json`). Both can be used together for optimal performance.
