# Screen Calibration Quick Start Guide

## What is Screen Calibration?

Screen calibration measures your monitor's actual refresh rate and the real flicker frequencies being displayed. This ensures the BCI classifier uses reference signals that match what your brain actually sees, improving SSVEP classification accuracy.

## Quick Commands

### Run Calibration
```bash
python screen_calibration.py
```

### Check if Calibrated
```bash
python screen_calibration.py --check
```

### View Calibration Info
```bash
python screen_calibration.py --info
```

### Delete Calibration (for recalibration)
```bash
python screen_calibration.py --delete
```

## Step-by-Step Calibration

1. **Ensure proper setup:**
   - Close other applications that might interfere
   - Use full screen mode (F11) if possible
   - Ensure monitor is at native refresh rate

2. **Run calibration:**
   ```bash
   python screen_calibration.py
   ```

3. **In the GUI:**
   - Click "Start Calibration"
   - Wait 5 seconds while measurements are taken
   - Click "Save Calibration" when complete

4. **Verify calibration:**
   ```bash
   python screen_calibration.py --info
   ```

## Expected Results

- **15Hz target**: Should measure ~15.00 Hz (±0.1 Hz is normal)
- **10Hz target**: Should measure ~10.00 Hz (±0.1 Hz is normal)
- **Refresh rate**: Your monitor's actual refresh rate (60Hz, 120Hz, etc.)

## When to Recalibrate

Recalibrate when:
- Switching to a different monitor
- Changing monitor refresh rate settings
- After system display driver updates
- If BCI classification accuracy seems degraded

## Troubleshooting

**Calibration incomplete:**
- Ensure window is visible (not minimized)
- Try full screen mode (F11)
- Close other applications

**Frequencies seem wrong:**
- Check monitor refresh rate settings
- Disable variable refresh rate (VRR/G-Sync/FreeSync) if enabled
- Ensure you're using the monitor's native refresh rate

**No calibration file:**
- System will use default frequencies (15Hz, 10Hz)
- Calibration is optional but recommended for best accuracy

## File Location

Calibration data is saved to:
```
screen_calibration.json
```

This file is automatically loaded by the BCI classifier when it starts.

## Integration

The calibration system integrates automatically:
- `src/bci/classifier.py` loads calibration on startup
- Uses measured frequencies for CCA reference signals
- Falls back to defaults if no calibration exists

---

**For detailed information, see:** `SCREEN_CALIBRATION.md`
