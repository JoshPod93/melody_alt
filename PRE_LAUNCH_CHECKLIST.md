# Pre-Launch Checklist - Main BCI Music App

## ✅ Code Fixes Applied
- [x] CCA DOWN target fix integrated (reference resizing instead of padding)
- [x] Applied to both main composition and validation methods
- [x] Phase alignment preserved for both UP (0°) and DOWN (π) targets

## Pre-Launch Verification

### 1. Environment Setup
- [ ] Conda environment activated: `conda activate hack`
- [ ] Python 3.11.14 ✓
- [ ] All dependencies installed (PyQt6, numpy, scipy, pylsl, etc.)

### 2. Hardware Setup
- [ ] g.tec Unicorn Black powered on and paired
- [ ] Unicorn Suite LSL Interface running
- [ ] Headset properly positioned (occipital channels: PO7, Oz, PO8)
- [ ] LSL stream visible (check with LSL viewer or `python -c "import pylsl; print([s.name() for s in pylsl.resolve_streams()])"`)

### 3. Calibration Data
- [ ] Screen calibration completed (`screen_calibration.json` exists)
- [ ] Subject-specific calibration completed (`calibration_data.json` exists)
- [ ] Calibration frequencies match screen calibration (14.20Hz/11.40Hz)

### 4. Data Validation (Run First!)
- [ ] Launch app: `python bci_main.py --mode gui`
- [ ] Click "Check Data" button
- [ ] Complete validation protocol:
  - 2s baseline (no flicker)
  - 10s top target (indicator on top)
  - 1s pause
  - 10s bottom target (indicator on bottom)
- [ ] Review validation report:
  - [ ] `corr_down` shows real values (not 0.0)
  - [ ] DOWN target accuracy > 0%
  - [ ] Both UP and DOWN predictions working
  - [ ] Check plots: `validation_plots/data_validation_*.png`

### 5. Main Composition Test
- [ ] If validation passes, proceed to main composition
- [ ] Click "Start Composition"
- [ ] Verify flickering is stable (both targets active)
- [ ] Test cursor movement:
  - Look at top target → cursor should move UP
  - Look at bottom target → cursor should move DOWN
- [ ] Complete composition (let playhead reach end)
- [ ] Verify score generation and playback

## Expected Results After Fix

### Data Validation
- **Before fix**: `corr_down = 0.0` (always zero)
- **After fix**: `corr_down ≈ 0.6-0.7` (real correlation values)
- **Accuracy**: DOWN target should have > 0% accuracy (ideally > 50%)

### Main Composition
- Cursor should respond to both UP and DOWN targets
- Classification should be more balanced (not always predicting UP)
- Confidence scores should reflect actual SSVEP responses

## Troubleshooting

### If `corr_down` is still 0.0:
1. Check that calibration data exists and is loaded
2. Verify references are being resized (check debug logs)
3. Ensure chunk size matches (0.3s = 75 samples at 250Hz)

### If classification is still biased:
1. Re-run calibration with proper attention to each target
2. Check that templates were generated correctly
3. Verify screen calibration frequencies match actual flicker rates

### If flickering is unstable:
1. Check screen refresh rate calibration
2. Verify high-precision timer is working
3. Check CPU usage (may need to optimize further)

## Files to Review After Testing

1. **Validation Report**: `validation_plots/validation_report_*.json`
   - Check `mean_corr_down` (should be > 0.0)
   - Check `bottom_target.accuracy` (should be > 0%)
   - Review individual predictions

2. **Validation Plots**: `validation_plots/data_validation_*.png`
   - Grand plot: All data (baseline + top + bottom)
   - Top plot: Top target epoched data
   - Bottom plot: Bottom target epoched data
   - Check power spectrum shows peaks at correct frequencies

3. **Debug Logs**: `validation_plots/cca_debug.log` (if exists)
   - Check CCA correlation values
   - Verify reference shapes match data shapes

## Launch Command

```bash
conda activate hack
python bci_main.py --mode gui
```

## Next Steps After Validation

1. If validation shows `corr_down` working → Proceed to main composition test
2. If validation still shows issues → Review debug logs and re-calibrate
3. Document results and any remaining issues
