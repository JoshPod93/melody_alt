# Data Validation Review Guide

## After Running "Check Data" Validation

Once you complete the validation protocol, here's what to check:

## 1. Check the Validation Report

The report will be saved to: `validation_plots/validation_report_{timestamp}.json`

### Key Metrics to Review:

#### Overall Results
- `mean_corr_down`: Should be > 0.0 (was 0.0 before fix)
- `mean_corr_up`: Should show real values
- `top_target.accuracy`: Accuracy for top target predictions
- `bottom_target.accuracy`: Should be > 0% (was 0% before fix)

#### Individual Predictions
- Check `predictions` array for each target
- `corr_down` values should vary (not all 0.0)
- `corr_up` values should vary
- Predictions should match ground truth for at least some chunks

## 2. Review Validation Plots

Three plots will be generated:

### `data_validation_grand_{timestamp}.png`
- Shows all captured data (baseline + top + bottom phases)
- Check raw EEG: Should show SSVEP responses during flicker phases
- Check processed EEG: Should be cleaner after filtering
- Check power spectrum: Should show peaks at target frequencies (14.20Hz and 11.40Hz)

### `data_validation_top_{timestamp}.png`
- Top target epoched data only
- Power spectrum should show peak near 14.20Hz

### `data_validation_bottom_{timestamp}.png`
- Bottom target epoched data only
- Power spectrum should show peak near 11.40Hz
- **This is the critical one** - verify corr_down is working

## 3. Check Debug Log (if exists)

`validation_plots/cca_debug.log` or `validation_plots/validation_debug_{timestamp}.log`

Look for:
- CCA correlation values for each chunk
- Reference shapes matching data shapes
- Any warnings about zero correlations

## 4. What Success Looks Like

### Before Fix:
- `corr_down = 0.0` (always)
- `bottom_target.accuracy = 0.0`
- All predictions biased toward UP

### After Fix (Expected):
- `corr_down ≈ 0.6-0.7` (real values)
- `bottom_target.accuracy > 0%` (ideally > 50%)
- Predictions balanced between UP and DOWN
- Power spectrum shows correct frequency peaks

## 5. If Issues Persist

If `corr_down` is still 0.0:
1. Check that calibration data is loaded correctly
2. Verify references are being resized (check log files)
3. Ensure chunk size matches (0.3s = 75 samples at 250Hz)
4. Re-run calibration if needed

## Quick Review Commands

After validation completes, you can quickly check:

```python
import json
from pathlib import Path

# Find latest validation report
reports = sorted(Path("validation_plots").glob("validation_report_*.json"))
if reports:
    latest = reports[-1]
    with open(latest, 'r') as f:
        data = json.load(f)
    
    print(f"Report: {latest.name}")
    print(f"Mean corr_down: {data.get('bottom_target', {}).get('mean_corr_down', 'N/A')}")
    print(f"Bottom accuracy: {data.get('bottom_target', {}).get('accuracy', 'N/A')}")
    print(f"Mean corr_up: {data.get('top_target', {}).get('mean_corr_up', 'N/A')}")
    print(f"Top accuracy: {data.get('top_target', {}).get('accuracy', 'N/A')}")
```
