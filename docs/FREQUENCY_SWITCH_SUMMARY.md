# Frequency Switch: 10Hz → 12Hz Summary

## Changes Made

Successfully switched from **10Hz + 15Hz** to **12Hz + 15Hz** configuration.

### Rationale

1. **Harmonic Interference Reduction**: 
   - Old: 10Hz and 15Hz shared harmonics at 30Hz and 60Hz
   - New: 12Hz and 15Hz only share 60Hz (high-order, weak signal)
   - Result: Minimal interference, better classification accuracy

2. **Both Frequencies are Perfect Factors of 60Hz**:
   - 15Hz = 60Hz / 4 (4 frames per cycle)
   - 12Hz = 60Hz / 5 (5 frames per cycle)
   - Result: Stable, consistent flickering

3. **Phase Offset Maintained**:
   - 15Hz: Phase = 0° (UP target)
   - 12Hz: Phase = 180° (DOWN target)
   - Phase offset helps decorrelate signals in CCA (though FFT doesn't use phase)

### Files Updated

#### Core BCI Components
- ✅ `src/bci/classifier.py`: Updated default frequencies, field names (`power_12hz`), comments
- ✅ `src/bci/stimulus.py`: Updated default frequencies, phase references, comments
- ✅ `src/bci/screen_config.py`: Updated field names (`target_12hz`, `actual_12hz`, `phase_12hz`), backward compatible loading
- ✅ `src/bci/calibration.py`: Updated data structures (`trials_12hz`, `template_12hz`), backward compatible loading
- ✅ `src/bci/interface.py`: Updated references, UI text, calibration messages

#### Screen Calibration
- ✅ `screen_calibration/screen_calibration.py`: Updated target frequency (12Hz), all references, backward compatible CLI output

#### Documentation
- ✅ `docs/FREQUENCY_ANALYSIS.md`: Created comprehensive analysis
- ✅ `docs/FREQUENCY_RECOMMENDATIONS.md`: Created implementation recommendations
- ✅ `docs/FREQUENCY_SWITCH_SUMMARY.md`: This file

### Backward Compatibility

The system maintains backward compatibility with old calibration files:
- `screen_config.py`: Loads both `target_10hz`/`actual_10hz` and `target_12hz`/`actual_12hz`
- `calibration.py`: Loads both `trials_10hz`/`template_10hz` and `trials_12hz`/`template_12hz`
- Old calibration files will still work, but new calibrations use 12Hz

### Harmonics Usage

**Yes, harmonics ARE being used**:
- `n_harmonics: int = 2` means fundamental + 1st harmonic
- FFT method: Includes harmonics in power calculation (`power_up = sum(get_band_power(freq * h) for h in range(1, n_harmonics + 1))`)
- CCA method: Includes harmonics in reference signals (`sin(2πhf + h*phase)`, `cos(2πhf + h*phase)`)

### Phase Offset Utilization

- **FFT Method**: Phase NOT used (only magnitude/power)
- **CCA Method**: Phase IS used in reference signals
- **Visual**: Phase offset helps users distinguish targets

### Next Steps

1. **Re-calibrate System**: Run screen calibration to measure actual 12Hz flicker frequency
2. **Test with Real Data**: Validate that 12Hz + 15Hz performs better than 10Hz + 15Hz
3. **Measure Harmonic Interference**: Compare interference levels between old and new frequencies
4. **Update User Documentation**: Update any user-facing docs that reference 10Hz

### Testing Checklist

- [ ] Screen calibration works with 12Hz target
- [ ] Classifier correctly detects 12Hz SSVEP responses
- [ ] Calibration system records 12Hz trials correctly
- [ ] Old calibration files still load (backward compatibility)
- [ ] Harmonics are included in classification
- [ ] Phase offset is applied correctly (12Hz at 180°)
