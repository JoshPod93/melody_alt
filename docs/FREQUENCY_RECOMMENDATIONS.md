# SSVEP Frequency Selection: Recommendations & Implementation

## Executive Summary

**Current Issues**:
1. ❌ Phase offset (180°) NOT utilized by FFT-based classification
2. ⚠️ Harmonic interference: 10Hz and 15Hz share harmonics at 30Hz and 60Hz
3. ✅ Both frequencies are factors of 60Hz (good for flicker stability)

**Recommended Solution**: Switch to **12Hz + 15Hz** combination

## Detailed Analysis

### Phase Utilization

#### FFT Method (`classify_fft`)
```python
fft_vals = np.abs(np.fft.rfft(data_avg))  # Only magnitude!
```
**Result**: Phase information is discarded. The 180° phase offset provides NO benefit for FFT classification.

**Impact**: Phase offset is only useful for:
- Visual distinction (helps user differentiate targets)
- CCA reference signals (if brain maintains phase relationship)

#### CCA Method (`classify_cca`)
```python
ref_signals.append(np.sin(2 * np.pi * h * freq * t + h * phase))
ref_signals.append(np.cos(2 * np.pi * h * freq * t + h * phase))
```
**Result**: Phase IS included in reference signals, BUT:
- Brain's SSVEP response may not maintain exact phase relationship
- Individual differences introduce phase shifts
- Phase decorrelation benefit is uncertain without validation

**Recommendation**: 
- Document that phase offset is primarily for visual distinction
- Validate phase benefit with real SSVEP data
- Consider removing phase offset if it doesn't improve CCA performance

### Harmonic Interference

#### Current: 10Hz + 15Hz

**Harmonics**:
- 10Hz: 10, 20, **30**, 40, 50, **60** Hz...
- 15Hz: 15, **30**, 45, **60** Hz...

**Shared Harmonics**: 30Hz (major), 60Hz (minor)

**Problem**: 
- When user attends to 10Hz, brain generates SSVEP at 10Hz, 20Hz, 30Hz...
- The 30Hz component interferes with 15Hz detection (30Hz = 2nd harmonic of 15Hz)
- Reduces classification accuracy and increases false positives

#### Proposed: 12Hz + 15Hz

**Harmonics**:
- 12Hz: 12, 24, 36, 48, **60** Hz...
- 15Hz: 15, 30, 45, **60** Hz...

**Shared Harmonics**: Only 60Hz (4th harmonic of 15Hz, 5th harmonic of 12Hz)

**Advantage**: 
- Minimal interference (60Hz is high-order harmonic, weak signal)
- Better discrimination between targets
- Both are perfect factors of 60Hz

### Frequency Selection for 60Hz Monitor

#### Perfect Factors (Optimal Flicker Stability)

Factors of 60Hz: 1, 2, 3, 4, 5, 6, **10**, **12**, **15**, 20, 30, 60Hz

**Within 8-15Hz SSVEP range**:
- ✅ **10Hz**: 6 frames/cycle
- ✅ **12Hz**: 5 frames/cycle  
- ✅ **15Hz**: 4 frames/cycle

#### Comparison Table

| Combination | Factor Status | Shared Harmonics | Separation | Interference Risk |
|------------|---------------|------------------|------------|-------------------|
| **10Hz + 15Hz** (current) | ✅ Both factors | 30Hz, 60Hz | 5Hz | ⚠️ **HIGH** |
| **12Hz + 15Hz** (recommended) | ✅ Both factors | 60Hz only | 3Hz | ✅ **LOW** |
| **10Hz + 12Hz** | ✅ Both factors | 60Hz only | 2Hz | ✅ Low (but close spacing) |

## Recommendations

### Option 1: Switch to 12Hz + 15Hz (RECOMMENDED)

**Rationale**:
- Minimal harmonic interference (only 60Hz shared)
- Both perfect factors of 60Hz
- Good frequency separation (3Hz)
- Within optimal SSVEP range (8-15Hz)

**Implementation Steps**:
1. Update default frequencies: `(15.0, 12.0)` instead of `(15.0, 10.0)`
2. Update screen calibration targets
3. Update stimulus generation
4. Update documentation
5. Re-calibrate system

**Files to Modify**:
- `src/bci/classifier.py`: `target_frequencies = (15.0, 12.0)`
- `src/bci/stimulus.py`: Update default frequencies
- `src/bci/screen_config.py`: Update default target frequencies
- `screen_calibration/screen_calibration.py`: Update target frequencies

### Option 2: Keep 10Hz + 15Hz, Add Harmonic Filtering

**Rationale**:
- No changes to existing setup
- Larger frequency separation (5Hz)

**Implementation Steps**:
1. Add harmonic exclusion filter in `classify_fft()`
2. Exclude shared harmonics (30Hz, 60Hz) from power calculation
3. Use only non-overlapping harmonics for each target
4. Document phase offset limitation

**Code Changes**:
```python
# In classify_fft(), exclude shared harmonics
shared_harmonics = [30.0, 60.0]  # Hz
power_up = sum(get_band_power(self.target_frequencies[0] * h) 
               for h in range(1, self.n_harmonics + 1)
               if self.target_frequencies[0] * h not in shared_harmonics)
```

### Option 3: Use CCA Exclusively, Remove Phase Offset

**Rationale**:
- CCA can utilize phase (if validated)
- Remove phase offset if it doesn't help
- Focus on frequency separation

**Implementation Steps**:
1. Set default method to "cca" in interface
2. Remove phase offset (set both to 0°)
3. Validate with real data
4. Document findings

## Implementation Priority

1. **Immediate** (This Session):
   - Document current limitations
   - Create analysis document
   - Propose frequency change

2. **Short-term** (Next Session):
   - Test 12Hz + 15Hz with real data
   - Compare performance vs. 10Hz + 15Hz
   - Measure harmonic interference reduction

3. **Medium-term**:
   - Implement harmonic filtering if keeping 10Hz + 15Hz
   - Validate phase offset benefit with CCA
   - Update calibration system for new frequencies

4. **Long-term**:
   - Consider adaptive frequency selection
   - User-specific frequency optimization
   - Multi-frequency SSVEP (3+ targets)

## Testing Plan

### Test 1: Harmonic Interference Measurement
- Record SSVEP responses to 10Hz and 15Hz targets separately
- Measure power at 30Hz and 60Hz harmonics
- Compare interference levels

### Test 2: 12Hz + 15Hz Performance
- Switch to 12Hz + 15Hz
- Measure classification accuracy
- Compare to 10Hz + 15Hz baseline
- Measure harmonic interference reduction

### Test 3: Phase Offset Validation
- Test CCA with and without phase offset
- Measure classification accuracy difference
- Determine if phase offset improves performance

## References

- SSVEP optimal range: 8-15Hz (MDPI, 2024)
- 60Hz monitor factors: 10Hz, 12Hz, 15Hz optimal
- Harmonic interference reduces SSVEP classification accuracy
- Phase decorrelation more effective with CCA than FFT
