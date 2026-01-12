# SSVEP Frequency Selection Analysis

## Current Configuration

- **Frequencies**: 10Hz (DOWN) and 15Hz (UP)
- **Phase Offset**: 180° (π radians) between targets
- **Monitor Refresh Rate**: 60Hz
- **Optimal SSVEP Range**: 8-15Hz

## Critical Issues Identified

### 1. Phase Utilization in Classification

#### FFT-Based Method (`classify_fft`)
**Status**: ❌ Phase NOT utilized

The FFT-based classifier only uses **magnitude/power** information:
```python
fft_vals = np.abs(np.fft.rfft(data_avg))  # Only magnitude!
```

**Implication**: The 180° phase offset provides NO benefit for FFT-based classification. The phase information is discarded.

#### CCA-Based Method (`classify_cca`)
**Status**: ✅ Phase IS utilized

The CCA method uses reference signals that include phase:
```python
ref_signals.append(np.sin(2 * np.pi * h * freq * t + h * phase))
ref_signals.append(np.cos(2 * np.pi * h * freq * t + h * phase))
```

**Implication**: Phase offset CAN help decorrelate signals in CCA, but only if:
1. The brain's SSVEP response maintains phase relationship with stimulus
2. The reference signals match the stimulus phase (currently implemented)
3. Individual differences don't introduce large phase shifts

**Recommendation**: 
- For FFT: Phase offset provides minimal benefit - consider removing or documenting as "visual distinction only"
- For CCA: Phase offset may help, but needs validation with real data

### 2. Harmonic Interference Problem

#### Shared Harmonics Analysis

**10Hz Harmonics**: 10, 20, 30, 40, 50, 60Hz...
**15Hz Harmonics**: 15, 30, 45, 60Hz...

**Shared Harmonics**:
- **30Hz**: 3rd harmonic of 10Hz, 2nd harmonic of 15Hz
- **60Hz**: 6th harmonic of 10Hz, 4th harmonic of 15Hz

**Problem**: When user attends to 10Hz target, their brain generates SSVEP at 10Hz, 20Hz, 30Hz, etc. The 30Hz component can interfere with detection of 15Hz target (which has 30Hz as its 2nd harmonic).

**Impact**: 
- Reduced classification accuracy
- Potential false positives
- Crosstalk between targets

### 3. Optimal Frequency Selection for 60Hz Monitor

#### Factors of 60Hz (Perfect Flicker Stability)

Factors of 60Hz: 1, 2, 3, 4, 5, 6, **10**, **12**, **15**, 20, 30, 60Hz

**Within 8-15Hz optimal SSVEP range**:
- ✅ **10Hz**: 6 frames/cycle (perfect factor)
- ✅ **12Hz**: 5 frames/cycle (perfect factor)
- ✅ **15Hz**: 4 frames/cycle (perfect factor)
- ⚠️ **8Hz**: 7.5 frames/cycle (NOT a factor, but close)
- ⚠️ **9Hz**: 6.67 frames/cycle (NOT a factor)

#### Frequency Combination Analysis

| Combination | Factor Status | Shared Harmonics | Separation | Recommendation |
|------------|---------------|------------------|------------|----------------|
| **10Hz + 15Hz** (current) | ✅ Both factors | 30Hz, 60Hz | 5Hz | ⚠️ Moderate interference |
| **12Hz + 15Hz** | ✅ Both factors | 60Hz only (4th/5th) | 3Hz | ✅ **BEST** - Minimal interference |
| **10Hz + 12Hz** | ✅ Both factors | 60Hz only (5th/6th) | 2Hz | ✅ Good - Close spacing |
| **8Hz + 12Hz** | ⚠️ 12Hz factor, 8Hz not | Minimal | 4Hz | ⚠️ 8Hz less stable |
| **9Hz + 12Hz** | ⚠️ 12Hz factor, 9Hz not | Minimal | 3Hz | ⚠️ 9Hz less stable |

#### Harmonic Overlap Analysis

**12Hz + 15Hz Combination**:
- 12Hz harmonics: 12, 24, 36, 48, 60Hz...
- 15Hz harmonics: 15, 30, 45, 60Hz...
- **Shared**: Only 60Hz (4th harmonic of 15Hz, 5th harmonic of 12Hz)
- **Separation**: 3Hz difference (acceptable)

**10Hz + 12Hz Combination**:
- 10Hz harmonics: 10, 20, 30, 40, 50, 60Hz...
- 12Hz harmonics: 12, 24, 36, 48, 60Hz...
- **Shared**: Only 60Hz (5th harmonic of 12Hz, 6th harmonic of 10Hz)
- **Separation**: 2Hz difference (close, but acceptable)

## Recommendations

### Option 1: Switch to 12Hz + 15Hz (RECOMMENDED)

**Advantages**:
- ✅ Both are perfect factors of 60Hz
- ✅ Minimal harmonic interference (only 60Hz shared)
- ✅ Good frequency separation (3Hz)
- ✅ Both within optimal 8-15Hz range
- ✅ 15Hz already in use (minimal code changes)

**Disadvantages**:
- ⚠️ Requires changing 10Hz to 12Hz
- ⚠️ Slightly closer spacing (3Hz vs 5Hz)

**Implementation**:
- Update `target_frequencies` to `(15.0, 12.0)`
- Update stimulus generation
- Update screen calibration targets
- Re-calibrate system

### Option 2: Keep 10Hz + 15Hz, Improve Classification

**Advantages**:
- ✅ No changes to existing setup
- ✅ Larger frequency separation (5Hz)

**Disadvantages**:
- ⚠️ Harmonic interference at 30Hz and 60Hz
- ⚠️ Phase offset not utilized by FFT

**Improvements**:
- Add harmonic filtering to exclude shared harmonics
- Use CCA exclusively (better phase utilization)
- Add frequency-specific bandpass filters

### Option 3: Switch to 10Hz + 12Hz

**Advantages**:
- ✅ Both perfect factors
- ✅ Minimal interference
- ✅ Good separation

**Disadvantages**:
- ⚠️ Closer spacing (2Hz) - may reduce discrimination
- ⚠️ Requires changing both frequencies

## Implementation Priority

1. **Immediate**: Document current limitations (phase not used in FFT, harmonic interference)
2. **Short-term**: Test 12Hz + 15Hz combination with real data
3. **Medium-term**: Implement harmonic filtering if keeping 10Hz + 15Hz
4. **Long-term**: Consider adaptive frequency selection based on user performance

## References

- SSVEP optimal range: 8-15Hz (MDPI, 2024)
- 60Hz monitor factors: 10Hz, 12Hz, 15Hz are optimal
- Phase decorrelation: More effective with CCA than FFT
- Harmonic interference: Shared harmonics reduce classification accuracy
