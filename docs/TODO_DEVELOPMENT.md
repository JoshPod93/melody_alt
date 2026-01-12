# BCI-UPIC Development TODO List

**Created**: 2024-12-19  
**Last Updated**: 2024-12-19  
**Status**: Planning Phase

---

## Overview

This document tracks development tasks for improving the BCI-UPIC system, focusing on:
1. Stimulus presentation improvements
2. Flashing pattern stability and accuracy
3. Harmonics integration for SSVEP classification

---

## Priority 1: Stimulus Presentation Issues

### Task 1.1: Change Target Shape from Bars to Central Squares
**Status**: ✅ Completed  
**Priority**: High  
**Estimated Time**: 1-2 hours  
**Files**: `src/bci/interface.py`, `src/bci/stimulus.py`

**Description**:
- Current implementation uses large horizontal bars (300x100px)
- Change to central square targets (e.g., 150x150px or 200x200px)
- Ensure targets are centered on screen
- Maintain aspect ratio (square)

**Implementation Steps**:
1. [x] Modify `FlickerWidget` in `src/bci/interface.py`:
   - Changed `setMinimumSize(300, 80)` to `setMinimumSize(200, 200)`
   - Updated `setMaximumHeight(100)` to `setMaximumHeight(200)`
   - Added `setMaximumWidth(200)` to enforce square
   - Added `alignment=Qt.AlignmentFlag.AlignHCenter` to layout.addWidget() calls
2. [x] Update `FlickerTarget.size` in `src/bci/stimulus.py`:
   - Changed default from `(200, 80)` to `(200, 200)`
   - Updated both targets in `SSVEPStimulus.__post_init__` from `(300, 100)` to `(200, 200)`
3. [ ] Test visual appearance:
   - Verify targets are square
   - Verify targets are centered
   - Verify flickering still works correctly

**Acceptance Criteria**:
- [x] Targets are square (width == height) - 200x200px
- [x] Targets are centered on screen - AlignHCenter applied
- [ ] Flickering frequency remains accurate - needs testing
- [ ] No visual glitches or rendering issues - needs testing

---

### Task 1.2: Investigate and Fix Flashing Pattern Inconsistencies
**Status**: 🔴 Not Started  
**Priority**: Critical  
**Estimated Time**: 3-4 hours  
**Files**: `src/bci/stimulus.py`, `src/bci/interface.py`

**Description**:
- Reports of inconsistent flashing patterns
- Instances where both 10Hz and 15Hz appear in phase
- Occasional complete stopping of flashing (possible buffer issue)
- May be related to screen refresh rate accuracy or timing issues

**Investigation Steps**:
1. [ ] **Diagnose timing issues**:
   - Add logging to `FlickerTarget.get_state()` to track actual flicker timing
   - Measure actual vs. target frequencies over extended periods
   - Check for frame rate synchronization issues
   - Verify `time.perf_counter()` accuracy

2. [ ] **Check screen refresh rate compatibility**:
   - Document screen refresh rate (60Hz, 120Hz, etc.)
   - Verify 15Hz and 10Hz are compatible with refresh rate
   - Consider using `QTimer` with precise timing instead of paint events
   - Test on different monitors/displays

3. [ ] **Investigate buffer issues**:
   - Check if `QTimer` update rate matches stimulus frequency
   - Verify no blocking operations in update loop
   - Check for event queue overflow
   - Profile update frequency in `_update_stimulus()` method

4. [ ] **Fix phase synchronization**:
   - Ensure both targets use same time reference (`_start_time`)
   - Verify phase offsets are correctly applied (0° for 15Hz, 180° for 10Hz)
   - Add phase verification test
   - Log phase relationship over time

**Implementation Fixes**:
1. [ ] **Implement frame-synchronized rendering**:
   - Use `QTimer` with precise interval (e.g., 16.67ms for 60Hz display)
   - Update flicker state based on elapsed time, not frame count
   - Ensure consistent timing regardless of frame drops

2. [ ] **Add timing diagnostics**:
   - Create `diagnose_timing()` function to measure actual frequencies
   - Log timing statistics during composition
   - Add visual indicator if timing drifts

3. [ ] **Fix phase relationship**:
   - Verify phase calculation: `phase = 2π * f * t + phase_offset`
   - Ensure 10Hz phase is exactly π radians offset from 15Hz
   - Add unit test for phase relationship

**Acceptance Criteria**:
- [ ] Both targets flicker at correct frequencies (±0.1Hz accuracy)
- [ ] Targets maintain 180° phase difference consistently
- [ ] No instances of both targets appearing in phase
- [ ] No instances of flashing stopping completely
- [ ] Timing diagnostics show stable frequencies over 10+ second periods

---

## Priority 2: Harmonics Integration

### Task 2.1: Explore Harmonics for 10Hz and 15Hz Frequencies
**Status**: 🔴 Not Started  
**Priority**: High  
**Estimated Time**: 2-3 hours  
**Files**: `src/bci/classifier.py`, `src/bci/stimulus.py`

**Description**:
- SSVEP responses contain harmonics (2f, 3f, etc.)
- Currently using only fundamental frequencies (10Hz, 15Hz)
- Need to explore using harmonics (20Hz, 30Hz, 30Hz, 45Hz, etc.) for better classification

**Investigation Steps**:
1. [ ] **Research SSVEP harmonics**:
   - Document typical harmonic content in SSVEP responses
   - Identify which harmonics are most prominent (usually 2nd, sometimes 3rd)
   - Determine optimal number of harmonics to include

2. [ ] **Analyze existing calibration data**:
   - Load existing `calibration_data.json`
   - Compute FFT of calibration trials
   - Identify harmonic peaks at 2f, 3f for both 10Hz and 15Hz
   - Measure harmonic amplitudes relative to fundamental

3. [ ] **Design harmonics integration**:
   - Decide on number of harmonics (suggest starting with 2 harmonics: 2f, 3f)
   - Plan how to combine harmonic evidence with fundamental
   - Consider weighting (fundamental may be stronger than harmonics)

**Implementation Steps**:
1. [ ] **Update classifier to detect harmonics**:
   - Modify `classify_fft()` to compute power at harmonic frequencies
   - For 10Hz: detect 20Hz, 30Hz
   - For 15Hz: detect 30Hz, 45Hz
   - Combine harmonic power with fundamental power

2. [ ] **Update CCA reference signals**:
   - Modify `_generate_reference_signals()` to include harmonics
   - Ensure harmonics maintain correct phase relationship
   - Update `n_harmonics` parameter (currently 2, may need 3)

3. [ ] **Test harmonics detection**:
   - Create test data with known harmonic content
   - Verify classifier can detect harmonics
   - Compare classification accuracy with vs. without harmonics

**Acceptance Criteria**:
- [ ] Classifier detects power at harmonic frequencies (2f, 3f)
- [ ] Harmonic detection improves classification accuracy
- [ ] CCA reference signals include harmonics with correct phase
- [ ] Documentation updated with harmonics strategy

---

### Task 2.2: Ensure Phase Matching Across Filter Bank
**Status**: 🔴 Not Started  
**Priority**: High  
**Estimated Time**: 3-4 hours  
**Files**: `src/bci/classifier.py`, `src/bci/calibration.py`

**Description**:
- CCA reference signals must match phase of:
  1. Visual stimulus (perfect sinusoids)
  2. Calibrated brain-derived signals (from calibration data)
  3. All harmonics in the filter bank
- Currently phase matching exists but needs verification and improvement

**Investigation Steps**:
1. [ ] **Audit current phase matching**:
   - Review `_generate_reference_signals()` in `classifier.py`
   - Verify phase propagation: `h * phase` for harmonic `h`
   - Check if calibration data phase is preserved
   - Document current phase relationships

2. [ ] **Analyze calibration data phase**:
   - Load calibration trials
   - Compute phase of SSVEP response at fundamental and harmonics
   - Compare to stimulus phase (should match if calibration is good)
   - Identify any phase shifts or delays

3. [ ] **Design phase alignment strategy**:
   - Decide how to align calibrated signals with synthetic references
   - Consider phase correction if calibration shows consistent offset
   - Plan for phase matching across all harmonics

**Implementation Steps**:
1. [ ] **Fix synthetic reference phase matching**:
   - Verify `_generate_reference_signals()` uses correct phase:
     - 15Hz: phase = 0° (0 radians)
     - 10Hz: phase = 180° (π radians)
   - For harmonics: phase = `h * phase_offset` where h is harmonic number
   - Add unit test to verify phase relationships

2. [ ] **Fix calibrated reference phase matching**:
   - Review `CalibrationData.get_cca_references()` in `calibration.py`
   - Ensure calibrated references preserve phase from recorded data
   - If calibration shows phase offset, apply correction
   - Verify phase alignment between calibrated and synthetic references

3. [ ] **Create phase verification tool**:
   - Add function to visualize phase relationships
   - Plot synthetic vs. calibrated reference signals
   - Show phase difference at each harmonic
   - Generate report of phase alignment quality

4. [ ] **Update CCA computation**:
   - Ensure `_cca_correlation()` handles multi-harmonic references correctly
   - Verify correlation is computed across all harmonics simultaneously
   - Test that phase-matched references give higher correlation

**Acceptance Criteria**:
- [ ] Synthetic references have correct phase (0° for 15Hz, 180° for 10Hz)
- [ ] Harmonic phases are correctly scaled (`h * phase_offset`)
- [ ] Calibrated references preserve phase from brain responses
- [ ] Phase alignment verified between synthetic and calibrated references
- [ ] CCA correlation improves with phase-matched references
- [ ] Phase verification tool confirms correct alignment

---

## Priority 3: Testing and Validation

### Task 3.1: Create Comprehensive Stimulus Timing Tests
**Status**: 🔴 Not Started  
**Priority**: Medium  
**Estimated Time**: 2 hours  
**Files**: `tests/test_stimulus_timing.py` (new)

**Description**:
- Need automated tests to verify stimulus timing accuracy
- Test frequency accuracy, phase relationships, and consistency

**Implementation Steps**:
1. [ ] Create `tests/test_stimulus_timing.py`
2. [ ] Test frequency accuracy:
   - Measure actual flicker frequency over 10 seconds
   - Verify within ±0.1Hz of target
3. [ ] Test phase relationship:
   - Verify 180° phase difference between 10Hz and 15Hz
   - Test over extended periods
4. [ ] Test timing consistency:
   - Verify no frame drops or timing glitches
   - Test under different system loads

**Acceptance Criteria**:
- [ ] All timing tests pass
- [ ] Tests run automatically in CI/CD
- [ ] Tests catch timing regressions

---

### Task 3.2: Create Harmonics Analysis Tool
**Status**: 🔴 Not Started  
**Priority**: Medium  
**Estimated Time**: 2-3 hours  
**Files**: `tools/analyze_harmonics.py` (new)

**Description**:
- Tool to analyze harmonic content in calibration data
- Visualize power spectrum showing fundamental and harmonics
- Help determine optimal harmonics to include

**Implementation Steps**:
1. [ ] Create `tools/analyze_harmonics.py`
2. [ ] Load calibration data
3. [ ] Compute FFT for each trial
4. [ ] Plot power spectrum with harmonics marked
5. [ ] Generate report with harmonic amplitudes

**Acceptance Criteria**:
- [ ] Tool successfully analyzes calibration data
- [ ] Visualizations clearly show harmonic content
- [ ] Report helps inform harmonics integration

---

## Progress Tracking

### Completed Tasks
- None yet

### In Progress
- None yet

### Blocked
- None yet

### Notes
- All tasks are planned but not yet started
- Priority order: Stimulus fixes → Harmonics → Testing
- Estimated total time: 15-20 hours

---

## Next Session Checklist

When starting the next development session:

1. [ ] Review this TODO list
2. [ ] Set development priorities for the session
3. [ ] Check current system state (run existing tests)
4. [ ] Start with Priority 1 tasks (stimulus presentation)
5. [ ] Document any findings or issues encountered
6. [ ] Update this TODO list with progress

---

## File Locations Reference

- **Stimulus Code**: `src/bci/stimulus.py`
- **Interface/UI**: `src/bci/interface.py`
- **Classifier**: `src/bci/classifier.py`
- **Calibration**: `src/bci/calibration.py`
- **Preprocessing**: `src/bci/preprocessing.py`
- **Calibration Data**: `calibration_data.json`

---

## Technical Notes

### Current Phase Settings
- 15Hz target: `phase_offset = 0.0` (0°)
- 10Hz target: `phase_offset = np.pi` (180°)
- Harmonic phase: `h * phase_offset` where h is harmonic number

### Current Harmonics
- `n_harmonics = 2` (fundamental + 1 harmonic)
- May need to increase to 3 for better detection

### Screen Refresh Rate Considerations
- Most displays: 60Hz (16.67ms frame time)
- 15Hz = 4 frames per cycle
- 10Hz = 6 frames per cycle
- Need to ensure timing is frame-synchronized

---

**Last Updated**: 2024-12-19  
**Next Review**: Next development session
