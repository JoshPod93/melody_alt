# P300 Migration Status

## Completed ✅
1. **P300 Stimulus Module** (`src/bci/p300_stimulus.py`)
   - Discrete flash targets with controlled timing
   - Flash duration: 150ms
   - Inter-stimulus interval: 750ms
   - Alternating flash pattern

2. **P300 Classifier Module** (`src/bci/p300_classifier.py`)
   - ERP epoching: -100ms to +800ms
   - Baseline correction: -100ms to 0ms
   - P300 peak detection: 250-450ms window
   - Epoch averaging for SNR improvement

3. **Preprocessing Updates**
   - Bandpass filter: 0.1-30 Hz (was 11-30 Hz for SSVEP)
   - Physics-based denoising retained (muscle/blink detection)

4. **GUI Widget** (`P300FlashWidget`)
   - Created widget for discrete flashes
   - Replaces `FlickerWidget` for P300 paradigm

## In Progress 🔄
1. **Interface Updates** (`src/bci/interface.py`)
   - ✅ Imports added (P300Stimulus, P300Classifier)
   - ✅ Stimulus/classifier initialization updated
   - ✅ UI setup updated to use P300FlashWidget
   - ⚠️ `_update_composition()` still has SSVEP code - NEEDS UPDATE
   - ⚠️ `_start_composition()` needs P300 stimulus.start() call

## Remaining Work ⚠️

### Critical (Must Complete):
1. **Update `_update_composition()` method** (line ~1137)
   - Replace SSVEP sliding window classification with P300 epoching
   - Track flash onsets from stimulus
   - Buffer EEG data with timestamps
   - Call `classifier.classify_averaged()` with epochs

2. **Update `_start_composition()` method** (line ~972)
   - Call `self.stimulus.start()` to begin flash sequence
   - Clear flash onsets buffer
   - Clear EEG buffer

3. **Fix AttentionTarget import conflict**
   - Both SSVEP and P300 classifiers define `AttentionTarget`
   - Need to use P300's version or create shared enum

### Important:
4. **Update controller integration**
   - Ensure P300 ClassificationResult works with BCICursorController
   - May need adapter if result structure differs

5. **Update calibration** (if needed)
   - P300 may need different calibration approach
   - Consider ERP template generation

## Testing Checklist:
- [ ] P300 flashes appear correctly
- [ ] Flash onsets are tracked accurately
- [ ] EEG epoching works correctly
- [ ] Classification produces reasonable results
- [ ] Cursor movement responds to P300 classification
- [ ] No crashes or errors

## Key Differences: SSVEP vs P300

| Aspect | SSVEP | P300 |
|--------|-------|------|
| Stimulus | Continuous flickering | Discrete flashes |
| Classification | Frequency-domain (CCA/FFT) | Time-domain (ERP) |
| Window | Sliding window (0.3-0.5s) | Epochs (-100ms to +800ms) |
| Timing | Continuous | Event-locked |
| Bandpass | 11-30 Hz | 0.1-30 Hz |
| Peak | Frequency peaks | ERP peak ~300ms |

## Notes:
- Physics-based denoising (muscle/blink detection) is retained and works for P300
- Preprocessing bandpass updated to include low frequencies needed for ERP
- Flash timing: 150ms flash, 750ms ISI = 900ms cycle
- Need at least 5 epochs for averaging (classifier default)
