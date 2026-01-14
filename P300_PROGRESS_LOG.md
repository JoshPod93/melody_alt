# P300 BCI System - Progress Log

## Overview
Migration from SSVEP to P300 oddball paradigm for BCI cursor control.

## Completed Work

### 1. P300 Paradigm Implementation
- **Stimulus System** (`src/bci/p300_stimulus.py`):
  - Discrete flash targets (top/bottom) with color cycling
  - Block-based color sequences: Each block contains all 8 colors exactly once (no repeats)
  - Randomized block permutations to prevent predictability
  - Red (target) appears once per block (~12.5% frequency)
  - No simultaneous red flashes on both targets
  - Flash duration: 62ms, ISI: 50ms (total cycle: 112ms, ~8.9 flashes/sec)

### 2. Color Sequence Generation
- **Block Structure**: Each block is a permutation of all 8 colors
  - Block 1: [red, blue, yellow, orange, green, cyan, purple, magenta] (randomized)
  - Block 2: [blue, orange, yellow, red, cyan, magenta, green, purple] (different order)
  - No color repeats within a block
  - Blocks are randomized and unique (checks last 3 blocks to prevent immediate repetition)
- **Verification**: Ensures each complete block contains all colors exactly once

### 3. P300 Classifier (`src/bci/p300_classifier.py`)
- ERP-based classification using Cz electrode
- Epoch extraction: -100ms to +800ms around flash onset
- Baseline correction: -100ms to 0ms
- P300 detection window: 250-450ms
- Averages multiple epochs (5) for better SNR
- Compares target (red) vs non-target amplitudes
- Binary classification: UP or DOWN (no NONE threshold)

### 4. Timestamp Synchronization (CRITICAL FIX)
- **Problem**: Flash timestamps and EEG timestamps were in different clock domains
- **Solution**: 
  - Flash timestamps use `pylsl.local_clock()` (LSL time domain)
  - EEG buffer timestamps use `pylsl.local_clock()` (LSL time domain)
  - Both now synchronized to LSL's clock for proper epoching
- **Diagnostics**: Added timestamp alignment warnings (>100ms difference)

### 5. Trigger Verification System
- Tracks expected vs actual flashes
- Compares color sequences to verify correct presentation
- Reports mismatches, missing flashes, and extra flashes
- Fixed string handling to prevent character iteration bugs

### 6. Data Logging
- **Session directories**: `p300_sessions/TIMESTAMP/`
- **Files**:
  - `triggers.jsonl`: All flash events with timestamps, colors, markers
  - `eeg_data.npy`: Raw EEG data (Cz channel only)
  - `metadata.json`: Session configuration, expected sequences, device info

### 7. UI Fixes
- **P300FlashWidget**: Discrete color changes, no flickering
  - Removed SSVEP-style intensity modulation
  - Solid color blocks that change discretely
  - State machine updates only from stimulus system (not on every paint)
  - `peek_state()` method to check state without updating

### 8. Preprocessing
- Bandpass: 0.1-30Hz (P300 ERP range)
- Cz channel extraction (index 2)
- Physics-based denoising (muscle artifacts, eye blinks)
- Common Average Reference (CAR)
- Notch filtering (50/60Hz)

## Current Status

### Working
- ✅ Block-based color sequences (all colors per block, no repeats)
- ✅ Timestamp synchronization (LSL clock)
- ✅ Discrete flash presentation (no flickering)
- ✅ Trigger verification system
- ✅ Data logging
- ✅ P300 classifier structure

### Needs Testing/Verification
- ⚠️ Classification accuracy (needs real data testing)
- ⚠️ Epoch extraction alignment (timestamp diagnostics will show issues)
- ⚠️ P300 detection sensitivity (amplitude thresholds)
- ⚠️ Color sequence synchronization (fixed but needs verification)

## Known Issues Fixed

1. **Color sequence desync**: Fixed independent color selection per target
2. **Trigger recording broken**: Fixed `_expected_flashes` population and `_actual_flashes` tracking
3. **Timestamp misalignment**: Fixed by using LSL clock for both flashes and EEG
4. **Block structure wrong**: Fixed to ensure all colors per block, no repeats
5. **Flickering instead of discrete flashes**: Fixed by removing state updates from paint events

## Technical Details

### Flash Timing
- Flash duration: 62ms
- ISI: 50ms
- Total cycle: 112ms
- Flashes per second: ~8.9
- Blocks per 10 seconds: ~11.2

### Color Sequence
- 8 colors total: red, blue, yellow, orange, green, cyan, purple, magenta
- Red is the target (oddball)
- Each block: permutation of all 8 colors
- Red frequency: 1/8 = 12.5% per block

### Classification
- Electrode: Cz (index 2)
- Epoch window: -100ms to +800ms
- P300 window: 250-450ms
- Baseline: -100ms to 0ms
- Epochs averaged: 5
- Classification: Binary (UP/DOWN based on amplitude difference)

## Next Steps (Future)
- Test with real user data
- Tune P300 detection parameters
- Verify classification accuracy
- Optimize flash timing for better P300 responses
- Consider calibration/training phase
