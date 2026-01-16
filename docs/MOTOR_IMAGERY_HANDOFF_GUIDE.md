# Motor Imagery BCI Handoff Guide

## Overview

This project implements a **Motor Imagery (MI) based Brain-Computer Interface** for drawing arcs/musical compositions. Users control vertical cursor movement by imagining left or right hand movements, which translates to UP/DOWN movement in the composition space.

**Reference Implementation:** [rishannp/Online-Motor-Imagery-Application](https://github.com/rishannp/Online-Motor-Imagery-Application/tree/main/Experiment%20Scripts/neurofeedback)

### Key Features
- **2-Class Motor Imagery**: Left hand imagery → UP movement, Right hand imagery → DOWN movement
- **Free Choice Paradigm**: After baseline, user can freely imagine left or right at any time
- **No Visual Targets**: Shows "Free Choice" prompt instead of alternating instructions
- **Baseline Normalization**: 10-second baseline capture for signal normalization
- **CSP-based Classification**: Common Spatial Patterns for feature extraction
- **Real-time Control**: Continuous classification with 2-second analysis windows

---

## System Architecture

### Paradigm Flow
1. **Baseline Capture** (10 seconds): User relaxes, system captures rest state
2. **Free Choice Period**: Shows "Free Choice - Imagine LEFT or RIGHT" prompt
3. **Motor Imagery**: User freely imagines left or right hand movement at any time
4. **Classification**: CSP extracts features continuously from mu (8-13 Hz) and beta (13-30 Hz) bands
5. **Control**: Classification result moves cursor UP (left) or DOWN (right) based on user's choice

---

## Required Scripts & Components

### Core Motor Imagery Modules

#### 1. `src/bci/motor_imagery_classifier.py`
**Purpose**: Real-time motor imagery classification using CSP + LDA

**Key Features**:
- CSP (Common Spatial Patterns) for spatial feature extraction
- LDA (Linear Discriminant Analysis) for classification
- Baseline normalization (Z-score using 10-second baseline)
- Uses sensorimotor channels: C3 (index 1), Cz (index 2), C4 (index 3)
- Frequency bands: Mu (8-13 Hz) and Beta (13-30 Hz)

**Key Methods**:
- `capture_baseline(eeg_data)`: Captures 10-second baseline for normalization
- `classify(eeg_data)`: Classifies motor imagery from EEG window
- `load_calibration(calibration_data)`: **NOT IMPLEMENTED** - placeholder for future CSP training

**Status**: ✅ Functional for basic classification (uses identity CSP filters without calibration)

---

#### 2. `src/bci/motor_imagery_stimulus.py`
**Purpose**: Free-choice stimulus system (no visual targets, no alternating instructions)

**Key Features**:
- Shows "Free Choice - Imagine LEFT or RIGHT" prompt after baseline
- User can imagine either direction at any time
- No alternating instructions - continuous free choice
- Classification runs continuously on 2-second windows

**Key Methods**:
- `start()`: Begin free choice period
- `update(current_time)`: Update stimulus state (always shows free choice prompt)
- `get_instruction_onsets()`: Returns empty list (no discrete instructions)

**Status**: ✅ Fully functional

---

#### 3. `src/bci/interface.py`
**Purpose**: Main GUI application (PyQt6)

**Key Components**:
- `BCICompositionWindow`: Main application window
- `MotorImageryInstructionWidget`: Visual instruction display with baseline indicator
- `_capture_baseline()`: Baseline capture workflow with visual feedback
- `_update_composition()`: Main composition loop with real-time classification

**Key Features**:
- Visual baseline indicator (blue background when capturing)
- LSL marker sending with proper timestamps
- Progress tracking and status updates
- Score generation and playback

**Status**: ✅ Functional, integrated with motor imagery system

---

### Supporting Modules

#### 4. `src/bci/controller.py`
**Purpose**: Cursor movement controller

**Key Features**:
- Automatic horizontal movement (playhead)
- BCI-controlled vertical movement (UP/DOWN)
- Smoothing for stable control
- Trail recording for score generation

**Status**: ✅ Functional, compatible with motor imagery results

---

#### 5. `src/bci/lsl_stream.py`
**Purpose**: LSL (Lab Streaming Layer) integration

**Key Components**:
- `LSLReceiver`: Receives EEG data from Unicorn
- `LSLMarkerSender`: Sends event markers with timestamps
- Channel mapping constants for Unicorn system

**Status**: ✅ Functional, includes proper timestamp handling

---

#### 6. `src/bci/preprocessing.py`
**Purpose**: EEG data preprocessing

**Key Components**:
- `LSLPreprocessor`: Handles LSL data with filtering
- Bandpass filtering, artifact removal
- Buffer management

**Status**: ✅ Functional

---

#### 7. `src/bci/score.py`
**Purpose**: Score generation and audio synthesis

**Status**: ✅ Functional

---

### Entry Points

#### 8. `bci_main.py`
**Main Script**: Entry point for the application

**Usage**:
```bash
python bci_main.py --mode gui
```

**Modes**:
- `gui` (default): Full GUI application
- `validate`: Validation tests
- `test`: Quick system test
- `demo`: Demo with random input

**Status**: ✅ Functional

---

#### 9. `start_bci.py`
**Helper Script**: Guided setup for Unicorn system

**Purpose**: Walks through Unicorn setup and launches the app

**Status**: ✅ Functional

---

## Hardware Requirements

### g.tec Unicorn Hybrid Black
- **8 EEG Channels**: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
- **Sampling Rate**: 250 Hz
- **Connection**: Bluetooth (via USB dongle)
- **Software**: Unicorn Suite with LSL streaming enabled

### Channel Mapping (0-indexed)
- **Index 0**: Fz (Frontal midline)
- **Index 1**: C3 (Left motor cortex) ← **Left hand imagery**
- **Index 2**: Cz (Central midline) ← **Reference**
- **Index 3**: C4 (Right motor cortex) ← **Right hand imagery**
- **Index 4**: Pz (Parietal midline)
- **Index 5**: PO7 (Left parieto-occipital)
- **Index 6**: Oz (Occipital midline)
- **Index 7**: PO8 (Right parieto-occipital)

---

## Dependencies

### Required Python Packages
- `numpy` - Numerical operations
- `scipy` - Signal processing (filtering, CSP)
- `scikit-learn` - LDA classifier (LinearDiscriminantAnalysis)
- `PyQt6` - GUI framework
- `pylsl` - Lab Streaming Layer for EEG streaming
- `matplotlib` - Plotting (for validation)
- `sounddevice` - Audio playback

**Note**: `scikit-learn` may need to be added to requirements.txt if not present.

### Hardware Software
- **Unicorn Suite** - g.tec software for Unicorn headset
- **Unicorn LSL** - LSL streaming component (included in Unicorn Suite)

---

## Setup Instructions

### 1. Environment Setup
```bash
# Activate conda environment
conda activate hack

# Install dependencies (if needed)
pip install -r requirements.txt

# If scikit-learn is missing:
pip install scikit-learn
```

### 2. Unicorn Setup
1. Plug in Bluetooth USB dongle
2. Power on Unicorn headset
3. Pair via Windows Bluetooth settings (PIN: 0000)
4. Launch "Unicorn LSL.exe" from Unicorn Suite
5. Select device and click "Start"

### 3. Run Application
```bash
# Option 1: Direct launch
python bci_main.py

# Option 2: Guided setup (recommended for first time)
python start_bci.py
```

### 4. In Application
1. Click "Connect LSL" button
2. Verify connection status shows "Connected: UN-XXXX"
3. Click "Start Composition"
4. **Baseline capture will start automatically** (10 seconds)
5. Follow instructions: "Imagine LEFT" → UP, "Imagine RIGHT" → DOWN

---

## What Needs Checking/Testing

### ✅ Critical Checks

1. **LSL Connection**
   - [ ] Verify LSL stream is detected
   - [ ] Check sample rate is 250 Hz
   - [ ] Verify 8 EEG channels are received
   - [ ] Test marker stream is working

2. **Baseline Capture**
   - [ ] Verify 10-second baseline is captured correctly
   - [ ] Check baseline statistics are computed (mu/beta power)
   - [ ] Verify baseline normalization is applied during classification
   - [ ] Test baseline visual indicator appears

3. **Channel Mapping**
   - [ ] Verify C3, Cz, C4 channels are correctly indexed (1, 2, 3)
   - [ ] Check channel names match Unicorn system
   - [ ] Verify sensorimotor data extraction is correct

4. **Classification**
   - [ ] Test left hand imagery produces UP movement
   - [ ] Test right hand imagery produces DOWN movement
   - [ ] Verify confidence scores are reasonable (0-1 range)
   - [ ] Check temporal smoothing is working

5. **Visual Feedback**
   - [ ] Verify instruction widget displays correctly
   - [ ] Check baseline mode visual indicator (blue background)
   - [ ] Test indicator lights for left/right feedback
   - [ ] Verify progress bar updates

6. **LSL Markers**
   - [ ] Verify "Baseline:Start" marker is sent with timestamp
   - [ ] Verify "Baseline:End" marker is sent with timestamp
   - [ ] Check "Composition Start" marker is sent
   - [ ] Verify markers use `local_clock()` for synchronization
   - [ ] Note: No instruction markers in free choice mode (user chooses freely)

7. **Data Logging**
   - [ ] Check session data is saved to `motor_imagery_sessions/`
   - [ ] Verify triggers.jsonl contains instruction onsets
   - [ ] Check metadata.json includes baseline duration

---

## What Needs Work/Implementation

### 🔴 High Priority

#### 1. **CSP Calibration System** (Not Implemented)
**Location**: `src/bci/motor_imagery_classifier.py` → `load_calibration()`

**Current Status**: Returns `False`, uses identity CSP filters (no spatial filtering)

**What's Needed**:
- Collect labeled training data (left/right hand imagery trials)
- Implement CSP filter training from calibration data
- Train LDA classifier on CSP features
- Save/load calibration data structure

**Reference**: See `src/bci/calibration.py` for SSVEP calibration pattern (adapt for MI)

**Impact**: Without calibration, classification uses simple heuristics and may have poor accuracy

---

#### 2. **Calibration Data Structure**
**Location**: `src/bci/calibration.py` (currently SSVEP-only)

**What's Needed**:
- Extend `CalibrationData` to support motor imagery trials
- Add `trials_left_hand` and `trials_right_hand` fields
- Implement trial collection interface
- Add CSP filter storage

**Reference**: Current structure uses `trials_15hz` and `trials_12hz` for SSVEP

---

#### 3. **Calibration UI/Workflow**
**Location**: `src/bci/interface.py`

**What's Needed**:
- Add motor imagery calibration button/workflow
- Implement trial collection (e.g., 20-30 trials per class)
- Show calibration progress
- Save calibration data to file

**Current**: Only baseline capture is implemented, no full calibration

---

### 🟡 Medium Priority

#### 4. **Performance Optimization**
**Current**: Classification runs every update cycle

**What's Needed**:
- Optimize CSP computation for real-time
- Consider caching filter computations
- Profile and optimize data pull rates
- Verify 50ms pull interval is optimal

---

#### 5. **Error Handling**
**What's Needed**:
- Better error messages for LSL connection failures
- Handle baseline capture failures gracefully
- Validate EEG data quality before classification
- Add timeout handling for LSL operations

---

#### 6. **Testing & Validation**
**What's Needed**:
- Unit tests for CSP computation
- Integration tests for classification pipeline
- Validation script for motor imagery accuracy
- Offline analysis tools for recorded data

---

### 🟢 Low Priority / Enhancements

#### 7. **Advanced Features**
- Filter-bank CSP (FBCSP) for better frequency selection
- Adaptive CSP that learns optimal bands per subject
- Cross-validation for calibration quality assessment
- Real-time performance metrics display

#### 8. **Documentation**
- API documentation for all modules
- User guide for experimenters
- Troubleshooting guide
- Performance tuning guide

---

## Known Issues & Limitations

### Current Limitations

1. **No CSP Training**: Uses identity filters → limited spatial discrimination
2. **Simple Heuristic Classification**: Without trained LDA, uses basic CSP feature differences
3. **No Subject-Specific Adaptation**: All users use same default parameters
4. **Limited Validation**: No automated accuracy testing

### Known Issues

1. **Baseline Required**: System won't start without baseline (by design, but could be more flexible)
2. **Fixed Parameters**: CSP components, frequency bands, window size are hardcoded
3. **No Calibration Persistence**: Baseline is not saved between sessions

---

## Key Parameters (Configurable)

### Motor Imagery Classifier (`motor_imagery_classifier.py`)
```python
sample_rate: float = 250.0              # Unicorn sample rate
window_seconds: float = 2.0             # Analysis window (2 seconds)
mu_band: Tuple[float, float] = (8.0, 13.0)    # Mu rhythm band
beta_band: Tuple[float, float] = (13.0, 30.0) # Beta rhythm band
n_csp_components: int = 4              # CSP components (2 per class)
sensorimotor_channels: List[int] = [1, 2, 3]  # C3, Cz, C4
threshold: float = 0.3                 # Confidence threshold
baseline_duration: float = 10.0        # Baseline capture duration
```

### Motor Imagery Stimulus (`motor_imagery_stimulus.py`)
```python
duration: float = 30.0                 # Total session duration
free_choice_text: str = "Free Choice - Imagine LEFT or RIGHT"  # Prompt text
```

---

## Data Flow

```
Unicorn Headset
    ↓ (Bluetooth)
Unicorn LSL Streamer
    ↓ (LSL)
LSLReceiver (lsl_stream.py)
    ↓
LSLPreprocessor (preprocessing.py)
    ↓ (Filtered EEG, 8 channels)
MotorImageryClassifier (motor_imagery_classifier.py)
    ↓ (ClassificationResult)
BCICursorController (controller.py)
    ↓ (CursorPosition)
CompositionCanvas (interface.py)
    ↓
BCIScore (score.py)
    ↓
Audio Synthesis
```

---

## File Structure

```
hackathon/
├── bci_main.py                    # Main entry point
├── start_bci.py                   # Setup helper
├── src/
│   └── bci/
│       ├── motor_imagery_classifier.py    # MI classification (CSP+LDA)
│       ├── motor_imagery_stimulus.py      # Instruction system
│       ├── interface.py                   # Main GUI
│       ├── controller.py                  # Cursor control
│       ├── lsl_stream.py                  # LSL integration
│       ├── preprocessing.py               # EEG preprocessing
│       ├── score.py                       # Score generation
│       └── calibration.py                 # Calibration (SSVEP only)
└── motor_imagery_sessions/         # Session data output
    └── YYYYMMDD_HHMMSS/
        ├── triggers.jsonl          # Instruction onsets
        ├── metadata.json           # Session metadata
        └── eeg_data.npy           # EEG data (if logged)
```

---

## Testing Checklist

### Pre-Experiment Checks
- [ ] Unicorn headset powered on and paired
- [ ] Unicorn LSL streaming active
- [ ] LSL stream detected in application
- [ ] Sample rate = 250 Hz confirmed
- [ ] All 8 channels receiving data

### During Experiment
- [ ] Baseline capture completes successfully
- [ ] Instructions display correctly
- [ ] Classification produces UP/DOWN movements
- [ ] Cursor responds to motor imagery
- [ ] No crashes or errors in console

### Post-Experiment
- [ ] Session data saved correctly
- [ ] Markers logged in triggers.jsonl
- [ ] Metadata includes all parameters
- [ ] Score can be played back
- [ ] Score can be exported

---

## Troubleshooting

### "No LSL stream found"
- Check Unicorn LSL is running and "Start" is clicked
- Verify Bluetooth connection
- Check firewall isn't blocking LSL (UDP multicast)
- Try: `python -c "from pylsl import resolve_streams; print(resolve_streams(2.0))"`

### "Baseline capture failed"
- Verify LSL is connected
- Check EEG data is being received
- Ensure at least 100 samples collected
- Check console for error messages

### "Classification not working"
- Verify baseline was captured successfully
- Check sensorimotor channels (C3, Cz, C4) are correct
- Verify frequency bands (mu: 8-13 Hz, beta: 13-30 Hz)
- Check confidence threshold (may be too high)

### "No cursor movement"
- Verify classification is producing results (check console logs)
- Check controller is receiving classification results
- Verify smoothing parameters aren't too aggressive
- Check vertical_speed parameter

---

## Next Steps for New Developer

1. **Understand the Reference Repo**
   - Review: https://github.com/rishannp/Online-Motor-Imagery-Application/tree/main/Experiment%20Scripts/neurofeedback
   - Compare baseline capture approach
   - Check pull rates and data handling

2. **Test Current System**
   - Run through testing checklist above
   - Verify baseline capture works
   - Test classification with real user

3. **Implement CSP Calibration** (Priority #1)
   - Study CSP algorithm implementation
   - Design calibration data structure
   - Implement trial collection
   - Train CSP filters and LDA

4. **Validate Performance**
   - Collect test data
   - Measure classification accuracy
   - Tune parameters for better performance

5. **Improve User Experience**
   - Add calibration workflow
   - Improve visual feedback
   - Add performance metrics display

---

## References

- **Reference Implementation**: [rishannp/Online-Motor-Imagery-Application](https://github.com/rishannp/Online-Motor-Imagery-Application/tree/main/Experiment%20Scripts/neurofeedback)
- **CSP Algorithm**: Common Spatial Patterns for motor imagery
- **Unicorn Documentation**: g.tec Unicorn Hybrid Black specifications
- **LSL Documentation**: Lab Streaming Layer protocol

---

## Contact & Support

For questions about:
- **Motor Imagery Implementation**: See `motor_imagery_classifier.py` and `motor_imagery_stimulus.py`
- **GUI Integration**: See `interface.py` → `BCICompositionWindow`
- **LSL Setup**: See `lsl_stream.py` and `start_bci.py`
- **Reference Repo**: Check GitHub repo for baseline approach and pull rates

---

---

## Quick Reference

### Running the System
```bash
conda activate hack
python bci_main.py
```

### Key Files
- **Main Entry**: `bci_main.py`
- **MI Classifier**: `src/bci/motor_imagery_classifier.py`
- **MI Stimulus**: `src/bci/motor_imagery_stimulus.py`
- **GUI**: `src/bci/interface.py`
- **Controller**: `src/bci/controller.py`

### Key Constants
- **Channels**: C3=1, Cz=2, C4=3 (Unicorn indices)
- **Frequency Bands**: Mu (8-13 Hz), Beta (13-30 Hz)
- **Window Size**: 2 seconds
- **Baseline Duration**: 10 seconds (configurable)
- **Sample Rate**: 250 Hz (Unicorn)

### Debugging
- Check console for `[MI BASELINE]` and `[MI CLASSIFY]` messages
- Verify LSL markers in triggers.jsonl
- Check session data in `motor_imagery_sessions/` directory

---

**Last Updated**: 2025-01-14
**System Status**: Functional for basic operation, CSP calibration pending
