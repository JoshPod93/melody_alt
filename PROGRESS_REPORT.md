# BCI-UPIC Project Progress Report
## Date: 2026-01-14 (Latest Update)

### Latest Changes (2026-01-14)
- ✅ **Fixed CCA DOWN target classification** - Critical fix for SSVEP classification accuracy
  - Root cause: Padding data with zeros broke phase alignment, especially for DOWN reference (π phase offset)
  - Solution: Resize references to match chunk size instead of padding data (preserves phase profile)
  - Applied to both main live composition and data validation methods
  - Test results: `corr_down` now shows real values (0.674976) instead of always 0.0
  - Impact: DOWN target predictions now work correctly, classification accuracy significantly improved
  - Files modified: `src/bci/classifier.py` - `classify_cca()` method

### Latest Changes (2026-01-27)
- ✅ **Fixed LSL data pulling method** - Now matches proven approach from unicorn_system_test
  - Extract EEG channels (first 8) immediately after pulling from LSL
  - Store only EEG channels in buffer (not all 17 channels)
  - Use uniform time axis based on sample rate (more reliable than LSL timestamps)
  - Background thread pulls every 10ms (was 100ms) for fresher buffer
- ✅ **Restricted analysis to occipital channels only** - PO7, Oz, PO8 (indices 5, 6, 7)
  - All preprocessing methods now return only occipital channels
  - CAR still uses all 8 channels (needed for proper referencing)
  - Classification receives only occipital channels (optimal for SSVEP)
  - Updated `pull_and_process()`, `get_recent_data()`, and `get_lsl_buffer()` to return 3 channels
- ✅ **Optimized timings**
  - Background thread: 10ms timeout (was 100ms) - keeps buffer fresh
  - Main thread: 50ms updates (unchanged) - 20Hz classification rate
  - Chunk size: 50ms worth (~12-13 samples) - matches update interval for better efficiency
- ✅ **Improved data flow**
  - LSL (17 channels) → Extract EEG (8 channels) immediately → Store in buffer (8 channels)
  - Process with CAR (all 8 channels) → Return occipital only (3 channels: PO7, Oz, PO8) → Classify

## Major Achievements

### 1. Complete BCI System Implementation
- ✅ SSVEP-based brain-computer interface for musical composition
- ✅ Real-time EEG signal processing pipeline
- ✅ Graphical additive synthesizer integration
- ✅ Automatic cursor movement (horizontal) with BCI-controlled vertical movement

### 2. g.tec Unicorn Black Integration
- ✅ LSL (Lab Streaming Layer) integration for live EEG streaming
- ✅ Automatic detection and handling of 17-channel Unicorn streams
- ✅ Extraction of 8 EEG channels from hybrid stream (8 EEG + 9 auxiliary)
- ✅ Support for Unicorn Suite LSL Interface
- ✅ Connection verification scripts and setup documentation

### 3. SSVEP Stimulus System
- ✅ Dual-frequency flickering targets (15Hz top, 12Hz bottom)
- ✅ Perfect phase alignment (0° for 15Hz, 180° for 12Hz)
- ✅ Real-time visual stimulus rendering with PyQt6
- ✅ Frame-synchronized flickering using paintEvent
- ✅ Screen calibration system for dynamic frequency adjustment
- ✅ Generic frequency labels (higher_freq/lower_freq) for portability
- ✅ Stable flickering protocol matching screen calibration

### 4. Signal Processing Pipeline
- ✅ **Common Average Reference (CAR)** - CRITICAL for SSVEP signal quality
  - Removes common-mode noise shared across all channels
  - Applied before filtering for optimal noise reduction
  - Standard practice for SSVEP studies
  - Optional mastoid reference support (if sensors are placed)
- ✅ Bandpass filtering (5-25Hz) for SSVEP frequency range + harmonics
- ✅ Notch filtering (50Hz/60Hz) for power line noise removal
- ✅ Running statistics for adaptive normalization
- ✅ Artifact detection and rejection
- ✅ Numerical stability improvements (NaN/Inf checks, variance clamping)
- ✅ Vectorized processing for improved performance

### 5. SSVEP Classification
- ✅ Canonical Correlation Analysis (CCA) implementation
- ✅ FFT-based power spectrum analysis
- ✅ Occipital channel selection (PO7, Oz, PO8)
- ✅ Real-time classification with configurable window size
- ✅ Calibration system for personalized reference signals
- ✅ Screen calibration integration for accurate frequency matching

### 6. Calibration System
- ✅ User-specific SSVEP response recording
- ✅ Template generation from recorded epochs
- ✅ Calibration data persistence (JSON format)
- ✅ Integration with CCA classifier for improved accuracy
- ✅ Calibration workflow with visual feedback
- ✅ Dynamic frequency sequence based on screen calibration

### 7. Music Generation Integration
- ✅ UPIC Arc generation from BCI cursor trail
- ✅ Score playback through existing synthesizer
- ✅ Wavetable synthesis integration
- ✅ Stereo audio output

### 8. User Interface
- ✅ PyQt6-based GUI for BCI composition
- ✅ Real-time visualization of cursor movement
- ✅ LSL connection status indicators
- ✅ Calibration controls and status
- ✅ Experiment mode (disables simulation, enforces real data)
- ✅ Indicator lights for calibration targets

### 9. Code Quality & Robustness
- ✅ Comprehensive error handling
- ✅ Extensive logging for debugging
- ✅ Numerical stability fixes
- ✅ Channel extraction fixes for Unicorn streams
- ✅ Filter state reset after calibration
- ✅ Screen calibration system for portability

## Current Status

### Working Features
- ✅ LSL connection to Unicorn Black
- ✅ Real-time EEG data streaming and processing
- ✅ SSVEP stimulus rendering (stable during calibration)
- ✅ Calibration data recording and saving
- ✅ Template generation from calibration
- ✅ Score generation from cursor trail
- ✅ Audio playback
- ✅ Screen calibration with dynamic frequency detection

### Recent Fixes (2026-01-12 to 2026-01-27)
- ✅ Fixed calibration sequence to use screen calibration frequencies (14.199Hz/11.397Hz) instead of hard-coded 15Hz/12Hz
- ✅ Fixed calibration target identification (TOP vs BOTTOM) - now correctly alternates
- ✅ Applied delayed-start pattern to main composition flickering (matches stable calibration protocol)
- ✅ Separated indicator lights from flickering targets to prevent interference
- ✅ Fixed flickering timing consistency during calibration (continuous flickering, no resets)
- ✅ Reduced LSL recording timer from 4ms to 16ms to reduce event loop blocking
- ✅ Fixed 17-channel Unicorn stream handling (extract 8 EEG channels before processing)
- ✅ Fixed numerical stability issues in preprocessing (filter initialization, variance overflow)
- ✅ Added calibration file cleanup at start of new sessions
- ✅ Added comprehensive logging throughout calibration process
- ✅ Fixed filter state corruption after long calibration sessions
- ✅ **Added Common Average Reference (CAR)** - Critical improvement for signal quality
  - Removes common-mode noise (powerline, muscle artifacts, etc.)
  - Applied before filtering in both sample-by-sample and chunk processing
  - Should significantly improve SSVEP classification accuracy
  - Optional mastoid reference support added (if sensors are placed)
- ✅ Updated documentation to clarify Unicorn Black reference/ground setup
- ✅ Clarified that mastoid sensors are optional and not required for SSVEP

### Known Issues / Future Improvements
- ⚠️ Timer precision issues on Windows (QTimer fires at ~21ms instead of 8ms)
- ⚠️ paintEvent throttling by Qt event loop (~100ms intervals)
- ⚠️ **HIGH PRIORITY**: Computational cost of LSL data pulling and flicker protocol needs radical optimization
  - Research document created: `docs/PERFORMANCE_OPTIMIZATION_RESEARCH.md`
  - Recommended solutions: Cython, OpenGL rendering, Windows multimedia timers, C++ extensions
- Could add more sophisticated artifact rejection
- Could add real-time visualization of EEG power spectrum
- Could add session recording/replay functionality

## Technical Stack
- Python 3.11 (conda environment: hack)
- PyQt6 for GUI
- NumPy/SciPy for signal processing
- pylsl for LSL integration
- sounddevice for audio playback
- Existing UPIC synthesizer core

## File Structure
- src/bci/ - BCI system modules
  - interface.py - Main GUI application
  - preprocessing.py - EEG signal processing
  - classifier.py - SSVEP classification
  - stimulus.py - Visual flickering targets
  - controller.py - Cursor movement control
  - score.py - Score generation and playback
  - calibration.py - Calibration system
  - lsl_stream.py - LSL integration
  - screen_config.py - Screen calibration configuration
- screen_calibration/ - Screen calibration tools
  - screen_calibration.py - Standalone calibration GUI
  - screen_calibration.json - Calibration results
- docs/ - Documentation
  - PERFORMANCE_OPTIMIZATION_RESEARCH.md - Performance optimization strategies
  - Integration guides for team members
  - Checklists and TODO lists
- excess/ - Unused/junk scripts

## Next Steps

### Tomorrow's Tasks (2026-01-28)
1. **Optimization of timings**
   - Fine-tune update intervals based on performance metrics
   - Optimize chunk sizes for best balance between responsiveness and CPU usage
   - Test and validate timing improvements

2. **Refinement of preprocessing and classification**
   - Tune filter parameters based on real data analysis
   - Optimize classification window size and overlap
   - Improve CCA reference signal generation
   - Fine-tune confidence thresholds

3. **CRITICAL: Implement identical data pipeline in calibration**
   - Apply same data pulling method (extract EEG immediately, uniform time axis)
   - Apply same preprocessing (CAR, filtering, occipital extraction)
   - Apply same classification method (CCA with same parameters)
   - **Ensure stimuli behave identically in calibration vs main experiment**
   - This is essential for calibration validity - calibration must match live capture exactly

### Future Improvements
1. **Performance Optimization** (HIGH PRIORITY)
   - Implement Windows multimedia timer for better flicker precision
   - Consider Cython for LSL data pulling
   - Evaluate OpenGL-based flicker rendering
   - See `docs/PERFORMANCE_OPTIMIZATION_RESEARCH.md` for detailed plan

2. Test full composition workflow with calibrated classifier
3. Verify audio playback quality
4. Monitor for any remaining stability issues
5. Consider adding session recording/replay
6. Optimize classifier parameters based on real-world performance

---
Generated: 2026-01-12 14:10:00
Last Updated: 2026-01-27