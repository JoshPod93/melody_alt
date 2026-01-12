# Classifier Integration Checklist
**Team Member**: Stephan  
**Reference**: See `INTEGRATION_CLASSIFIER.md` for detailed instructions

## Required Implementation

### Core Interface Methods
- [ ] Implement `classify(eeg_data, method)` returning `ClassificationResult`
- [ ] Implement `load_calibration(calibration_data)` returning bool
- [ ] Implement `is_calibrated` property
- [ ] Implement `reset()` method for state cleanup

### Required Data Structures
- [ ] Use `AttentionTarget` enum (UP, DOWN, NONE)
- [ ] Return `ClassificationResult` with all required fields:
  - `target`: AttentionTarget
  - `confidence`: float (0-1)
  - `power_15hz`: float
  - `power_10hz`: float
  - `raw_score`: float

### Input Specifications
- [ ] Accept EEG data shape `(n_samples, 8)`
- [ ] Focus on occipital channels (3, 6, 7: Oz, PO7, PO8) for SSVEP
- [ ] Handle empty input data gracefully
- [ ] Process data in under 10ms for real-time operation

### Target Frequencies
- [ ] 15 Hz: Top target (UP movement), phase 0 degrees
- [ ] 10 Hz: Bottom target (DOWN movement), phase 180 degrees
- [ ] Ensure phase alignment with visual stimulus

### Calibration Support
- [ ] Load calibration data from `CalibrationData` object
- [ ] Use calibrated references if available, fallback to synthetic
- [ ] Verify calibration improves classification accuracy

### Integration Steps
- [ ] Create classifier module in `src/bci/` directory
- [ ] Update `src/bci/interface.py` to import your classifier
- [ ] Replace `SSVEPClassifier` instantiation with your class
- [ ] Test classification with known test data
- [ ] Verify real-time performance meets requirements

### Testing
- [ ] Test with simulated SSVEP data (15Hz and 10Hz)
- [ ] Verify correct target detection (UP vs DOWN)
- [ ] Measure classification time (should be < 10ms)
- [ ] Test with and without calibration data
- [ ] Verify confidence scores are reasonable (0-1 range)

### Documentation
- [ ] Document classification algorithm used
- [ ] Note any parameters that can be tuned
- [ ] Update this checklist when implementation is complete

---

**Questions?** Refer to `INTEGRATION_CLASSIFIER.md` for detailed examples and algorithm implementations.
