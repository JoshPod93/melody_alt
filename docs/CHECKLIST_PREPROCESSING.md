# Preprocessing Integration Checklist
**Team Members**: Guy Davies, Ido  
**Reference**: See `INTEGRATION_PREPROCESSING.md` for detailed instructions

## Required Implementation

### Core Interface Methods
- [ ] Implement `connect_lsl(stream_name)` method returning bool
- [ ] Implement `disconnect_lsl()` method
- [ ] Implement `is_lsl_connected` property
- [ ] Implement `pull_and_process(n_samples)` returning NDArray
- [ ] Implement `get_recent_data(seconds)` returning NDArray
- [ ] Implement `reset()` method for state cleanup

### Critical Requirements
- [ ] Extract exactly 8 EEG channels from 17-channel Unicorn stream
- [ ] Return data as `numpy.ndarray` with dtype `np.float64`
- [ ] Process data in under 20ms per update for real-time operation
- [ ] Handle sample rate from actual LSL stream (typically 250 Hz)
- [ ] Reset filter states properly after calibration

### Data Format
- [ ] Input: 17-channel LSL stream from Unicorn Black
- [ ] Processing: Extract channels 0-7 (EEG only, ignore channels 8-16)
- [ ] Output: Processed data shape `(n_samples, 8)`

### Integration Steps
- [ ] Create preprocessing module in `src/bci/` directory
- [ ] Update `src/bci/interface.py` to import your preprocessor
- [ ] Replace `LSLPreprocessor` instantiation with your class
- [ ] Test LSL connection and data reception
- [ ] Verify real-time performance meets requirements

### Testing
- [ ] Test with actual Unicorn Black hardware
- [ ] Verify 8-channel output (not 17 channels)
- [ ] Measure processing time (should be < 20ms per update)
- [ ] Test reset functionality after calibration
- [ ] Verify no memory leaks in long sessions

### Documentation
- [ ] Document any new parameters or configuration options
- [ ] Note any deviations from standard preprocessing pipeline
- [ ] Update this checklist when implementation is complete

---

**Questions?** Refer to `INTEGRATION_PREPROCESSING.md` for detailed examples and troubleshooting.
