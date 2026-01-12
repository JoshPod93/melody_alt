# Integration and Stimulus Interface Checklist
**Team Lead**: Josh  
**Reference**: See `TODO_DEVELOPMENT.md` for development tasks

## Stimulus Interface Tasks

### Task 1: Change Target Shape
- [ ] Modify `FlickerWidget` in `src/bci/interface.py` to use square dimensions
- [ ] Update `FlickerTarget.size` in `src/bci/stimulus.py` to square (e.g., 150x150px)
- [ ] Center targets vertically on screen
- [ ] Test visual appearance and verify flickering still works

### Task 2: Fix Flashing Pattern Issues
- [ ] Diagnose timing accuracy (measure actual vs target frequencies)
- [ ] Check screen refresh rate compatibility
- [ ] Investigate buffer issues in update loop
- [ ] Fix phase synchronization (ensure 180 degree difference)
- [ ] Implement frame-synchronized rendering if needed
- [ ] Add timing diagnostics and logging

### Task 3: Harmonics Integration
- [ ] Research SSVEP harmonic content (2f, 3f for 10Hz and 15Hz)
- [ ] Analyze calibration data for harmonic peaks
- [ ] Update classifier to detect harmonics (20Hz, 30Hz, 30Hz, 45Hz)
- [ ] Update CCA reference signals to include harmonics
- [ ] Verify phase matching for harmonics (h * phase_offset)

### Task 4: Phase Matching Verification
- [ ] Audit current phase matching in `_generate_reference_signals()`
- [ ] Verify synthetic references: 15Hz at 0 degrees, 10Hz at 180 degrees
- [ ] Verify harmonic phases are correctly scaled
- [ ] Check calibrated reference phase alignment
- [ ] Create phase verification tool
- [ ] Test that phase-matched references improve CCA correlation

## Component Integration

### Preprocessing Integration
- [ ] Review Guy and Ido's preprocessing implementation
- [ ] Verify interface compatibility (required methods present)
- [ ] Test integration with full system
- [ ] Verify 8-channel output and real-time performance

### Classifier Integration
- [ ] Review Stephan's classifier implementation
- [ ] Verify `ClassificationResult` structure is correct
- [ ] Test classification accuracy with real EEG data
- [ ] Verify calibration integration works
- [ ] Test real-time performance meets requirements

### Usability Integration
- [ ] Review Aminata's UI improvements
- [ ] Verify core BCI functionality preserved
- [ ] Test that visual changes do not affect classification
- [ ] Verify accessibility features work as intended

## System Testing

### Integration Testing
- [ ] Test full pipeline: LSL → Preprocessing → Classifier → Controller
- [ ] Verify end-to-end composition workflow
- [ ] Test calibration with all components integrated
- [ ] Measure overall system latency

### Performance Testing
- [ ] Verify preprocessing meets 20ms requirement
- [ ] Verify classification meets 10ms requirement
- [ ] Test system under extended use (no memory leaks)
- [ ] Verify stimulus timing accuracy over 10+ second periods

### Validation Testing
- [ ] Test with actual Unicorn Black hardware
- [ ] Verify classification accuracy with real user data
- [ ] Test stimulus presentation on different displays
- [ ] Validate phase relationships are maintained

## Documentation and Coordination

### Documentation
- [ ] Update main project documentation with integration details
- [ ] Document any interface changes or new requirements
- [ ] Update `TODO_DEVELOPMENT.md` with completed tasks
- [ ] Create integration test results report

### Team Coordination
- [ ] Coordinate interface changes with team members
- [ ] Resolve any integration conflicts or issues
- [ ] Ensure all components work together seamlessly
- [ ] Schedule integration review meeting if needed

---

**Priority Order**: Stimulus fixes → Harmonics → Integration → Testing

**Questions?** Refer to `TODO_DEVELOPMENT.md` for detailed task breakdowns and `INTEGRATION_OVERVIEW.md` for system architecture.
