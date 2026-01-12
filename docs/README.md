# BCI-UPIC Integration Documentation

This directory contains integration guides for team members working on different aspects of the BCI-UPIC system.

## Quick Start

### For Users

**Screen Calibration** (recommended before first use):
```bash
python screen_calibration/screen_calibration.py          # Run calibration GUI
python screen_calibration/screen_calibration.py --info   # View calibration info
```

See [`screen_calibration/CALIBRATION_QUICKSTART.md`](../screen_calibration/CALIBRATION_QUICKSTART.md) for details.

### For Developers

1. **Start Here**: All team members should begin with [`CHECKLIST_GENERAL.md`](CHECKLIST_GENERAL.md) - a workflow checklist covering research, design, testing, optimization, and integration.

2. **Read the Overview**: Then read [`INTEGRATION_OVERVIEW.md`](INTEGRATION_OVERVIEW.md) for a high-level understanding of the system architecture and integration workflow.

3. **Find Your Guide**: Based on your role, read the appropriate integration guide:
   - **Preprocessing**: [`INTEGRATION_PREPROCESSING.md`](INTEGRATION_PREPROCESSING.md)
   - **Classifier**: [`INTEGRATION_CLASSIFIER.md`](INTEGRATION_CLASSIFIER.md)
   - **Usability/UX**: [`INTEGRATION_USABILITY.md`](INTEGRATION_USABILITY.md)

## Development TODO

**For active development tasks and progress tracking**: See [`TODO_DEVELOPMENT.md`](TODO_DEVELOPMENT.md)

## Team Checklists

**General Workflow** (All Team Members): [`CHECKLIST_GENERAL.md`](CHECKLIST_GENERAL.md)

Role-specific checklists:
- **Preprocessing** (Guy Davies, Ido): [`CHECKLIST_PREPROCESSING.md`](CHECKLIST_PREPROCESSING.md)
- **Classifier** (Stephan): [`CHECKLIST_CLASSIFIER.md`](CHECKLIST_CLASSIFIER.md)
- **Usability** (Aminata): [`CHECKLIST_USABILITY.md`](CHECKLIST_USABILITY.md)
- **Integration** (Josh): [`CHECKLIST_INTEGRATION.md`](CHECKLIST_INTEGRATION.md)

## Document Structure

### INTEGRATION_OVERVIEW.md
- System architecture overview
- Data flow diagrams
- Integration workflow
- Common patterns and testing strategies
- Team coordination guidelines

### INTEGRATION_PREPROCESSING.md
- Preprocessing architecture
- Required interface methods
- Step-by-step integration instructions
- Channel extraction (17 → 8 channels)
- Real-time performance requirements
- Example implementations

### INTEGRATION_CLASSIFIER.md
- Classification architecture
- Required interface methods
- Input/output data specifications
- Algorithm examples (FFT, CCA, ML)
- Calibration integration
- Performance requirements

### INTEGRATION_USABILITY.md
- UI component structure
- Stimulus interface modifications
- Visual accessibility features
- Workflow improvements
- Configuration system
- Backwards compatibility strategies

## Integration Checklist

Before starting:
- [ ] Read the overview document
- [ ] Read your specific integration guide
- [ ] Review existing code implementations
- [ ] Understand data formats and interfaces
- [ ] Set up development environment

During development:
- [ ] Implement required interfaces
- [ ] Test component in isolation
- [ ] Integrate with full system
- [ ] Verify performance requirements
- [ ] Test edge cases

Before submitting:
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Backwards compatibility maintained
- [ ] Performance requirements met
- [ ] Code reviewed

## Getting Help

If you encounter issues:
1. Review your specific integration guide
2. Check existing implementations as examples
3. Test with minimal changes first
4. Contact the project maintainer

---

**Happy integrating!** 🚀
