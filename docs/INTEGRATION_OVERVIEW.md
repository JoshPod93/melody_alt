# BCI-UPIC Integration Overview

## Introduction

This document provides an overview of the integration guides for team members working on different aspects of the BCI-UPIC system. Each team member should refer to their specific integration guide for detailed instructions.

## Team Roles and Integration Guides

### 1. Preprocessing Team Member

**Focus**: EEG signal preprocessing (filtering, artifact rejection, normalization)

**Integration Guide**: [`INTEGRATION_PREPROCESSING.md`](INTEGRATION_PREPROCESSING.md)

**Key Responsibilities**:
- Implement custom preprocessing pipeline
- Handle LSL stream connection and data extraction
- Ensure real-time performance (< 20ms per update)
- Extract 8 EEG channels from 17-channel Unicorn stream

**Main Integration Point**: `src/bci/preprocessing.py` → Replace `LSLPreprocessor` class

---

### 2. Classifier Team Member

**Focus**: SSVEP classification algorithms (detecting 15Hz vs 10Hz attention)

**Integration Guide**: [`INTEGRATION_CLASSIFIER.md`](INTEGRATION_CLASSIFIER.md)

**Key Responsibilities**:
- Implement SSVEP classification algorithm
- Return `ClassificationResult` with correct structure
- Support calibration data for personalized templates
- Ensure real-time performance (< 10ms per classification)

**Main Integration Point**: `src/bci/classifier.py` → Replace `SSVEPClassifier` class

---

### 3. Usability/UX Team Member

**Focus**: Patient accessibility and user experience improvements

**Integration Guide**: [`INTEGRATION_USABILITY.md`](INTEGRATION_USABILITY.md)

**Key Responsibilities**:
- Improve visual design for patient accessibility
- Modify stimulus interface (target sizes, colors, labels)
- Simplify workflow and controls
- Add progress indicators and feedback

**Main Integration Points**: 
- `src/bci/interface.py` → Modify `BCICompositionWindow` and `FlickerWidget`
- `src/bci/stimulus.py` → Modify `SSVEPStimulus` and `FlickerTarget`

---

## System Architecture

### Data Flow

```
LSL Stream (17 channels)
    ↓
Preprocessor (extract 8 EEG channels, filter, normalize)
    ↓
Classifier (detect 15Hz vs 10Hz attention)
    ↓
Controller (update cursor position)
    ↓
Canvas (visualize trail)
    ↓
Score Generator (convert to music)
    ↓
Synthesizer (play audio)
```

### Component Dependencies

```
Interface (GUI)
├── Stimulus (flickering targets)
├── Preprocessor (EEG processing)
│   └── LSL Stream (data source)
├── Classifier (SSVEP detection)
│   └── Calibration Data (optional)
├── Controller (cursor movement)
└── Score (music generation)
    └── Synthesizer (audio playback)
```

## Integration Workflow

### Step 1: Understand Your Component

1. Read your specific integration guide
2. Review the existing implementation in the codebase
3. Understand the data formats and interfaces
4. Identify what needs to be changed

### Step 2: Create Your Implementation

1. Create new files or modify existing ones
2. Implement required interfaces
3. Test your component in isolation
4. Ensure backwards compatibility

### Step 3: Integrate with System

1. Update imports in `src/bci/interface.py`
2. Replace component instantiation
3. Test full system integration
4. Verify no regressions

### Step 4: Test and Validate

1. Test with simulated data
2. Test with real hardware (if applicable)
3. Verify performance requirements
4. Check for edge cases and errors

## Common Integration Patterns

### Pattern 1: Extending Existing Classes

```python
# Create new class extending existing one
class YourCustomPreprocessor(LSLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Your custom initialization
    
    def pull_and_process(self, n_samples: int = 16):
        # Override method with your implementation
        pass
```

### Pattern 2: Configuration-Based Approach

```python
# Add configuration to existing class
class BCICompositionWindow(QMainWindow):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DefaultConfig()
        
        if self.config.accessibility_mode:
            self._setup_accessibility()
```

### Pattern 3: Factory Pattern

```python
# Create factory function to instantiate correct component
def create_preprocessor(mode: str = "default"):
    if mode == "custom":
        return YourCustomPreprocessor()
    else:
        return LSLPreprocessor()
```

## Testing Strategy

### Unit Tests

Test your component in isolation:

```python
# Example: Test preprocessor
def test_preprocessor():
    preprocessor = YourCustomPreprocessor()
    test_data = np.random.randn(64, 8)
    result = preprocessor._your_processing_pipeline(test_data)
    assert result.shape == (64, 8)
```

### Integration Tests

Test your component with the full system:

```python
# Example: Test full pipeline
def test_full_pipeline():
    window = BCICompositionWindow()
    window.preprocessor = YourCustomPreprocessor()
    window.classifier = YourCustomClassifier()
    
    # Test composition workflow
    window._start_composition()
    # ... verify behavior ...
```

### Performance Tests

Ensure real-time performance:

```python
# Example: Performance test
def test_performance():
    preprocessor = YourCustomPreprocessor()
    start = time.perf_counter()
    for _ in range(100):
        preprocessor.pull_and_process(16)
    elapsed = time.perf_counter() - start
    assert elapsed / 100 < 0.02  # < 20ms per call
```

## File Structure

```
src/bci/
├── __init__.py              # Module exports
├── interface.py             # Main GUI (usability team modifies)
├── stimulus.py              # Flickering targets (usability team modifies)
├── preprocessing.py         # EEG processing (preprocessing team replaces)
├── classifier.py            # SSVEP detection (classifier team replaces)
├── controller.py            # Cursor control (usually unchanged)
├── score.py                 # Music generation (usually unchanged)
├── calibration.py           # Calibration system (usually unchanged)
└── lsl_stream.py            # LSL integration (usually unchanged)

docs/
├── INTEGRATION_OVERVIEW.md  # This file
├── INTEGRATION_PREPROCESSING.md
├── INTEGRATION_CLASSIFIER.md
└── INTEGRATION_USABILITY.md
```

## Important Constraints

### Preprocessing Team

- **MUST** return exactly 8 channels (extract from 17-channel stream)
- **MUST** process data in < 20ms for real-time operation
- **MUST** handle LSL connection/disconnection
- **MUST** reset filter states after calibration

### Classifier Team

- **MUST** return `ClassificationResult` with correct structure
- **MUST** process data in < 10ms for real-time operation
- **MUST** handle empty input data gracefully
- **SHOULD** support calibration data for better accuracy

### Usability Team

- **SHOULD** preserve core BCI functionality
- **SHOULD** maintain backwards compatibility
- **CAN** create new UI components alongside existing ones
- **SHOULD** use configuration system for easy toggling

## Communication Between Teams

### Data Formats

All teams must agree on data formats:

- **Preprocessor → Classifier**: `NDArray[np.float64]` of shape `(n_samples, 8)`
- **Classifier → Controller**: `ClassificationResult` object
- **Controller → Canvas**: `List[CursorPosition]` objects

### Interface Contracts

Teams should document their interfaces:

```python
# Example: Preprocessor interface contract
class PreprocessorInterface:
    """
    Interface contract for preprocessors.
    
    All preprocessors must implement:
    - connect_lsl() -> bool
    - pull_and_process(n_samples) -> NDArray
    - get_recent_data(seconds) -> NDArray
    - reset() -> None
    """
    pass
```

## Version Control Strategy

### Branching

Each team should work on their own branch:

```bash
# Preprocessing team
git checkout -b feature/preprocessing-custom

# Classifier team
git checkout -b feature/classifier-ml

# Usability team
git checkout -b feature/usability-accessible
```

### Merging

1. Test your changes thoroughly
2. Ensure backwards compatibility
3. Update documentation
4. Create pull request with description
5. Coordinate with other teams if interfaces change

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure new modules are exported in `__init__.py`
2. **Shape Mismatches**: Verify data shapes match expected formats
3. **Performance Issues**: Profile code to find bottlenecks
4. **Integration Failures**: Test components individually first

### Getting Help

1. Review your specific integration guide
2. Check existing implementations as examples
3. Test with minimal changes first
4. Contact project maintainer for interface questions

## Next Steps

1. **Read your specific integration guide** in detail
2. **Review existing code** to understand current implementation
3. **Plan your changes** before starting implementation
4. **Test incrementally** as you develop
5. **Coordinate with other teams** if interfaces need to change

---

**Questions?** Contact the project maintainer or refer to your specific integration guide.
