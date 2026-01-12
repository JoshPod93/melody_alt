# Classifier Integration Guide

## Overview

This guide explains how to integrate a custom SSVEP classifier into the BCI-UPIC system. The classifier is responsible for decoding user intent (attending to 15Hz top target = move up, 10Hz bottom target = move down) from preprocessed EEG signals.

## Current Architecture

The classification system is located in `src/bci/classifier.py` and consists of:

1. **`SSVEPClassifier`** - Main classifier with FFT and CCA methods
2. **`AttentionTarget`** - Enum for classification output (UP, DOWN, NONE)
3. **`ClassificationResult`** - Dataclass containing classification results

### Data Flow

```
Preprocessed EEG (8 channels) → Classifier.classify() → 
ClassificationResult → Controller.update() → Cursor Movement
```

## Integration Points

### 1. Replace the Classifier Class

The main integration point is the `SSVEPClassifier` class in `src/bci/classifier.py`. The interface (`src/bci/interface.py`) uses it like this:

```python
# In BCICompositionWindow.__init__()
self.classifier = SSVEPClassifier()

# During composition (in _update_composition method)
eeg_buffer = self.preprocessor.get_recent_data(0.5)  # 0.5 second window
result = self.classifier.classify(eeg_buffer, method="cca")
controller.update(result)
```

### 2. Required Interface

Your custom classifier **must** implement these methods to be compatible:

#### Essential Methods

```python
class YourCustomClassifier:
    def __init__(self, sample_rate: float = 250.0):
        """Initialize your classifier."""
        pass
    
    def classify(
        self,
        eeg_data: NDArray[np.float64],
        method: str = "default"
    ) -> ClassificationResult:
        """
        Classify EEG data to determine SSVEP attention.
        
        Args:
            eeg_data: Preprocessed EEG data of shape (n_samples, 8)
            method: Classification method (optional, for compatibility)
            
        Returns:
            ClassificationResult with target, confidence, and scores
        """
        pass
    
    def load_calibration(self, calibration_data: 'CalibrationData') -> bool:
        """
        Load personalized calibration data (optional but recommended).
        
        Args:
            calibration_data: CalibrationData object with user's SSVEP templates
            
        Returns:
            True if calibration loaded successfully
        """
        pass
    
    @property
    def is_calibrated(self) -> bool:
        """Check if using calibration data."""
        pass
    
    def reset(self) -> None:
        """Reset classifier state (called between sessions)."""
        pass
```

#### Required Data Structures

You must use these existing types (defined in `src/bci/classifier.py`):

```python
from .classifier import AttentionTarget, ClassificationResult

# AttentionTarget enum:
# - AttentionTarget.UP (user attending to 15Hz top target)
# - AttentionTarget.DOWN (user attending to 10Hz bottom target)
# - AttentionTarget.NONE (no clear attention detected)

# ClassificationResult dataclass:
@dataclass
class ClassificationResult:
    target: AttentionTarget      # Which target user is attending to
    confidence: float            # 0-1, how confident the classification is
    power_15hz: float            # Power/correlation at 15Hz (for debugging)
    power_10hz: float            # Power/correlation at 10Hz (for debugging)
    raw_score: float             # Raw classification score (positive = up, negative = down)
```

### 3. Integration Steps

#### Step 1: Create Your Classifier

Create a new file `src/bci/your_classifier.py`:

```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .classifier import AttentionTarget, ClassificationResult
from .calibration import CalibrationData

class YourCustomClassifier:
    """Your custom SSVEP classifier implementation."""
    
    def __init__(self, sample_rate: float = 250.0):
        self.sample_rate = sample_rate
        self._using_calibration = False
        # ... your initialization code ...
    
    def classify(
        self,
        eeg_data: NDArray[np.float64],
        method: str = "default"
    ) -> ClassificationResult:
        """
        Classify EEG data.
        
        Args:
            eeg_data: Shape (n_samples, 8) - preprocessed EEG
            method: Ignored (for compatibility)
            
        Returns:
            ClassificationResult
        """
        if len(eeg_data) == 0:
            # No data - return neutral result
            return ClassificationResult(
                target=AttentionTarget.NONE,
                confidence=0.0,
                power_15hz=0.0,
                power_10hz=0.0,
                raw_score=0.0
            )
        
        # TODO: Implement your classification algorithm here
        # Example workflow:
        # 1. Extract features from eeg_data
        # 2. Compare features to 15Hz and 10Hz templates
        # 3. Determine which target user is attending to
        # 4. Calculate confidence score
        
        # Placeholder implementation:
        target = AttentionTarget.UP  # or DOWN or NONE
        confidence = 0.5
        power_15hz = 0.0
        power_10hz = 0.0
        raw_score = 0.0
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_15hz=power_15hz,
            power_10hz=power_10hz,
            raw_score=raw_score
        )
    
    def load_calibration(self, calibration_data: CalibrationData) -> bool:
        """
        Load calibration data for personalized classification.
        
        This is highly recommended for better accuracy!
        """
        try:
            # Extract reference signals from calibration
            # calibration_data contains recorded SSVEP responses
            # See src/bci/calibration.py for CalibrationData structure
            
            # Example: Get CCA references
            window_seconds = 0.5  # Match your analysis window
            ref_up, ref_down = calibration_data.get_cca_references(window_seconds)
            
            if ref_up is not None and ref_down is not None:
                # Store references for use in classify()
                self._ref_up = ref_up
                self._ref_down = ref_down
                self._using_calibration = True
                return True
            
            return False
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    @property
    def is_calibrated(self) -> bool:
        return self._using_calibration
    
    def reset(self) -> None:
        """Reset classifier state."""
        # Clear any internal buffers, history, etc.
        pass
```

#### Step 2: Update the Interface

Modify `src/bci/interface.py` to use your classifier:

```python
# At the top of the file, change the import:
from .your_classifier import YourCustomClassifier  # Instead of SSVEPClassifier

# In BCICompositionWindow.__init__(), change:
self.classifier = YourCustomClassifier()

# The rest of the code should work without changes!
```

#### Step 3: Update Exports (Optional)

Update `src/bci/__init__.py`:

```python
from .your_classifier import YourCustomClassifier

__all__ = [
    # ... existing exports ...
    'YourCustomClassifier',
]
```

### 4. Input Data Specifications

#### EEG Data Format

- **Shape**: `(n_samples, 8)` where `n_samples` varies (typically 64-256 samples for 0.5s window at 250Hz)
- **Channels**: 8 EEG channels from Unicorn Black (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8)
- **Data Type**: `np.float64`
- **Preprocessing**: Data is already filtered, normalized, and artifact-rejected

#### Channel Mapping

The 8 channels correspond to:
- Channel 0: Fz (Frontal)
- Channel 1: Cz (Central)
- Channel 2: Pz (Parietal)
- Channel 3: Oz (Occipital)
- Channel 4: P3 (Left Parietal)
- Channel 5: P4 (Right Parietal)
- Channel 6: PO7 (Left Parieto-Occipital) ← **Best for SSVEP**
- Channel 7: PO8 (Right Parieto-Occipital) ← **Best for SSVEP**

**Recommendation**: Focus on channels 3, 6, 7 (Oz, PO7, PO8) for SSVEP detection.

#### Target Frequencies

- **15 Hz**: Top target (UP movement)
- **10 Hz**: Bottom target (DOWN movement)
- **Phase**: 15Hz at 0°, 10Hz at 180° (out of phase)

### 5. Classification Algorithm Examples

#### Example 1: FFT-Based Power Ratio

```python
def classify(self, eeg_data: NDArray, method: str = "default") -> ClassificationResult:
    """Simple FFT-based classifier."""
    # Use occipital channels (best for SSVEP)
    occipital_channels = [3, 6, 7]  # Oz, PO7, PO8
    data = eeg_data[:, occipital_channels]
    
    # Average across channels
    data_avg = np.mean(data, axis=1)
    
    # Compute FFT
    n_samples = len(data_avg)
    freqs = np.fft.rfftfreq(n_samples, 1/self.sample_rate)
    fft_vals = np.abs(np.fft.rfft(data_avg))
    
    # Get power at target frequencies
    bandwidth = 1.0  # Hz
    power_15hz = self._get_band_power(fft_vals, freqs, 15.0, bandwidth)
    power_10hz = self._get_band_power(fft_vals, freqs, 10.0, bandwidth)
    
    # Calculate ratio
    total_power = power_15hz + power_10hz + 1e-10
    ratio = (power_15hz - power_10hz) / total_power
    
    # Determine target and confidence
    confidence = min(abs(ratio) * 2, 1.0)
    if confidence < 0.1:  # Threshold
        target = AttentionTarget.NONE
    elif ratio > 0:
        target = AttentionTarget.UP
    else:
        target = AttentionTarget.DOWN
    
    return ClassificationResult(
        target=target,
        confidence=confidence,
        power_15hz=power_15hz,
        power_10hz=power_10hz,
        raw_score=ratio
    )
```

#### Example 2: Canonical Correlation Analysis (CCA)

```python
def classify(self, eeg_data: NDArray, method: str = "default") -> ClassificationResult:
    """CCA-based classifier with calibration support."""
    # Use occipital channels
    occipital_channels = [3, 6, 7]
    data = eeg_data[:, occipital_channels]
    
    # Generate or load reference signals
    if self._using_calibration:
        ref_up = self._ref_up
        ref_down = self._ref_down
    else:
        # Generate synthetic references
        ref_up = self._generate_reference(15.0, phase=0.0, n_samples=len(data))
        ref_down = self._generate_reference(10.0, phase=np.pi, n_samples=len(data))
    
    # Compute CCA correlations
    corr_up = self._cca_correlation(data, ref_up)
    corr_down = self._cca_correlation(data, ref_down)
    
    # Determine target
    cca_diff = corr_up - corr_down
    confidence = min(abs(cca_diff) * 2, 1.0)
    
    if confidence < 0.1:
        target = AttentionTarget.NONE
    elif cca_diff > 0:
        target = AttentionTarget.UP
    else:
        target = AttentionTarget.DOWN
    
    return ClassificationResult(
        target=target,
        confidence=confidence,
        power_15hz=corr_up,
        power_10hz=corr_down,
        raw_score=cca_diff
    )
```

#### Example 3: Machine Learning Classifier

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

class MLClassifier(YourCustomClassifier):
    """Machine learning-based classifier."""
    
    def __init__(self, sample_rate: float = 250.0):
        super().__init__(sample_rate)
        self.model = RandomForestClassifier(n_estimators=100)
        self._is_trained = False
    
    def train(self, X: NDArray, y: NDArray) -> None:
        """
        Train the classifier.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels (0=DOWN, 1=UP, 2=NONE)
        """
        self.model.fit(X, y)
        self._is_trained = True
    
    def extract_features(self, eeg_data: NDArray) -> NDArray:
        """Extract features from EEG data."""
        features = []
        
        # Example features:
        # - Power at 10Hz and 15Hz
        # - Spectral entropy
        # - Channel correlations
        # - etc.
        
        return np.array(features)
    
    def classify(self, eeg_data: NDArray, method: str = "default") -> ClassificationResult:
        """Classify using trained model."""
        if not self._is_trained:
            # Fallback to simple method
            return super().classify(eeg_data, method)
        
        # Extract features
        features = self.extract_features(eeg_data)
        features = features.reshape(1, -1)  # Single sample
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Map to AttentionTarget
        if prediction == 0:
            target = AttentionTarget.DOWN
        elif prediction == 1:
            target = AttentionTarget.UP
        else:
            target = AttentionTarget.NONE
        
        confidence = np.max(probabilities)
        
        return ClassificationResult(
            target=target,
            confidence=confidence,
            power_15hz=probabilities[1] if len(probabilities) > 1 else 0.0,
            power_10hz=probabilities[0] if len(probabilities) > 0 else 0.0,
            raw_score=probabilities[1] - probabilities[0] if len(probabilities) > 1 else 0.0
        )
```

### 6. Calibration Integration

The system includes a calibration phase that records user-specific SSVEP responses. **Highly recommended** to use this for better accuracy!

#### Calibration Data Structure

```python
from .calibration import CalibrationData

# CalibrationData contains:
# - trials: List of CalibrationTrial objects
# - Each trial has: frequency (15 or 10), eeg_data, timestamps
# - Methods: get_cca_references(), get_average_response(), etc.
```

#### Using Calibration

```python
def load_calibration(self, calibration_data: CalibrationData) -> bool:
    """Load calibration for personalized classification."""
    # Get reference signals (averaged SSVEP responses)
    window_seconds = 0.5
    ref_up, ref_down = calibration_data.get_cca_references(window_seconds)
    
    if ref_up is not None and ref_down is not None:
        # Store for use in classify()
        self._ref_up = ref_up
        self._ref_down = ref_down
        self._using_calibration = True
        return True
    
    return False
```

### 7. Testing Your Integration

#### Test 1: Basic Classification

```python
from src.bci.your_classifier import YourCustomClassifier
import numpy as np

classifier = YourCustomClassifier(sample_rate=250)

# Create test data with 15Hz SSVEP
n_samples = 125  # 0.5 seconds at 250Hz
t = np.arange(n_samples) / 250
eeg_data = np.random.randn(n_samples, 8) * 0.5
# Add 15Hz signal to occipital channels
eeg_data[:, 6] += 0.3 * np.sin(2 * np.pi * 15 * t)  # PO7
eeg_data[:, 7] += 0.3 * np.sin(2 * np.pi * 15 * t)  # PO8

# Classify
result = classifier.classify(eeg_data)
print(f"Target: {result.target.name}, Confidence: {result.confidence:.2f}")
# Should detect UP with reasonable confidence
```

#### Test 2: Real-time Performance

```python
import time

classifier = YourCustomClassifier()
test_data = np.random.randn(125, 8)  # 0.5s window

start = time.perf_counter()
for _ in range(100):
    result = classifier.classify(test_data)
elapsed = time.perf_counter() - start

print(f"100 classifications: {elapsed:.3f}s ({elapsed/100*1000:.2f}ms each)")
# Should be < 10ms for real-time operation
```

#### Test 3: Integration Test

Run the full interface and verify:
1. Classifier initializes without errors
2. Classification results are reasonable
3. Cursor moves correctly based on attention
4. Calibration improves accuracy

### 8. Key Considerations

#### Real-time Constraints

- Classification must complete in < 10ms
- Use efficient algorithms (pre-compute reference signals)
- Avoid heavy computations in the hot path

#### Temporal Smoothing

The existing classifier includes temporal smoothing via a history buffer. Consider implementing similar smoothing to reduce jitter:

```python
def __init__(self, ...):
    self._history = []
    self._history_size = 3

def classify(self, eeg_data, method="default"):
    result = self._classify_once(eeg_data)
    
    # Add to history
    self._history.append(result)
    if len(self._history) > self._history_size:
        self._history.pop(0)
    
    # Smooth result
    return self._smooth_result(result)
```

#### Always Return a Direction

For better user experience, consider always returning UP or DOWN (never NONE) to keep the cursor moving:

```python
# Instead of returning NONE when confidence is low:
if confidence < threshold:
    target = AttentionTarget.NONE  # Bad - cursor stops

# Consider:
if confidence < threshold:
    # Use previous direction or default to slight movement
    target = self._last_target or AttentionTarget.UP
    confidence = 0.2  # Low but non-zero
```

#### Confidence Calibration

Ensure confidence scores are well-calibrated (0.8 confidence should mean 80% accuracy). Test with known data to verify.

### 9. Troubleshooting

#### Issue: Classifier always returns NONE

**Solutions**:
- Check that you're using occipital channels (3, 6, 7)
- Verify frequency detection (check power_15hz and power_10hz values)
- Lower the confidence threshold
- Check that input data is properly preprocessed

#### Issue: Wrong direction detected

**Solutions**:
- Verify phase alignment with stimulus (15Hz at 0°, 10Hz at 180°)
- Check channel mapping (ensure correct electrode positions)
- Use calibration data for personalized templates
- Verify reference signal generation matches stimulus

#### Issue: Low confidence scores

**Solutions**:
- Improve signal-to-noise ratio in preprocessing
- Use calibration data for better templates
- Increase analysis window size (if latency allows)
- Check for artifacts in input data

### 10. Integration Checklist

- [ ] Classifier implements `classify()` method
- [ ] Returns `ClassificationResult` with correct structure
- [ ] Handles empty input data gracefully
- [ ] Processes data in real-time (< 10ms)
- [ ] Supports calibration data loading
- [ ] Uses occipital channels for SSVEP detection
- [ ] Handles edge cases (NaN, Inf, short windows)
- [ ] Updated interface.py to use new classifier
- [ ] Tested with real EEG data
- [ ] Documentation updated

### 11. Advanced Topics

#### Multi-harmonic Analysis

SSVEP responses contain harmonics. Consider analyzing multiple harmonics:

```python
harmonics = [1, 2, 3]  # Fundamental + 2 harmonics
for h in harmonics:
    power_15hz += get_power_at_freq(eeg_data, 15.0 * h)
    power_10hz += get_power_at_freq(eeg_data, 10.0 * h)
```

#### Adaptive Thresholds

Adjust classification thresholds based on user performance:

```python
def update_threshold(self, recent_results: List[ClassificationResult]):
    """Adaptively adjust threshold based on performance."""
    # Analyze recent results and adjust threshold
    pass
```

#### Multi-class Extension

If you want to support more than 2 targets, you'll need to:
1. Extend `AttentionTarget` enum
2. Update stimulus system to show more targets
3. Modify controller to handle more directions

---

**Questions?** Contact the project maintainer or refer to the main project documentation.
