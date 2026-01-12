# Preprocessing Integration Guide

## Overview

This guide explains how to integrate a custom preprocessing pipeline into the BCI-UPIC system. The preprocessing module is responsible for cleaning and preparing raw EEG signals for SSVEP classification.

## Current Architecture

The preprocessing system is located in `src/bci/preprocessing.py` and consists of:

1. **`EEGPreprocessor`** - Base class for real-time EEG preprocessing
2. **`LSLPreprocessor`** - Extends `EEGPreprocessor` with LSL streaming support
3. **`SimulatedEEGSource`** - For testing without hardware

### Data Flow

```
LSL Stream (17 channels) → LSLPreprocessor → Extract 8 EEG channels → 
Filter → Normalize → Buffer → Classifier
```

## Integration Points

### 1. Replace the Preprocessor Class

The main integration point is the `LSLPreprocessor` class in `src/bci/preprocessing.py`. The interface (`src/bci/interface.py`) uses it like this:

```python
# In BCICompositionWindow.__init__()
self.preprocessor = LSLPreprocessor(sample_rate=250, n_channels=8)

# During composition
processed = self.preprocessor.pull_and_process(n_samples=16)
eeg_buffer = self.preprocessor.get_recent_data(0.5)  # 0.5 second window
```

### 2. Required Interface

Your custom preprocessor **must** implement these methods to be compatible:

#### Essential Methods

```python
class YourCustomPreprocessor:
    def __init__(self, sample_rate: float = 250.0, n_channels: int = 8):
        """Initialize your preprocessor."""
        pass
    
    def connect_lsl(self, stream_name: Optional[str] = None) -> bool:
        """
        Connect to LSL stream.
        
        Returns:
            True if connected successfully
        """
        pass
    
    def disconnect_lsl(self) -> None:
        """Disconnect from LSL stream."""
        pass
    
    @property
    def is_lsl_connected(self) -> bool:
        """Check if LSL is connected."""
        pass
    
    def pull_and_process(self, n_samples: int = 16) -> NDArray[np.float64]:
        """
        Pull data from LSL and process it.
        
        Args:
            n_samples: Number of samples to pull
            
        Returns:
            Processed EEG data of shape (n_samples, 8) - MUST be 8 channels
        """
        pass
    
    def get_recent_data(self, seconds: float) -> NDArray[np.float64]:
        """
        Get the most recent N seconds of processed data.
        
        Args:
            seconds: Number of seconds of data
            
        Returns:
            Processed data of shape (n_samples, 8)
        """
        pass
    
    def reset(self) -> None:
        """Reset all filter states and buffers (called after calibration)."""
        pass
```

#### Important Constraints

1. **Channel Count**: The system expects **exactly 8 EEG channels** (Unicorn Black has 8 EEG + 9 auxiliary). Extract only the first 8 channels from the 17-channel LSL stream.

2. **Data Format**: All methods must return `numpy.ndarray` with dtype `np.float64`.

3. **Real-time Processing**: Methods must be fast enough for real-time operation (~60 FPS updates).

4. **Sample Rate**: The system uses 250 Hz (Unicorn Black default), but your preprocessor should handle the actual stream rate.

### 3. Integration Steps

#### Step 1: Create Your Preprocessor

Create a new file `src/bci/your_preprocessing.py`:

```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .lsl_stream import LSLReceiver

class YourCustomPreprocessor:
    """Your custom preprocessing implementation."""
    
    def __init__(self, sample_rate: float = 250.0, n_channels: int = 8):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self._lsl_receiver = None
        self._lsl_connected = False
        # ... your initialization code ...
    
    def connect_lsl(self, stream_name: Optional[str] = None) -> bool:
        """Connect to LSL stream."""
        try:
            self._lsl_receiver = LSLReceiver(stream_name=stream_name)
            if self._lsl_receiver.connect(stream_name):
                self.sample_rate = self._lsl_receiver.sample_rate
                self._lsl_connected = True
                self._lsl_receiver.start_receiving()
                return True
            return False
        except Exception as e:
            print(f"LSL connection failed: {e}")
            return False
    
    def disconnect_lsl(self) -> None:
        """Disconnect from LSL."""
        if self._lsl_receiver:
            self._lsl_receiver.disconnect()
            self._lsl_connected = False
    
    @property
    def is_lsl_connected(self) -> bool:
        return self._lsl_connected and self._lsl_receiver is not None
    
    def pull_and_process(self, n_samples: int = 16) -> NDArray[np.float64]:
        """Pull and process data from LSL."""
        if not self.is_lsl_connected:
            return np.array([])
        
        # Pull raw data from LSL
        samples, timestamps = self._lsl_receiver.pull_chunk(n_samples)
        
        if len(samples) == 0:
            return np.array([])
        
        # CRITICAL: Extract only first 8 EEG channels (Unicorn sends 17 total)
        if samples.shape[1] > 8:
            samples = samples[:, :8]
        
        # Apply your preprocessing pipeline here
        processed = self._your_processing_pipeline(samples)
        
        return processed
    
    def get_recent_data(self, seconds: float) -> NDArray[np.float64]:
        """Get recent processed data."""
        if not self.is_lsl_connected:
            return np.array([])
        
        # Get raw data from LSL buffer
        raw_data = self._lsl_receiver.get_recent_data(seconds)
        
        if len(raw_data) == 0:
            return np.array([])
        
        # Extract 8 EEG channels
        if raw_data.shape[1] > 8:
            raw_data = raw_data[:, :8]
        
        # Process
        return self._your_processing_pipeline(raw_data)
    
    def _your_processing_pipeline(self, raw_data: NDArray) -> NDArray[np.float64]:
        """
        Your custom preprocessing pipeline.
        
        Args:
            raw_data: Raw EEG data of shape (n_samples, 8)
            
        Returns:
            Processed data of same shape
        """
        # TODO: Implement your preprocessing steps:
        # - Filtering (bandpass, notch, etc.)
        # - Artifact rejection
        # - Normalization
        # - Any other steps
        
        processed = raw_data.copy()  # Placeholder
        
        # Example: Bandpass filter
        # processed = your_bandpass_filter(processed, ...)
        
        # Example: Artifact rejection
        # processed = your_artifact_rejection(processed, ...)
        
        # Example: Normalization
        # processed = your_normalization(processed, ...)
        
        return processed.astype(np.float64)
    
    def reset(self) -> None:
        """Reset all internal state."""
        # Reset filters, buffers, statistics, etc.
        pass
```

#### Step 2: Update the Interface

Modify `src/bci/interface.py` to use your preprocessor:

```python
# At the top of the file, change the import:
from .preprocessing import YourCustomPreprocessor  # Instead of LSLPreprocessor

# In BCICompositionWindow.__init__(), change:
self.preprocessor = YourCustomPreprocessor(sample_rate=250, n_channels=8)
```

#### Step 3: Update Exports (Optional)

If you want to export your preprocessor from the module, update `src/bci/__init__.py`:

```python
from .your_preprocessing import YourCustomPreprocessor

__all__ = [
    # ... existing exports ...
    'YourCustomPreprocessor',
]
```

### 4. Testing Your Integration

#### Test 1: Basic Functionality

```python
from src.bci.your_preprocessing import YourCustomPreprocessor

preprocessor = YourCustomPreprocessor(sample_rate=250, n_channels=8)

# Test LSL connection
if preprocessor.connect_lsl():
    print("Connected!")
    
    # Test data pulling
    data = preprocessor.pull_and_process(n_samples=64)
    print(f"Received {len(data)} samples, shape: {data.shape}")
    
    # Should be (64, 8) or less if no data available
    assert data.shape[1] == 8, "Must return 8 channels!"
    
    preprocessor.disconnect_lsl()
```

#### Test 2: Real-time Performance

```python
import time

preprocessor = YourCustomPreprocessor()
preprocessor.connect_lsl()

start = time.perf_counter()
for _ in range(100):
    data = preprocessor.pull_and_process(n_samples=16)
elapsed = time.perf_counter() - start

print(f"100 calls took {elapsed:.3f}s ({elapsed/100*1000:.2f}ms per call)")
# Should be < 20ms per call for real-time operation
```

#### Test 3: Integration with Interface

Run the full interface and verify:
1. LSL connection works
2. Composition starts without errors
3. Cursor moves based on classification
4. No crashes or performance issues

### 5. Key Considerations

#### Channel Extraction

The Unicorn Black sends 17 channels total:
- Channels 0-7: EEG electrodes (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8)
- Channels 8-16: Auxiliary channels (battery, etc.)

**You MUST extract only channels 0-7** before processing:

```python
if samples.shape[1] > 8:
    samples = samples[:, :8]  # Extract first 8 channels
```

#### Real-time Constraints

- Processing must complete in < 20ms per update
- Use efficient algorithms (avoid heavy FFTs in the hot path)
- Consider buffering strategies for batch processing

#### Filter State Management

- Maintain filter states (`zi` in scipy) for real-time filtering
- Reset states when `reset()` is called (after calibration)
- Handle edge cases (NaN, Inf, empty buffers)

#### Memory Management

- Use fixed-size buffers (deque with maxlen)
- Avoid memory leaks in long-running sessions
- Clear buffers appropriately on reset

### 6. Example: Adding ICA

Here's an example of how you might add Independent Component Analysis (ICA):

```python
from sklearn.decomposition import FastICA

class ICAPreprocessor(YourCustomPreprocessor):
    """Preprocessor with ICA for artifact removal."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ica = FastICA(n_components=8, random_state=42)
        self._ica_fitted = False
        self._calibration_buffer = []
    
    def fit_ica(self, calibration_data: NDArray) -> None:
        """Fit ICA on calibration data."""
        # calibration_data shape: (n_samples, 8)
        self.ica.fit(calibration_data)
        self._ica_fitted = True
    
    def _your_processing_pipeline(self, raw_data: NDArray) -> NDArray:
        """Apply ICA and other preprocessing."""
        # Standard preprocessing first
        processed = super()._your_processing_pipeline(raw_data)
        
        # Apply ICA if fitted
        if self._ica_fitted:
            processed = self.ica.transform(processed)
        
        return processed
```

### 7. Troubleshooting

#### Issue: "Expected 8 channels, got 17"

**Solution**: Make sure you extract 8 channels before processing:
```python
if samples.shape[1] > 8:
    samples = samples[:, :8]
```

#### Issue: Performance too slow

**Solutions**:
- Profile your code to find bottlenecks
- Use vectorized NumPy operations
- Consider downsampling if sample rate is very high
- Cache filter coefficients

#### Issue: Filter state corruption

**Solution**: Properly initialize and reset filter states:
```python
def reset(self):
    # Reinitialize all filter states
    self._init_filters()
    self._init_buffers()
```

### 8. Integration Checklist

- [ ] Preprocessor implements all required methods
- [ ] Returns exactly 8 channels (extracted from 17-channel stream)
- [ ] Handles LSL connection/disconnection
- [ ] Processes data in real-time (< 20ms per update)
- [ ] Resets properly after calibration
- [ ] Handles edge cases (empty buffers, NaN, Inf)
- [ ] Tested with actual Unicorn Black hardware
- [ ] No memory leaks in long sessions
- [ ] Updated interface.py to use new preprocessor
- [ ] Documentation updated

### 9. Getting Help

If you encounter issues:

1. Check the existing `LSLPreprocessor` implementation in `src/bci/preprocessing.py` as a reference
2. Review the interface code in `src/bci/interface.py` to see how preprocessing is used
3. Test with simulated data first before using real hardware
4. Use logging to debug data flow issues

### 10. Next Steps

After integrating your preprocessing:

1. Test thoroughly with real EEG data
2. Compare performance with baseline preprocessor
3. Document any new parameters or configuration options
4. Consider adding configuration file support for easy tuning

---

**Questions?** Contact the project maintainer or refer to the main project documentation.
