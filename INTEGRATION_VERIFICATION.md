# CCA Fix Integration Verification

## ✅ Fix is Integrated for Both Paths

The reference resizing fix is **already integrated** into both the main live composition and data validation methods.

### Call Chain

1. **Main Live Composition** (`src/bci/interface.py:1077`):
   ```python
   eeg_buffer = self.preprocessor.get_recent_data(0.3)  # 0.3s chunks
   result = self.classifier.classify(eeg_buffer, method="cca")
   ```
   ↓
   `classifier.classify(method="cca")` → `classify_cca()` (line 530)
   ↓
   **Uses resized references** (lines 361-372)

2. **Data Validation** (`src/bci/interface.py:2406, 2421`):
   ```python
   result = self.classifier.classify(chunk, method="cca")  # 0.3s chunks
   ```
   ↓
   `classifier.classify(method="cca")` → `classify_cca()` (line 530)
   ↓
   **Uses resized references** (lines 361-372)

### Fix Location

**File**: `src/bci/classifier.py`  
**Method**: `classify_cca()` (lines 334-426)  
**Fix**: Lines 361-372 - Resizes references to match data chunk size instead of padding data

```python
# Resize reference signals to match data length (preserves phase profile)
n_data = data.shape[0]
n_ref = self._ref_signals_up.shape[0]

if n_data != n_ref:
    # Resize references to match data chunk size (preserves phase)
    ref_up_resized = self._resize_reference(self._ref_signals_up, n_data)
    ref_down_resized = self._resize_reference(self._ref_signals_down, n_data)
else:
    ref_up_resized = self._ref_signals_up
    ref_down_resized = self._ref_signals_down
```

### Verification

- ✅ Both paths call `classify(method="cca")`
- ✅ Both route to `classify_cca()`
- ✅ `classify_cca()` contains the fix
- ✅ References are resized for any chunk size (0.3s, 0.5s, etc.)
- ✅ Phase alignment is preserved (especially for DOWN target with π offset)

### Result

**No additional changes needed** - the fix is already integrated and will work for:
- Main live composition (0.3s chunks every 50ms)
- Data validation/checking (0.3s chunks for testing)
- Any future use of `classify(method="cca")`
