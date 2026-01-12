# Performance Optimization Research: LSL Data Pulling & Flicker Protocol

## Date: 2026-01-12

## Problem Statement

Current system experiences performance bottlenecks:
1. **LSL Data Pulling**: Python `pylsl` library pulling data at 16ms intervals (62.5Hz) is blocking the event loop
2. **Flicker Protocol**: PyQt6 `QTimer` on Windows fires at ~21ms instead of requested 8ms, and `paintEvent` is throttled to ~100ms intervals
3. **Event Loop Blocking**: LSL data processing and GUI updates compete for CPU time, causing flickering inconsistencies

## Current Architecture

### LSL Data Pulling
- **Location**: `src/bci/lsl_stream.py` → `LSLReceiver.pull_chunk()`
- **Frequency**: 16ms intervals (~62.5Hz) during calibration, same during composition
- **Method**: Python `pylsl` wrapper around C++ LSL library
- **Threading**: Background thread with queue-based communication
- **Bottleneck**: Python GIL, queue operations, NumPy array conversions

### Flicker Protocol
- **Location**: `src/bci/interface.py` → `FlickerWidget.paintEvent()`
- **Frequency**: Requested 8ms intervals (~125Hz), actual ~21ms (timer) + ~100ms (paintEvent)
- **Method**: PyQt6 `QTimer` → `update()` → `paintEvent()` → real-time intensity calculation
- **Bottleneck**: Windows timer precision, Qt event loop batching, Python overhead

## Research Findings: Radical Solutions

### 1. **C/C++ Native Implementation** (Highest Impact)

#### LSL Data Pulling
- **Option A**: Rewrite `LSLReceiver` in C++ as a Python extension module
  - Direct C++ LSL API calls (no Python wrapper overhead)
  - Zero-copy data transfer using NumPy C API
  - Native threading (no GIL)
  - **Expected improvement**: 5-10x faster data pulling

- **Option B**: Use existing C++ LSL library directly via ctypes/ctypes++
  - Lower overhead than `pylsl` wrapper
  - Still requires Python GIL management
  - **Expected improvement**: 2-3x faster

#### Flicker Protocol
- **Option A**: OpenGL-based rendering (C++ backend)
  - Hardware-accelerated rendering on GPU
  - Frame-synchronized updates (vsync)
  - Bypass Qt's paintEvent entirely
  - **Expected improvement**: 10-20x faster, true frame synchronization

- **Option B**: Direct Windows API rendering (C++/Win32)
  - `SetTimer` with `WM_TIMER` messages (higher precision than Qt)
  - Direct window drawing via GDI/GDI+
  - **Expected improvement**: 3-5x faster, better timer precision

### 2. **Cython/Numba JIT Compilation** (Medium Impact, Easier Integration)

#### LSL Data Pulling
- **Cython**: Compile critical paths to C extensions
  - Type-annotated functions for zero-overhead NumPy operations
  - Release GIL during data pulling
  - **Expected improvement**: 2-4x faster

- **Numba**: JIT-compile NumPy-heavy functions
  - Automatic optimization of array operations
  - **Expected improvement**: 1.5-3x faster

#### Flicker Protocol
- **Cython**: Compile intensity calculation to C
  - Real-time sine wave calculation in C
  - **Expected improvement**: 2-3x faster calculation

### 3. **Hardware Acceleration** (GPU/FPGA)

#### GPU Acceleration
- **OpenGL/Vulkan**: Hardware-accelerated flicker rendering
  - Offload rendering to GPU
  - True vsync synchronization
  - **Expected improvement**: 10-20x faster rendering, frame-perfect timing

- **CUDA/OpenCL**: Parallel signal processing
  - Offload filtering/classification to GPU
  - **Expected improvement**: 5-10x faster processing (for large buffers)

#### FPGA (Extreme Solution)
- Custom hardware for flicker generation
- Hardware LSL receiver
- **Expected improvement**: Microsecond-level precision
- **Cost**: High development time, requires FPGA hardware

### 4. **Alternative Frameworks** (Complete Rewrite)

#### For Flicker Protocol
- **SDL2** (C library, Python bindings available)
  - Lower-level than Qt, better timer precision
  - Hardware-accelerated rendering
  - **Expected improvement**: 3-5x faster

- **GLFW** (C library, Python bindings)
  - Minimal overhead OpenGL windowing
  - **Expected improvement**: 5-10x faster

- **DirectX/Windows API** (C++/Win32)
  - Native Windows rendering
  - **Expected improvement**: 5-10x faster

#### For LSL
- **Native C++ application** with Python scripting layer
  - Core processing in C++, Python for high-level control
  - **Expected improvement**: 5-10x faster overall

### 5. **System-Level Optimizations** (Lower Impact, Easier)

#### Windows-Specific
- **High-precision timers**: `timeSetEvent` (multimedia timer API)
  - Better precision than `QTimer` on Windows
  - **Expected improvement**: 2-3x better timer precision

- **Thread priorities**: Set LSL thread to high priority
  - Reduce scheduling delays
  - **Expected improvement**: 10-20% reduction in jitter

- **CPU affinity**: Pin threads to specific cores
  - Reduce context switching overhead
  - **Expected improvement**: 5-15% reduction in latency

#### LSL Configuration
- **Network tuning**: Adjust `TimeProbeMaxRTT`, `TimeProbeInterval`
- **Buffer sizes**: Optimize chunk sizes for minimal overhead
- **Expected improvement**: 10-30% reduction in latency

## Recommended Approach: Hybrid Solution

### Phase 1: Quick Wins (1-2 days)
1. **Cython for LSL data pulling**
   - Compile `pull_chunk()` and array operations
   - Release GIL during data acquisition
   - **Expected**: 2-3x improvement

2. **Windows multimedia timer for flicker**
   - Replace `QTimer` with `timeSetEvent` (via ctypes)
   - **Expected**: 2-3x better timer precision

3. **Thread priority optimization**
   - Set LSL thread to high priority
   - **Expected**: 10-20% reduction in jitter

### Phase 2: Medium-Term (1-2 weeks)
1. **OpenGL-based flicker rendering**
   - Replace `paintEvent` with OpenGL shader
   - Hardware-accelerated, vsync-synchronized
   - **Expected**: 10-20x improvement, frame-perfect timing

2. **Cython for signal processing**
   - Compile filtering and classification
   - **Expected**: 2-4x improvement

### Phase 3: Long-Term (1-2 months, if needed)
1. **C++ extension module for LSL**
   - Native C++ implementation
   - Zero-copy NumPy integration
   - **Expected**: 5-10x improvement

2. **Complete C++ backend with Python bindings**
   - Core processing in C++, Python for control
   - **Expected**: 10-20x overall improvement

## Implementation Priority

### Critical (Do First)
1. ✅ **Windows multimedia timer** - Easy, high impact on flicker precision
2. ✅ **Cython for LSL pulling** - Moderate effort, high impact on data acquisition
3. ✅ **Thread priorities** - Trivial, moderate impact

### High Priority
4. **OpenGL flicker rendering** - High effort, very high impact
5. **Cython for signal processing** - Moderate effort, high impact

### Medium Priority
6. **C++ LSL extension** - High effort, very high impact
7. **Complete C++ backend** - Very high effort, maximum impact

## References

- LSL Performance: https://sccn.ucsd.edu/githubwiki/files/20210615-EEGLAB_workshop.pdf
- PyQt6 Timer Issues: Windows timer precision limitations
- OpenGL for SSVEP: Hardware-accelerated stimulus rendering
- Cython Performance: https://cython.readthedocs.io/
- Windows Multimedia Timer: `timeSetEvent` API documentation

## Next Steps

1. **Profile current system** to quantify bottlenecks
2. **Implement Phase 1 quick wins** (multimedia timer, thread priorities)
3. **Benchmark improvements** before proceeding to Phase 2
4. **Evaluate OpenGL approach** for flicker rendering
5. **Consider Cython** for LSL and signal processing if Phase 1 insufficient
