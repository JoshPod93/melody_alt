# Unicorn Black System Test Setup Guide

## Run Command (If Already in unicorn_system_test Folder)

**From Command Prompt (cmd), run:**

```cmd
conda activate unicorn_test && python test_system.py
```

**Prerequisites:**
- UnicornLSL.exe must be running and streaming
- Unicorn Black device powered on and paired via Bluetooth
- Conda environment `unicorn_test` must be created (see Environment Setup below)

---

This guide provides step-by-step instructions for setting up and testing the g.tec Unicorn Black EEG system for live data capture via LSL.

## Required Software

### 1. Unicorn Suite
- **Download**: https://www.gtec.at/product/unicorn-suite/
- **Installation**: Run installer with default settings
- **Purpose**: Provides drivers and UnicornPy Python library
- **Location**: Installed to `C:\Program Files\gtec\Unicorn Suite\`

### 2. Python Environment
- **Python Version**: 3.11
- **Package Manager**: Conda (recommended) or pip

### 3. Required Python Packages
- `numpy` (>= 1.24.0)
- `pylsl` (>= 1.17.0)
- `matplotlib` (>= 3.7.0) - for data visualization plots

## Hardware Setup

### 1. Power On Device
- Press and hold power button until LED indicator blinks
- LED should show steady blinking pattern when ready

### 2. Bluetooth Pairing
1. Open Windows Settings
2. Navigate to Bluetooth & devices
3. Click "Add device"
4. Select "Bluetooth"
5. Find "UN-XXXX-XXXX" (your device serial number)
6. Pair using PIN: `0000` or `1234` (check device documentation)
7. Verify connection shows as "Connected" or "Paired"

### 3. Electrode Placement
- Ensure all 8 electrodes are properly positioned:
  - Fz: Frontal midline
  - C3: Left central
  - Cz: Central midline
  - C4: Right central
  - Pz: Parietal midline
  - PO7: Left occipital
  - Oz: Occipital midline
  - PO8: Right occipital
- Apply conductive gel if required
- Ensure good skin contact (impedance should be low)

## Environment Setup

### Option 1: Conda (Recommended)

1. Navigate to test directory:
   ```bash
   cd unicorn_system_test
   ```

2. Create conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate environment:
   ```bash
   conda activate unicorn_test
   ```

4. Verify installation:
   ```bash
   python -c "import numpy; import pylsl; print('Dependencies OK')"
   ```

### Option 2: Pip

1. Navigate to test directory:
   ```bash
   cd unicorn_system_test
   ```

2. Create virtual environment (optional):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

### Quick Test
Runs basic connectivity and battery check:
```bash
python test_system.py --quick
```

### Full Test Suite
Runs comprehensive system validation:
```bash
python test_system.py
```

### Custom Options
```bash
# Specify device serial number
python test_system.py --device UN-XXXX-XXXX

# Set LSL streaming duration (default: 10 seconds)
python test_system.py --stream-duration 15.0
```

## Test Components

The full test suite validates:

1. **LSL Stream Verification**: Confirms UnicornLSL.exe stream is available
2. **Electrode Validation**: Checks all 8 EEG channels are functioning
3. **Bandwidth Check**: Verifies data throughput meets 250 Hz specification
4. **Battery Status**: Reports current battery level (auto-detects channel)
5. **Data Capture and Visualization**: Captures 2 seconds of EEG data and generates plot

## Test Output

After running the test, you will find:

- **Test Report**: `reports/test_report_YYYYMMDD_HHMMSS.json`
  - Contains all test results in JSON format
  - Includes timestamps, status, and detailed metrics
  
- **EEG Plot**: `reports/eeg_plot_YYYYMMDD_HHMMSS.png`
  - Visual representation of captured EEG signals
  - All 8 channels displayed with time axis
  - Saved alongside report with matching timestamp

Both files are saved in the `reports/` directory with synchronized timestamps for easy pairing.

## Viewing Live Stream Data

### Option 1: Unicorn Suite (Recommended for Live Visualization)

Unicorn Suite provides built-in visualization tools for viewing the live EEG stream:

1. **Launch Unicorn Suite**
   - Open Unicorn Suite from Start Menu or desktop shortcut
   - Location: `C:\Program Files\gtec\Unicorn Suite\`

2. **Connect to Device**
   - Select your Unicorn device from the dropdown menu
   - Click "Open" to establish connection
   - Wait for connection status to show "Connected"

3. **Start LSL Streaming**
   - Navigate to: **Apps → LSL Interface**
   - Click "Start" to begin streaming
   - Stream will appear as `UN-XXXX-XXXX` (your device serial)

4. **View Live Data in Unicorn Suite**
   - Go to **View → Signal Display** or click the signal display icon
   - All 8 EEG channels will be displayed in real-time
   - Use zoom and scroll controls to examine specific time windows
   - Channel names and scales are automatically configured

5. **Additional Visualization Options**
   - **Spectrum Display**: View frequency domain (FFT) of signals
   - **Impedance Check**: Verify electrode contact quality
   - **Signal Quality**: Monitor signal-to-noise ratios

**Note**: Unicorn Suite must remain open while streaming. The LSL stream is available to other applications simultaneously.

### Option 2: Python Script (This Test Suite)

The test script automatically captures and plots 2 seconds of data:
- Plot is saved alongside the test report
- Both files use matching timestamps for easy pairing
- Location: `reports/eeg_plot_YYYYMMDD_HHMMSS.png`

### Option 3: Custom Python Code

```python
from pylsl import resolve_streams, StreamInlet

# Find the stream
streams = resolve_streams(timeout=5.0)
unicorn_stream = [s for s in streams if s.channel_count() >= 8][0]

# Create inlet
inlet = StreamInlet(unicorn_stream)

# Pull data
sample, timestamp = inlet.pull_sample()
```

### Option 4: Other LSL-Compatible Applications

- Use any LSL-compatible application
- Stream name: `UN-XXXX-XXXX` (device serial number)
- Type: EEG
- Channels: 17 (8 EEG + accelerometer + gyroscope + status channels)
- Sample rate: 250 Hz

## Troubleshooting

### "UnicornPy not available"
- Verify Unicorn Suite is installed
- Check path exists: `C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib`
- Restart terminal after installation

### "No devices found"
- Ensure Unicorn Black is powered on (LED blinking)
- Verify Bluetooth pairing in Windows Settings
- Try restarting Bluetooth adapter
- Close Unicorn Suite if open (only one connection allowed)

### "Connection failed"
- Close Unicorn Suite application
- Restart Unicorn Black device
- Re-pair Bluetooth connection
- Check device is not connected to another application

### "pylsl not available"
- Install: `pip install pylsl`
- Verify: `python -c "import pylsl; print(pylsl.__version__)"`

### Low bandwidth or missing samples
- Check Bluetooth signal strength
- Ensure no other applications are using the device
- Verify device battery is adequately charged
- Try moving closer to Bluetooth adapter

### Electrode not working
- Check electrode placement and contact
- Verify conductive gel is applied (if required)
- Check for loose connections
- Clean electrode surface

### Battery low warning
- Charge device before extended use
- Battery should be above 20% for reliable operation

## Function Library

The `unicorn_functions.py` module provides:

- `check_dependencies()`: Verify required libraries
- `list_devices()`: Get available device serial numbers
- `connect_device()`: Establish device connection
- `check_electrodes()`: Validate all EEG channels
- `check_bandwidth()`: Measure data throughput
- `check_battery()`: Get battery level
- `stream_to_lsl()`: Create live LSL data stream
- `verify_lsl_connection()`: Confirm stream accessibility

See function docstrings for detailed usage information.

## Next Steps

After successful testing:
1. Verify all electrodes show activity
2. Confirm bandwidth meets 250 Hz specification
3. Ensure battery level is adequate
4. Test LSL stream in your target application
5. Proceed with main BCI system integration
