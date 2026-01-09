# g.tec Unicorn Black Setup Guide

## Option 1: Python-Native Streaming (Recommended for Hackathon)

This approach streams directly from Python without the Unicorn Suite GUI.

### Prerequisites

1. **Install Unicorn Suite** (required for drivers and UnicornPy)
   - Download from: https://www.gtec.at/product/unicorn-suite/
   - Install with default settings

2. **Pair Unicorn Black via Bluetooth**
   - Power on the Unicorn Black
   - Go to Windows Bluetooth settings
   - Pair the device (PIN: usually 0000 or 1234)

3. **Add UnicornPy to Python Path**
   
   The UnicornPy module is installed with Unicorn Suite at:
   ```
   C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib
   ```
   
   Either:
   - Copy contents to your conda environment's site-packages, OR
   - The code auto-detects this path

### Usage

```python
# Start the streamer
from src.bci.unicorn_streamer import UnicornLSLStreamer

streamer = UnicornLSLStreamer()
streamer.start()  # Connects and starts LSL stream

# Your BCI code runs here...
# The stream is available as "Unicorn" in LSL

streamer.stop()  # When done
```

Or from command line:
```bash
conda activate hack
python -m src.bci.unicorn_streamer
```

### Check Setup
```bash
conda activate hack
python -c "from src.bci.unicorn_streamer import check_unicorn_setup; check_unicorn_setup()"
```

---

## Option 2: Unicorn Suite LSL Interface

Use this if you prefer the GUI or need to visualize raw signals.

### Steps

1. **Open Unicorn Suite**
2. **Connect to your device**
3. **Go to Apps → LSL Interface**
4. **Click "Start" to begin streaming**
5. **Run BCI-UPIC and click "Connect LSL"**

The stream will appear as "UN-XXXX-XXXX" (your device serial).

---

## Troubleshooting

### "No devices found"
- Ensure Unicorn is powered on (LED blinking)
- Check Bluetooth pairing in Windows settings
- Try restarting Bluetooth adapter

### "UnicornPy not available"
- Install Unicorn Suite
- Check the path exists: `C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib`
- Manually add to sys.path if needed

### "Connection failed"
- Close Unicorn Suite if open (can't have two connections)
- Restart the Unicorn Black
- Re-pair Bluetooth

### LSL stream not detected
- Check firewall isn't blocking LSL (UDP multicast)
- Verify stream exists: `python -c "from pylsl import resolve_streams; print(resolve_streams(2.0))"`

---

## Channel Configuration

The Unicorn Black has 8 EEG channels:

| Index | Name | Location | Best For |
|-------|------|----------|----------|
| 0 | Fz | Frontal midline | - |
| 1 | C3 | Left motor | Motor imagery |
| 2 | Cz | Central midline | - |
| 3 | C4 | Right motor | Motor imagery |
| 4 | Pz | Parietal midline | P300 |
| 5 | PO7 | Left occipital | **SSVEP** |
| 6 | Oz | Occipital midline | **SSVEP** |
| 7 | PO8 | Right occipital | **SSVEP** |

For SSVEP (our 15Hz/10Hz paradigm), channels **PO7, Oz, PO8** (indices 5, 6, 7) are most important.

---

## Quick Test

```bash
# 1. Activate environment
conda activate hack

# 2. Check setup
python -c "from src.bci.unicorn_streamer import check_unicorn_setup; check_unicorn_setup()"

# 3. Start streamer (if using Python-native)
python -m src.bci.unicorn_streamer

# 4. In another terminal, run the BCI app
python bci_main.py --mode gui

# 5. Click "Connect LSL" in the GUI
```
