# g.tec Unicorn Black Setup Guide

> **Note**: g.Pype (g.tec's official Python SDK) does NOT yet support Unicorn Black directly.
> Unicorn support is "planned for upcoming releases" per their FAQ.
> Use one of the options below instead.

## Option 1: Unicorn Suite LSL Interface (Easiest - Recommended)

Use the built-in LSL streaming from Unicorn Suite. No extra code needed.

### Prerequisites

1. **Install Unicorn Suite** from g.tec
   - Download: https://www.gtec.at/product/unicorn-suite/
   - Install with default settings

2. **Pair Unicorn Black via Bluetooth**
   - Power on the Unicorn Black (LED blinking)
   - Windows Settings → Bluetooth → Add device
   - Pair the device (PIN: usually `0000` or `1234`)

### Steps

1. **Open Unicorn Suite**
2. **Connect to your device** (select from dropdown)
3. **Go to Apps → LSL Interface**
4. **Click "Start"** to begin streaming
5. **In a terminal, run BCI-UPIC:**
   ```bash
   conda activate hack
   python bci_main.py --mode gui
   ```
6. **Click "Connect LSL"** in the GUI

The stream appears as `UN-XXXX-XXXX` (your device serial).

---

## Option 2: Python-Native Streaming (Advanced)

Stream directly from Python without the Unicorn Suite GUI.
Useful for programmatic control or running everything from code.

### Prerequisites

1. **Install Unicorn Suite** (required for drivers and UnicornPy module)
2. **Pair via Bluetooth** (same as Option 1)

### Setup UnicornPy

The UnicornPy module is installed with Unicorn Suite at:
```
C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib
```

Our code auto-detects this path. If it doesn't work, copy the `Lib` contents to your conda site-packages.

### Usage

**From Python:**
```python
from src.bci.unicorn_streamer import UnicornLSLStreamer

streamer = UnicornLSLStreamer()
streamer.start()  # Connects and starts LSL stream

# Your BCI code runs here...
# Stream available as "Unicorn" in LSL

streamer.stop()
```

**From command line (two terminals):**

Terminal 1 - Start streamer:
```bash
conda activate hack
python -m src.bci.unicorn_streamer
```

Terminal 2 - Run BCI app:
```bash
conda activate hack
python bci_main.py --mode gui
# Click "Connect LSL"
```

### Check Setup
```bash
python -c "from src.bci.unicorn_streamer import check_unicorn_setup; check_unicorn_setup()"
```

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

# 2. Check setup (shows if UnicornPy is available)
python -c "from src.bci.unicorn_streamer import check_unicorn_setup; check_unicorn_setup()"

# 3. Start Unicorn Suite LSL Interface (Option 1)
#    OR run Python streamer (Option 2):
python -m src.bci.unicorn_streamer

# 4. In another terminal, run the BCI app
python bci_main.py --mode gui

# 5. Click "Connect LSL" in the GUI
```
