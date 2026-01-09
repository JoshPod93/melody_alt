#!/usr/bin/env python
"""
BCI-UPIC Startup Helper

Guides you through the Unicorn Black setup and launches the BCI app.
"""

import subprocess
import sys
import time
import os

# Paths
UNICORN_LSL_PATH = r"C:\Users\joshp\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn LSL\Unicorn LSL.exe"

def print_header():
    print("=" * 60)
    print("  BCI-UPIC Startup")
    print("=" * 60)

def print_step(num, text):
    print(f"\n[Step {num}] {text}")

def wait_for_enter(prompt="Press ENTER when done..."):
    input(f"  >>> {prompt}")

def check_lsl_stream():
    """Check if Unicorn LSL stream is available."""
    try:
        from pylsl import resolve_streams
        print("  Searching for LSL streams (5 sec)...")
        streams = resolve_streams(5.0)
        
        unicorn = None
        for s in streams:
            if s.channel_count() >= 8:
                unicorn = s
                break
        
        if unicorn:
            print(f"  [OK] Found: {unicorn.name()} ({unicorn.channel_count()} channels @ {unicorn.nominal_srate()} Hz)")
            return True
        else:
            print("  [!] No Unicorn stream found")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    print_header()
    
    # Step 1: Bluetooth dongle
    print_step(1, "INSERT BLUETOOTH DONGLE")
    print("  - Plug in the Unicorn Bluetooth USB dongle")
    wait_for_enter()
    
    # Step 2: Power on headset
    print_step(2, "POWER ON UNICORN HEADSET")
    print("  - Hold power button for 2 seconds")
    print("  - LED should start blinking (pairing mode)")
    print("  - If already paired, LED will be solid")
    wait_for_enter()
    
    # Step 3: Check Bluetooth pairing
    print_step(3, "CHECK BLUETOOTH PAIRING")
    print("  - If not paired: Windows Settings > Bluetooth > Add device")
    print("  - Select 'UN-XXXX.XX.XX' from list")
    print("  - PIN is usually: 0000")
    wait_for_enter("Press ENTER once paired (or already paired)...")
    
    # Step 4: Launch Unicorn LSL
    print_step(4, "LAUNCH UNICORN LSL")
    if os.path.exists(UNICORN_LSL_PATH):
        print(f"  Opening: {UNICORN_LSL_PATH}")
        try:
            subprocess.Popen([UNICORN_LSL_PATH], shell=True)
            print("  [OK] Unicorn LSL launched")
        except Exception as e:
            print(f"  [!] Could not auto-launch: {e}")
            print(f"  Manually open: {UNICORN_LSL_PATH}")
    else:
        print(f"  [!] Not found at expected path")
        print(f"  Manually open: Unicorn LSL.exe")
    
    print("\n  In Unicorn LSL:")
    print("    1. Select your device from dropdown")
    print("    2. Click 'Open'")
    print("    3. Click 'Start'")
    wait_for_enter("Press ENTER once LSL is streaming...")
    
    # Step 5: Verify LSL stream
    print_step(5, "VERIFY LSL STREAM")
    if check_lsl_stream():
        print("  [OK] Unicorn is streaming!")
    else:
        print("\n  Troubleshooting:")
        print("    - Make sure you clicked 'Start' in Unicorn LSL")
        print("    - Check device is connected (not showing error)")
        print("    - Try: Close Unicorn LSL, restart headset, try again")
        
        retry = input("\n  Retry check? (y/n): ").strip().lower()
        if retry == 'y':
            if not check_lsl_stream():
                print("\n  [!] Still no stream. Please troubleshoot and run again.")
                sys.exit(1)
    
    # Step 6: Launch BCI app
    print_step(6, "LAUNCH BCI APP")
    print("  Starting BCI-UPIC interface...")
    print("\n  In the app:")
    print("    1. Click 'Connect LSL' button")
    print("    2. Status should change to 'Connected: UN-XXXX'")
    print("    3. Click 'Start Composition' to begin!")
    print("\n" + "=" * 60)
    print("  Launching GUI...")
    print("=" * 60)
    
    # Launch the BCI app
    os.system("python bci_main.py --mode gui")

if __name__ == "__main__":
    main()
