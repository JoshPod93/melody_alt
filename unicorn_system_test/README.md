# Unicorn Black System Test

Lightweight testing suite for validating g.tec Unicorn Black EEG system.

## Run Command (If Already in unicorn_system_test Folder)

**From Command Prompt (cmd):**

```cmd
conda activate unicorn_test && python test_system.py
```

**First Time Setup:**

```cmd
conda env create -f environment.yml
conda activate unicorn_test
python test_system.py
```

## Test Coverage

- LSL stream verification
- Electrode validation (8 channels)
- Bandwidth/throughput check
- Battery status
- Impedance estimation
- Data capture and visualization

See `SETUP_GUIDE.md` for detailed instructions.
