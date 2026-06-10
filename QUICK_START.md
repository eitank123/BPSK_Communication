# Quick Start Guide - Refactored QPSK Simulation

## What Changed?

Your original code has been professionally refactored into a modular, maintainable architecture:

**Before:** 1 massive main.py (600+ lines)
**After:** 9 focused modules with clear responsibilities

## File Organization

```
src/
├── main_refactored.py         ← START HERE (run this)
├── config.py                  ← Change parameters here
├── simulator.py               ← Orchestrates simulation
├── utilities.py               ← Reusable functions
├── channel.py                 ← Channel operations
├── signal_processing.py       ← Signal utilities
├── client_Tx.py               ← Transmitter (updated docs)
├── client_Rx.py               ← Receiver (updated docs)
├── plots.py                   ← Plotting (updated docs)
└── RRC_Implementation.py       ← Filter design (updated docs)
```

## Running the Simulation

### Step 1: Open Terminal
Navigate to the src folder:
```bash
cd c:\Users\eitan\OneDrive\project\ haim\comms_code\src
```

### Step 2: Run Simulation
```bash
python main_refactored.py
```

That's it! The simulation will:
- Run Phase 1 (SNR/K-factor sweep)
- Run Phase 2 (SPS sweep)
- Generate all plots automatically
- Print summary tables

## Changing Parameters (Most Common)

Open `config.py` and edit:

### Example 1: Reduce Number of Bits (Faster Testing)
```python
# Find this line:
NUMBER_OF_BITS = 10000

# Change to:
NUMBER_OF_BITS = 1000  # 10x faster
```

### Example 2: Change SNR Values
```python
# Find this line:
SNR_VALUES = [0, 2, 4, 6, 8, 10]

# Change to:
SNR_VALUES = [6, 8, 10, 12]  # Higher SNR only
```

### Example 3: Adjust RRC Filter
```python
ROLLOFF_FACTOR = 0.5   # ← Beta (0=sharp, 1=gradual)
FILTER_SPAN = 10       # ← Symbol periods (larger = longer)
```

### Example 4: Test One SPS Only
```python
# Find this line:
SPS_SWEEP_VALUES = [1, 2, 4, 8, 16]

# Change to:
SPS_SWEEP_VALUES = [8]  # Test only SPS=8
```

### Example 5: Change Timing Loop Gains
```python
EARLY_LATE_KP = 0.01   # Proportional gain (smaller = slower)
EARLY_LATE_KI = 0.001  # Integral gain (smaller = slower)

GARDNER_KP = 0.08      # Gardner proportional gain
GARDNER_KI = 0.01      # Gardner integral gain

LMS_MU_PHASE = 0.01    # LMS phase learning rate
LMS_MU_DRIFT = 0.0001  # LMS drift learning rate
```

**Done!** Just save config.py and run main_refactored.py again.

## Understanding the Output

### Console Output
```
PHASE 1: SNR SWEEP (Fixed SPS=8)
BER Results Summary (Delay=200.4):
K-Factor (dB)  M1          M2          M3          ...
0.00           0.05        0.04        0.03        ...
2.00           0.04        0.03        0.02        ...
...
```
Shows BER (Bit Error Rate) for each method at each K-factor.

### Generated Plots

1. **BER vs K-Factor** - How error rate changes with fading
2. **Delay Tracking** - How accurately each method estimates delay
3. **Range Error** - Physical distance error (in meters)
4. **Eye Diagrams** - Constellation quality visualization
5. **BER vs SPS** - Performance across different sampling rates

## Common Tasks

### Task 1: Make Simulation Run Faster
```python
# In config.py:
NUMBER_OF_BITS = 1000        # ← Reduce bits
SPS_SWEEP_VALUES = [8]       # ← Test one SPS
RICIAN_K_FACTORS = [0, 6]    # ← Fewer K values
SNR_VALUES = [6, 8, 10]      # ← Fewer SNR values
```

### Task 2: Test Only One Timing Method
```python
# In main_refactored.py, change:
for method_id in range(1, 7):  # Currently 1-6
# To:
for method_id in [1]:          # Test only Method 1
```

### Task 3: Disable Plotting
```python
# In config.py:
ENABLE_PLOTTING = False
```

### Task 4: Compare Two Configurations
```python
# In config.py, keep first config
python main_refactored.py     # Run first test, note results

# Change parameters, run again
python main_refactored.py     # Run second test, compare
```

## Module Quick Reference

### config.py
👉 **Use this to change parameters**
- All constants in one place
- Organized by category
- Well-commented

### main_refactored.py
👉 **This is your entry point**
- Shows overall simulation flow
- Two phases clearly separated
- High-level, easy to understand

### simulator.py
- Orchestrates transmitter/receiver
- Manages sweeps
- Don't need to edit normally

### utilities.py
- Helper functions
- Use these when adding new code

### channel.py
- Fading, CP, preamble generation
- Use these for channel modifications

### signal_processing.py
- Interpolation, filtering, FFT
- Use these for signal analysis

### client_Tx.py, client_Rx.py
- Updated with better documentation
- Timing methods implemented here
- Don't need to edit normally

### plots.py, RRC_Implementation.py
- Updated with better documentation
- Use existing functions

## Troubleshooting

### Problem: Import Error
**Solution:** Make sure you're in the src/ folder:
```bash
cd src
python main_refactored.py
```

### Problem: Slow Running
**Solution:** Reduce parameters in config.py:
```python
NUMBER_OF_BITS = 100        # Much smaller
SPS_SWEEP_VALUES = [8]      # One value only
```

### Problem: Memory Error
**Solution:** Reduce data size:
```python
NUMBER_OF_BITS = 1000       # Smaller
```

### Problem: Plots Won't Display
**Solution:** Check matplotlib backend or disable:
```python
# In config.py:
ENABLE_PLOTTING = False
```

## Architecture Overview (5-Minute Explanation)

```
main_refactored.py
    ├─ Initializes SimulationEngine
    ├─ Runs Phase 1 (SNR sweep)
    │   └─ For each config: engine.run_snr_sweep()
    └─ Runs Phase 2 (SPS sweep)
        └─ For each config: engine.run_sps_sweep()

SimulationEngine (simulator.py)
    ├─ Manages Client_Tx (transmitter)
    ├─ Manages Client_Rx (receiver)
    ├─ Creates signals via channel.py
    ├─ Processes through 6 timing methods
    └─ Returns BER/delay results

For Each Signal:
    ├─ Transmitter: bits → QPSK symbols → RRC filter
    ├─ Channel: Add Rician fading + AWGN
    ├─ Receiver: Apply RRC matched filter
    ├─ Timing Recovery: Extract symbols (Method 1-6)
    ├─ Equalization: Remove channel distortion
    └─ Analysis: Calculate BER, delay error

Results Processing:
    ├─ Organize BER/delay by method
    ├─ Calculate range errors
    └─ Generate plots
```

## Next Steps

1. **Run it:** `python main_refactored.py`
2. **Modify:** Edit parameters in `config.py`
3. **Understand:** Read docstrings in each module
4. **Extend:** Add new methods in `client_Rx.py`

## Key Concepts (Refresher)

- **QPSK**: 2 bits per symbol (I/Q constellation)
- **RRC Filter**: Pulse shaping for zero-ISI transmission
- **Rician Fading**: Realistic channel with LoS/NLoS components
- **K-Factor**: Ratio of LoS to NLoS power (dB)
- **BER**: Bit Error Rate (lower is better)
- **Timing Synchronization**: Recover symbol clock from received signal
- **SC-FDE**: Single carrier with frequency-domain equalization

## Getting Help

### Understand a Function
```python
# In any module file:
def function_name(param1, param2):
    """
    This docstring explains what it does,
    inputs, and outputs.
    """
```
Read the docstring!

### Understand a Module
Start with the top docstring:
```python
"""
This module does X, Y, Z.

Functions:
  - func1(): Does A
  - func2(): Does B
"""
```

### Understand Config
Every parameter in `config.py` has a comment:
```python
SAMPLES_PER_SYMBOL = 8  # Default SPS
```

## Summary

✅ **Refactored:** 600-line monolith → 9 focused modules
✅ **Documented:** Docstrings everywhere
✅ **Configured:** All settings in config.py
✅ **Organized:** Clear module responsibilities
✅ **Ready to Use:** Just run main_refactored.py
✅ **Easy to Modify:** Change config.py
✅ **Easy to Extend:** Clear patterns to follow

**Start here:** `python main_refactored.py`

Happy simulating! 🚀
