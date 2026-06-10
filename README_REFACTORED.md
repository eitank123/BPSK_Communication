# QPSK Communication System - Refactored

## Overview

A comprehensive simulation framework for analyzing QPSK communication systems with emphasis on timing synchronization methods and Rician fading channels.

## Project Structure

```
src/
├── main_refactored.py          # Main entry point (clean, orchestrates all modules)
├── config.py                   # Centralized configuration and constants
├── utilities.py                # Common utility functions (BER, conversions)
├── channel.py                  # Channel operations (CP, preamble, fading)
├── signal_processing.py        # Signal analysis utilities
├── simulator.py                # Simulation engine orchestrator
├── client_Tx.py                # Transmitter class (bit mapping, pulse shaping)
├── client_Rx.py                # Receiver class (6 timing recovery methods)
├── plots.py                    # Visualization functions
├── RRC_Implementation.py        # Root-Raised-Cosine filter design
└── README.md                   # This file
```

## Module Descriptions

### Core Infrastructure

#### `config.py`
Centralizes ALL constants and configuration parameters:
- System parameters (sample rate, number of bits)
- RRC filter configuration
- Channel parameters (SNR, K-factor, delays)
- Timing recovery algorithm hyperparameters
- Method labels and styling

**Benefits:**
- Single source of truth for all settings
- Easy parameter sweeps and variations
- Reduced code duplication

#### `utilities.py`
Common helper functions used throughout the project:
- `symbols_to_bits()` - Convert symbols to bit stream
- `calculate_ber()` - Bit Error Rate calculation
- `to_scalar()` - Extract scalar from arrays/lists
- `calculate_range_error()` - Convert delay error to physical distance
- Signal validation functions

#### `signal_processing.py`
Signal analysis and manipulation utilities:
- `cubic_interpolate()` - Fractional sample interpolation
- `correlate_signals()` - Cross-correlation
- `convolve_signals()` - Filtering operations
- `create_complex_interpolator()` - For phase-preserving interpolation
- `estimate_signal_power()`, `normalize_signal()` - Power analysis
- FFT operations and frequency domain utilities

#### `channel.py`
Channel-related operations:
- `add_cyclic_prefix()` - SC-FDE CP insertion
- `generate_zadoff_chu_preamble()` - Synchronization preamble
- `add_rician_fading()` - Realistic fading channel simulator
- `create_formatted_payload()` - Format data with CP
- `extract_block_payload()` - Recover formatted data

### Communication System

#### `client_Tx.py` (Transmitter)
QPSK transmitter with:
- Bit generation and mapping to QPSK symbols
- RRC pulse shaping filtering
- Fractional delay application (Farrow interpolation)
- Frequency offset insertion
- Upsampling and signal preparation

Key methods:
- `generate_bit_array()` - Generate random bits
- `map_bits()` - QPSK constellation mapping
- `set_responses()` - Configure RRC filter
- `prepare_x_t()` - Create transmitted signal

#### `client_Rx.py` (Receiver)
QPSK receiver with 6 timing synchronization methods:

1. **Integer Correlation** (M1)
   - Simple cross-correlation with preamble
   - Integer sample delay only

2. **Parabolic Interpolation** (M2)
   - Fractional delay via parabolic fit
   - Lower complexity than M3

3. **ML Grid Search** (M3)
   - Maximum likelihood fractional estimation
   - Fine grid search approach

4. **Early-Late Loop** (M4)
   - Tracking-based synchronization
   - Proportional-Integral feedback control

5. **Gardner Loop** (M5)
   - Classic timing recovery loop
   - Gardner Timing Error Detector

6. **LMS Adaptive** (M6)
   - Adaptive filter-based timing recovery
   - Converges over payload duration

Also includes:
- Channel estimation (`estimate_channel_and_weights()`)
- SC-FDE equalization (`equalize_sc_fde()`, `equalize_blocks_only()`)
- Signal filtering (`filter_signal()`)

#### `RRC_Implementation.py` (Filter Design)
Root-Raised-Cosine filter implementation:
- `rrc_design()` - Generate RRC filter taps
- `get_rrc_freq_response()` - Frequency domain response
- `plot_rrc_impulse()`, `plot_rrc_freq()` - Visualization

#### `plots.py` (Visualization)
Comprehensive plotting functions:
- `plot_ber_vs_k()` - BER vs Rician K-factor
- `plot_ber_comparison()` - Simulated vs theoretical
- `plot_eye_diagram()` - I/Q constellation eye patterns
- `plot_delay_tracking()` - Timing error vs channel condition
- `plot_noisy_signal()` - Time-domain waveforms

### Simulation Engine

#### `simulator.py` (SimulationEngine)
High-level simulation orchestration:
- Manages transmitter/receiver initialization
- Coordinates signal creation and processing
- Implements all 6 methods uniformly
- Runs sweeps (SNR and SPS)

Key methods:
- `initialize_transmitter()`, `initialize_preamble()`
- `create_transmitted_signal()` - Full signal generation
- `process_received_signal()` - Route to appropriate method
- `run_snr_sweep()` - Phase 1 analysis
- `run_sps_sweep()` - Phase 2 analysis

#### `main_refactored.py` (Entry Point)
Clean main simulation runner:
- Phase 1: SNR/Rician sweep (fixed SPS)
- Phase 2: SPS sweep (fixed SNR)
- Console output with summary tables
- Automatic plotting

## Key Improvements

### 1. **Modularity**
- Each module has a single, clear responsibility
- Functions are focused and reusable
- Reduces coupling between components

### 2. **Configuration Management**
- All magic numbers in `config.py`
- Easy to adjust parameters without code changes
- Single source of truth for constants

### 3. **Code Reusability**
- Utility functions (`utilities.py`, `signal_processing.py`)
- Channel operations (`channel.py`)
- Eliminates code duplication from original main.py

### 4. **Readability**
- Clear function and module names
- Comprehensive docstrings
- Organized into logical sections
- Type hints in documentation

### 5. **Maintainability**
- Easy to debug (isolated modules)
- Easy to extend (add new methods or simulations)
- Easy to test (pure functions)

### 6. **Scalability**
- `SimulationEngine` abstracts complexity
- Easy to add new methods (6+ methods)
- Sweep logic is generalized

## Configuration Examples

### Adjust SNR Values
```python
# In config.py
SNR_VALUES = [0, 2, 4, 6, 8, 10, 12, 14]  # Add more values
```

### Change RRC Filter Parameters
```python
# In config.py
ROLLOFF_FACTOR = 0.5  # Beta
FILTER_SPAN = 10      # Symbol periods
```

### Modify Timing Recovery Gains
```python
# In config.py
EARLY_LATE_KP = 0.02    # Increase proportional gain
EARLY_LATE_KI = 0.0015  # Increase integral gain
```

## Running the Simulation

### Basic Execution
```bash
python main_refactored.py
```

### With Custom Configuration
Edit `config.py` then run:
```bash
python main_refactored.py
```

### Example: Quick Test with 1 SPS
```python
# In config.py, change:
SPS_SWEEP_VALUES = [1]  # Test only SPS=1
NUMBER_OF_BITS = 1000   # Smaller dataset
```

## Output

### Console Output
- Configuration parameters
- Summary tables (BER for each method vs channel condition)
- Method comparison metrics

### Plots Generated
- **Graph 1-3**: BER vs K-factor (per method) at fixed SNR
- **Graph 4**: Delay tracking performance vs K-factor
- **Graph 5**: Range Error vs SPS
- **Graph 6**: Eye diagrams (constellation quality)
- **Graph 7**: BER vs SPS

## Extending the System

### Adding a New Timing Recovery Method
1. Implement method in `Client_Rx` class
2. Add to `SimulationEngine.process_received_signal()`
3. Update `config.py` with method label
4. Update METHOD_LABELS list

### Adding New Channel Model
1. Create function in `channel.py`
2. Add to sweep loop in `simulator.py`
3. Update configuration parameters

### New Simulation Type
1. Add phase to `main_refactored.py`
2. Create sweep method in `SimulationEngine`
3. Implement plotting if needed

## Performance Considerations

- **Memory**: ~500 MB for full simulation
- **Runtime**: ~2-5 minutes (depends on parameters)
- **Optimization**: SimulationEngine caches interpolators

## References

- RRC Filter: IEEE standard pulse-shaping filter
- Rician Channel: K-factor controls LOS/NLOS ratio
- Gardner Loop: Classic timing recovery technique
- Early-Late Loop: Common in practical systems
- ML Estimation: Maximum likelihood principles

## Future Enhancements

- [ ] OFDM support (extends SC-FDE)
- [ ] Multiple antenna systems (MIMO)
- [ ] Adaptive modulation (switch between BPSK/QPSK)
- [ ] Phase recovery algorithms
- [ ] Frequency offset estimation
- [ ] Multi-path equalization comparison
- [ ] Real-time simulation visualization
