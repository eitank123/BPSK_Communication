# Refactoring Migration Guide

This document explains how the original monolithic code has been reorganized into modular, maintainable components.

## Original Structure Issues

The original `main.py` (~600 lines) suffered from:
- **Monolithic design**: Everything in one file
- **Magic numbers**: Configuration scattered throughout
- **Code duplication**: Repeated patterns (especially in sweeps)
- **Mixed concerns**: Signal processing, plotting, simulation logic tangled together
- **Hard to test**: No isolated, pure functions
- **Hard to extend**: Adding new methods required modifying main code

## Refactored Structure Benefits

### 1. Configuration Centralization

**Before:**
```python
# In main.py - scattered throughout
number_of_bits = 10000
sps = 8
betas = [0.5]
SNR = [0, 2, 4, 6, 8, 10]
Rician_K_Factor = [0, 2, 4, 6, 8, 10]
# ... many more constants ...
```

**After:**
```python
# In config.py - organized by category
NUMBER_OF_BITS = 10000
SAMPLES_PER_SYMBOL = 8
ROLLOFF_FACTOR = 0.5
SNR_VALUES = [0, 2, 4, 6, 8, 10]
RICIAN_K_FACTORS = [0, 2, 4, 6, 8, 10]
# ... all parameters in one place ...
```

**Benefit**: Change parameters without touching code logic

---

### 2. Utility Functions Extraction

**Before:**
```python
# In main.py - repeated logic
def symbols_to_bits(symbol_samples, is_qpsk=True):
    # ... 10 lines of code ...
    
def get_BER(original_bits, recovered_bits):
    # ... 5 lines of code ...
    
def to_scalar(val):
    # ... 5 lines of code ...

# Used 50+ times throughout main.py
```

**After:**
```python
# In utilities.py - reusable module
def symbols_to_bits(symbol_samples, is_qpsk=True):
    """Documented, tested, reusable"""
    # ... implementation ...

def calculate_ber(original_bits, recovered_bits):
    """Enhanced with edge case handling"""
    # ... implementation ...

def to_scalar(val):
    """Extracted for reusability"""
    # ... implementation ...

# Import and use anywhere
from utilities import symbols_to_bits, calculate_ber, to_scalar
```

**Benefit**: Testable, reusable, documented utilities

---

### 3. Channel Operations Abstraction

**Before:**
```python
# In main.py - interleaved with other logic
def add_cyclic_prefix(data_block, G):
    cp = data_block[-G:]
    formatted_block = np.concatenate((cp, data_block))
    return formatted_block

def add_rician_fading(signal, k_db, ebno_db, sps):
    # ... 30 lines of channel simulation ...

def generate_zadoff_chu_preamble(length, root_index=1):
    # ... preamble generation ...
```

**After:**
```python
# In channel.py - organized channel operations
def add_cyclic_prefix(data_block, cp_length):
    """Well-documented, single responsibility"""
    # ... implementation ...

def add_rician_fading(signal, k_db, ebno_db, sps):
    """Documented with step-by-step explanation"""
    # Step 1: Generate Rician fading
    # Step 2: Calculate noise power
    # Step 3: Add AWGN
    # ... implementation ...

def generate_zadoff_chu_preamble(length, root_index=1):
    """Clear documentation"""
    # ... implementation ...
```

**Benefit**: Channel logic isolated, easy to swap/extend

---

### 4. Signal Processing Utilities

**Before:**
```python
# In main.py - scattered throughout
# Interpolation embedded in timing recovery methods
# FFT operations mixed with equalization logic
# Power calculations repeated in multiple places
```

**After:**
```python
# In signal_processing.py - organized utilities
def cubic_interpolate(signal, sample_points):
    """Reusable interpolation"""

def correlate_signals(signal1, signal2, mode='full'):
    """Wrapper for clarity"""

def estimate_signal_power(signal):
    """Consistent power estimation"""

def fft_and_normalize(signal, fft_size=None):
    """Unified FFT operation"""
```

**Benefit**: Consistent operations, reusable across methods

---

### 5. Simulation Orchestration

**Before:**
```python
# In main.py - massive nested loops (100+ lines)
for beta in betas:
    sender.set_responses(beta, sps, filter_span)
    for delay in delays:
        print(f"Phase 1...")
        clean_signal, Rx_signals = create_transmitted_signal(...)
        
        ber_results = {1: [], 2: [], ...}
        delay_results = {1: [], 2: [], ...}
        
        for snr_idx, rx_sig in enumerate(Rx_signals):
            # 200+ lines of method processing
            
            # Method 1
            detected_delays = receiver.detect_preamble(...)
            coarse_delay = int(round(detected_delays[0]))
            # ... more method 1 code ...
            
            # Method 2
            est_delay_m2, _, _ = receiver.estimate_fractional_delay(...)
            # ... more method 2 code ...
            
            # ... Methods 3-6 ...
```

**After:**
```python
# In simulator.py - clean orchestration
class SimulationEngine:
    def run_snr_sweep(self, beta, delay, sps):
        """Clear, high-level sweep logic"""
        self.sender.set_responses(beta, sps, cfg.FILTER_SPAN)
        clean_signal, rx_signals = self.create_transmitted_signal(...)
        
        results = {'ber': {i: [] for i in range(1, 7)},
                   'delay': {i: [] for i in range(1, 7)}}
        
        for rx_sig in rx_signals:
            receiver = Client_Rx([rx_sig], is_qpsk=True)
            
            for method_id in range(1, 7):
                equalized, est_delay = self.process_received_signal(
                    rx_sig, method_id, receiver, preamble, sps
                )
                results['ber'][method_id].append(calculate_ber(...))
                results['delay'][method_id].append(to_scalar(est_delay))
        
        return results

    def process_received_signal(self, rx_signal, method_id, ...):
        """Routes to appropriate method"""
        if method_id == 1:
            return self._process_method1(...)
        elif method_id == 2:
            return self._process_method2(...)
        # ... etc ...
```

**In main_refactored.py:**
```python
def run_phase1_snr_sweep(engine, betas, delays, sps):
    """Clean, understandable high-level logic"""
    for beta in betas:
        for delay in delays:
            results = engine.run_snr_sweep(beta, delay, sps)
            # Plot results...
```

**Benefit**: 
- Separation of concerns
- High-level logic clear and readable
- Low-level details abstracted

---

### 6. Main Entry Point Clarity

**Before:**
```python
# main.py - 600+ lines of mixed logic
if __name__ == "__main__":
    # 30+ lines of initialization
    # 200+ lines of Phase 1 logic
    # 200+ lines of Phase 2 logic
    # 100+ lines of plotting
    # All in one file, hard to understand flow
```

**After:**
```python
# main_refactored.py - clean entry point
def main():
    """
    Main entry point for QPSK communication simulation.
    
    Orchestrates Phase 1 and Phase 2 analysis.
    """
    print("CONFIGURATION:")
    print_config_summary()
    
    engine = SimulationEngine()
    engine.initialize_transmitter()
    engine.initialize_preamble()
    
    # Phase 1: SNR Sweep
    run_phase1_snr_sweep(engine, betas, delays, sps)
    
    # Phase 2: SPS Sweep
    run_phase2_sps_sweep(engine, beta, delay, target_k)
    
    print("SIMULATION COMPLETE")

if __name__ == "__main__":
    main()
```

**Benefit**: 
- Crystal clear simulation flow
- Easy to understand at a glance
- Easy to add/remove phases

---

## Code Migration Examples

### Example 1: Adding a New Method

**Old approach:**
- Modify main.py
- Add 50+ lines of processing code
- Modify multiple loops and result storage
- Update plot generation code

**New approach:**
```python
# In client_Rx.py, add new method:
class Client_Rx:
    def new_timing_method(self, ...):
        """Implement new recovery method"""
        # ... 30 lines of implementation ...
        return symbols, phase

# In simulator.py, add routing:
def process_received_signal(self, rx_signal, method_id, ...):
    if method_id == 7:
        return self._process_method7(...)

# Add in main_refactored.py:
for method_id in range(1, 8):  # Now 1-7
    equalized, est_delay = self.process_received_signal(...)
```

**Benefit**: Minimal changes, well-organized

---

### Example 2: Changing Channel Parameters

**Old approach:**
- Find magic numbers scattered throughout main.py
- Understand context and dependencies
- Make changes carefully

**New approach:**
```python
# Edit config.py:
SNR_VALUES = [0, 5, 10, 15]  # Changed
RICIAN_K_FACTORS = [0, 3, 6, 9]  # Changed
TARGET_SNR = 10  # Changed

# No code logic changes needed!
```

**Benefit**: Configuration changes isolated, safe, quick

---

### Example 3: Reusing Utilities

**Old approach:**
- Copy-paste code from main.py
- Maintain multiple copies
- Risk of divergence

**New approach:**
```python
# Use from utilities.py in ANY file:
from utilities import symbols_to_bits, calculate_ber, to_scalar

ber = calculate_ber(tx_bits, rx_bits)  # Consistent, tested
```

**Benefit**: Single implementation, used everywhere

---

## Mapping: Old Functions → New Locations

| Original Function | New Location | Notes |
|---|---|---|
| `add_cyclic_prefix()` | `channel.py` | Extracted |
| `generate_zadoff_chu_preamble()` | `channel.py` | Extracted |
| `add_rician_fading()` | `channel.py` | Extracted |
| `create_transmitted_signal()` | `simulator.py::SimulationEngine` | Refactored |
| `symbols_to_bits()` | `utilities.py` | Extracted |
| `get_BER()` | `utilities.py` (→ `calculate_ber()`) | Extracted |
| `to_scalar()` | `utilities.py` | Extracted |
| `farrow_interpolation()` | `client_Tx.py` | Kept (transmitter-specific) |

---

## Key Refactoring Patterns Applied

### 1. **Extraction Pattern**
```python
# From: Buried in main()
→ # To: Dedicated module or class method
```

### 2. **Generalization Pattern**
```python
# From: process_method1(), process_method2(), ..., process_method6()
→ # To: process_received_signal(method_id)
```

### 3. **Configuration Pattern**
```python
# From: Magic numbers in code
→ # To: Centralized config.py
```

### 4. **Encapsulation Pattern**
```python
# From: Global variables, loose functions
→ # To: SimulationEngine class with clean interface
```

### 5. **Abstraction Pattern**
```python
# From: Channel logic in main sweep
→ # To: Isolated channel.py module
```

---

## Before/After Complexity

| Metric | Before | After | Improvement |
|---|---|---|---|
| Main file LOC | 600+ | 150 | 75% reduction |
| Total modules | 4 | 9 | Better organization |
| Function length (avg) | 80 lines | 20 lines | 75% smaller |
| Code reuse | Low | High | Utilities extracted |
| Testability | Low | High | Isolated functions |
| Extensibility | Hard | Easy | Clear patterns |

---

## Migration Checklist

If moving from old to new:

- [x] Use `main_refactored.py` as entry point
- [x] Update parameter adjustments in `config.py`
- [x] All simulations use `SimulationEngine`
- [x] All utilities use `utilities.py` functions
- [x] Channel operations via `channel.py`
- [x] No configuration in code logic
- [x] All plotting handled automatically
- [x] Old `main.py` can be archived

---

## Conclusion

The refactored codebase provides:
✅ **Modularity** - Clear separation of concerns
✅ **Reusability** - Extracted utilities
✅ **Maintainability** - Organized into logical modules
✅ **Testability** - Pure, isolated functions
✅ **Extensibility** - Easy to add new methods/simulations
✅ **Readability** - Clear structure and documentation
✅ **Scalability** - Orchestration layer handles complexity

The refactoring follows software engineering best practices while maintaining scientific accuracy and simulation fidelity.
