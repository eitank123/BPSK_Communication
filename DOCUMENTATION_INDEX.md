# 📚 Complete Documentation Index

## Getting Started (Choose Your Path)

### ⚡ Quick Start (5 minutes)
**Best for**: Just want to run the code
1. Read: `QUICK_START.md`
2. Edit: `config.py` (adjust parameters)
3. Run: `python main_refactored.py`

### 📖 Complete Understanding (30 minutes)
**Best for**: Want to understand the structure
1. Read: `README_REFACTORED.md` (overview)
2. Read: `ARCHITECTURE.md` (design)
3. Browse: Module docstrings (details)

### 🔄 Migrating from Old Code (20 minutes)
**Best for**: Understanding what changed
1. Read: `REFACTORING_GUIDE.md` (before/after)
2. Compare: Old vs new code
3. Update: References/scripts

### 🎯 Extending/Customizing (varies)
**Best for**: Adding new features
1. Read: `ARCHITECTURE.md` → "Extensibility Points"
2. Choose: Pattern to follow
3. Implement: New feature

---

## 📄 Document Guide

### 1. `README_REFACTORED.md`
**What**: Complete project documentation
**When to read**: Want full understanding of modules
**Length**: 15 minutes
**Contains**:
- Project overview
- Module descriptions with examples
- Configuration guide
- Running instructions
- Future enhancements

**Read if you want to**:
✅ Understand what each module does
✅ See code examples
✅ Know how to configure the system
✅ Understand improvement made

---

### 2. `QUICK_START.md`
**What**: Fast guide to running and modifying
**When to read**: Want to use code immediately
**Length**: 5 minutes
**Contains**:
- What changed overview
- Running the simulation
- Parameter modification guide
- Common tasks
- Troubleshooting

**Read if you want to**:
✅ Run the code quickly
✅ Make common parameter changes
✅ Fix issues
✅ Understand basic flow

---

### 3. `REFACTORING_GUIDE.md`
**What**: Detailed before/after mapping
**When to read**: Curious about improvements
**Length**: 20 minutes
**Contains**:
- Problem analysis of original code
- Refactoring patterns applied
- Before/after code examples
- Function mapping table
- Complexity metrics

**Read if you want to**:
✅ Understand code improvements
✅ Learn software engineering patterns
✅ See before/after comparisons
✅ Find where old functions went

---

### 4. `ARCHITECTURE.md`
**What**: Design decisions and patterns
**When to read**: Extending or deep understanding
**Length**: 20 minutes
**Contains**:
- Architectural layers
- Design patterns used
- Key decisions with rationale
- Modularity principles
- Extensibility points
- Performance considerations
- Testing approach

**Read if you want to**:
✅ Understand system design
✅ Add new features
✅ Learn design patterns used
✅ Understand trade-offs made

---

### 5. `REFACTORING_SUMMARY.md` (This file's companion)
**What**: Executive summary of changes
**When to read**: Quick overview of what was done
**Length**: 10 minutes
**Contains**:
- What was done
- Files created
- Key improvements
- Code metrics
- Next steps
- Common customizations

**Read if you want to**:
✅ Quick summary of changes
✅ Overview of new files
✅ See improvements made
✅ Know next steps

---

## 📂 Module Quick Reference

### `src/main_refactored.py`
**Purpose**: Entry point for simulation
**When to use**: Run simulation
**Key functions**:
- `main()` - Start here
- `run_phase1_snr_sweep()` - SNR analysis
- `run_phase2_sps_sweep()` - SPS analysis

**Example**:
```bash
python main_refactored.py
```

---

### `src/config.py`
**Purpose**: Centralized configuration
**When to use**: Change parameters
**Key settings**:
- System parameters (bits, SPS)
- Channel parameters (SNR, K-factor)
- Algorithm hyperparameters
- Plotting options

**Example**:
```python
# Edit to change:
NUMBER_OF_BITS = 1000  # Faster
SNR_VALUES = [8, 10]   # Different range
```

---

### `src/simulator.py`
**Purpose**: Orchestrate simulation
**When to use**: Understand simulation flow
**Key class**: `SimulationEngine`
- `run_snr_sweep()` - Phase 1
- `run_sps_sweep()` - Phase 2
- `process_received_signal()` - Route methods

---

### `src/utilities.py`
**Purpose**: Reusable utility functions
**When to use**: Calculate metrics or conversions
**Key functions**:
- `calculate_ber()` - BER calculation
- `symbols_to_bits()` - Symbol conversion
- `calculate_range_error()` - Delay to distance
- `to_scalar()` - Array to scalar

**Example**:
```python
from utilities import calculate_ber
ber = calculate_ber(tx_bits, rx_bits)
```

---

### `src/channel.py`
**Purpose**: Channel operations
**When to use**: Model channel behavior
**Key functions**:
- `add_rician_fading()` - Fading simulation
- `add_cyclic_prefix()` - SC-FDE formatting
- `generate_zadoff_chu_preamble()` - Preamble
- `create_formatted_payload()` - Data formatting

---

### `src/signal_processing.py`
**Purpose**: Signal analysis utilities
**When to use**: Process/analyze signals
**Key functions**:
- `cubic_interpolate()` - Fractional sampling
- `correlate_signals()` - Cross-correlation
- `estimate_signal_power()` - Power analysis
- `fft_and_normalize()` - FFT operations

---

### `src/client_Tx.py`
**Purpose**: QPSK transmitter
**When to use**: Understand transmission
**Key class**: `Client_Tx`
- `generate_bit_array()` - Random bits
- `prepare_x_t()` - Create transmitted signal
- Includes Farrow interpolation

---

### `src/client_Rx.py`
**Purpose**: QPSK receiver with 6 methods
**When to use**: Understand reception
**Key class**: `Client_Rx`
- Method 1: Integer Correlation
- Method 2: Parabolic Interpolation
- Method 3: ML Grid Search
- Method 4: Early-Late Loop
- Method 5: Gardner Loop
- Method 6: LMS Adaptive
- Plus channel estimation & equalization

---

### `src/plots.py`
**Purpose**: Visualization functions
**When to use**: Generate plots
**Key functions**:
- `plot_ber_vs_k()` - BER vs Rician K
- `plot_eye_diagram()` - Constellation
- `plot_delay_tracking()` - Timing performance

---

### `src/RRC_Implementation.py`
**Purpose**: Root-Raised-Cosine filter
**When to use**: Understand pulse shaping
**Key functions**:
- `rrc_design()` - Generate filter taps
- `get_rrc_freq_response()` - Frequency response
- `plot_rrc_impulse()` - Visualize impulse
- `get_impulse_and_freq_response()` - Get both

---

## 🎓 Learning Path

### For Understanding the System

**Level 1: Overview (10 min)**
1. Read `QUICK_START.md` sections "What Changed" + "Understanding Output"
2. Skim `README_REFACTORED.md` module list

**Level 2: Architecture (20 min)**
1. Read `README_REFACTORED.md` → "Module Descriptions"
2. Read `ARCHITECTURE.md` → "Architectural Layers"

**Level 3: Implementation (30 min)**
1. Read `ARCHITECTURE.md` → "Design Patterns"
2. Read module docstrings in each `.py` file
3. Browse actual code

**Level 4: Mastery (varies)**
1. Run with different configs
2. Modify code
3. Add new features
4. Run your own analyses

---

## 🔧 Common Tasks & Where to Find Help

| Task | Where to Look | Doc Section |
|---|---|---|
| Run simulation | `QUICK_START.md` | "Running the Simulation" |
| Change parameters | `QUICK_START.md` | "Changing Parameters" |
| Add timing method | `ARCHITECTURE.md` | "Extensibility Points" |
| Add channel model | `ARCHITECTURE.md` | "Extensibility Points" |
| Understand Module X | `README_REFACTORED.md` | Module descriptions |
| See before/after | `REFACTORING_GUIDE.md` | "Code Migration Examples" |
| Troubleshoot | `QUICK_START.md` | "Troubleshooting" |
| Understand design | `ARCHITECTURE.md` | "Design Decisions" |
| Make faster | `QUICK_START.md` | "Task 1" |
| Compare methods | `config.py` | RICIAN_K_FACTORS |
| Test single method | `QUICK_START.md` | "Task 2" |

---

## 📊 Documentation Statistics

| Document | Length | Read Time | Best For |
|---|---|---|---|
| `QUICK_START.md` | 5 pages | 5 min | Quick use |
| `README_REFACTORED.md` | 12 pages | 15 min | Understanding |
| `ARCHITECTURE.md` | 15 pages | 20 min | Design/Extension |
| `REFACTORING_GUIDE.md` | 12 pages | 20 min | Improvements |
| `REFACTORING_SUMMARY.md` | 8 pages | 10 min | Overview |
| Docstrings (all files) | ~50 pages | As needed | Details |
| **TOTAL** | **52+ pages** | **70 min** | Everything |

**Quick Facts**:
- 📚 **5** comprehensive guides
- 📝 **50+** pages of documentation
- 💻 **9** code modules
- ✅ **100%** documented functions
- 🎯 **6** different learning paths

---

## 🚀 Getting Started NOW

### The Absolute Fastest Way (2 minutes)

```bash
# 1. Open terminal in src/ folder
cd src

# 2. Run it
python main_refactored.py

# 3. View plots that appear
# Done!
```

### Still Want More? (5 minutes)

```bash
# Edit one parameter
# In config.py, change:
NUMBER_OF_BITS = 1000

# Run again
python main_refactored.py

# Compare results
# Done!
```

### Want to Understand? (30 minutes)

```
1. Read QUICK_START.md (5 min)
2. Read README_REFACTORED.md (15 min)
3. Read ARCHITECTURE.md (10 min)
4. Explore code with understanding
```

---

## 📖 Recommended Reading Order

### For Users
1. `QUICK_START.md` - Get running
2. `config.py` - Understand parameters
3. `README_REFACTORED.md` → Module descriptions

### For Developers
1. `QUICK_START.md` - Understand basics
2. `README_REFACTORED.md` - Module overview
3. `ARCHITECTURE.md` - Design patterns
4. `REFACTORING_GUIDE.md` - Improvements

### For Maintainers
1. All of the above
2. Code docstrings
3. Specific module implementation
4. `ARCHITECTURE.md` → Extensibility

### For Contributors
1. `QUICK_START.md` - Baseline
2. `ARCHITECTURE.md` → Extensibility Points
3. Specific module for change
4. Related tests/examples

---

## 🎯 Navigation by Question

**Q: How do I run this?**
→ `QUICK_START.md` → "Running the Simulation"

**Q: How do I change parameters?**
→ `QUICK_START.md` → "Changing Parameters"

**Q: What modules exist?**
→ `README_REFACTORED.md` → "Module Descriptions"

**Q: How do I add a new feature?**
→ `ARCHITECTURE.md` → "Extensibility Points"

**Q: What changed from old code?**
→ `REFACTORING_GUIDE.md` → "Code Migration Examples"

**Q: Why was it designed this way?**
→ `ARCHITECTURE.md` → "Design Decisions"

**Q: How does timing recovery work?**
→ `client_Rx.py` docstring + `README_REFACTORED.md`

**Q: What are the design patterns?**
→ `ARCHITECTURE.md` → "Design Patterns Used"

**Q: Can I extend it?**
→ `ARCHITECTURE.md` → "Extensibility Points" + `REFACTORING_GUIDE.md`

---

## 📋 Checklist: Before Running

- [ ] Read `QUICK_START.md` (5 min)
- [ ] Navigate to `src/` folder
- [ ] Edit `config.py` if needed
- [ ] Run `python main_refactored.py`
- [ ] Wait for plots to appear
- [ ] Read plots and results
- [ ] Try modifying parameters

---

## 🎓 Learning Objectives

After reading these docs, you should understand:

**Basic Level**:
- ✅ How to run the simulation
- ✅ How to change parameters
- ✅ What each module does
- ✅ How to interpret results

**Intermediate Level**:
- ✅ Overall system architecture
- ✅ Why code is organized this way
- ✅ How modules interact
- ✅ How to modify code

**Advanced Level**:
- ✅ Design patterns used
- ✅ How to add new methods
- ✅ How to extend system
- ✅ Trade-offs in design

---

## 💡 Pro Tips

1. **Start Simple**: Edit one parameter in `config.py`, run simulation
2. **Read Docstrings**: Every function has helpful documentation
3. **Explore Gradually**: Don't need to understand everything at once
4. **Use Config.py**: Change parameters there, not in code
5. **Look at Examples**: Each doc has code examples
6. **Check Docstrings**: `help(function_name)` in Python

---

## 🔗 Document Cross-References

Each document references others where relevant:

```
QUICK_START.md
    ├─ Refers to QUICK_START.md for details
    ├─ Refers to config.py for parameters
    └─ Refers to README_REFACTORED.md for architecture

README_REFACTORED.md
    ├─ Refers to QUICK_START.md for usage
    ├─ Refers to ARCHITECTURE.md for design
    └─ Refers to module docstrings for details

ARCHITECTURE.md
    ├─ Refers to REFACTORING_GUIDE.md for comparisons
    ├─ Refers to module docstrings
    └─ Refers to code examples

REFACTORING_GUIDE.md
    ├─ Refers to ARCHITECTURE.md for patterns
    ├─ References old code
    └─ References new code location
```

---

## 📞 Support Resources

| Need | Resource | Time |
|---|---|---|
| Quick help | `QUICK_START.md` | 5 min |
| Module info | `README_REFACTORED.md` | 15 min |
| Design help | `ARCHITECTURE.md` | 20 min |
| Code location | `REFACTORING_GUIDE.md` | 20 min |
| Function details | Module docstrings | As needed |

---

## Final Note

You now have:
✅ Professional, modular code
✅ Comprehensive documentation
✅ Multiple learning paths
✅ Clear extension points
✅ Best practices applied

**Start with**: `QUICK_START.md` (5 minutes)
**Then run**: `python main_refactored.py`
**Then explore**: Based on your needs

**Happy coding!** 🚀
