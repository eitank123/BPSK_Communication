"""
Configuration module for BPSK Communication Simulation.

Centralizes all system constants, parameters, and configuration settings
to improve maintainability and reduce code duplication.
"""

import numpy as np

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

# Transmission parameters
SAMPLE_RATE = 4e6  # 4 MHz symbol rate (Rs)
SAMPLES_PER_SYMBOL = 8  # Default SPS
SAMPLING_FREQ = SAMPLE_RATE * SAMPLES_PER_SYMBOL  # 32 MHz
CARRIER_FREQ = 2.4e9  # 2.4 GHz carrier frequency

# Bit and data parameters
NUMBER_OF_BITS = 10000
BIT_MAPPING_QPSK = {
    (0, 0): (1 + 1j) / np.sqrt(2),
    (0, 1): (1 - 1j) / np.sqrt(2),
    (1, 0): (-1 + 1j) / np.sqrt(2),
    (1, 1): (-1 - 1j) / np.sqrt(2)
}

# ============================================================================
# RRC FILTER CONFIGURATION
# ============================================================================

ROLLOFF_FACTOR = 0.5  # Beta parameter
FILTER_SPAN = 10  # Number of symbol periods for filter length
FARROW_INTERPOLATION_DEGREE = 3

# ============================================================================
# SC-FDE (SINGLE CARRIER FREQUENCY DOMAIN EQUALIZATION) PARAMETERS
# ============================================================================

CP_LENGTH = 16  # Cyclic Prefix length
DATA_BLOCK_SIZE = 256  # FFT size for equalization

# ============================================================================
# PREAMBLE CONFIGURATION
# ============================================================================

PREAMBLE_LENGTH = 211
PREAMBLE_ROOT_INDEX = 1

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

SPEED_OF_LIGHT = 3e8  # m/s
ANTENNAS_DISTANCE = 5e-2 #Distance between two receivers in meters, used for calculating DOA
DOA = 30  # Angle of Arrival in degrees, used for calculating DOA-based delay differences

# ============================================================================
# CHANNEL SIMULATION PARAMETERS
# ============================================================================

# SNR/EbN0 sweep values
SNR_VALUES = [4, 6, 8, 10, 12, 14, 16]  # dB
TARGET_SNR = 8  # Default SNR for simulations

# Rician fading K-factors
RICIAN_K_FACTORS = [4, 6, 8, 10, 12, 14, 16]  # dB
TARGET_K_FACTOR = 10

# Frequency offset
FREQ_OFFSET = 0

# Delay parameters
ANTENNA1_DELAY_VALUES = [165.2]  # Fractional delay for testing
ANTENNA2_PHASE_SHIFT_VALUES = []
ANTENNA2_PHASE_SHIFT_VALUES.append(2*np.pi*CARRIER_FREQ*ANTENNAS_DISTANCE * np.sin(np.radians(DOA))/ SPEED_OF_LIGHT)
print(f"ANTENNA2_PHASE_SHIFT_VALUES: {ANTENNA2_PHASE_SHIFT_VALUES}")
SPS_SWEEP_VALUES = [1, 2, 4, 8, 16]  # SPS values for sweep analysis

# ============================================================================
# TIMING RECOVERY ALGORITHM PARAMETERS
# ============================================================================

# Early-Late Loop parameters
EARLY_LATE_DEPTH = 0.25  # d parameter for early-late gate
EARLY_LATE_KP = 0.015  # Proportional gain
EARLY_LATE_KI = 0.002  # Integral gain

# Gardner Loop parameters
GARDNER_KP = 0.08
GARDNER_KI = 0.01

# LMS Adaptive Timing Recovery parameters
LMS_MU_PHASE = 0.01  # Learning rate for phase adaptation
LMS_MU_DRIFT = 0.0001  # Learning rate for clock drift

# ML Grid Search parameters
ML_GRID_RESOLUTION = 0.01  # Fractional delay grid resolution

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

EYE_DIAGRAM_SNR = 10  # SNR for eye diagram plotting
PLOT_NUM_SYMBOLS = 40  # Number of symbols to show in eye diagram
PLOT_SKIP_SYMBOLS = 5  # Skip transients in eye diagram

# ============================================================================
# METHOD LABELS AND STYLING
# ============================================================================

METHOD_LABELS = [
    "Integer Correlation (M1)",
    "Parabolic Interp (M2)",
    "ML Grid Search (M3)",
    "Early-Late Loop (M4)",
    "Gardner Loop (M5)",
    "LMS Adaptive (M6)"
]

METHOD_STYLES = {
    1: {'label': 'Integer Correlation (M1)', 'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
    2: {'label': 'Parabolic Interp (M2)', 'color': '#ff7f0e', 'marker': 's', 'ls': '-'},
    3: {'label': 'ML Grid Search (M3)', 'color': '#2ca02c', 'marker': '^', 'ls': '-'},
    4: {'label': 'Early-Late Loop (M4)', 'color': '#d62728', 'marker': 'x', 'ls': '-'},
    5: {'label': 'Gardner Loop (M5)', 'color': '#9467bd', 'marker': 'd', 'ls': '-'},
    6: {'label': 'LMS Adaptive (M6)', 'color': '#8c564b', 'marker': 'v', 'ls': '-'}
}

# ============================================================================
# SIMULATION CONTROL FLAGS
# ============================================================================

ENABLE_PLOTTING = True
ENABLE_CONSOLE_OUTPUT = True
ENABLE_EYE_DIAGRAMS = True
