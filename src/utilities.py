"""
Utility functions for signal processing and data conversion.

Provides common helper functions for bit manipulation, BER calculation,
and scalar extraction used throughout the simulation.
"""

import numpy as np


def symbols_to_bits(symbol_samples, is_qpsk=True):
    """
    Convert complex symbols back to bit stream.
    
    For QPSK:
        - Bit 0 (I-channel): 1 if real(symbol) >= 0, else 0
        - Bit 1 (Q-channel): 1 if imag(symbol) >= 0, else 0
    
    Parameters
    ----------
    symbol_samples : ndarray
        Received symbols (complex for QPSK, real for BPSK)
    is_qpsk : bool
        If True, assumes QPSK (2 bits per symbol); else BPSK (1 bit per symbol)
    
    Returns
    -------
    ndarray
        Recovered bit stream (1D array of 0s and 1s)
    """
    if is_qpsk:
        bit_i = (np.real(symbol_samples) < 0).astype(int)
        bit_q = (np.imag(symbol_samples) < 0).astype(int)
        bits = np.zeros(len(bit_i) * 2, dtype=int)
        bits[0::2] = bit_i
        bits[1::2] = bit_q
    else:
        bits = (np.real(symbol_samples) < 0).astype(int)
    
    return bits


def calculate_ber(original_bits, recovered_bits):
    """
    Calculate Bit Error Rate (BER) between two bit streams.
    
    Handles length mismatch by aligning to the shorter sequence.
    Safely handles edge cases (all zeros, perfect reception, etc.).
    
    Parameters
    ----------
    original_bits : ndarray
        Original transmitted bits
    recovered_bits : ndarray
        Recovered/received bits
    
    Returns
    -------
    float
        BER value in range [0.0, 1.0]
    """
    min_len = min(len(original_bits), len(recovered_bits))
    
    if min_len == 0:
        return 1.0
    
    num_correct = np.sum(original_bits[:min_len] == recovered_bits[:min_len])
    ber = float((min_len - num_correct) / min_len)
    
    return ber


def to_scalar(val):
    """
    Safely extract a single scalar float from various input types.
    
    Useful for extracting converged values from tracking loops,
    integrators, or adaptive algorithm states.
    
    If input is array/list, extracts the last (converged) value.
    
    Parameters
    ----------
    val : float, int, list, ndarray
        Value to convert to scalar
    
    Returns
    -------
    float
        Scalar float value
    """
    if isinstance(val, (list, np.ndarray)):
        flat = np.asarray(val).ravel()
        return float(flat[-1]) if flat.size > 0 else 0.0
    return float(val)


def calculate_range_error(estimated_delay, true_delay, sampling_freq, speed_of_light=3e8):
    """
    Convert delay estimation error to physical range error.
    
    Range Error = (Estimated Delay - True Delay) * c / fs
    
    Parameters
    ----------
    estimated_delay : float
        Estimated delay in samples
    true_delay : float
        True (reference) delay in samples
    sampling_freq : float
        Sampling frequency in Hz
    speed_of_light : float
        Speed of light (default: 3e8 m/s)
    
    Returns
    -------
    float
        Range error in meters
    """
    delay_error_samples = estimated_delay - true_delay
    range_error = delay_error_samples * (speed_of_light / sampling_freq)
    return range_error


def validate_signal_length(signal_length, required_length, name="signal"):
    """
    Validate that a signal has sufficient length.
    
    Parameters
    ----------
    signal_length : int
        Actual signal length
    required_length : int
        Minimum required length
    name : str
        Signal name for error messages
    
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
    """
    if signal_length < required_length:
        raise ValueError(
            f"{name} length ({signal_length}) is less than "
            f"required length ({required_length})"
        )
    return True


def clip_to_db_range(signal_db, min_db=-60, max_db=10):
    """
    Clip dB values to a reasonable visualization range.
    
    Useful for preventing visualization artifacts when plotting
    magnitude responses with extreme dB values.
    
    Parameters
    ----------
    signal_db : ndarray
        Signal magnitude in dB
    min_db : float
        Minimum dB value (default: -60 dB)
    max_db : float
        Maximum dB value (default: 10 dB)
    
    Returns
    -------
    ndarray
        Clipped dB values
    """
    return np.clip(signal_db, min_db, max_db)


def ensure_complex(signal):
    """
    Ensure signal is complex-valued.
    
    Converts real signals to complex with zero imaginary part.
    
    Parameters
    ----------
    signal : ndarray
        Input signal (real or complex)
    
    Returns
    -------
    ndarray
        Complex signal
    """
    if not np.iscomplexobj(signal):
        return signal.astype(complex)
    return signal


def ensure_real(signal):
    """
    Extract real part of signal.
    
    Parameters
    ----------
    signal : ndarray
        Input signal (real or complex)
    
    Returns
    -------
    ndarray
        Real part of signal
    """
    return np.real(signal)
