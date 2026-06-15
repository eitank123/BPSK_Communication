"""
Channel operations module for signal degradation and formatting.

Handles cyclic prefix insertion, Zadoff-Chu preamble generation,
and Rician fading channel simulation.
"""

import numpy as np
from config import SPEED_OF_LIGHT
import matplotlib.pyplot as plt


def add_cyclic_prefix(data_block, cp_length):
    """
    Add Cyclic Prefix (CP) to a data block for SC-FDE transmission.
    
    The CP is created by prepending the last G samples of the data block
    to its front, enabling simple equalization even with multipath channels.
    
    Parameters
    ----------
    data_block : ndarray
        Time-domain block of symbols (length N)
    cp_length : int
        Length of cyclic prefix (number of samples to copy from end)
    
    Returns
    -------
    ndarray
        Formatted block with CP prepended (length N + cp_length)
    """
    if cp_length <= 0:
        return data_block
    
    # Extract the last cp_length samples
    cp = data_block[-cp_length:]
    
    # Prepend CP to the original block
    formatted_block = np.concatenate((cp, data_block))
    
    return formatted_block


def generate_zadoff_chu_preamble(length, root_index=1):
    """
    Generate a Zadoff-Chu sequence for preamble/pilot signals.
    
    Zadoff-Chu sequences have excellent autocorrelation properties
    (sharp peak, low sidelobe) making them ideal for synchronization.
    
    Parameters
    ----------
    length : int
        Length of the preamble sequence
    root_index : int
        Root index of the sequence (coprime with length)
    
    Returns
    -------
    ndarray
        Complex Zadoff-Chu preamble (length `length`)
    """
    n = np.arange(length)
    exponent = -1j * np.pi * root_index * n * (n + 1) / length
    zadoff_chu = np.exp(exponent)
    return zadoff_chu



def add_rician_fading(signal, k_db, ebno_db, sps, num_taps=8, decay_factor=2):
    """
    Apply a frequency-selective Rician fading channel using a Tapped Delay Line
    model with an exponential Power Delay Profile and AWGN.

    Parameters
    ----------
    signal : ndarray
        Input oversampled baseband signal.
    k_db : float
        Rician K-factor in dB.
    ebno_db : float
        Target Eb/N0 in dB.
    sps : int
        Samples per symbol.
    num_taps : int, optional
        Number of multipath taps.
    decay_factor : float, optional
        Exponential PDP decay factor.

    Returns
    -------
    ndarray
        Faded signal with AWGN added.
    """

    N = len(signal)
    K_linear = 10.0 ** (k_db / 10.0)

    # ------------------------------------------------------------------
    # STEP 1: Generate tap delays
    # ------------------------------------------------------------------
    # Spread taps over roughly 4 symbol durations
    max_delay = max(num_taps, 4 * sps)

    tap_delays = np.sort(
        np.random.choice(
            np.arange(max_delay),
            size=num_taps,
            replace=False
        )
    )

    # Force LOS path to arrive first
    tap_delays[0] = 0

    # ------------------------------------------------------------------
    # STEP 2: Power Delay Profile
    # ------------------------------------------------------------------
    p_nlos = np.exp(-np.arange(num_taps) / decay_factor)
    p_nlos /= np.sum(p_nlos)

    # ------------------------------------------------------------------
    # STEP 3: Generate Rician tap gains
    # ------------------------------------------------------------------
    h_los = np.zeros(num_taps, dtype=complex)
    h_los[0] = np.sqrt(K_linear / (K_linear + 1.0))

    sigma_nlos = np.sqrt(
        p_nlos / (2.0 * (K_linear + 1.0))
    )

    h_nlos = sigma_nlos * (
        np.random.randn(num_taps) +
        1j * np.random.randn(num_taps)
    )

    tap_gains = h_los + h_nlos

    # ------------------------------------------------------------------
    # STEP 4: Build sparse channel impulse response
    # ------------------------------------------------------------------
    h = np.zeros(max_delay + 1, dtype=complex)

    for delay, gain in zip(tap_delays, tap_gains):
        h[delay] += gain

    # Normalize average channel power
    h /= np.sqrt(np.sum(np.abs(h) ** 2))

    # ------------------------------------------------------------------
    # STEP 5: Apply channel
    # ------------------------------------------------------------------
    faded_signal = np.convolve(signal, h, mode='full')[:N]

    # ------------------------------------------------------------------
    # STEP 6: Add AWGN
    # ------------------------------------------------------------------
    signal_power = np.mean(np.abs(signal) ** 2)

    # Symbol energy
    Es = signal_power * sps

    # QPSK: Es = 2Eb
    EbN0 = 10.0 ** (ebno_db / 10.0)
    EsN0 = 2.0 * EbN0

    N0 = Es / EsN0

    noise = np.sqrt(N0 / 2.0) * (
        np.random.randn(N) +
        1j * np.random.randn(N)
    )

    return faded_signal + noise

def upsample_symbols(symbols, sps, is_complex=True):
    """
    Upsample symbol stream by inserting zeros between symbols.
    
    Parameters
    ----------
    symbols : ndarray
        Symbol stream to upsample
    sps : int
        Samples per symbol (upsampling factor)
    is_complex : bool
        If True, output is complex; else float
    
    Returns
    -------
    ndarray
        Upsampled signal with length len(symbols) * sps
    """
    dtype = complex if is_complex else float
    upsampled = np.zeros(len(symbols) * sps, dtype=dtype)
    upsampled[::sps] = symbols
    return upsampled


def downsample_symbols(signal, sps, offset=0):
    """
    Downsample signal to symbol rate.
    
    Parameters
    ----------
    signal : ndarray
        Input signal at sps * symbol_rate
    sps : int
        Samples per symbol
    offset : int
        Sample offset before downsampling (default: 0)
    
    Returns
    -------
    ndarray
        Downsampled signal
    """
    return signal[offset::sps]


def create_formatted_payload(data_symbols, data_block_size, cp_length):
    """
    Format data symbols with SC-FDE structure (CP + blocks).
    
    Segments data into blocks, pads final block if needed,
    and adds cyclic prefix to each block.
    
    Parameters
    ----------
    data_symbols : ndarray
        Symbols to format
    data_block_size : int
        Size of each SC-FDE block (FFT size)
    cp_length : int
        Cyclic prefix length
    
    Returns
    -------
    ndarray
        Formatted payload with CP on each block
    """
    formatted_data = []
    
    # Segment into blocks and add CP
    for i in range(0, len(data_symbols), data_block_size):
        block = data_symbols[i : i + data_block_size]
        
        # Zero-pad final block if needed
        if len(block) < data_block_size:
            block = np.pad(block, (0, data_block_size - len(block)), mode='constant')
        
        # Add cyclic prefix
        block_with_cp = add_cyclic_prefix(block, cp_length)
        formatted_data.append(block_with_cp)
    
    # Concatenate all blocks
    return np.concatenate(formatted_data)


def extract_block_payload(rx_signal, sps, preamble_length, data_block_size, cp_length):
    """
    Extract and unformat received payload (remove CP, downsample).
    
    Parameters
    ----------
    rx_signal : ndarray
        Received signal (at sample rate)
    sps : int
        Samples per symbol
    preamble_length : int
        Length of preamble symbols to skip
    data_block_size : int
        Size of each SC-FDE block
    cp_length : int
        Cyclic prefix length per block
    
    Returns
    -------
    ndarray
        Extracted data symbols (CP removed, downsampled)
    """
    # Downsample to symbol rate
    rx_symbols = rx_signal[::sps]
    
    # Skip preamble
    data_stream = rx_symbols[preamble_length:]
    
    # Remove CP from each block and concatenate
    block_stride = data_block_size + cp_length
    payload_symbols = []
    
    for i in range(0, len(data_stream), block_stride):
        block_with_cp = data_stream[i : i + block_stride]
        
        if len(block_with_cp) < block_stride:
            break
        
        # Remove CP (first cp_length samples)
        block_data = block_with_cp[cp_length:]
        payload_symbols.extend(block_data)
    
    return np.array(payload_symbols)
