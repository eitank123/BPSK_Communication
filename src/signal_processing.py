"""
Signal processing module for signal analysis and downsampling.

Provides functions for analysis and symbol-rate recovery operations
used in the timing synchronization and equalization pipeline.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal as scipy_signal


def downsample_to_symbol_rate(rx_signal, sps, offset=0):
    """
    Downsample received signal to symbol rate.
    
    Parameters
    ----------
    rx_signal : ndarray
        Received signal at sampling rate
    sps : int
        Samples per symbol
    offset : int
        Sample offset before downsampling
    
    Returns
    -------
    ndarray
        Downsampled symbols
    """
    return rx_signal[offset::sps]


def create_complex_interpolator(signal):
    """
    Create a cubic interpolator for complex signals.
    
    Separately interpolates real and imaginary parts for accurate
    phase preservation during fractional sample delay estimation.
    
    Parameters
    ----------
    signal : ndarray
        Complex input signal
    
    Returns
    -------
    callable
        Interpolation function accepting sample points
    """
    t = np.arange(len(signal))
    
    if np.iscomplexobj(signal):
        interp_real = interp1d(
            t, np.real(signal), kind='cubic',
            bounds_error=False, fill_value=0.0
        )
        interp_imag = interp1d(
            t, np.imag(signal), kind='cubic',
            bounds_error=False, fill_value=0.0
        )
        return lambda pts: interp_real(pts) + 1j * interp_imag(pts)
    else:
        return interp1d(
            t, signal, kind='cubic',
            bounds_error=False, fill_value=0.0
        )


def cubic_interpolate(signal, sample_points):
    """
    Interpolate signal at arbitrary sample points using cubic splines.
    
    Supports both real and complex signals with proper handling of
    phase information in complex case.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    sample_points : ndarray or list
        Points where signal should be evaluated
    
    Returns
    -------
    ndarray
        Interpolated samples
    """
    t = np.arange(len(signal))
    
    if np.iscomplexobj(signal):
        interp_real = interp1d(
            t, np.real(signal), kind='cubic',
            bounds_error=False, fill_value=0.0
        )
        interp_imag = interp1d(
            t, np.imag(signal), kind='cubic',
            bounds_error=False, fill_value=0.0
        )
        return interp_real(sample_points) + 1j * interp_imag(sample_points)
    else:
        interp_func = interp1d(
            t, signal, kind='cubic',
            bounds_error=False, fill_value=0.0
        )
        return interp_func(sample_points)


def correlate_signals(signal1, signal2, mode='full'):
    """
    Compute cross-correlation between two signals.
    
    Wrapper around scipy.signal.correlate for consistency.
    
    Parameters
    ----------
    signal1 : ndarray
        First signal
    signal2 : ndarray
        Second signal
    mode : str
        'full', 'same', or 'valid'
    
    Returns
    -------
    ndarray
        Cross-correlation result
    """
    return scipy_signal.correlate(signal1, signal2, mode=mode)


def convolve_signals(signal, filter_coeffs, mode='full'):
    """
    Convolve signal with filter coefficients.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    filter_coeffs : ndarray
        Filter coefficients
    mode : str
        'full', 'same', or 'valid'
    
    Returns
    -------
    ndarray
        Convolved signal
    """
    return np.convolve(signal, filter_coeffs, mode=mode)


def fft_and_normalize(signal, fft_size=None):
    """
    Compute FFT and normalize by signal length.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    fft_size : int, optional
        FFT size (pads if larger than signal)
    
    Returns
    -------
    ndarray
        Normalized FFT
    """
    if fft_size is None:
        fft_size = len(signal)
    
    spectrum = np.fft.fft(signal, fft_size)
    return spectrum / len(signal)


def estimate_signal_power(signal):
    """
    Estimate average power of a signal.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    
    Returns
    -------
    float
        Average power (mean squared magnitude)
    """
    return np.mean(np.abs(signal) ** 2)


def estimate_energy_per_symbol(signal, sps):
    """
    Calculate energy per symbol.
    
    Parameters
    ----------
    signal : ndarray
        Input signal at sample rate
    sps : int
        Samples per symbol
    
    Returns
    -------
    float
        Energy per symbol
    """
    signal_power = estimate_signal_power(signal)
    return signal_power * sps


def normalize_signal(signal, target_power=1.0):
    """
    Normalize signal to target average power.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    target_power : float
        Desired average power
    
    Returns
    -------
    ndarray
        Normalized signal
    """
    current_power = estimate_signal_power(signal)
    
    if current_power < 1e-12:
        return signal
    
    scale_factor = np.sqrt(target_power / current_power)
    return signal * scale_factor


def find_peak(correlation):
    """
    Find peak index in correlation sequence.
    
    Handles both real and complex correlation.
    
    Parameters
    ----------
    correlation : ndarray
        Correlation sequence
    
    Returns
    -------
    int
        Index of maximum magnitude
    """
    return np.argmax(np.abs(correlation))


def apply_phase_rotation(signal, phase_rad):
    """
    Apply complex phase rotation to signal.
    
    Parameters
    ----------
    signal : ndarray
        Complex input signal
    phase_rad : float
        Phase rotation in radians
    
    Returns
    -------
    ndarray
        Phase-rotated signal
    """
    return signal * np.exp(1j * phase_rad)


def remove_dc_component(signal):
    """
    Remove DC (average) component from signal.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    
    Returns
    -------
    ndarray
        DC-removed signal
    """
    return signal - np.mean(signal)


def estimate_snr_db(signal, noise):
    """
    Estimate SNR between signal and noise.
    
    Parameters
    ----------
    signal : ndarray
        Signal component
    noise : ndarray
        Noise component
    
    Returns
    -------
    float
        SNR in dB
    """
    signal_power = estimate_signal_power(signal)
    noise_power = estimate_signal_power(noise)
    
    if noise_power < 1e-12:
        return np.inf
    
    return 10 * np.log10(signal_power / noise_power)
