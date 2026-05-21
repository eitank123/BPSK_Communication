import numpy as np
from RRC_Implementation import *


def farrow_interpolation(signal, fractional_delay, degree=3):
    """
    Farrow structure interpolation for fractional time delay.
    
    signal: input signal (can be complex)
    fractional_delay: desired fractional delay in samples (0 < fractional_delay < 1)
    degree: polynomial degree (1=linear, 2=quadratic, 3=cubic)
    
    Returns: interpolated signal with fractional delay applied
    """
    from scipy.interpolate import interp1d
    
    is_complex = np.iscomplexobj(signal)
    
    # Original signal time axis
    t_orig = np.arange(len(signal), dtype=float)
    
    # Delayed time axis
    t_delayed = t_orig - fractional_delay
    
    if is_complex:
        # Handle complex signals separately (real and imaginary parts)
        f_real = interp1d(t_orig, np.real(signal), kind=degree if degree <= 3 else 3, 
                         fill_value='extrapolate', bounds_error=False)
        f_imag = interp1d(t_orig, np.imag(signal), kind=degree if degree <= 3 else 3, 
                         fill_value='extrapolate', bounds_error=False)
        output = f_real(t_delayed) + 1j * f_imag(t_delayed)
    else:
        # For real signals
        f = interp1d(t_orig, signal, kind=degree if degree <= 3 else 3, 
                    fill_value='extrapolate', bounds_error=False)
        output = f(t_delayed)
    
    return output


class Client_Tx:
    def __init__(self, num_of_bits, bit_mapping, is_qpsk=False, farrow_degree=3):
        self.num_of_bits = num_of_bits
        self.bit_mapping = bit_mapping
        self.is_qpsk = is_qpsk
        self.farrow_degree = farrow_degree
        self.impulse_response = None
        self.freq_response = None
        self.bit_array = []
        self.mapped_bits = []

    def set_responses(self, beta, sps, filter_span):
        self.impulse_response, self.freq_response = get_impulse_and_freq_response(beta, sps, filter_span)

    def map_bits(self):
        if self.is_qpsk:
            self.mapped_bits = [self.bit_mapping[b] for b in self.bit_array]
        else:
            self.mapped_bits = [self.bit_mapping[b] for b in self.bit_array]

    def generate_bit_array(self):
        if self.is_qpsk:
            # Generate 2-bit tuples for QPSK
            self.bit_array = [tuple(np.random.randint(0, 2, 2)) for _ in range(self.num_of_bits // 2)]
        else:
            self.bit_array = [np.random.randint(0, 2) for bit in range(self.num_of_bits)]
        self.map_bits()

    def upscale_array(self, sps):
        # Create complex array if QPSK, otherwise real array for BPSK
        dtype = complex if self.is_qpsk else float
        upsampled = np.zeros(len(self.mapped_bits) * sps, dtype=dtype)
        upsampled[::sps] = self.mapped_bits
        return upsampled

    def add_delay(self, signal, delay, use_farrow=False):
        """
        Apply delay to signal using Farrow interpolation for fractional delays.
        
        signal: input signal
        delay: delay in samples (can be fractional)
        use_farrow: if True, use Farrow interpolation; if False, use simple zero-padding (default False for robustness)
        """
        if delay <= 0:
            return signal
        
        # Split into integer and fractional parts
        integer_delay = int(np.floor(delay))
        frac_delay = delay - integer_delay
        
        # Apply Farrow interpolation for fractional part if significant
        if use_farrow and frac_delay > 1e-6:
            delayed_signal = farrow_interpolation(signal, frac_delay, degree=self.farrow_degree)
        else:
            delayed_signal = signal.copy()
        
        # Add integer delay as zero-padding
        if integer_delay > 0:
            delayed_signal = np.concatenate([np.zeros(integer_delay, dtype=delayed_signal.dtype), delayed_signal])
        
        return delayed_signal

    def add_frequency_offset(self, signal, freq_offset, sample_rate=8):
        if freq_offset == 0:
            return signal
        t = np.arange(len(signal)) / sample_rate
        return signal * np.exp(1j * 2 * np.pi * freq_offset * t)

    def prepare_x_t(self, upsampled_signal, delay=0, freq_offset=0, sample_rate=1):
        transmitted_signal = np.convolve(upsampled_signal, self.impulse_response, mode='full')
        transmitted_signal = self.add_frequency_offset(transmitted_signal, freq_offset, sample_rate)
        return self.add_delay(transmitted_signal, delay)
