"""QPSK Transmitter - Generates bit streams and applies pulse shaping."""

import numpy as np
from scipy.interpolate import interp1d
from RRC_Implementation import get_impulse_and_freq_response


def farrow_interpolation(signal, fractional_delay, degree=3):
    """Cubic interpolation for fractional time delay."""
    is_complex = np.iscomplexobj(signal)
    t_orig = np.arange(len(signal), dtype=float)
    t_delayed = t_orig - fractional_delay
    kind = min(degree, 3)
    
    if is_complex:
        f_real = interp1d(t_orig, np.real(signal), kind=kind, 
                          fill_value='extrapolate', bounds_error=False)
        f_imag = interp1d(t_orig, np.imag(signal), kind=kind, 
                          fill_value='extrapolate', bounds_error=False)
        return f_real(t_delayed) + 1j * f_imag(t_delayed)
    else:
        f = interp1d(t_orig, signal, kind=kind, 
                     fill_value='extrapolate', bounds_error=False)
        return f(t_delayed)


class Client_Tx:
    """QPSK transmitter with pulse shaping and delay application."""
    
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
        """Configure RRC filter."""
        self.impulse_response, self.freq_response = get_impulse_and_freq_response(beta, sps, filter_span)

    def generate_bit_array(self):
        """Generate random QPSK bit pairs or BPSK bits."""
        if self.is_qpsk:
            self.bit_array = [tuple(np.random.randint(0, 2, 2)) for _ in range(self.num_of_bits // 2)]
        else:
            self.bit_array = [np.random.randint(0, 2) for _ in range(self.num_of_bits)]
        self.map_bits()

    def map_bits(self):
        """Map bit tuples to constellation symbols."""
        self.mapped_bits = [self.bit_mapping[b] for b in self.bit_array]

    def add_delay(self, signal, delay, use_farrow=True):
        """Apply integer and fractional delay to signal."""
        if delay <= 0:
            return signal
        
        integer_delay = int(np.floor(delay))
        frac_delay = delay - integer_delay
        
        if use_farrow and frac_delay > 1e-6:
            delayed_signal = farrow_interpolation(signal, frac_delay, degree=self.farrow_degree)
        else:
            delayed_signal = signal.copy()
        
        if integer_delay > 0:
            delayed_signal = np.concatenate([np.zeros(integer_delay, dtype=delayed_signal.dtype), delayed_signal])
        
        return delayed_signal

    def add_frequency_offset(self, signal, freq_offset, sample_rate=8):
        """Apply complex exponential frequency shift."""
        if freq_offset == 0:
            return signal
        t = np.arange(len(signal)) / sample_rate
        return signal * np.exp(1j * 2 * np.pi * freq_offset * t)

    def prepare_x_t(self, upsampled_signal, delay=0, freq_offset=0, sample_rate=1):
        """Create transmitted signal: filter → frequency offset → delay."""
        transmitted_signal = np.convolve(upsampled_signal, self.impulse_response, mode='full')
        transmitted_signal = self.add_frequency_offset(transmitted_signal, freq_offset, sample_rate)
        return self.add_delay(transmitted_signal, delay, use_farrow=True)