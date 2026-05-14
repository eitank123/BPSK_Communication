import numpy as np
from RRC_Implementation import *


class Client_Tx:
    def __init__(self, num_of_bits, bit_mapping):
        self.num_of_bits = num_of_bits
        self.bit_mapping = bit_mapping
        self.impulse_response = None
        self.freq_response = None
        self.bit_array = []
        self.mapped_bits = []

    def set_responses(self, beta, sps, filter_span):
        self.impulse_response, self.freq_response = get_impulse_and_freq_response(beta, sps, filter_span)

    def map_bits(self):
        self.mapped_bits = [self.bit_mapping[b] for b in self.bit_array]

    def generate_bit_array(self):
        self.bit_array = [np.random.randint(0, 2) for bit in range(self.num_of_bits)]
        self.map_bits()

    def upscale_array(self, sps):
        upsampled = np.zeros(len(self.mapped_bits) * sps)
        upsampled[::sps] = self.mapped_bits
        return upsampled

    def add_delay(self, signal, delay):
        if delay <= 0:
            return signal
        delayed_signal = np.concatenate((np.zeros(delay), signal))
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
