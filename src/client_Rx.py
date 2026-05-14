from RRC_Implementation import *


class Client_Rx:
    def __init__(self, signal):
        self.signal = signal
        self.impulse_response = None
        self.freq_response = None
        self.filtered_signal = []

    def filter_signal(self):
        for signal in self.signal:
            self.filtered_signal.append(np.convolve(signal, self.impulse_response, mode='full'))

    def set_responses(self, beta, sps, filter_span):
        self.impulse_response, self.freq_response = get_impulse_and_freq_response(beta, sps, filter_span)

    def sample_receiver(self, sps, span):
        """
        Samples the filtered Rx signal at the optimal points.

        filtered_rx_signal: The output of the Rx matched filter
        sps: Samples per symbol
        span: The span of the RRC filter
        """
        # 1. Calculate the total group delay in samples
        # TX delay (span*sps/2) + RX delay (span*sps/2) = span*sps
        total_delay = sps * span
        # 2. Extract samples
        # We start at total_delay and jump by sps
        # We stop before the tail end of the convolution
        sampled_output = []
        for filt_signal in self.filtered_signal:
            symbol_samples = np.array(filt_signal[total_delay:: sps])
            decided_bits = np.sign(np.real(symbol_samples))
            decided_bits[decided_bits == 0] = 1
            sampled_output.append(decided_bits)
        return sampled_output
