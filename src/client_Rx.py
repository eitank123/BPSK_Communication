from RRC_Implementation import *
from scipy import signal


class Client_Rx:
    def __init__(self, signal, is_qpsk=False):
        self.signal = signal
        self.is_qpsk = is_qpsk
        self.impulse_response = None
        self.freq_response = None
        self.filtered_signal = []
        self.preamble_delay = None  # Will store detected preamble delay for each SNR level
        self.detected_delays_list = []  # Store all detected delays

    def filter_signal(self):
        for signal in self.signal:
            self.filtered_signal.append(np.convolve(signal, self.impulse_response, mode='full'))

    def set_responses(self, beta, sps, filter_span):
        self.impulse_response, self.freq_response = get_impulse_and_freq_response(beta, sps, filter_span)

    def detect_preamble(self, preamble, sps, preamble_length, filter_span):
        """
        Detect Zadoff-Chu preamble in filtered signals using correlation.
        Stores detected delays for each SNR level.
        
        filter_span: RRC filter span (needed for sample_receiver to account for group delay)
        """
        self.detected_delays_list = []
        self.filter_span = filter_span  # Store for use in sample_receiver
        preamble_upscaled = np.repeat(preamble, sps)
        
        for filt_signal in self.filtered_signal:
            # Correlate using scipy.signal.correlate with 'full' mode
            correlation = signal.correlate(filt_signal, preamble_upscaled, mode='full')
            # Find the peak of the correlation
            peak_idx = np.argmax(np.abs(correlation))
            
            # In 'full' mode, convert peak index to signal position
            # peak_idx corresponds to template starting at: peak_idx - (len(template) - 1)
            detected_sample = peak_idx - (len(preamble_upscaled) - 1)
            detected_sample = max(0, detected_sample)
            
            self.detected_delays_list.append(detected_sample)
            print(f"Preamble detected at sample {detected_sample} (peak at {peak_idx})")
        
        return self.detected_delays_list

    def sample_receiver(self, sps, span, preamble_length):
        """
        Samples the filtered Rx signal at the optimal points, starting AFTER the preamble.
        
        preamble_length: number of preamble symbols
        sps: Samples per symbol
        span: The span of the RRC filter
        """
        # Calculate preamble length in samples
        preamble_samples = preamble_length * sps
        
        sampled_output = []
        for snr_idx, filt_signal in enumerate(self.filtered_signal):
            # Start position: detected preamble location + preamble length
            # detected_delay already accounts for all delays (TX delay + filter effects)
            detected_delay = self.detected_delays_list[snr_idx] if snr_idx < len(self.detected_delays_list) else 0
            start_sample = detected_delay + preamble_samples
            
            # Extract symbol samples starting from message (skip preamble)
            symbol_samples = np.array(filt_signal[start_sample:: sps])
            
            if self.is_qpsk:
                # Decode QPSK: extract 2 bits from real and imaginary parts
                # Bit 0 from sign of real part, Bit 1 from sign of imaginary part
                bit0 = (np.sign(np.real(symbol_samples)) + 1) // 2  # 0->-1->0, 1->1->1
                bit0 = 1 - bit0  # Invert: -1 sign -> 1 bit, 1 sign -> 0 bit
                bit1 = (np.sign(np.imag(symbol_samples)) + 1) // 2  # Same for imag
                bit1 = 1 - bit1
                # Flatten bits into a 1D array (interleave or concatenate)
                decided_bits = np.zeros(len(bit0) * 2, dtype=int)
                decided_bits[0::2] = bit0
                decided_bits[1::2] = bit1
            else:
                # BPSK decoding
                decided_bits = np.sign(np.real(symbol_samples))
                decided_bits[decided_bits == 0] = 1
                decided_bits = (decided_bits + 1) // 2  # Convert -1,1 to 0,1
            
            sampled_output.append(decided_bits)
        return sampled_output
