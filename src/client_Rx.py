"""
QPSK Receiver - 6 timing synchronization methods + channel estimation/equalization.

Methods:
  1. Integer Correlation - Coarse integer delay
  2. Parabolic Interpolation - Fractional delay via parabola fit
  3. ML Grid Search - Maximum likelihood fractional estimation
  4. Early-Late Loop - Tracking-based (PI control)
  5. Gardner Loop - Classic timing recovery loop
  6. LMS Adaptive - Adaptive filter-based tracking
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from signal_processing import cubic_interpolate
import matplotlib.pyplot as plt

class Client_Rx:
    def __init__(self, signal_list, is_qpsk=False):
        self.signal = signal_list
        self.is_qpsk = is_qpsk
        self.impulse_response = None
        self.freq_response = None
        self.filtered_signal = []
        self.detected_delays_list = []

    def set_responses(self, beta, sps, filter_span):
        from RRC_Implementation import get_impulse_and_freq_response
        self.impulse_response, self.freq_response = get_impulse_and_freq_response(beta, sps, filter_span)

    def filter_signal(self, sps, filter_span):
        self.filtered_signal = []
        total_filter_delay = filter_span * sps
        
        for rx_signal in self.signal:
            filtered = np.convolve(rx_signal, self.impulse_response, mode='full')
            # Aligns the signal to 0 filter group delay
            aligned_filtered = filtered[total_filter_delay : len(rx_signal) + total_filter_delay]
            self.filtered_signal.append(aligned_filtered)

    def generate_reference_preamble(self, preamble, sps):
        """Generate expected preamble for correlation."""
        preamble_upsampled = np.zeros(len(preamble) * sps, dtype=complex)
        preamble_upsampled[::sps] = preamble
        reference = np.convolve(preamble_upsampled, self.impulse_response, mode='full')
        return reference

    # --- TIMING RECOVERY METHODS (1-6) ---
    def detect_preamble(self, preamble, sps, filter_span):
        self.detected_delays_list = []
        reference = self.generate_reference_preamble(preamble, sps)
        reference_length = len(reference)
        
        # Reference contains a group delay of (filter_span * sps) / 2
        reference_group_delay = (filter_span * sps) // 2

        for filt_signal in self.filtered_signal:
            # 1. Switch to 'valid' mode. It only computes correlation where the signals fully overlap.
            # This makes the base alignment index exactly equal to the peak_index.
            correlation = signal.correlate(filt_signal, reference, mode='valid')
            abs_corr = np.abs(correlation)
            
            # Find the coarse integer peak index
            peak_index = np.argmax(abs_corr)
            
            # --- 2. Parabolic Interpolation for Sub-Sample Precision ---
            # Protect array boundaries
            if 0 < peak_index < len(abs_corr) - 1:
                alpha = abs_corr[peak_index - 1]  # Left neighbor
                beta = abs_corr[peak_index]       # Discrete peak
                gamma = abs_corr[peak_index + 1]  # Right neighbor
                
                denominator = alpha - 2 * beta + gamma
                if np.abs(denominator) > 1e-6:
                    fractional_offset = 0.5 * (alpha - gamma) / denominator
                else:
                    fractional_offset = 0.0
            else:
                fractional_offset = 0.0
                
            # Combine the integer peak with its sub-sample fractional adjustment
            precise_peak = float(peak_index) + fractional_offset
            
            # 3. Calculate corrected delay without any arbitrary safety hard-coding
            # In 'valid' mode, the index directly represents the start delay.
            corrected_delay = precise_peak + reference_group_delay
            
            # Save the exact precise delay
            self.detected_delays_list.append(corrected_delay)

        return self.detected_delays_list
    

    # --- METHOD 2: Parabolic Interpolation ---
    def parabolic_interpolation(self, correlation):
        """Fit parabola to correlation peak for fractional delay."""
        peak = np.argmax(np.abs(correlation))
        if peak == 0 or peak == len(correlation) - 1:
            return peak, 0.0

        y1 = np.abs(correlation[peak - 1])
        y2 = np.abs(correlation[peak])
        y3 = np.abs(correlation[peak + 1])

        denominator = 2 * (y1 - 2 * y2 + y3)
        delta = 0.0 if abs(denominator) < 1e-12 else (y1 - y3) / denominator
        delta = np.clip(delta, -0.5, 0.5)
        return peak + delta, delta

    def estimate_fractional_delay(self, rx_signal, full_preamble, cp_length, sps, filter_span):
        """
        Estimates true fractional delay and Doppler shift using a full conjugate ZC frame.
        
        Parameters
        ----------
        rx_signal : ndarray
            The received signal vector.
        full_preamble : ndarray
            The complete concatenated transmission block: [CP1 + ZC1(u) + CP2 + ZC2(N-u)]
        cp_length : int
            The number of samples in the Cyclic Prefix before upsampling.
        sps : int
            Samples per symbol.
        filter_span : int
            The span of the pulse shaping filter, used to calculate group delay.
        """
        # --- Dynamically extract structures from the full preamble ---
        total_len = len(full_preamble)
        half_len = total_len // 2  # Length of one block: (N_zc + CP)
        
        # Extract pure ZC sequences (stripping the CP ensures a perfectly sharp correlation peak)
        preamble_u = full_preamble[cp_length : half_len]
        preamble_v = full_preamble[half_len + cp_length : total_len]
        
        # Calculate the physical design gap in the upsampled domain
        # The distance between the start of ZC1 and the start of ZC2 is exactly one block length
        n_gap_samples = half_len * sps
        
        # 1. Generate the upsampled/pulse-shaped references for both pure roots
        ref_u = self.generate_reference_preamble(preamble_u, sps)
        ref_v = self.generate_reference_preamble(preamble_v, sps)
        
        # 2. Run parallel cross-correlations against the received signal
        corr_u = signal.correlate(rx_signal, ref_u, mode='full')
        corr_v = signal.correlate(rx_signal, ref_v, mode='full')
        
        # 3. Find precise fractional peaks for both
        refined_peak_u, _ = self.parabolic_interpolation(corr_u)
        refined_peak_v, _ = self.parabolic_interpolation(corr_v)
        
        # 4. Map 'full' correlation indices back to the physical signal time-base
        t1 = refined_peak_u - (len(ref_u) - 1)
        t2 = refined_peak_v - (len(ref_v) - 1)
        
        # 5. Apply Conjugate Math to isolate True Time and Doppler Shift
        t2_aligned = t2 - n_gap_samples
        
        # True start time of ZC1 (Doppler time-shifts cancel out)
        true_time_zc1 = (t1 + t2_aligned) / 2.0
        
        # The true start of the entire payload is BEFORE ZC1 (we must step back over CP1)
        # Multiplying cp_length by sps converts it to the upsampled time-domain
        true_time_start = true_time_zc1 - (cp_length * sps)
        
        # The difference isolates the time-shift caused purely by Carrier Frequency Offset
        doppler_shift_samples = (t2_aligned - t1) / 2.0
        
        # 6. Correct for the filter group delay (Subtracting to find the true physical start)
        reference_group_delay = (filter_span * sps) // 2
        true_time_start += reference_group_delay
        
        # 7. Split into integer index and fractional component for the cubic interpolator
        estimated_delay = int(np.floor(true_time_start))
        frac = true_time_start - estimated_delay
        
        return estimated_delay, frac, doppler_shift_samples, (corr_u, corr_v)

    # --- METHOD 3: Maximum Likelihood Grid Search ---
    def ml_fractional_delay(self, rx_signal, preamble, sps, filter_span, grid_resolution=0.01):
        """Estimate fractional delay via ML grid search."""
        reference = self.generate_reference_preamble(preamble, sps)
        correlation = signal.correlate(rx_signal, reference, mode='full')
        coarse_peak = np.argmax(np.abs(correlation))
        
        fractional_grid = np.arange(-1.0, 1.0, grid_resolution)
        search_points = coarse_peak + fractional_grid
        
        interpolated_corr = cubic_interpolate(correlation, search_points)
        best_idx = np.argmax(np.abs(interpolated_corr))
        
        refined_peak = coarse_peak + fractional_grid[best_idx]
        coarse_delay = refined_peak - (len(reference) - 1)
        reference_group_delay = (filter_span * sps) // 2
        estimated_delay = coarse_delay + reference_group_delay
        
        return estimated_delay, fractional_grid[best_idx], correlation

    # --- METHOD 4: Early-Late Loop ---
    def early_late_block_recovery(self, interp_func, start_idx, block_symbol_offset, block_size, sps, timing_phase, d=0.25):
        """
        Processes a single block of symbols using a FIXED timing phase 
        to preserve circular convolution for SC-FDE.
        """
        block_symbols = []
        block_error = 0.0

        for m in range(block_size):
            # Calculate the absolute symbol index within the frame sequence
            global_symbol_idx = block_symbol_offset + m
            
            # Apply the current uniform timing phase for this entire block
            current_symbol_idx = start_idx + global_symbol_idx * sps + timing_phase
            early_idx = current_symbol_idx - d * sps
            late_idx = current_symbol_idx + d * sps

            # Interpolate early, current, and late samples
            y_early = interp_func([early_idx])[0]
            y_curr = interp_func([current_symbol_idx])[0]
            y_late = interp_func([late_idx])[0]

            block_symbols.append(y_curr)

            # Accumulate the timing error across the block
            error = np.abs(y_early)**2 - np.abs(y_late)**2
            block_error += error

        return np.array(block_symbols), block_error

    # --- METHOD 5: Gardner Loop ---
    def Gardner_recovery(self, rx_signal, start_idx, sps=8, kp=0.1, ki=0.01, max_symbols=500):
        """Timing recovery using Gardner Timing Error Detector."""
        timing_phase = 0.0
        integrator = 0.0
        recovered_symbols = []

        t = np.arange(len(rx_signal))
        if np.iscomplexobj(rx_signal):
            interp_real = interp1d(t, np.real(rx_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_imag = interp1d(t, np.imag(rx_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_func = lambda pts: interp_real(pts) + 1j * interp_imag(pts)
        else:
            interp_func = interp1d(t, rx_signal, kind='cubic', bounds_error=False, fill_value=0.0)

        for symbol_index in range(max_symbols):
            current_symbol_idx = start_idx + symbol_index * sps + timing_phase
            prev_symbol_idx = current_symbol_idx - sps
            mid_idx = current_symbol_idx - sps / 2

            if prev_symbol_idx < 0 or current_symbol_idx >= len(rx_signal) or np.isnan(current_symbol_idx):
                break

            y_prev = interp_func([prev_symbol_idx])[0]
            y_mid = interp_func([mid_idx])[0]
            y_curr = interp_func([current_symbol_idx])[0]

            # Gardner Timing Error Detector
            error = np.real(np.conj(y_mid) * (y_curr - y_prev))
            integrator += ki * error
            timing_phase -= (kp * error + integrator)
            timing_phase = np.clip(timing_phase, -sps, sps)

            recovered_symbols.append(y_curr)

        if len(recovered_symbols) == 0:
            return np.zeros(max_symbols, dtype=complex), 0.0
        return np.array(recovered_symbols), timing_phase

    # --- METHOD 6: LMS Adaptive Timing Recovery ---
    def lms_adaptive_timing_recovery(self, filt_signal, start_idx, sps, max_symbols):
        """Adaptive timing recovery using LMS filter + Gardner TED."""
        recovered_symbols = []
        
        # Create interpolator once outside loop
        t = np.arange(len(filt_signal))
        if np.iscomplexobj(filt_signal):
            interp_real = interp1d(t, np.real(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_imag = interp1d(t, np.imag(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_func = lambda pts: interp_real(pts) + 1j * interp_imag(pts)
        else:
            interp_func = interp1d(t, filt_signal, kind='cubic', bounds_error=False, fill_value=0.0)
        
        mu_phase, mu_drift = 0.02, 0.0002
        phase_offset = clock_drift = 0.0
        current_idx = float(start_idx)
        
        for _ in range(max_symbols):
            eval_idx = current_idx + phase_offset
            if eval_idx + sps >= len(filt_signal) or eval_idx - sps < 0:
                break
            
            y_curr = interp_func([eval_idx])[0]
            y_mid = interp_func([eval_idx - sps / 2.0])[0]
            y_prev = interp_func([eval_idx - sps])[0]
            recovered_symbols.append(y_curr)
            
            ted_error = np.sign(np.real((y_curr - y_prev) * np.conj(y_mid)))
            clock_drift += mu_drift * ted_error
            phase_offset -= (mu_phase * ted_error + clock_drift)
            phase_offset = np.clip(phase_offset, -sps, sps)
            current_idx += sps

        return np.array(recovered_symbols), phase_offset

    def plot_channel_estimation(self, h_time, h_block):
        """
        h_time: Your raw impulse response estimate (ifft of H_est)
        h_block: Your final MMSE-equalized channel (FFT of h_padded)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 1. Time Domain Impulse Response
        axes[0].stem(np.abs(h_time), label='Raw Estimate')
        axes[0].set_title("Time Domain Impulse Response")
        axes[0].set_xlabel("Sample Delay")
        axes[0].set_ylabel("Magnitude")
        axes[0].grid(True, linestyle='--')
        
        # 2. Frequency Domain Channel Response
        axes[1].plot(np.abs(h_block), label='Equalized Channel')
        axes[1].set_title("Frequency Domain Channel Response")
        axes[1].set_xlabel("Subcarrier Index")
        axes[1].set_ylabel("Magnitude")
        axes[1].grid(True, linestyle='--')
        
        plt.tight_layout()
        plt.show()

    def estimate_channel_and_weights(self, rx_preamble, ideal_preamble, data_block_size, current_snr, cp_length):
        """Estimate channel via preamble, denoise in time-domain, and compute MMSE weights."""

        Y_preamble = np.fft.fft(rx_preamble)
        X_preamble = np.fft.fft(ideal_preamble)
        # Raw channel estimate at preamble resolution
        H_est = Y_preamble / X_preamble

        # Transform to Time Domain (Impulse Response)
        h_time = np.fft.ifft(H_est)

        # --- FIX: Denoise by keeping only the taps within the Cyclic Prefix span ---
        # The CP length defines the maximum physical delay spread of your channel.
        h_denoised = np.zeros_like(h_time)
        h_denoised[:cp_length] = h_time[:cp_length]
        # Zero-pad the denoised impulse response to target data block size
        h_padded = np.zeros(data_block_size, dtype=complex)
        h_padded[:len(h_denoised)] = h_denoised
        # Transform back to Frequency Domain at the new resolution
        H_block = np.fft.fft(h_padded)

        #self.plot_channel_estimation(h_time, H_block)

        # Calculate actual average channel power to correctly scale the noise ratio
        channel_power = np.mean(np.abs(H_block)**2)
        snr_linear = 10 ** (current_snr / 10.0)
        
        # Scale your noise variance ratio relative to the actual estimated channel power
        noise_variance_ratio = channel_power / snr_linear

        W_mmse = np.conj(H_block) / (np.abs(H_block)**2 + noise_variance_ratio)
        return H_block, W_mmse

    def equalize_sc_fde(self, rx_signal, sps, ideal_preamble, current_snr, data_block_size, cp_length):
        """SC-FDE equalization: downsample, remove CP, equalize in frequency domain."""
        # 1. Downsample down to symbol rate
        rx_symbols = rx_signal[::sps]
        
        # ideal_preamble is the full [CP1 + ZC1 + CP2 + ZC2] block
        # Calculate the size of just a single block sequence
        single_block_len = len(ideal_preamble) // 2  
        
        # Extract only the first preamble block
        rx_preamble_block = rx_symbols[:single_block_len]
        ideal_preamble_block = ideal_preamble[:single_block_len]

        # STRIP the CP from both before running channel estimation 
        # to force them to match data_block_size (e.g., 286 - 30 = 256)
        rx_preamble_pure = rx_preamble_block[cp_length:]
        ideal_preamble_pure = ideal_preamble_block[cp_length:]

        # Now pass the clean, size-256 blocks down
        H_est, W_mmse = self.estimate_channel_and_weights(
            rx_preamble_pure, ideal_preamble_pure, data_block_size, current_snr, cp_length
        )
        
        # Jump completely over BOTH preambles to reach the data payload
        header_offset = 2 * single_block_len
        data_stream = rx_symbols[header_offset:]
        
        block_stride = data_block_size + cp_length
        equalized_symbols = []

        for i in range(0, len(data_stream), block_stride):
            block_with_cp = data_stream[i : i + block_stride]
            if len(block_with_cp) < block_stride:
                break

            block_data = block_with_cp[cp_length:]
            Y_block = np.fft.fft(block_data)
            X_hat_freq = Y_block * W_mmse
            equalized_symbols.append(np.fft.ifft(X_hat_freq))

        if len(equalized_symbols) == 0:
            return np.array([], dtype=complex)
            
        return np.concatenate(equalized_symbols), H_est

    def equalize_blocks_only(self, time_symbols, W_mmse, data_block_size, cp_length, header_offset=0):
        """
        Equalize downsampled symbols using MMSE weights.
        
        Parameters:
        -----------
        time_symbols : ndarray
            The full downsampled received signal (or just the payload).
        header_offset : int
            The number of samples to skip to get past the preambles.
        """
        # FIX: Slice the array to skip over the preambles completely
        data_stream = time_symbols[header_offset:]
        
        block_stride = data_block_size + cp_length
        equalized_list = []

        for i in range(0, len(data_stream), block_stride):
            block_with_cp = data_stream[i : i + block_stride]
            
            # If the remaining data isn't a full block, drop it
            if len(block_with_cp) < block_stride:
                break

            # Strip the CP from the payload block
            block_data = block_with_cp[cp_length:]
            
            # Transform, apply the MMSE weights, and return to time domain
            Y = np.fft.fft(block_data)
            X_hat = Y * W_mmse
            equalized_list.append(np.fft.ifft(X_hat))

        if len(equalized_list) == 0:
            return np.array([], dtype=complex)
            
        return np.concatenate(equalized_list)