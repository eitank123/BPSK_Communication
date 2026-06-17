import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


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
        preamble_upsampled = np.zeros(len(preamble) * sps, dtype=complex)
        preamble_upsampled[::sps] = preamble
        reference = np.convolve(preamble_upsampled, self.impulse_response, mode='full')
        return reference

    def cubic_interpolate(self, signal_in, sample_points):
        """Helper for standalone vector interpolations (Approaches 2 and 3)"""
        t = np.arange(len(signal_in))
        if np.iscomplexobj(signal_in):
            interp_real = interp1d(t, np.real(signal_in), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_imag = interp1d(t, np.imag(signal_in), kind='cubic', bounds_error=False, fill_value=0.0)
            return interp_real(sample_points) + 1j * interp_imag(sample_points)
        
        interp_func = interp1d(t, signal_in, kind='cubic', bounds_error=False, fill_value=0.0)
        return interp_func(sample_points)

    # --- APPROACH 1: Integer Correlation ---
    def detect_preamble(self, preamble, sps, filter_span):
        self.detected_delays_list = []
        reference = self.generate_reference_preamble(preamble, sps)
        reference_length = len(reference)
        # reference contains a group delay of (filter_span * sps) / 2 = 40 samples
        reference_group_delay = (filter_span * sps) // 2

        for filt_signal in self.filtered_signal:
            correlation = signal.correlate(filt_signal, reference, mode='full')
            peak_index = np.argmax(np.abs(correlation))
            coarse_delay = peak_index - (reference_length - 1)
            
            # Compensate only for the reference filter shift
            corrected_delay = coarse_delay + reference_group_delay
            self.detected_delays_list.append(corrected_delay)

        return self.detected_delays_list

    # --- APPROACH 2: Parabolic Interpolation ---
    def parabolic_interpolation(self, correlation):
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

    def estimate_fractional_delay(self, rx_signal, preamble, sps, filter_span):
        reference = self.generate_reference_preamble(preamble, sps)
        correlation = signal.correlate(rx_signal, reference, mode='full')
        refined_peak, frac = self.parabolic_interpolation(correlation)
        
        coarse_delay = refined_peak - (len(reference) - 1)
        reference_group_delay = (filter_span * sps) // 2
        estimated_delay = coarse_delay + reference_group_delay
        return estimated_delay, frac, correlation

    # --- APPROACH 3: Maximum Likelihood (ML) Fractional Delay Estimation ---
    def ml_fractional_delay(self, rx_signal, preamble, sps, filter_span, grid_resolution=0.01):
        reference = self.generate_reference_preamble(preamble, sps)
        correlation = signal.correlate(rx_signal, reference, mode='full')
        coarse_peak = np.argmax(np.abs(correlation))
        
        fractional_grid = np.arange(-1.0, 1.0, grid_resolution)
        search_points = coarse_peak + fractional_grid
        
        interpolated_corr = self.cubic_interpolate(correlation, search_points)
        best_idx = np.argmax(np.abs(interpolated_corr))
        
        refined_peak = coarse_peak + fractional_grid[best_idx]
        coarse_delay = refined_peak - (len(reference) - 1)
        reference_group_delay = (filter_span * sps) // 2
        estimated_delay = coarse_delay + reference_group_delay
        
        return estimated_delay, fractional_grid[best_idx], correlation

    # --- APPROACH 4: Early-Late Timing Recovery (Optimized Fast Interpolation) ---
    def early_late_recovery(self, rx_signal, start_idx, sps=8, d=0.25, kp=0.01, ki=0.001, max_symbols=500):
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
            early_idx = current_symbol_idx - d * sps
            late_idx = current_symbol_idx + d * sps

            if early_idx < 0 or late_idx >= len(rx_signal) or np.isnan(current_symbol_idx):
                break

            y_early = interp_func([early_idx])[0]
            y_curr = interp_func([current_symbol_idx])[0]
            y_late = interp_func([late_idx])[0]

            # Error calculation
            error = np.abs(y_early)**2 - np.abs(y_late)**2
            integrator += ki * error
            
            # CHANGED: Use negative feedback (-=) to drive error toward zero
            timing_phase -= (kp * error + integrator)
            timing_phase = np.clip(timing_phase, -sps, sps)

            recovered_symbols.append(y_curr)

        if len(recovered_symbols) == 0:
            return np.zeros(max_symbols, dtype=complex), 0.0
        return np.array(recovered_symbols), timing_phase  # CHANGED: Return the actual scalar phase

    # --- APPROACH 5: Gardner Timing Recovery (Optimized Fast Interpolation) ---
    def Gardner_recovery(self, rx_signal, start_idx, sps=8, kp=0.1, ki=0.01, max_symbols=500):
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

            # Gardner Timing Error Detector Formula
            error = np.real(np.conj(y_mid) * (y_curr - y_prev))
            integrator += ki * error
            
            # CHANGED: Use negative feedback (-=) to track properly
            timing_phase -= (kp * error + integrator)
            timing_phase = np.clip(timing_phase, -sps, sps)

            recovered_symbols.append(y_curr)

        if len(recovered_symbols) == 0:
            return np.zeros(max_symbols, dtype=complex), 0.0
        return np.array(recovered_symbols), timing_phase  # CHANGED: Return the actual scalar phase
    


    def lms_adaptive_timing_recovery(self, filt_signal, start_idx, sps, max_symbols):
        """
        Recovers timing synchronization using a Low-Complexity LMS Adaptive Filter
        driven by a Gardner Timing Error Detector (TED).
        """
        recovered_symbols = []
        
        # --- OPTIMIZATION: Create the interpolator ONCE outside the loop ---
        t = np.arange(len(filt_signal))
        if np.iscomplexobj(filt_signal):
            interp_real = interp1d(t, np.real(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_imag = interp1d(t, np.imag(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_func = lambda pts: interp_real(pts) + 1j * interp_imag(pts)
        else:
            interp_func = interp1d(t, filt_signal, kind='cubic', bounds_error=False, fill_value=0.0)
        
        # LMS Hyperparameters
        mu_phase = 0.01  
        mu_drift = 0.0001 
        
        # Initialize adaptive parameters (scalars)
        phase_offset = 0.0
        clock_drift = 0.0
        
        current_idx = float(start_idx)
        
        for _ in range(max_symbols):
            # Apply the current adaptive timing correction
            eval_idx = current_idx + phase_offset
            
            # Guard rails for signal boundaries
            if eval_idx + sps >= len(filt_signal) or eval_idx - sps < 0:
                break
                
            t_curr = eval_idx
            t_mid  = eval_idx - (sps / 2.0)
            t_prev = eval_idx - sps
            
            # Evaluate the pre-computed interpolator
            y_curr = interp_func([t_curr])[0]
            y_mid  = interp_func([t_mid])[0]
            y_prev = interp_func([t_prev])[0]
            
            recovered_symbols.append(y_curr)
            
            # Calculate the raw Gardner TED error
            raw_error = np.real((y_curr - y_prev) * np.conj(y_mid))
            
            # Normalize to a "Bang-Bang" error (+1, -1, or 0)
            ted_error = np.sign(raw_error)
            
            # FIX 1: Align the integrator (clock_drift) sign (+ instead of -) 
            # so it works WITH the proportional term, not against it.
            clock_drift += (mu_drift * ted_error)
            
            # FIX 2: Group the PI terms so negative feedback pulls them both correctly
            phase_offset -= (mu_phase * ted_error + clock_drift)
            
            # FIX 3: Bound the phase offset. This prevents the Bang-Bang logic 
            # from jumping entire symbols during initial transients.
            phase_offset = np.clip(phase_offset, -sps, sps)
            
            # Move the base index forward by a nominal symbol period
            current_idx += sps

        final_phase_offset = phase_offset
        return np.array(recovered_symbols), final_phase_offset
    