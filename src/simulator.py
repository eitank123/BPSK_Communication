"""
Simulation orchestrator module.

Coordinates the overall simulation flow, managing sweeps and method comparisons.
Keeps main logic clean and readable by abstracting away sweep complexity.
"""

import numpy as np
from client_Tx import Client_Tx
from client_Rx import Client_Rx
from channel import (
    generate_zadoff_chu_preamble,
    add_cyclic_prefix,
    add_rician_fading,
    create_formatted_payload,
    upsample_symbols
)
from utilities import calculate_ber, to_scalar, symbols_to_bits
from signal_processing import downsample_to_symbol_rate
import config as cfg
from signal_processing import cubic_interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class SimulationEngine:
    """
    Main simulation engine coordinating transmitter, receiver, and channel.
    """
    
    def __init__(self):
        self.sender = None
        self.preamble = None
        self.original_bits_flat = None
        self.num_data_symbols = None
        
    def initialize_transmitter(self):
        """Initialize transmitter and bit stream."""
        self.sender = Client_Tx(
            cfg.NUMBER_OF_BITS,
            cfg.BIT_MAPPING_QPSK,
            is_qpsk=True,
            farrow_degree=cfg.FARROW_INTERPOLATION_DEGREE
        )
        self.sender.generate_bit_array()
        
        # Store original bits for BER calculation
        self.original_bits_flat = np.array(
            [bit for tuple_bits in self.sender.bit_array for bit in tuple_bits]
        )
        self.num_data_symbols = len(self.sender.mapped_bits)
    
    def initialize_preamble(self):
        """Generate Zadoff-Chu preamble."""
        self.preamble = generate_zadoff_chu_preamble(
            cfg.PREAMBLE_LENGTH,
            cfg.PREAMBLE_ROOT_INDEX
        )
    
    def create_transmitted_signal(self, sender, preamble, delay, freq_offset, sps):
        """
        Create transmitted signal with preamble and payload.
        
        Applies RRC filtering, cyclic prefix insertion, and channel distortion.
        
        Parameters
        ----------
        sender : Client_Tx
            Transmitter object
        preamble : ndarray
            Preamble symbols
        delay : float
            Transmission delay in samples
        freq_offset : float
            Frequency offset in Hz
        sps : int
            Samples per symbol
        
        Returns
        -------
        tuple
            (clean_signal, rx_signals_per_k_factor)
        """
        # ====================================================================
        # STEP 1: Format payload with cyclic prefixes
        # ====================================================================
        data_symbols = sender.mapped_bits
        formatted_data = create_formatted_payload(
            data_symbols,
            cfg.DATA_BLOCK_SIZE,
            cfg.CP_LENGTH
        )
        
        # Combine preamble with payload
        full_symbols = np.concatenate([preamble, formatted_data])
        
        # ====================================================================
        # STEP 2: Upsample and apply RRC filtering
        # ====================================================================
        upsampled_full = upsample_symbols(full_symbols, sps, is_complex=True)
        x_t = sender.prepare_x_t(upsampled_full, delay, freq_offset, sample_rate=sps)
        
        # ====================================================================
        # STEP 3: Apply channel (Rician fading + AWGN for each K-factor)
        # ====================================================================
        rx_signals = []
        for k_db in cfg.RICIAN_K_FACTORS:
            rx_signal = add_rician_fading(
                np.array(x_t),
                k_db,
                cfg.TARGET_SNR,
                sps
            )
            rx_signals.append(rx_signal)
        
        return x_t, rx_signals
    
    def process_received_signal(self, rx_signal, method_id, receiver,
                                preamble, sps, coarse_delay=None):
        """
        Process received signal through selected recovery method.
        
        Parameters
        ----------
        rx_signal : ndarray
            Received signal
        method_id : int
            Timing recovery method ID (1-6)
        receiver : Client_Rx
            Receiver object
        preamble : ndarray
            Reference preamble
        sps : int
            Samples per symbol
        coarse_delay : float, optional
            Coarse delay estimate for methods 4-6
        
        Returns
        -------
        tuple
            (equalized_symbols, estimated_delay)
        """
        filt_signal = receiver.filtered_signal[0]
        
        if method_id == 1:
            return self._process_method1(
                filt_signal, receiver, preamble, sps
            )
        elif method_id == 2:
            return self._process_method2(
                filt_signal, receiver, preamble, sps
            )
        elif method_id == 3:
            return self._process_method3(
                filt_signal, receiver, preamble, sps
            )
        elif method_id == 4:
            return self._process_method4(
                filt_signal, receiver, preamble, sps, coarse_delay
            )
        elif method_id == 5:
            return self._process_method5(
                filt_signal, receiver, preamble, sps, coarse_delay
            )
        elif method_id == 6:
            return self._process_method6(
                filt_signal, receiver, preamble, sps, coarse_delay
            )
        else:
            raise ValueError(f"Unknown method_id: {method_id}")
    
    def plot_constellation(self, equalized_symbols, title="Equalized Constellation"):
        """
        Plots the I/Q constellation diagram for equalized symbols.
        
        Parameters:
        - equalized_symbols: 1D array of complex numbers representing the symbols.
        - title: String title for the plot.
        """
        # Ensure input is a flattened numpy array
        symbols = np.asarray(equalized_symbols).flatten()
        
        # Extract Real (In-phase) and Imaginary (Quadrature) components
        i_component = np.real(symbols)
        q_component = np.imag(symbols)
        
        plt.figure(figsize=(6, 6))
        
        # Plot symbols as points
        plt.scatter(i_component, q_component, color='blue', marker='.', alpha=0.6, edgecolors='none')
        
        # Add axes lines through the origin
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        
        # Formatting
        plt.title(title)
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Equalize axis scaling so circles don't look like ellipses
        plt.axis('equal')
        
        # Add a slight margin around the furthest points
        max_val = np.max(np.abs(symbols)) * 1.2 if len(symbols) > 0 else 1.5
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        
        plt.show()

    def _process_method1(self, filt_signal, receiver, preamble, sps):
        """Method 1: Integer Correlation."""
        detected_delays = receiver.detect_preamble(preamble, sps, cfg.FILTER_SPAN)
        coarse_delay = int(round(detected_delays[0]))
        receiver.coarse_delay = max(coarse_delay - 1, 0)
        print(receiver.coarse_delay)
        total_symbols = cfg.PREAMBLE_LENGTH + self.num_data_symbols
        frame_samples = filt_signal[coarse_delay : coarse_delay + total_symbols * sps]
        
        equalized = receiver.equalize_sc_fde(
            frame_samples, sps, preamble, cfg.TARGET_SNR,
            cfg.DATA_BLOCK_SIZE, cfg.CP_LENGTH
        )
        #self.plot_constellation(equalized, title="Method 1: Integer Correlation Constellation")
        return equalized, to_scalar(coarse_delay)
    
    def _process_method2(self, filt_signal, receiver, preamble, sps):
        """Method 2: Parabolic Fractional Interpolation."""
        est_delay, _, _ = receiver.estimate_fractional_delay(
            filt_signal, preamble, sps, cfg.FILTER_SPAN
        )
        
        total_symbols = cfg.PREAMBLE_LENGTH + self.num_data_symbols
        t_samples = est_delay + np.arange(total_symbols * sps)
        frame_samples = cubic_interpolate(filt_signal, t_samples)
        
        equalized = receiver.equalize_sc_fde(
            frame_samples, sps, preamble, cfg.TARGET_SNR,
            cfg.DATA_BLOCK_SIZE, cfg.CP_LENGTH
        )
        #self.plot_constellation(equalized, title="Method 2: Parabolic Interpolation Constellation")
        
        return equalized, to_scalar(est_delay)
    
    def _process_method3(self, filt_signal, receiver, preamble, sps):
        """Method 3: Maximum Likelihood Grid Search."""
        est_delay, _, _ = receiver.ml_fractional_delay(
            filt_signal, preamble, sps, cfg.FILTER_SPAN,
            grid_resolution=cfg.ML_GRID_RESOLUTION
        )
        
        total_symbols = cfg.PREAMBLE_LENGTH + self.num_data_symbols
        t_samples = est_delay + np.arange(total_symbols * sps)
        frame_samples = cubic_interpolate(filt_signal, t_samples)
        
        equalized = receiver.equalize_sc_fde(
            frame_samples, sps, preamble, cfg.TARGET_SNR,
            cfg.DATA_BLOCK_SIZE, cfg.CP_LENGTH
        )
        
        return equalized, to_scalar(est_delay)
    
    def _process_method4(self, filt_signal, receiver, preamble, sps, coarse_delay):
        """Method 4: Block-by-Block Early-Late Loop Tracking with FDE Phase Correction."""
        coarse_delay = receiver.coarse_delay
        payload_start = coarse_delay + (cfg.PREAMBLE_LENGTH * sps)
        
        num_blocks = int(np.ceil(self.num_data_symbols / cfg.DATA_BLOCK_SIZE))
        block_size = cfg.DATA_BLOCK_SIZE + cfg.CP_LENGTH  # Total symbols per FDE block
        
        # 1. Estimate base static MMSE weights from the unshifted preamble
        rx_preamble = filt_signal[coarse_delay : coarse_delay + cfg.PREAMBLE_LENGTH * sps : sps]
        _, W_mmse = receiver.estimate_channel_and_weights(
            rx_preamble, preamble, cfg.DATA_BLOCK_SIZE, cfg.TARGET_SNR, cfg.CP_LENGTH
        )
        
        # 2. Setup the cubic interpolator for the entire received frame
        t = np.arange(len(filt_signal))
        interp_real = interp1d(t, np.real(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
        interp_imag = interp1d(t, np.imag(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
        interp_func = lambda pts: interp_real(pts) + 1j * interp_imag(pts)
        
        # Initialize Loop Filter variables
        timing_phase = 0.0
        integrator = 0.0
        kp, ki = 0.01, 0.001  # PI gains
        
        equalized_payload = []

        # 3. Process the frame block-by-block
        for b in range(num_blocks):
            block_symbol_offset = b * block_size
            
            # Save the phase value applied to extract this specific block
            applied_phase = timing_phase
            
            # Extract the uniform block symbols and calculate the collective block error
            block_symbols, block_error = receiver.early_late_block_recovery(
                interp_func, payload_start, block_symbol_offset, block_size, sps, applied_phase
            )
            
            # Update the PI loop filter for the NEXT block boundary
            avg_error = block_error / block_size
            integrator += ki * avg_error
            timing_phase -= (kp * avg_error + integrator)
            timing_phase = np.clip(timing_phase, -sps, sps)
            
            # 4. Single-Carrier Frequency Domain Equalization (SC-FDE) Engine
            # Strip the Cyclic Prefix (assuming CP is located at the front of the block)
            data_symbols = block_symbols[cfg.CP_LENGTH:]
            
            # Transform block to the frequency domain
            R_block = np.fft.fft(data_symbols)
            
            # Phase Correction: A time shift of 'applied_phase' samples creates a linear phase slope
            # across the FFT bins. We counter-rotate the weights to maintain perfect alignment.
            k_bins = np.arange(cfg.DATA_BLOCK_SIZE)
            phase_correction = np.exp(-1j * 2 * np.pi * k_bins * applied_phase / (cfg.DATA_BLOCK_SIZE * sps))
            W_adapted = W_mmse * phase_correction
            
            # Apply the adapted equalizer weights
            X_hat = R_block * W_adapted
            
            # Convert back to the time domain
            equalized_block = np.fft.ifft(X_hat)

            # --- ADD THIS: Decision-Directed Carrier Phase Correction ---
            # 1. Map each symbol to its closest ideal QPSK constellation point
            ideal_qpsk = (np.sign(np.real(equalized_block)) + 1j * np.sign(np.imag(equalized_block))) / np.sqrt(2)
            
            # 2. Find the average bulk phase rotation vector for this block
            # Multiplying by the conjugate calculates the phase difference
            avg_rotation_vector = np.mean(equalized_block * np.conj(ideal_qpsk))
            block_phase_tilt = np.angle(avg_rotation_vector)
            
            # 3. Counter-rotate the entire block to snap it back to 0 degrees
            equalized_block = equalized_block * np.exp(-1j * block_phase_tilt)
            # ------------------------------------------------------------
            equalized_payload.extend(equalized_block)
            
        equalized_payload = np.array(equalized_payload)
        #self.plot_constellation(equalized_payload, title="Method 4: Block-by-Block Tracking Constellation")
        
        return equalized_payload, to_scalar(coarse_delay + timing_phase)
    
    def _process_method5(self, filt_signal, receiver, preamble, sps, coarse_delay):
        """Method 5: Gardner Loop Tracking (Block-by-Block FDE)."""
        from scipy.interpolate import interp1d
        
        coarse_delay = receiver.coarse_delay
        payload_start = coarse_delay + (cfg.PREAMBLE_LENGTH * sps)
        
        num_blocks = int(np.ceil(self.num_data_symbols / cfg.DATA_BLOCK_SIZE))
        
        # 1. Estimate base channel weights from the static preamble
        rx_preamble = filt_signal[coarse_delay : coarse_delay + cfg.PREAMBLE_LENGTH * sps : sps]
        _, W_mmse = receiver.estimate_channel_and_weights(
            rx_preamble, preamble, cfg.DATA_BLOCK_SIZE, cfg.TARGET_SNR, cfg.CP_LENGTH
        )
        
        # 2. Setup cubic interpolator strictly for the Timing Error Detector (TED)
        t = np.arange(len(filt_signal))
        if np.iscomplexobj(filt_signal):
            interp_real = interp1d(t, np.real(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_imag = interp1d(t, np.imag(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_func = lambda pts: interp_real(pts) + 1j * interp_imag(pts)
        else:
            interp_func = interp1d(t, filt_signal, kind='cubic', bounds_error=False, fill_value=0.0)

        # Loop Parameters
        timing_phase = 0.0
        integrator = 0.0
        kp, ki = 0.02, 0.001  # Damped for stability
        
        k_bins = np.fft.fftfreq(cfg.DATA_BLOCK_SIZE) * cfg.DATA_BLOCK_SIZE
        equalized_payload = []
        
        for b in range(num_blocks):
            block_start_samples = payload_start + b * (cfg.DATA_BLOCK_SIZE + cfg.CP_LENGTH) * sps
            data_start_samples = block_start_samples + cfg.CP_LENGTH * sps
            
            # --- PHASE 1: Gardner Timing Error Detection ---
            block_error = 0.0
            for i in range(cfg.DATA_BLOCK_SIZE):
                curr_idx = data_start_samples + i * sps + timing_phase
                prev_idx = curr_idx - sps
                mid_idx = curr_idx - sps / 2.0
                
                y_curr = interp_func([curr_idx])[0]
                y_prev = interp_func([prev_idx])[0]
                y_mid = interp_func([mid_idx])[0]
                
                block_error += np.real(np.conj(y_mid) * (y_curr - y_prev))
            
            # Average error over the block
            avg_error = block_error / cfg.DATA_BLOCK_SIZE
            
            # Update PI Filter
            integrator += ki * avg_error
            timing_phase -= (kp * avg_error + integrator)
            timing_phase = np.clip(timing_phase, -sps, sps)
            
            # --- PHASE 2: Frequency Domain Correction & Equalization ---
            # Extract raw block (downsampled integer indices)
            raw_block = filt_signal[int(data_start_samples) : int(data_start_samples) + cfg.DATA_BLOCK_SIZE * sps : sps]
            if len(raw_block) < cfg.DATA_BLOCK_SIZE:
                raw_block = np.pad(raw_block, (0, cfg.DATA_BLOCK_SIZE - len(raw_block)))
            
            R_block = np.fft.fft(raw_block)
            
            # Apply fractional sample shift in frequency domain (Negative sign fixed)
            phase_correction = np.exp(-1j * 2 * np.pi * k_bins * timing_phase / (cfg.DATA_BLOCK_SIZE * sps))
            W_adapted = W_mmse * phase_correction
            X_hat = R_block * W_adapted
            
            equalized_block = np.fft.ifft(X_hat)
            
            # --- PHASE 3: Bulk Carrier Phase De-rotation ---
            ideal_qpsk = (np.sign(np.real(equalized_block)) + 1j * np.sign(np.imag(equalized_block))) / np.sqrt(2)
            avg_rotation_vector = np.mean(equalized_block * np.conj(ideal_qpsk))
            block_phase_tilt = np.angle(avg_rotation_vector)
            
            equalized_block = equalized_block * np.exp(-1j * block_phase_tilt)
            
            equalized_payload.extend(equalized_block)
            
        return np.array(equalized_payload), float(coarse_delay + timing_phase)
    
    def _process_method6(self, filt_signal, receiver, preamble, sps, coarse_delay):
        """Method 6: LMS Adaptive Timing Recovery (Block-by-Block FDE)."""
        from scipy.interpolate import interp1d
        
        coarse_delay = receiver.coarse_delay
        payload_start = coarse_delay + (cfg.PREAMBLE_LENGTH * sps)
        
        num_blocks = int(np.ceil(self.num_data_symbols / cfg.DATA_BLOCK_SIZE))
        
        # 1. Estimate base channel weights
        rx_preamble = filt_signal[coarse_delay : coarse_delay + cfg.PREAMBLE_LENGTH * sps : sps]
        _, W_mmse = receiver.estimate_channel_and_weights(
            rx_preamble, preamble, cfg.DATA_BLOCK_SIZE, cfg.TARGET_SNR, cfg.CP_LENGTH
        )
        
        # 2. Setup cubic interpolator strictly for the LMS TED
        t = np.arange(len(filt_signal))
        if np.iscomplexobj(filt_signal):
            interp_real = interp1d(t, np.real(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_imag = interp1d(t, np.imag(filt_signal), kind='cubic', bounds_error=False, fill_value=0.0)
            interp_func = lambda pts: interp_real(pts) + 1j * interp_imag(pts)
        else:
            interp_func = interp1d(t, filt_signal, kind='cubic', bounds_error=False, fill_value=0.0)

        # Loop Parameters
        phase_offset = 0.0
        clock_drift = 0.0
        mu_phase, mu_drift = 0.01, 0.0001
        
        k_bins = np.fft.fftfreq(cfg.DATA_BLOCK_SIZE) * cfg.DATA_BLOCK_SIZE
        equalized_payload = []
        
        for b in range(num_blocks):
            block_start_samples = payload_start + b * (cfg.DATA_BLOCK_SIZE + cfg.CP_LENGTH) * sps
            data_start_samples = block_start_samples + cfg.CP_LENGTH * sps
            
            # --- PHASE 1: LMS Error Detection ---
            block_error = 0.0
            for i in range(cfg.DATA_BLOCK_SIZE):
                curr_idx = data_start_samples + i * sps + phase_offset
                prev_idx = curr_idx - sps
                mid_idx = curr_idx - sps / 2.0
                
                y_curr = interp_func([curr_idx])[0]
                y_prev = interp_func([prev_idx])[0]
                y_mid = interp_func([mid_idx])[0]
                
                block_error += np.sign(np.real((y_curr - y_prev) * np.conj(y_mid)))
            
            # Average LMS gradient over the block
            avg_error = block_error / cfg.DATA_BLOCK_SIZE
            
            # Update LMS Filter
            clock_drift += mu_drift * avg_error
            phase_offset -= (mu_phase * avg_error + clock_drift)
            phase_offset = np.clip(phase_offset, -sps, sps)
            
            # --- PHASE 2: Frequency Domain Correction & Equalization ---
            # Extract raw block
            raw_block = filt_signal[int(data_start_samples) : int(data_start_samples) + cfg.DATA_BLOCK_SIZE * sps : sps]
            if len(raw_block) < cfg.DATA_BLOCK_SIZE:
                raw_block = np.pad(raw_block, (0, cfg.DATA_BLOCK_SIZE - len(raw_block)))
                
            R_block = np.fft.fft(raw_block)
            
            # Apply fractional sample shift in frequency domain (Negative sign fixed)
            phase_correction = np.exp(-1j * 2 * np.pi * k_bins * phase_offset / (cfg.DATA_BLOCK_SIZE * sps))
            W_adapted = W_mmse * phase_correction
            X_hat = R_block * W_adapted
            
            equalized_block = np.fft.ifft(X_hat)
            
            # --- PHASE 3: Bulk Carrier Phase De-rotation ---
            ideal_qpsk = (np.sign(np.real(equalized_block)) + 1j * np.sign(np.imag(equalized_block))) / np.sqrt(2)
            avg_rotation_vector = np.mean(equalized_block * np.conj(ideal_qpsk))
            block_phase_tilt = np.angle(avg_rotation_vector)
            
            equalized_block = equalized_block * np.exp(-1j * block_phase_tilt)
            
            equalized_payload.extend(equalized_block)
            
        return np.array(equalized_payload), float(coarse_delay + phase_offset)
    
    def run_snr_sweep(self, beta, delay, sps):
        """
        Run SNR sweep test (varying Rician K-factor).
        
        Returns
        -------
        dict
            Results dictionary with BER and delay for each method
        """
        print(f"\n{'='*60}")
        print(f"SNR Sweep: Beta={beta}, Delay={delay}, SPS={sps}")
        print(f"{'='*60}")
        
        # Prepare transmitter
        self.sender.set_responses(beta, sps, cfg.FILTER_SPAN)
        
        # Generate signal
        clean_signal, rx_signals = self.create_transmitted_signal(
            self.sender, self.preamble, delay, cfg.FREQ_OFFSET, sps
        )
        
        results = {
            'ber': {i: [] for i in range(1, 7)},
            'delay': {i: [] for i in range(1, 7)}
        }
        
        # Process each received signal
        for rx_sig in rx_signals:
            receiver = Client_Rx([rx_sig], is_qpsk=True)
            receiver.set_responses(beta, sps, cfg.FILTER_SPAN)
            receiver.filter_signal(sps, cfg.FILTER_SPAN)
            
            # Process through all 6 methods
            for method_id in range(1, 7):
                try:
                    equalized, est_delay = self.process_received_signal(
                        rx_sig, method_id, receiver,
                        self.preamble, sps
                    )
                    
                    ber = calculate_ber(
                        self.original_bits_flat,
                        symbols_to_bits(equalized)
                    )
                    
                    results['ber'][method_id].append(ber)
                    results['delay'][method_id].append(est_delay)
                    
                except Exception as e:
                    print(f"Error in method {method_id}: {e}")
                    results['ber'][method_id].append(1.0)
                    results['delay'][method_id].append(0.0)
        
        return results
    
    def run_sps_sweep(self, beta, delay, target_k_factor):
        """
        Run SPS sweep test (varying samples per symbol).
        
        Returns
        -------
        dict
            Results dictionary with BER and delay for each SPS value
        """
        print(f"\n{'='*60}")
        print(f"SPS Sweep: Beta={beta}, Delay={delay}, K={target_k_factor} dB")
        print(f"{'='*60}")
        
        results = {
            'ber': {i: [] for i in range(1, 7)},
            'delay': {i: [] for i in range(1, 7)},
            'sps_values': cfg.SPS_SWEEP_VALUES
        }
        
        for test_sps in cfg.SPS_SWEEP_VALUES:
            print(f"Testing SPS = {test_sps}...")
            
            # Update transmitter for this SPS
            self.sender.set_responses(beta, test_sps, cfg.FILTER_SPAN)
            
            # Create signal
            _, rx_signals = self.create_transmitted_signal(
                self.sender, self.preamble, delay, cfg.FREQ_OFFSET, test_sps
            )
            
            # Get signal for target K-factor
            k_idx = cfg.RICIAN_K_FACTORS.index(target_k_factor)
            rx_sig = rx_signals[k_idx]
            
            # Process through all methods
            receiver = Client_Rx([rx_sig], is_qpsk=True)
            receiver.set_responses(beta, test_sps, cfg.FILTER_SPAN)
            receiver.filter_signal(test_sps, cfg.FILTER_SPAN)
            
            for method_id in range(1, 7):
                try:
                    equalized, est_delay = self.process_received_signal(
                        rx_sig, method_id, receiver,
                        self.preamble, test_sps
                    )
                    
                    ber = calculate_ber(
                        self.original_bits_flat,
                        symbols_to_bits(equalized)
                    )
                    
                    results['ber'][method_id].append(ber)
                    results['delay'][method_id].append(est_delay)
                    
                except Exception as e:
                    print(f"Error in method {method_id}: {e}")
                    results['ber'][method_id].append(1.0)
                    results['delay'][method_id].append(0.0)
        
        return results
