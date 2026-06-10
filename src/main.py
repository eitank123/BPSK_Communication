import numpy as np
import matplotlib.pyplot as plt
from client_Tx import Client_Tx
from client_Rx import Client_Rx
from RRC_Implementation import *
import plots  # Imports your updated plotting code package

np.random.seed(42)
number_of_bits = 10000
sps = 8
betas = [0.5]
delays = [200.4]  # Testing fractional delay positioning
filter_span = 10

SNR = [0, 2, 4, 6, 8, 10]
Rician_K_Factor = [0, 2, 4, 6, 8, 10]

target_snr = 8

preamble_length = 127
preamble_root_index = 1 
farrow_degree = 3  

CP_length = 16
data_block_size = 512

Rs = 4e6
fs = Rs * sps  # 4 MHz sampling rate
c = 3e8   # Speed of light


qpsk_normalized = 1 / np.sqrt(2)
bit_mapping_QPSK = {
    (0, 0): qpsk_normalized * (1 + 1j),
    (0, 1): qpsk_normalized * (1 - 1j),
    (1, 0): qpsk_normalized * (-1 + 1j),
    (1, 1): qpsk_normalized * (-1 - 1j)
}

FREQ_OFFSET = 0

def add_cyclic_prefix(data_block, G):
    """
    Prepends the last G samples of a data block to the front.
    
    Parameters:
    -----------
    data_block : ndarray
        The time-domain block of symbols (length N)
    G : int
        The number of samples to use for the Guard Interval (Cyclic Prefix)
        
    Returns:
    --------
    ndarray
        The formatted block with the CP included (length N + G)
    """
    if G <= 0:
        return data_block
        
    # Extract the last G samples
    cp = data_block[-G:]
    
    # Concatenate the CP to the front of the original block
    formatted_block = np.concatenate((cp, data_block))
    
    return formatted_block

def generate_zadoff_chu_preamble(length, root_index=1):
    n = np.arange(length)
    zadoff_chu = np.exp(-1j * np.pi * root_index * n * (n + 1) / length)
    return zadoff_chu

def add_rician_fading(signal, k_db, ebno_db, sps):
    """
    Mathematically precise Rician fading channel.
    Dynamically measures signal power to guarantee accurate Eb/N0 
    regardless of pulse-shaping filter gains.
    """
    N = len(signal)
    
    # 1. Channel Power Normalization & Fading
    K_linear = 10**(k_db / 10)
    los_amp = np.sqrt(K_linear / (K_linear + 1))
    nlos_sigma = np.sqrt(1 / (2 * (K_linear + 1)))
    
    # Calculate number of actual symbols
    N = len(signal)
    
    # 1. Channel Power Normalization
    K_linear = 10**(k_db / 10)
    los_amp = np.sqrt(K_linear / (K_linear + 1))
    nlos_sigma = np.sqrt(1 / (2 * (K_linear + 1)))
    
    # Calculate required symbols, rounding UP to cover filter tails
    num_symbols = int(np.ceil(N / sps))
    
    # 2. Generate fading coefficients at the SYMBOL rate
    h_los = los_amp * np.ones(num_symbols, dtype=complex)
    h_nlos = nlos_sigma * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    h_symbols = h_los + h_nlos
    
    # 3. Repeat across sps, then TRUNCATE exactly to length N
    h = np.repeat(h_symbols, sps)[:N]
    
    # Apply flat fading to the signal safely
    faded_signal = h * signal
    
    # 2. Universal AWGN Calculation
    # Measure the TRUE average power of the transmitted pulse-shaped signal
    clean_sig_power = np.mean(np.abs(signal)**2)
    
    # Energy per symbol (Es) = Average Power * samples_per_symbol
    es = clean_sig_power * sps
    
    # Convert target Eb/N0 to linear scale
    ebno_linear = 10**(ebno_db / 10)
    esno_linear = 2 * ebno_linear  # 2 bits per symbol for QPSK
    
    # Noise Power Spectral Density (N0)
    # This equals the variance of the complex noise samples
    N0 = es / esno_linear
    
    # 3. Add Noise
    # Split total variance (N0) equally across Real and Imaginary components (N0/2 each)
    noise = np.sqrt(N0 / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    
    return faded_signal + noise

def create_transmitted_signal(sender: Client_Tx, preamble, delay, freq_offset, sps):
    # --- 1. Block Segmentation and CP Insertion ---
    data_symbols = sender.mapped_bits
    formatted_data = []
    
    # Slice the data into chunks of size N and add the CP (length G) to each
    for i in range(0, len(data_symbols), data_block_size):
        block = data_symbols[i:i+data_block_size]
        
        # If the last block is smaller than N, zero-pad it to maintain fixed block size
        if len(block) < data_block_size:
            block = np.pad(block, (0, data_block_size - len(block)), mode='constant')
            
        # Add the cyclic prefix (using the global variable G)
        block_with_cp = add_cyclic_prefix(block, CP_length)
        formatted_data.append(block_with_cp)
        
    # Concatenate all CP-padded data blocks together
    all_data_blocks = np.concatenate(formatted_data)
    
    # Combine the initial frame preamble with the formatted SC-FDE data block stream
    full_symbols = np.concatenate([preamble, all_data_blocks])
    
    # --- 2. Upsampling and Signal Preparation ---
    dtype = complex if sender.is_qpsk else float
    upsampled_full = np.zeros(len(full_symbols) * sps, dtype=dtype)
    upsampled_full[::sps] = full_symbols
    
    x_t = sender.prepare_x_t(upsampled_full, delay, freq_offset, sample_rate=sps)
    
    # --- 3. Channel Simulation ---
    r_t = []
    ebno = target_snr  # Start with the first SNR value for the initial AWGN addition
    for K_value in Rician_K_Factor:
        r_t.append(add_rician_fading(np.array(x_t), K_value, ebno, sps))

    return x_t, r_t

def symbols_to_bits(symbol_samples, is_qpsk=True):
    if is_qpsk:
        bit_i = (np.real(symbol_samples) < 0).astype(int)
        bit_q = (np.imag(symbol_samples) < 0).astype(int)
        bits = np.zeros(len(bit_i) * 2, dtype=int)
        bits[0::2] = bit_i
        bits[1::2] = bit_q
    else:
        bits = (np.real(symbol_samples) < 0).astype(int)
    return bits

def get_BER(original_bits, recovered_bits):
    min_len = min(len(original_bits), len(recovered_bits))
    num_correct = np.sum(original_bits[:min_len] == recovered_bits[:min_len])
    ber = float((min_len - num_correct) / min_len)
    return ber

def to_scalar(val):
    """
    Safely extracts a single scalar float. If 'val' is an array/list 
    (like a tracking history vector), it grabs the last converged value.
    """
    if isinstance(val, (list, np.ndarray)):
        flat = np.asarray(val).ravel()
        return float(flat[-1]) if flat.size > 0 else 0.0
    return float(val)

        

if __name__ == "__main__":

    preamble = generate_zadoff_chu_preamble(preamble_length, preamble_root_index)
    
    sender = Client_Tx(number_of_bits, bit_mapping_QPSK, is_qpsk=True, farrow_degree=farrow_degree)
    sender.generate_bit_array()
    original_bits_flat = np.array([bit for tuple_bits in sender.bit_array for bit in tuple_bits])
    num_data_symbols = len(sender.mapped_bits)

    # Dictionaries to store the final sweep data for the new assignment plots
    range_error_vs_snr_final = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    # Define the SPS values to sweep for Graphs 5 and 7
    sps_sweep_values = [1, 2, 4, 8, 16]
    range_error_vs_sps_final = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    ber_vs_sps_final = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    # =========================================================================
    # SWEEP 1: Varying SNR (Fixed SPS) - Graphs 1, 2, 3, 4, 6
    # =========================================================================
    for beta in betas:
        sender.set_responses(beta, sps, filter_span)
        for delay in delays:
            print(f"\n=======================================================")
            print(f"Phase 1: Simulating SNR Sweep (Beta={beta}, Delay={delay} samples, SPS={sps})")
            print(f"=======================================================")

            clean_signal, Rx_signals = create_transmitted_signal(sender, preamble, delay, FREQ_OFFSET, sps)
            
            ber_results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
            delay_results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
            
            for snr_idx, rx_sig in enumerate(Rx_signals):
                # Note: Rx_signals are indexed by Rician K-factor, not SNR
                # All signals generated with target_snr = 8 dB
                current_snr = target_snr
                
                receiver = Client_Rx([rx_sig], is_qpsk=True)
                receiver.set_responses(beta, sps, filter_span)
                receiver.filter_signal(sps, filter_span)
                filt_signal = receiver.filtered_signal[0]
                
                # We still need a rough baseline to tell the feedback loops where to start tracking
                detected_delays = receiver.detect_preamble(preamble, sps, filter_span)
                coarse_delay = int(round(detected_delays[0]))
                payload_start_idx = coarse_delay + (preamble_length * sps)
                total_sc_fde_symbols = preamble_length + num_data_symbols 

                # =========================================================================
                # --- APPROACH 1: Integer Correlation ---
                # =========================================================================
                # 1. Timing Method complete (coarse_delay found)
                frame_samples_m1 = filt_signal[coarse_delay : coarse_delay + total_sc_fde_symbols * sps]
                
                # 2. Estimate Channel & Equalize (Strictly after timing)
                equalized_symbols_m1 = receiver.equalize_sc_fde(
                    frame_samples_m1, sps, preamble, current_snr, data_block_size, CP_length
                )
                ber_results[1].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m1)))
                delay_results[1].append(to_scalar(coarse_delay))

                # =========================================================================
                # --- APPROACH 2: Parabolic Fractional Interpolation ---
                # =========================================================================
                # 1. Timing Method complete (fractional delay found)
                est_delay_m2, _, _ = receiver.estimate_fractional_delay(filt_signal, preamble, sps, filter_span)
                t_samples_m2 = est_delay_m2 + np.arange(total_sc_fde_symbols * sps)
                frame_samples_m2 = receiver.cubic_interpolate(filt_signal, t_samples_m2)
                
                # 2. Estimate Channel & Equalize (Strictly after timing)
                equalized_symbols_m2 = receiver.equalize_sc_fde(
                    frame_samples_m2, sps, preamble, current_snr, data_block_size, CP_length
                )
                ber_results[2].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m2)))
                delay_results[2].append(to_scalar(est_delay_m2))
                
                # =========================================================================
                # --- APPROACH 3: Maximum Likelihood Fractional Estimation ---
                # =========================================================================
                # 1. Timing Method complete (ML delay found)
                est_delay_m3, _, _ = receiver.ml_fractional_delay(filt_signal, preamble, sps, filter_span, grid_resolution=0.01)
                t_samples_m3 = est_delay_m3 + np.arange(total_sc_fde_symbols * sps)
                frame_samples_m3 = receiver.cubic_interpolate(filt_signal, t_samples_m3)
                
                # 2. Estimate Channel & Equalize (Strictly after timing)
                equalized_symbols_m3 = receiver.equalize_sc_fde(
                    frame_samples_m3, sps, preamble, current_snr, data_block_size, CP_length
                )
                ber_results[3].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m3)))
                delay_results[3].append(to_scalar(est_delay_m3))
                
                # =========================================================================
                # --- APPROACH 4, 5, & 6: Tracking Loops & Equalization ---
                # =========================================================================
                # Calculate total payload symbols including all Cyclic Prefixes
                num_blocks = int(np.ceil(num_data_symbols / data_block_size))
                total_payload_symbols = num_blocks * (data_block_size + CP_length)

                # --- APPROACH 4: Early-Late Loop Tracking ---
                symbols_m4, final_phase_m4 = receiver.early_late_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=total_payload_symbols
                )
                rx_preamble_m4 = filt_signal[coarse_delay : coarse_delay + preamble_length * sps : sps]
                _, W_mmse_m4 = receiver.estimate_channel_and_weights(rx_preamble_m4, preamble, data_block_size, current_snr)
                equalized_symbols_m4 = receiver.equalize_blocks_only(symbols_m4, W_mmse_m4, data_block_size, CP_length)
                
                ber_results[4].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m4)))
                delay_results[4].append(to_scalar(coarse_delay + final_phase_m4))

                # --- APPROACH 5: Gardner Loop Tracking ---
                symbols_m5, final_phase_m5 = receiver.Dynamic_Gardner_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=total_payload_symbols
                ) if hasattr(receiver, 'Dynamic_Gardner_recovery') else receiver.Gardner_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=total_payload_symbols
                )
                rx_preamble_m5 = filt_signal[coarse_delay : coarse_delay + preamble_length * sps : sps]
                _, W_mmse_m5 = receiver.estimate_channel_and_weights(rx_preamble_m5, preamble, data_block_size, current_snr)
                equalized_symbols_m5 = receiver.equalize_blocks_only(symbols_m5, W_mmse_m5, data_block_size, CP_length)
                
                ber_results[5].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m5)))
                delay_results[5].append(to_scalar(coarse_delay + final_phase_m5))

                # --- APPROACH 6: LMS Adaptive Timing Recovery ---
                symbols_m6, final_phase_m6 = receiver.lms_adaptive_timing_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=total_payload_symbols
                )
                rx_preamble_m6 = filt_signal[coarse_delay : coarse_delay + preamble_length * sps : sps]
                _, W_mmse_m6 = receiver.estimate_channel_and_weights(rx_preamble_m6, preamble, data_block_size, current_snr)
                equalized_symbols_m6 = receiver.equalize_blocks_only(symbols_m6, W_mmse_m6, data_block_size, CP_length)
                
                ber_results[6].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m6)))
                delay_results[6].append(to_scalar(coarse_delay + final_phase_m6))

                if current_snr == 10:
                    plots.plot_eye_diagram(rx_sig, filt_signal, sps, current_snr, coarse_delay, preamble_length)

            # Print console layout summary comparison
            print(f"\nSummary of BER results (%) for Corrected Delay = {delay}:")
            print(f"{'SNR (dB)':<10}{'Method 1':<12}{'Method 2':<12}{'Method 3':<12}{'Method 4':<12}{'Method 5':<12}{'Method 6':<12}")
            print("-" * 82)
            for i, snr_val in enumerate(SNR):
                print(f"{snr_val:<10}"
                      f"{ber_results[1][i]*100:<12.2f}"
                      f"{ber_results[2][i]*100:<12.2f}"
                      f"{ber_results[3][i]*100:<12.2f}"
                      f"{ber_results[4][i]*100:<12.2f}"
                      f"{ber_results[5][i]*100:<12.2f}"
                      f"{ber_results[6][i]*100:<12.2f}")
            
            # Map tracking datasets over to plotting modules (Legacy Plots)
            matrix_ber = np.array([ber_results[1], ber_results[2], ber_results[3], ber_results[4], ber_results[5], ber_results[6]])
            matrix_delay = np.array([delay_results[1], delay_results[2], delay_results[3], delay_results[4], delay_results[5], delay_results[6]])
            labels = ["Integer Correlation (M1)", "Parabolic Interp (M2)", "ML Grid Search (M3)", "Early-Late Loop (M4)", "Gardner Loop (M5)", "LMS Adaptive (M6)"]
            
            # Fire legacy performance analytical plot interfaces
            plots.plot_ber_vs_k(Rician_K_Factor, matrix_ber, target_snr, series_labels=labels)           
            plots.plot_delay_tracking(SNR, matrix_delay, labels, true_delay=delay)
            
            # Populate Range Error vs SNR dictionary (Graph 4)
            for m_id in delay_results:
                # Range Error = (Estimated Delay - True Delay) * c / fs
                errors = [(est - delay) * (c / fs) for est in delay_results[m_id]]
                range_error_vs_snr_final[m_id] = errors

    # =========================================================================
    # SWEEP 2: Varying SPS (Fixed SNR = 8 dB) - Graphs 5 and 7
    # =========================================================================
    print(f"\n=======================================================")
    print(f"Phase 2: Simulating SPS Sweep (SNR=8 dB, Beta={betas[0]}, Delay={delays[0]})")
    print(f"=======================================================")
    
    target_k_sps_sweep = 6
    target_delay = delays[0]
    target_beta = betas[0]
    
    for test_sps in sps_sweep_values:
        print(f"Testing SPS = {test_sps}...")
        
        # Recalculate sampling rate for this iteration
        current_fs = Rs * test_sps
        
        # Re-initialize transmitter with new SPS
        sender.set_responses(target_beta, test_sps, filter_span)
        
        # Create signal for just this 1 target SNR
        # Note: Ensure your create_transmitted_signal can accept just one SNR or parse accordingly
        # We temporarily overwrite the global SNR array if the function depends on it, or pass a list with 1 element
        _, rx_signals_sps_sweep = create_transmitted_signal(sender, preamble, target_delay, FREQ_OFFSET, test_sps)
        
        # Grab the RX signal corresponding to 6 dB (Assuming create_transmitted_signal generates for all SNRs in the global list)
        K_idx_6db = Rician_K_Factor.index(target_k_sps_sweep)  # Find index for 6 dB
        rx_sig_test = rx_signals_sps_sweep[K_idx_6db]
        
        receiver = Client_Rx([rx_sig_test], is_qpsk=True)
        receiver.set_responses(target_beta, test_sps, filter_span)
        receiver.filter_signal(test_sps, filter_span)
        filt_signal = receiver.filtered_signal[0]
        
        # M1
        det_delays = receiver.detect_preamble(preamble, test_sps, filter_span)
        c_delay = int(round(det_delays[0]))
        p_start = c_delay + (preamble_length * test_sps)
        sym_m1 = filt_signal[p_start : p_start + num_data_symbols * test_sps : test_sps]
        ber_vs_sps_final[1].append(get_BER(original_bits_flat, symbols_to_bits(sym_m1)))
        range_error_vs_sps_final[1].append((to_scalar(c_delay) - target_delay) * (c / current_fs))
        
        # M2
        e_delay_m2, _, _ = receiver.estimate_fractional_delay(filt_signal, preamble, test_sps, filter_span)
        start_m2 = e_delay_m2 + (preamble_length * test_sps)
        t_samp_m2 = start_m2 + np.arange(num_data_symbols) * test_sps
        sym_m2 = receiver.cubic_interpolate(filt_signal, t_samp_m2)
        ber_vs_sps_final[2].append(get_BER(original_bits_flat, symbols_to_bits(sym_m2)))
        range_error_vs_sps_final[2].append((to_scalar(e_delay_m2) - target_delay) * (c / current_fs))
        
        # M3
        e_delay_m3, _, _ = receiver.ml_fractional_delay(filt_signal, preamble, test_sps, filter_span, grid_resolution=0.01)
        start_m3 = e_delay_m3 + (preamble_length * test_sps)
        t_samp_m3 = start_m3 + np.arange(num_data_symbols) * test_sps
        sym_m3 = receiver.cubic_interpolate(filt_signal, t_samp_m3)
        ber_vs_sps_final[3].append(get_BER(original_bits_flat, symbols_to_bits(sym_m3)))
        range_error_vs_sps_final[3].append((to_scalar(e_delay_m3) - target_delay) * (c / current_fs))
        
        # M4
        num_blocks = int(np.ceil(num_data_symbols / data_block_size))
        total_payload_symbols = num_blocks * (data_block_size + CP_length)

        symbols_m4, f_phase_m4 = receiver.early_late_recovery(filt_signal, start_idx=p_start, sps=test_sps, max_symbols=total_payload_symbols)
        rx_preamble_m4 = filt_signal[c_delay : c_delay + preamble_length * test_sps : test_sps]
        _, W_mmse_m4 = receiver.estimate_channel_and_weights(rx_preamble_m4, preamble, data_block_size, target_snr)
        equalized_symbols_m4 = receiver.equalize_blocks_only(symbols_m4, W_mmse_m4, data_block_size, CP_length)
        
        ber_vs_sps_final[4].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m4)))
        range_error_vs_sps_final[4].append((to_scalar(c_delay + f_phase_m4) - target_delay) * (c / current_fs))
        
        # M5
        symbols_m5, f_phase_m5 = receiver.Dynamic_Gardner_recovery(
            filt_signal, start_idx=p_start, sps=test_sps, max_symbols=total_payload_symbols
        ) if hasattr(receiver, 'Dynamic_Gardner_recovery') else receiver.Gardner_recovery(
            filt_signal, start_idx=p_start, sps=test_sps, max_symbols=total_payload_symbols
        )
        rx_preamble_m5 = filt_signal[c_delay : c_delay + preamble_length * test_sps : test_sps]
        _, W_mmse_m5 = receiver.estimate_channel_and_weights(rx_preamble_m5, preamble, data_block_size, target_snr)
        equalized_symbols_m5 = receiver.equalize_blocks_only(symbols_m5, W_mmse_m5, data_block_size, CP_length)
        
        ber_vs_sps_final[5].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m5)))
        range_error_vs_sps_final[5].append((to_scalar(c_delay + f_phase_m5) - target_delay) * (c / current_fs))

        # M6
        symbols_m6, f_phase_m6 = receiver.lms_adaptive_timing_recovery(
            filt_signal, start_idx=p_start, sps=test_sps, max_symbols=total_payload_symbols
        )
        rx_preamble_m6 = filt_signal[c_delay : c_delay + preamble_length * test_sps : test_sps]
        _, W_mmse_m6 = receiver.estimate_channel_and_weights(rx_preamble_m6, preamble, data_block_size, target_snr)
        equalized_symbols_m6 = receiver.equalize_blocks_only(symbols_m6, W_mmse_m6, data_block_size, CP_length)
        
        ber_vs_sps_final[6].append(get_BER(original_bits_flat, symbols_to_bits(equalized_symbols_m6)))
        range_error_vs_sps_final[6].append((to_scalar(c_delay + f_phase_m6) - target_delay) * (c / current_fs))
    # =========================================================================
    # RENDER NEW ASSIGNMENT PLOTS (Graphs 4, 5, 7)
    # =========================================================================
    import matplotlib.pyplot as plt
    method_styles = {
        1: {'label': 'Integer Correlation (M1)',    'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
        2: {'label': 'Parabolic Interp (M2)',       'color': '#ff7f0e', 'marker': 's', 'ls': '-'},
        3: {'label': 'ML Grid Search (M3)',          'color': '#2ca02c', 'marker': '^', 'ls': '-'},
        4: {'label': 'Early-Late Loop (M4)',        'color': '#d62728', 'marker': 'x', 'ls': '-'},
        5: {'label': 'Gardner Loop (M5)',           'color': '#9467bd', 'marker': 'd', 'ls': '-'},
        6: {'label': 'LMS Adaptive (M6)',    'color': '#8c564b', 'marker': 'v', 'ls': '-'}
    }

    # Graph 4: Range Error vs Rician K-Factor
    plt.figure(figsize=(12, 6))
    for m_id, style in method_styles.items():
        # Sweep 1 actually iterates over Rician_K_Factor, not SNR
        plt.plot(Rician_K_Factor, np.abs(range_error_vs_snr_final[m_id]), color=style['color'], marker=style['marker'], 
                 linestyle=style['ls'], label=style['label'])
    plt.title(f'Graph 4: Range Error vs. Rician K-Factor (SNR = {target_snr} dB, SPS = {sps})')
    plt.xlabel('Rician K-Factor (dB)')
    plt.ylabel('Absolute Range Error (Meters)')
    plt.axhline(0, color='black', linestyle=':')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # Graph 5: Range Error vs SPS
    plt.figure(figsize=(12, 6))
    for m_id, style in method_styles.items():
        plt.plot(sps_sweep_values, np.abs(range_error_vs_sps_final[m_id]), color=style['color'], marker=style['marker'], 
                 linestyle=style['ls'], label=style['label'])
    # Note: Phase 2 pulls from index 4 of Rician_K_Factor (K = 8 dB) because target_k_sps_sweep = 6
    plt.title(f'Graph 5: Absolute Range Error vs. SPS over Rician Channel (SNR = {target_snr} dB, K = {Rician_K_Factor[K_idx_6db]} dB)')
    plt.xlabel('Samples Per Symbol (SPS)')
    plt.ylabel('Absolute Range Error (Meters)')
    plt.xscale('log', base=2)
    plt.xticks(sps_sweep_values, labels=[str(s) for s in sps_sweep_values])
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # Graph 7: BER vs SPS
    plt.figure(figsize=(12, 6))
    for m_id, style in method_styles.items():
        clean_ber = [max(b, 1e-6) for b in ber_vs_sps_final[m_id]]
        plt.semilogy(sps_sweep_values, clean_ber, color=style['color'], marker=style['marker'], 
                     linestyle=style['ls'], label=style['label'])
    plt.title(f'Graph 7: BER vs. SPS over Rician Channel (SNR = {target_snr} dB, K = {Rician_K_Factor[K_idx_6db]} dB)')
    plt.xlabel('Samples Per Symbol (SPS)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.xscale('log', base=2)
    plt.xticks(sps_sweep_values, labels=[str(s) for s in sps_sweep_values])
    plt.ylim(1e-6, 0.5)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
