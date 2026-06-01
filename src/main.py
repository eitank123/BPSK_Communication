import numpy as np
import matplotlib.pyplot as plt
from client_Tx import Client_Tx
from client_Rx import Client_Rx
from RRC_Implementation import *
import plots  # Imports your updated plotting code package

np.random.seed(42)
number_of_bits = 10000
sps = 2
betas = [0.5]
delays = [200.4]  # Testing fractional delay positioning
filter_span = 10

SNR = [0, 2, 4, 6, 8, 10]

preamble_length = 127
preamble_root_index = 1 
farrow_degree = 3  

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

def generate_zadoff_chu_preamble(length, root_index=1):
    n = np.arange(length)
    zadoff_chu = np.exp(-1j * np.pi * root_index * n * (n + 1) / length)
    return zadoff_chu

def add_awgn(signal, ebno_db, sps):
    """
    Mathematically precise AWGN function that calculates power from the active 
    filtered baseline and correctly accounts for bits/symbol and multi-rate upsampling.
    """
    ebno_linear = 10**(ebno_db / 10)
    sig_power = np.mean(np.abs(signal)**2)
    
    # QPSK contains 2 bits per symbol.
    # Total sample noise variance formula:
    total_noise_variance = sig_power * (sps / (2 * ebno_linear))

    # Split noise power evenly between Real (I) and Imaginary (Q) structures
    noise = np.sqrt(total_noise_variance / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise

def create_transmitted_signal(sender: Client_Tx, preamble, delay, freq_offset, sps):
    full_symbols = np.concatenate([preamble, sender.mapped_bits])
    
    dtype = complex if sender.is_qpsk else float
    upsampled_full = np.zeros(len(full_symbols) * sps, dtype=dtype)
    upsampled_full[::sps] = full_symbols
    
    x_t = sender.prepare_x_t(upsampled_full, delay, freq_offset, sample_rate=sps)
    
    r_t = []
    for ebno in SNR:
        r_t.append(add_awgn(np.array(x_t), ebno, sps))

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
                current_snr = SNR[snr_idx]
                
                receiver = Client_Rx([rx_sig], is_qpsk=True)
                receiver.set_responses(beta, sps, filter_span)
                receiver.filter_signal(sps, filter_span)
                filt_signal = receiver.filtered_signal[0]
                
                # --- APPROACH 1: Integer Correlation ---
                detected_delays = receiver.detect_preamble(preamble, sps, filter_span)
                coarse_delay = int(round(detected_delays[0]))
                payload_start_idx = coarse_delay + (preamble_length * sps)

                symbols_m1 = filt_signal[payload_start_idx : payload_start_idx + num_data_symbols * sps : sps]
                ber_results[1].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m1)))
                delay_results[1].append(to_scalar(coarse_delay))

                # --- APPROACH 2: Parabolic Fractional Interpolation ---
                est_delay_m2, _, _ = receiver.estimate_fractional_delay(filt_signal, preamble, sps, filter_span)
                start_m2 = est_delay_m2 + (preamble_length * sps)
                t_samples_m2 = start_m2 + np.arange(num_data_symbols) * sps

                symbols_m2 = receiver.cubic_interpolate(filt_signal, t_samples_m2)
                ber_results[2].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m2)))
                delay_results[2].append(to_scalar(est_delay_m2))
                
                # --- APPROACH 3: Maximum Likelihood Fractional Estimation ---
                est_delay_m3, _, _ = receiver.ml_fractional_delay(filt_signal, preamble, sps, filter_span, grid_resolution=0.01)
                start_m3 = est_delay_m3 + (preamble_length * sps)
                t_samples_m3 = start_m3 + np.arange(num_data_symbols) * sps

                symbols_m3 = receiver.cubic_interpolate(filt_signal, t_samples_m3)
                ber_results[3].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m3)))
                delay_results[3].append(to_scalar(est_delay_m3))
                
                # --- APPROACH 4: Early-Late Loop Tracking ---
                symbols_m4, final_phase_m4 = receiver.early_late_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=num_data_symbols
                )
                ber_results[4].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m4)))
                delay_results[4].append(to_scalar(coarse_delay + final_phase_m4))

                # --- APPROACH 5: Gardner Loop Tracking ---
                symbols_m5, final_phase_m5 = receiver.Dynamic_Gardner_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=num_data_symbols
                ) if hasattr(receiver, 'Dynamic_Gardner_recovery') else receiver.Gardner_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=num_data_symbols
                )
                ber_results[5].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m5)))
                delay_results[5].append(to_scalar(coarse_delay + final_phase_m5))

                # --- APPROACH 6: LMS Adaptive Timing Recovery ---
                symbols_m6, final_phase_m6 = receiver.lms_adaptive_timing_recovery(
                    filt_signal, start_idx=payload_start_idx, sps=sps, max_symbols=num_data_symbols
                )
                ber_results[6].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m6)))
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
            plots.plot_ber_comparison(SNR, matrix_ber, labels)
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
    
    target_snr_sps_sweep = 8
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
        snr_idx_6db = SNR.index(target_snr_sps_sweep)
        rx_sig_test = rx_signals_sps_sweep[snr_idx_6db]
        
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
        sym_m4, f_phase_m4 = receiver.early_late_recovery(filt_signal, start_idx=p_start, sps=test_sps, max_symbols=num_data_symbols)
        ber_vs_sps_final[4].append(get_BER(original_bits_flat, symbols_to_bits(sym_m4)))
        range_error_vs_sps_final[4].append((to_scalar(c_delay + f_phase_m4) - target_delay) * (c / current_fs))
        
        # M5
        sym_m5, f_phase_m5 = receiver.Dynamic_Gardner_recovery(
            filt_signal, start_idx=p_start, sps=test_sps, max_symbols=num_data_symbols
        ) if hasattr(receiver, 'Dynamic_Gardner_recovery') else receiver.Gardner_recovery(
            filt_signal, start_idx=p_start, sps=test_sps, max_symbols=num_data_symbols
        )
        ber_vs_sps_final[5].append(get_BER(original_bits_flat, symbols_to_bits(sym_m5)))
        range_error_vs_sps_final[5].append((to_scalar(c_delay + f_phase_m5) - target_delay) * (c / current_fs))

        # M6
        sym_m6, f_phase_m6 = receiver.lms_adaptive_timing_recovery(
            filt_signal, start_idx=p_start, sps=test_sps, max_symbols=num_data_symbols
        )
        ber_vs_sps_final[6].append(get_BER(original_bits_flat, symbols_to_bits(sym_m6)))
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

    # Graph 4: Range Error vs SNR
    plt.figure(figsize=(12, 6))
    for m_id, style in method_styles.items():
        plt.plot(SNR, np.abs(range_error_vs_snr_final[m_id]), color=style['color'], marker=style['marker'], 
                 linestyle=style['ls'], label=style['label'])
    plt.title(f'Graph 4: Range Error vs. SNR (SPS = {sps})')
    plt.xlabel('E_b/N_0 (dB)')
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
    plt.title(f'Graph 5: Absolute Range Error vs. SPS (SNR = {target_snr_sps_sweep} dB)')
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
    plt.title(f'Graph 7: BER vs. SPS (SNR = {target_snr_sps_sweep} dB)')
    plt.xlabel('Samples Per Symbol (SPS)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.xscale('log', base=2)
    plt.xticks(sps_sweep_values, labels=[str(s) for s in sps_sweep_values])
    plt.ylim(1e-6, 0.5)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
