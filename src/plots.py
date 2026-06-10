"""
Plotting and visualization module for QPSK communication analysis.

Provides functions for:
- BER vs K-factor (Rician fading) visualization
- BER vs SNR performance curves
- Eye diagrams (I/Q constellation visualization)
- Timing and delay tracking analysis
- Signal quality assessment
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from scipy.integrate import quad

def rician_qpsk_ber_integrand(theta, EbNo_linear, K_linear):
    """Integrand for Rician QPSK BER (Craig's formula)."""
    sin2 = np.sin(theta) ** 2
    numerator = (1 + K_linear) * sin2
    denominator = (1 + K_linear) * sin2 + EbNo_linear
    exponent = - (K_linear * EbNo_linear) / denominator
    return (numerator / denominator) * np.exp(exponent)

def get_rician_theoretical_ber(ebno_db, k_db):
    """Calculate theoretical Rician QPSK BER."""
    EbNo_linear = 10 ** (ebno_db / 10)
    K_linear = 10 ** (k_db / 10) if k_db > -np.inf else 0.0
    integral, _ = quad(rician_qpsk_ber_integrand, 0, np.pi / 2, args=(EbNo_linear, K_linear))
    return (1 / np.pi) * integral

def plot_ber_vs_k(k_db_range, simulated_ber_matrix, target_snr, series_labels=None):
    """Plot BER vs Rician K-factor at fixed SNR with theoretical baseline."""
    plt.figure(figsize=(10, 6.5))
    markers = ['o', 's', '^', 'x', 'd', 'v', 'p', '*']
    
    theoretical_ber = [get_rician_theoretical_ber(target_snr, k) for k in k_db_range]
    plt.semilogy(k_db_range, theoretical_ber, 'k-', linewidth=3, alpha=0.9, 
                 label=f'Theoretical QPSK Bound (SNR = {target_snr} dB)')

    if simulated_ber_matrix is not None:
        sim_arr = np.array(simulated_ber_matrix)
        for idx in range(sim_arr.shape[0]):
            clean_sim_ber = np.clip(sim_arr[idx], 1e-6, 1.0)
            label = series_labels[idx] if series_labels is not None else f'Method {idx + 1}'
            plt.semilogy(k_db_range, clean_sim_ber, linestyle='--', marker=markers[idx % len(markers)], 
                         markersize=6, alpha=0.8, linewidth=1.8, label=label)

    plt.title(f'Timing Synchronization BER vs. Rician K-Factor (Fixed $E_b/N_0$ = {target_snr} dB)', fontsize=12, fontweight='bold')
    plt.xlabel('Rician K-Factor (dB)', fontsize=11)
    plt.ylabel('Bit Error Rate (BER)', fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=9, ncol=1)
    plt.tight_layout()
    plt.show()


def plot_BER_vs_SNR(ber_plot_values, series_labels=None, snr_range=None):
    """Plot BER vs SNR for multiple methods."""
    if snr_range is None:
        snr_range = [0, 5, 10, 15]
    plt.figure(figsize=(8, 6))

    if isinstance(ber_plot_values, (list, np.ndarray)) and np.array(ber_plot_values).ndim == 2:
        for idx, ber_curve in enumerate(ber_plot_values):
            label = series_labels[idx] if series_labels is not None else f"Curve {idx + 1}"
            ber_curve_arr = np.array(ber_curve, dtype=float)
            valid = ber_curve_arr > 0
            if not np.any(valid):
                continue
            plt.semilogy(np.array(snr_range)[valid], ber_curve_arr[valid], marker='o', linewidth=2, markersize=8, label=label)
        plt.legend()
    else:
        ber_curve_arr = np.array(ber_plot_values, dtype=float)
        valid = ber_curve_arr > 0
        if np.any(valid):
            plt.semilogy(np.array(snr_range)[valid], ber_curve_arr[valid], 'b-o', linewidth=2, markersize=8)

    plt.title('QPSK Bit Error Rate (BER) vs. Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()


def plot_eye_diagram(rx_before, rx_after, sps, snr, detected_delay=0, preamble_length=0):
    """Plot I/Q eye diagram for pre and post-filter signals."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    t_norm = np.linspace(-1, 1, 2 * sps)

    preamble_samples = preamble_length * sps
    start_idx = detected_delay + preamble_samples + 5 * sps
    end_idx = min(start_idx + 40 * sps, len(rx_before) - 2*sps, len(rx_after) - 2*sps)
    
    if end_idx <= start_idx:
        print(f"Warning: Not enough samples for eye diagram (SNR={snr} dB, start={start_idx}, end={end_idx})")
        return
    
    print(f"Eye diagram SNR={snr} dB: plotting from sample {start_idx} to {end_idx}")

    # Plot "Before" - Real part (In-phase)
    for i in range(start_idx, end_idx, sps):  # Step by sps for overlapping windows
        segment = rx_before[i: i + 2 * sps]
        if len(segment) == 2 * sps:
            axes[0, 0].plot(t_norm, np.real(segment), 'r', alpha=0.2)
    axes[0, 0].set_title(f'Eye Diagram BEFORE - Real Part, Eb/N0={snr} dB')
    axes[0, 0].set_xlabel('Time (Normalized to Symbol Period)')
    axes[0, 0].set_ylabel('Real (I) Amplitude')
    axes[0, 0].grid(True)

    # Plot "Before" - Imaginary part (Quadrature)
    for i in range(start_idx, end_idx, sps):
        segment = rx_before[i: i + 2 * sps]
        if len(segment) == 2 * sps:
            axes[0, 1].plot(t_norm, np.imag(segment), 'r', alpha=0.2)
    axes[0, 1].set_title(f'Eye Diagram BEFORE - Imaginary Part, Eb/N0={snr} dB')
    axes[0, 1].set_xlabel('Time (Normalized to Symbol Period)')
    axes[0, 1].set_ylabel('Imaginary (Q) Amplitude')
    axes[0, 1].grid(True)

    # Plot "After" - Real part
    for i in range(start_idx, end_idx, sps):
        segment = rx_after[i: i + 2 * sps]
        if len(segment) == 2 * sps:
            axes[1, 0].plot(t_norm, np.real(segment), 'b', alpha=0.2)
    axes[1, 0].set_title(f'Eye Diagram AFTER (Matched Filter) - Real Part, Eb/N0={snr} dB')
    axes[1, 0].set_xlabel('Time (Normalized to Symbol Period)')
    axes[1, 0].set_ylabel('Real (I) Amplitude')
    axes[1, 0].grid(True)

    # Plot "After" - Imaginary part
    for i in range(start_idx, end_idx, sps):
        segment = rx_after[i: i + 2 * sps]
        if len(segment) == 2 * sps:
            axes[1, 1].plot(t_norm, np.imag(segment), 'b', alpha=0.2)
    axes[1, 1].set_title(f'Eye Diagram AFTER (Matched Filter) - Imaginary Part, Eb/N0={snr} dB')
    axes[1, 1].set_xlabel('Time (Normalized to Symbol Period)')
    axes[1, 1].set_ylabel('Imaginary (Q) Amplitude')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_ber_comparison(ebno_db_range, simulated_ber, series_labels=None):
    """Compare simulated BER with theoretical QPSK baseline."""
    ebno_linear = 10 ** (np.array(ebno_db_range) / 10)
    theoretical_ber = 0.5 * erfc(np.sqrt(ebno_linear))

    plt.figure(figsize=(9, 6))
    plt.semilogy(ebno_db_range, theoretical_ber, 'k-', label='Theoretical QPSK Baseline', linewidth=2.5)

    simulated_ber_array = np.array(simulated_ber)
    markers = ['o', 's', '^', 'x', 'd']
    
    if simulated_ber_array.ndim == 1:
        ber_curve_arr = np.array(simulated_ber_array, dtype=float)
        valid = ber_curve_arr > 0
        if np.any(valid):
            plt.semilogy(np.array(ebno_db_range)[valid], ber_curve_arr[valid], 'ro', label='Simulated Receiver Performance')
    else:
        for idx, ber_curve in enumerate(simulated_ber_array):
            label = series_labels[idx] if series_labels is not None else f"Simulated Curve {idx + 1}"
            ber_curve_arr = np.array(ber_curve, dtype=float)
            clean_ber = np.clip(ber_curve_arr, 1e-6, 1.0)
            plt.semilogy(ebno_db_range, clean_ber, marker=markers[idx % len(markers)], 
                         linewidth=1.8, markersize=6, label=label)

    plt.title('BER Performance Evaluation: Simulated vs. Theoretical Bounds', fontsize=12, fontweight='bold')
    plt.xlabel('$E_b/N_0$ (dB)', fontsize=11)
    plt.ylabel('Bit Error Rate (BER)', fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_noisy_signal(clean_signal, noisy_signal, sps, snr, num_symbols_to_show=10):
    num_samples = min(len(clean_signal), num_symbols_to_show * sps)
    time_axis = np.arange(num_samples) / sps

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, np.real(noisy_signal[:num_samples]), color='silver', label='Noisy Signal (I)', alpha=0.7)
    plt.plot(time_axis, np.real(clean_signal[:num_samples]), color='#1f77b4', linewidth=2, label='Clean RRC Signal (I)')

    plt.title(f'Time Domain Signal Waveform (Eb/N0: {snr} dB)', fontsize=12)
    plt.xlabel('Symbol Periods (T)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_sent_vs_sampled(sent_bits, sampled_output):
    num_samples = min(len(sent_bits), len(sampled_output))
    indices = np.arange(num_samples)

    plt.figure(figsize=(12, 6))
    plt.step(indices, sent_bits[:num_samples], where='mid', label='Sent Bits (Input)', color='gray', linestyle='--', alpha=0.6)
    plt.stem(indices, sampled_output[:num_samples], linefmt='C0-', markerfmt='C0o', label='Sampled Output (Rx)', basefmt=" ")

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('Sent Bits vs. Sampled Matched Filter Output (Alignment Check)')
    plt.xlabel('Symbol Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_delay_tracking(snr_range, delay_matrix, labels, true_delay):
    """Plot absolute delay estimation error vs SNR."""
    plt.figure(figsize=(9, 6))
    markers = ['o', 's', '^', 'x', 'd']
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, 
                label='Perfect Synchronization (Zero Error)')
    
    for i in range(delay_matrix.shape[0]):
        abs_error = np.abs(delay_matrix[i] - true_delay)
        plt.plot(snr_range, abs_error, marker=markers[i % len(markers)],
                 linestyle='-', alpha=0.8, linewidth=1.5, label=labels[i])
        
    plt.title("Delay Estimation Absolute Error vs. SNR", fontsize=12, fontweight='bold')
    plt.xlabel("$E_b/N_0$ (dB)", fontsize=11)
    plt.ylabel("Absolute Delay Error (Samples)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.show()


