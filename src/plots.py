import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


SNR = [0, 5, 10, 15]  # Values in dB


def plot_BER_vs_SNR(ber_plot_values, series_labels=None):
    plt.figure(figsize=(8, 6))

    # Use semilogy for the logarithmic Y-axis
    if isinstance(ber_plot_values, (list, np.ndarray)) and np.array(ber_plot_values).ndim == 2:
        for idx, ber_curve in enumerate(ber_plot_values):
            label = series_labels[idx] if series_labels is not None else f"Curve {idx + 1}"
            ber_curve_arr = np.array(ber_curve, dtype=float)
            valid = ber_curve_arr > 0
            if not np.any(valid):
                continue
            plt.semilogy(np.array(SNR)[valid], ber_curve_arr[valid], marker='o', linewidth=2, markersize=8, label=label)
        plt.legend()
    else:
        ber_curve_arr = np.array(ber_plot_values, dtype=float)
        valid = ber_curve_arr > 0
        if np.any(valid):
            plt.semilogy(np.array(SNR)[valid], ber_curve_arr[valid], 'b-o', linewidth=2, markersize=8)

    plt.title('QPSK Bit Error Rate (BER) vs. Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.show()


def plot_eye_diagram(rx_before, rx_after, sps, snr, detected_delay=0, preamble_length=0):
    """
    Plot eye diagram for QPSK signals (handles complex signals).
    
    rx_before: received signal before matched filter
    rx_after: received signal after matched filter
    sps: samples per symbol
    snr: SNR value in dB
    detected_delay: detected preamble position in samples
    preamble_length: length of preamble in symbols (to skip entire preamble)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Create a time vector normalized to the symbol period (-1 to +1)
    t_norm = np.linspace(-1, 1, 2 * sps)

    # Determine start position: skip to message region (after preamble)
    preamble_samples = preamble_length * sps
    start_idx = detected_delay + preamble_samples
    
    # Add offset to skip first 5 symbols of message for transient settling
    start_idx = start_idx + 5 * sps
    end_idx = start_idx + 40 * sps  # Plot 40 symbols worth
    
    # Ensure we don't go past signal length
    end_idx = min(end_idx, len(rx_before) - 2*sps, len(rx_after) - 2*sps)
    
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
    """
    Plots simulated BER vs Theoretical BPSK BER.
    ebno_db_range: Array of Eb/N0 values in dB (e.g., np.arange(0, 11))
    simulated_ber: Array of BER values calculated from your simulation.
                   Can be a 1D array for a single curve or a 2D array for multiple curves.
    series_labels: Optional list of labels for each simulated BER curve.
    """
    # 1. Calculate Theoretical BER for QPSK
    # Convert Eb/N0 from dB to linear
    ebno_linear = 10 ** (ebno_db_range / 10)

    # QPSK has same BER as BPSK (for the same Eb/N0)
    # Pb = Q(sqrt(2 * Eb/N0)) -> 0.5 * erfc(sqrt(Eb/N0))
    theoretical_ber = 0.5 * erfc(np.sqrt(ebno_linear))

    # 2. Plotting
    plt.figure(figsize=(8, 6))

    # Use a semi-log scale (y-axis is logarithmic)
    plt.semilogy(ebno_db_range, theoretical_ber, 'b-', label='Theoretical QPSK', linewidth=2)

    simulated_ber_array = np.array(simulated_ber)
    if simulated_ber_array.ndim == 1:
        ber_curve_arr = np.array(simulated_ber_array, dtype=float)
        valid = ber_curve_arr > 0
        if np.any(valid):
            plt.semilogy(np.array(ebno_db_range)[valid], ber_curve_arr[valid], 'ro', label='Simulated QPSK RRC Matched Filter')
    else:
        for idx, ber_curve in enumerate(simulated_ber_array):
            label = series_labels[idx] if series_labels is not None else f"Simulated Curve {idx + 1}"
            ber_curve_arr = np.array(ber_curve, dtype=float)
            valid = ber_curve_arr > 0
            if not np.any(valid):
                continue
            plt.semilogy(np.array(ebno_db_range)[valid], ber_curve_arr[valid], marker='o', linewidth=2, markersize=6, label=label)

    plt.title('BER Performance: Simulated vs. Theoretical')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


def plot_noisy_signal(clean_signal, noisy_signal, sps, snr, num_symbols_to_show=10):
    """
    Plots the filtered signal before and after noise.

    clean_signal: The RRC filtered signal (no noise)
    noisy_signal: The signal after add_awgn()
    sps: Samples per symbol
    num_symbols_to_show: Number of symbol periods to display on the x-axis
    """
    # Calculate how many samples to plot based on symbols
    num_samples = min(len(clean_signal), num_symbols_to_show * sps)

    # Create a time axis in terms of symbol periods
    time_axis = np.arange(num_samples) / sps

    plt.figure(figsize=(12, 5))

    # Plot noisy signal in the background
    plt.plot(time_axis, noisy_signal[:num_samples], color='silver',
             label='Noisy Signal', alpha=0.7)

    # Plot clean signal in the foreground
    plt.plot(time_axis, clean_signal[:num_samples], color='#1f77b4',
             linewidth=2, label='Clean RRC Signal')

    plt.title(f'Time Domain Signal, SNR value: {snr} dB (First {num_symbols_to_show} Symbols)')
    plt.xlabel('Symbol Periods (T)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    # Highlight symbol sampling points (the centers)
    # Note: Depending on your filter delay/span, you may need an offset here
    plt.show()


def plot_sent_vs_sampled(sent_bits, sampled_output):
    """
    Plots the original bits vs the recovered symbols.

    sent_bits: The original sequence of 1s and -1s.
    sampled_output: The output from your sample_receiver() function.
    """
    # Ensure both arrays are the same length for comparison
    # The receiver might have fewer samples due to filter truncation
    num_samples = min(len(sent_bits), len(sampled_output))

    indices = np.arange(num_samples)

    plt.figure(figsize=(12, 6))

    # Plot Sent Bits (Digital level)
    plt.step(indices, sent_bits[:num_samples], where='mid',
             label='Sent Bits (Input)', color='gray', linestyle='--', alpha=0.6)

    # Plot Sampled Symbols (Analog level at sampling instance)
    plt.stem(indices, sampled_output[:num_samples],
             linefmt='C0-', markerfmt='C0o', label='Sampled Output (Rx)',
             basefmt=" ")

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'Sent Bits vs. Sampled Matched Filter Output (Alignment Check)')
    plt.xlabel('Symbol Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()