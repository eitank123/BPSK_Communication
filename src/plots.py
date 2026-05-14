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

    plt.title('BPSK Bit Error Rate (BER) vs. SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.show()


def plot_eye_diagram(rx_before, rx_after, sps, snr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create a time vector normalized to the symbol period (-1 to +1)
    # This centers the "eye" opening exactly at time 0.
    t_norm = np.linspace(-1, 1, 2 * sps)

    # Plot "Before" Eye Diagram
    # Starting at i=20 is good practice to skip initial filter transients
    for i in range(20, 60):
        segment = rx_before[i * sps: (i + 2) * sps]
        ax1.plot(t_norm, segment, 'r', alpha=0.2)

    ax1.set_title(f'Eye Diagram BEFORE Matched Filter (RRC Only), SNR={snr}')
    ax1.set_xlabel('Time (Normalized to Symbol Period T)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Plot "After" Eye Diagram
    for i in range(20, 60):
        segment = rx_after[i * sps: (i + 2) * sps]
        ax2.plot(t_norm, segment, 'b', alpha=0.2)

    ax2.set_title(f'Eye Diagram AFTER Matched Filter (Combined RC), SNR={snr}')
    ax2.set_xlabel('Time (Normalized to Symbol Period T)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

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
    # 1. Calculate Theoretical BER for BPSK
    # Convert Eb/N0 from dB to linear
    ebno_linear = 10 ** (ebno_db_range / 10)

    # The Q-function Q(x) can be calculated using the complementary
    # error function: Q(x) = 0.5 * erfc(x / sqrt(2))
    # Pb = Q(sqrt(2 * Eb/N0)) -> 0.5 * erfc(sqrt(Eb/N0))
    theoretical_ber = 0.5 * erfc(np.sqrt(ebno_linear))

    # 2. Plotting
    plt.figure(figsize=(8, 6))

    # Use a semi-log scale (y-axis is logarithmic)
    plt.semilogy(ebno_db_range, theoretical_ber, 'b-', label='Theoretical BPSK', linewidth=2)

    simulated_ber_array = np.array(simulated_ber)
    if simulated_ber_array.ndim == 1:
        ber_curve_arr = np.array(simulated_ber_array, dtype=float)
        valid = ber_curve_arr > 0
        if np.any(valid):
            plt.semilogy(np.array(ebno_db_range)[valid], ber_curve_arr[valid], 'ro', label='Simulated RRC Matched Filter')
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