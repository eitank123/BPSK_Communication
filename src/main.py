from client_Tx import Client_Tx
from client_Rx import Client_Rx
from RRC_Implementation import *
from plots import *

np.random.seed(42)
number_of_bits = 100000
sps = 8
betas = [0.25]
delays = [0.5]  # Fractional delay in samples
filter_span = 10
# Zadoff-Chu preamble parameters
preamble_length = 128
preamble_root_index = 1  # coprime to preamble_length
# Farrow interpolation parameters
farrow_degree = 3  # Polynomial degree (1=linear, 2=quadratic, 3=cubic)
# QPSK constellation: 2 bits -> complex symbol (normalized to energy 1)
# Energy per symbol = 1, so each dimension is 1/sqrt(2)
qpsk_normalized = 1 / np.sqrt(2)
bit_mapping_QPSK = {
    (0, 0): qpsk_normalized * (1 + 1j),
    (0, 1): qpsk_normalized * (1 - 1j),
    (1, 0): qpsk_normalized * (-1 + 1j),
    (1, 1): qpsk_normalized * (-1 - 1j)
}


# Global transmit frequency offset multiplier parameter
FREQ_OFFSET = 0


def generate_zadoff_chu_preamble(length, root_index=1):
    """
    Generate a Zadoff-Chu sequence (constant amplitude zero autocorrelation).
    length: sequence length
    root_index: root index (should be coprime to length)
    """
    n = np.arange(length)
    # Zadoff-Chu: u(n) = exp(-j * pi * root_index * n * (n + 1) / length)
    zadoff_chu = np.exp(-1j * np.pi * root_index * n * (n + 1) / length)
    # Normalize to have energy = 1
    zadoff_chu = zadoff_chu / np.sqrt(length)
    return zadoff_chu


def bit_generation_handling(client: Client_Tx):
    client.generate_bit_array()
    upscaled_bit_array = client.upscale_array(sps)
    return upscaled_bit_array


def add_awgn(signal, ebno_db):
    ebno = 10**(ebno_db / 10)

    noise_variance = 1 / (2 * ebno)

    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_variance / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
    else:
        noise = np.sqrt(noise_variance) * np.random.randn(len(signal))

    return signal + noise


def create_transmitted_signal(sender: Client_Tx, upsampled_signal, preamble, delay, freq_offset, sample_rate):
    """Create transmitted signal with preamble prepended"""
    # Upscale preamble to match sps
    preamble_upscaled = np.repeat(preamble, sps)
    # Concatenate preamble and message
    full_signal = np.concatenate([preamble_upscaled, upsampled_signal])
    
    x_t = sender.prepare_x_t(full_signal, delay, freq_offset, sample_rate)
    r_t = []
    for ebno in SNR:
        # For QPSK: Energy per symbol is normalized to 1
        # Eb/N0 (energy per bit) needs adjustment: QPSK has 2 bits per symbol
        # Eb = Es/2 for QPSK, so Eb/N0 = (Es/2)/N0
        r_t.append(add_awgn(np.array(x_t), ebno))

    return x_t, r_t, preamble_upscaled


def get_BER(original_bits, recovered_bits):
    # Convert to numpy arrays to be safe
    orig = original_bits
    recv = recovered_bits

    # Find the common length
    min_len = min(len(orig), len(recv))

    # Slice both to the same size and compare
    num_correct = np.sum(orig[:min_len] == recv[:min_len])

    # Calculate Accuracy and BER
    total_bits = min_len
    ber = float((total_bits - num_correct) / total_bits)

    print(f"Comparison length: {total_bits}")
    print(f"Correct: {num_correct}")
    print(f"BER: {ber*100:.2f}%")
    # Try shifting the comparison to see if accuracy jumps up
    return ber


def plots(sender: Client_Tx, receiver: Client_Rx, series_labels, ber_matrix):
    plot_responses(sender.impulse_response, sender.freq_response, sps)
    plot_BER_vs_SNR(ber_matrix, series_labels)
    plot_ber_comparison(np.array(SNR), ber_matrix, series_labels)
    for snr in range(len(SNR)):
        # Get detected delay for this SNR level
        detected_delay = receiver.detected_delays_list[snr] if snr < len(receiver.detected_delays_list) else 0
        plot_eye_diagram(receiver.signal[snr], receiver.filtered_signal[snr], sps, SNR[snr], 
                        detected_delay=detected_delay, preamble_length=preamble_length)


if __name__ == "__main__":
    # Generate Zadoff-Chu preamble
    preamble = generate_zadoff_chu_preamble(preamble_length, preamble_root_index)
    
    sender = Client_Tx(number_of_bits, bit_mapping_QPSK, is_qpsk=True, farrow_degree=farrow_degree)
    upscaled_bit_array = bit_generation_handling(sender)

    all_ber = []
    series_labels = []
    last_sender = None
    last_receiver = None

    for beta in betas:
        sender.set_responses(beta, sps, filter_span)
        for delay in delays:
            print(f"\n=== Simulating QPSK with Zadoff-Chu Preamble + Farrow Interpolation ===")
            print(f"    Beta={beta}, Fractional Delay={delay:.2f} samples, Farrow Degree={farrow_degree}")

            clean_signal, Rx_signal, preamble_upscaled = create_transmitted_signal(sender, upscaled_bit_array, preamble, delay, FREQ_OFFSET, sps)
            receiver = Client_Rx(Rx_signal, is_qpsk=True)
            receiver.set_responses(beta, sps, filter_span)
            receiver.filter_signal()
            
            # Detect preamble using receiver's FFT-based correlation
            receiver.detect_preamble(preamble, sps, preamble_length, filter_span)
            
            # Print detected delays
            for snr_idx, detected_delay in enumerate(receiver.detected_delays_list):
                print(f"Eb/N0 {SNR[snr_idx]} dB: Preamble detected at sample {detected_delay}, Expected delay: {delay}")
            
            # Sample receiver output, skipping preamble
            sampled_output = receiver.sample_receiver(sps, filter_span, preamble_length)

            # Flatten original bits from 2-bit tuples for BER comparison
            original_bits_flat = np.array([bit for tuple_bits in sender.bit_array for bit in tuple_bits])

            ber = []
            for i, recovered_bits in enumerate(sampled_output):
                print(f"\n--- Testing Eb/N0: {SNR[i]} dB ---")
                ber.append(get_BER(original_bits_flat, recovered_bits))

            all_ber.append(ber)
            series_labels.append(f"beta={beta}, delay={delay}")
            last_sender = sender
            last_receiver = receiver

    all_ber = np.array(all_ber)
    plots(last_sender, last_receiver, series_labels, all_ber)

