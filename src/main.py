from client_Tx import Client_Tx
from client_Rx import Client_Rx
from RRC_Implementation import *
from plots import *

np.random.seed(42)
number_of_bits = 100000
sps = 8
betas = [1]
delays = [0]
filter_span = 10
bit_mapping_BPSK = {0: -1, 1: 1}


# Global transmit frequency offset multiplier parameter
FREQ_OFFSET = 0


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


def create_transmitted_signal(sender: Client_Tx, upsampled_signal, delay, freq_offset, sample_rate):
    x_t = sender.prepare_x_t(upsampled_signal, delay, freq_offset, sample_rate)
    r_t = []
    for snr in SNR:
        r_t.append(add_awgn(np.array(x_t), snr))

    return x_t, r_t


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
        plot_eye_diagram(receiver.signal[snr], receiver.filtered_signal[snr], sps, SNR[snr])


if __name__ == "__main__":
    sender = Client_Tx(number_of_bits, bit_mapping_BPSK)
    upscaled_bit_array = bit_generation_handling(sender)

    all_ber = []
    series_labels = []
    last_sender = None
    last_receiver = None

    for beta in betas:
        sender.set_responses(beta, sps, filter_span)
        for delay in delays:
            print(f"\n=== Simulating beta={beta}, delay={delay} ===")

            clean_signal, Rx_signal = create_transmitted_signal(sender, upscaled_bit_array, delay, FREQ_OFFSET, sps)
            receiver = Client_Rx(Rx_signal)
            receiver.set_responses(beta, sps, filter_span)
            receiver.filter_signal()
            sampled_output = receiver.sample_receiver(sps, filter_span)

            ber = []
            for i, recovered_bits in enumerate(sampled_output):
                print(f"\n--- Testing SNR: {SNR[i]} dB ---")
                ber.append(get_BER(sender.mapped_bits, recovered_bits))

            all_ber.append(ber)
            series_labels.append(f"beta={beta}, delay={delay}")
            last_sender = sender
            last_receiver = receiver

    all_ber = np.array(all_ber)
    plots(last_sender, last_receiver, series_labels, all_ber)

