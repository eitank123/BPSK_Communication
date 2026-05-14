import numpy as np
import math
import matplotlib.pyplot as plt


def rrc_design(beta=0.25, sps=8, span=10):

    num_taps = sps * span + 1
    taps = []

    half_width = (num_taps - 1) // 2

    for i in range(-half_width, half_width + 1):

        n = float(i)

        # beta = 0 --> sinc
        if beta == 0:
            val = (1 / math.sqrt(sps)) * np.sinc(n / sps)

        # center tap
        elif n == 0:
            val = (1.0 / math.sqrt(sps)) * (
                1.0 + beta * (4.0 / math.pi - 1.0)
            )

        # singularity at t = ±T/(4β)
        elif abs(abs(n) - sps / (4.0 * beta)) < 1e-12:

            val = (beta / math.sqrt(2.0 * sps)) * (
                (1.0 + 2.0 / math.pi)
                * math.sin(math.pi / (4.0 * beta))
                +
                (1.0 - 2.0 / math.pi)
                * math.cos(math.pi / (4.0 * beta))
            )

        else:

            angle = math.pi * n / sps

            num = (
                math.cos((1 + beta) * angle)
                +
                math.sin((1 - beta) * angle)
                / (4 * beta * n / sps)
            )

            den = 1 - (4 * beta * n / sps) ** 2

            val = (
                (4 * beta)
                / (math.pi * math.sqrt(sps))
            ) * (num / den)

        taps.append(val)

    # Normalize energy
    energy = math.sqrt(sum(x*x for x in taps))

    return [x / energy for x in taps]


def get_rrc_freq_response(filter_array, n_fft=2048):
    h_fft = np.fft.fft(filter_array, n_fft)
    return np.fft.fftshift(h_fft)


def plot_rrc_impulse(taps, sps=8):
    """
    Plots the RRC filter impulse response.

    taps: List of coefficients from rrc_design()
    sps: Samples per symbol used in design
    """
    num_taps = len(taps)
    half_width = (num_taps - 1) / 2

    # Create time index normalized to symbol periods
    # (Matches the "Time Index" in the reference image)
    time_index = [i / sps for i in range(int(-half_width), int(half_width) + 1)]

    plt.figure(figsize=(10, 5))

    # Plot the continuous-like shape and the discrete taps (stems)
    plt.plot(time_index, taps, color='#1f77b4', alpha=0.7, label='RRC Shape')
    markerline, stemlines, _ = plt.stem(time_index, taps, label='Taps (Discrete)')

    # Styling to match technical standards
    plt.setp(markerline, 'markersize', 4)
    plt.title(f'RRC Filter Impulse Response (sps={sps})')
    plt.xlabel('Time Index (Symbols)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    plt.legend()
    plt.show()


def plot_rrc_freq(filter_frequencies, sps=8, n_fft=2048):
    magnitude = np.abs(filter_frequencies)

    # Convert to decibels (dB)
    # Adding a tiny epsilon to avoid log(0)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    # Frequency axis normalized to the sampling rate
    # Frequencies will range from -sps/2 to +sps/2 (relative to symbol rate)
    freqs = np.linspace(-sps / 2, sps / 2, n_fft)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, magnitude_db)
    plt.title('RRC Filter Frequency Response')
    plt.xlabel('Normalized Frequency (f/Rs)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim([-60, 10])  # Focus on the passband and first few sidelobes
    plt.show()


def get_impulse_and_freq_response(beta, sps, filter_span):
    rrc_impulse_response = rrc_design(beta, sps, filter_span)
    rrc_freq_response = get_rrc_freq_response(rrc_impulse_response)

    return rrc_impulse_response, rrc_freq_response


def plot_responses(impulse_response, freq_response, sps):
    plot_rrc_impulse(impulse_response, sps)
    plot_rrc_freq(freq_response, sps)
