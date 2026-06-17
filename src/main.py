import numpy as np
import matplotlib.pyplot as plt
from client_Tx import Client_Tx
from client_Rx import Client_Rx
from RRC_Implementation import *
import plots  # Imports your updated plotting code package
import config as cfg
import scipy.linalg as la
import DOA_estimate_model as NN

np.random.seed(42)
number_of_bits = cfg.BIT_AMOUNT
sps = 8
betas = [0.5]
delays = [200.4]  # Testing fractional delay positioning
filter_span = 10


preamble_length = 127
preamble_root_index = 1 
farrow_degree = 3  

Rs = 4e6
fs = Rs * sps  # 4 MHz sampling rate
c = 3e8   # Speed of light

SNR = cfg.SNR

qpsk_normalized = 1 / np.sqrt(2)
bit_mapping_QPSK = {
    (0, 0): qpsk_normalized * (1 + 1j),
    (0, 1): qpsk_normalized * (1 - 1j),
    (1, 0): qpsk_normalized * (-1 + 1j),
    (1, 1): qpsk_normalized * (-1 - 1j)
}


def generate_zadoff_chu_preamble(length=cfg.PREAMBLE_LENGTH, root_index=1):
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

def plot_absolute_errors_comparison(methods_dict):
    """
    Plots absolute DOA error curves for multiple estimation methods side-by-side.
    
    Parameters:
    methods_dict (dict): A dictionary where keys are method names (str) 
                         and values are lists/arrays of errors across the SNR sweep.
    """
    plt.figure(figsize=(10, 6))

    global_max_val = 0.0

    # Loop through each method and plot its error curve dynamically
    for method_name, errors in methods_dict.items():
        plt.plot(SNR, errors, marker='o', linestyle='-', linewidth=2, label=f'{method_name}')
        
        # Track the largest error across all methods for axis scaling
        if errors:
            global_max_val = max(global_max_val, max(errors))

    # Draw a horizontal line representing perfect estimation (0 degrees of error)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Ideal Error (0°)')

    # Graph formatting
    plt.title('DOA Estimation Absolute Error Comparison vs. SNR', fontsize=12, fontweight='bold')
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=10)
    plt.ylabel('Absolute Error [Degrees]', fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')

    # Dynamically scale y-axis top to be 10% larger than the worst error across all methods
    # Set bottom slightly below 0 (e.g., -0.2) so the red dashed baseline isn't cut off by the border
    y_top = global_max_val * 1.1 if global_max_val > 0 else 1.0
    plt.ylim(-0.2, y_top)

    plt.tight_layout()
    plt.show()

def esprit_1d(filt_signal1, filt_signal2, coarse_delay, preamble_len=cfg.PREAMBLE_LENGTH, sps=cfg.SPS, num_sources=1):
    """
    Classic LS-ESPRIT for a Uniform Linear Array (ULA) using custom geometry.
    """
    # 1. Get the synchronized preamble blocks (Full SPS for processing gain)
    p1 = filt_signal1[coarse_delay : coarse_delay + (preamble_len * sps)]
    p2 = filt_signal2[coarse_delay : coarse_delay + (preamble_len * sps)]

    # 2. Compute the 2x2 Spatial Covariance Matrix R
    X = np.vstack((p1, p2))
    R = (X @ X.conj().T) / len(p1)
    
    # 3. Compute Eigendecomposition and extract Signal Subspace (U_s)
    # la.eigh handles the Hermitian covariance matrix efficiently
    _, U = la.eigh(R)
    
    # Extract the signal subspace (strongest eigenvector)
    U_s = U[:, -num_sources:]
    
    # 4. Shift-Invariance Selection (Split into overlapping subarrays)
    U_s1 = U_s[:-1, :]
    U_s2 = U_s[1:, :]
    
    # 5. Solve the invariance equation: U_s2 = U_s1 @ Phi via pseudo-inverse
    Phi = la.pinv(U_s1) @ U_s2
    
    # 6. Extract spatial frequencies from the eigenvalues of Phi
    eigenvalues = la.eigvals(Phi)
    phases = np.angle(eigenvalues)
    
    # 7. Map spatial phase to physical angle using actual array dimensions
    wavelength = cfg.SPEED_OF_LIGHT / cfg.CARRIER_FREQ
    sin_theta = (phases * wavelength) / (2 * np.pi * cfg.ANTENNAS_DISTANCE[0])
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    
    estimated_angles = np.arcsin(sin_theta)
    return np.degrees(estimated_angles)

def music_1d(filt_signal1, filt_signal2, coarse_delay, preamble_len=cfg.PREAMBLE_LENGTH, sps=cfg.SPS, num_sources=1):
    """
    Vectorized 1D MUSIC algorithm for a Uniform Linear Array (ULA).
    """
    # 1. Get the synchronized preamble blocks
    p1 = filt_signal1[coarse_delay : coarse_delay + (preamble_len * sps)]
    p2 = filt_signal2[coarse_delay : coarse_delay + (preamble_len * sps)]

    # 2. Compute the 2x2 Spatial Covariance Matrix R
    X = np.vstack((p1, p2))
    R = (X @ X.conj().T) / len(p1)
    
    # 3. Compute Eigendecomposition
    # la.eigh returns eigenvalues in ASCENDING order
    eigenvalues, U = la.eigh(R)
    
    # 4. Extract the Noise Subspace (U_n)
    # Total antennas (N) - signal sources (M) = noise dimensions
    # For 2 antennas and 1 source, it extracts the first column (smallest eigenvalue)
    N = R.shape[0]
    U_n = U[:, :(N - num_sources)]
    
    # 5. Define the Angular Search Grid (e.g., -90 to 90 degrees with 0.1° resolution)
    angle_grid = np.linspace(-90.0, 90.0, 1801)
    theta_rad = np.radians(angle_grid)
    
    # 6. Physical Geometry Variables
    wavelength = cfg.SPEED_OF_LIGHT / cfg.CARRIER_FREQ
    d = cfg.ANTENNAS_DISTANCE[0]
    
    # 7. Generate Steering Vectors across the entire grid simultaneously
    # Phase shift vector for each angle candidate: shape (1801,)
    phase_shifts = 2 * np.pi * d * np.sin(theta_rad) / wavelength
    
    # Matrix of steering vectors: shape (2, 1801)
    A = np.vstack((np.ones_like(phase_shifts), np.exp(1j * phase_shifts)))
    
    # 8. Compute MUSIC Pseudo-Spectrum cleanly without loops
    # Project steering vectors onto the noise subspace
    # Matrix dimension trick: (N_noise x 2) @ (2 x 1801) -> (N_noise x 1801)
    projection = U_n.conj().T @ A
    
    # Denominator equals the squared Euclidean norm of the projection column-by-column
    denominator = np.sum(np.abs(projection) ** 2, axis=0)
    
    # Avoid zero division inside ideal noiseless scenarios
    pseudo_spectrum = 1.0 / np.where(denominator == 0, 1e-15, denominator)
    
    # 9. Find the location of the highest peak
    peak_idx = np.argmax(pseudo_spectrum)
    estimated_angle = angle_grid[peak_idx]
    
    return estimated_angle


def esprit_4ant(x, d, wavelength):
    """
    x: complex array shape (4, N) - received signals (4 antennas, N snapshots)
    d: antenna spacing
    wavelength: signal wavelength
    """

    x = np.atleast_2d(x)

    # 1. Form covariance matrix
    R = x @ x.conj().T / x.shape[1]

    # 2. Eigen-decomposition (signal subspace)
    eigvals, eigvecs = np.linalg.eigh(R)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Assume 1 source (extendable)
    Es = eigvecs[:, :1]   # signal subspace (4 x 1)

    # 3. Subarray selection matrices (shift invariance)
    E1 = Es[0:3, :]
    E2 = Es[1:4, :]

    # 4. ESPRIT rotational operator
    Psi = np.linalg.pinv(E1) @ E2

    # 5. Eigenvalues -> phase shifts
    eig = np.linalg.eigvals(Psi)

    phi = np.angle(eig)[0]

    # 6. DOA conversion
    sin_theta = (phi * wavelength) / (2 * np.pi * d)
    sin_theta = np.clip(sin_theta.real, -1.0, 1.0)

    theta = np.arcsin(sin_theta)

    return np.degrees(theta)

def steering_vector(theta, d, wavelength, M=4):

    k = 2 * np.pi / wavelength
    m = np.arange(M)

    return np.exp(1j * k * d * m * np.sin(theta))

def ml_4ant(x, d, wavelength=cfg.wavelength, grid=1000):

    M = 4

    # covariance (single snapshot)
    R = np.outer(x, np.conj(x))

    angles = np.linspace(-np.pi/2, np.pi/2, grid)

    best_theta = None
    best_val = -np.inf

    for theta in angles:

        a = steering_vector(theta, d, wavelength, M)

        metric = np.real(np.conj(a) @ R @ a)

        if metric > best_val:
            best_val = metric
            best_theta = theta

    return np.degrees(best_theta)

def mvdr_4ant(R, d, wavelength=cfg.wavelength, grid=1000):

    M = R.shape[0]  # 4 antennas

    # regularization MUST match R size
    R = R + 1e-6 * np.eye(M)

    R_inv = np.linalg.inv(R)

    angles = np.linspace(-np.pi/2, np.pi/2, grid)

    best_theta = None
    best_val = -np.inf

    for th in angles:

        a = steering_vector(th, d, wavelength, M)

        denom = np.real(np.conj(a) @ R_inv @ a)

        if denom <= 0:
            continue

        metric = 1 / denom

        if metric > best_val:
            best_val = metric
            best_theta = th

    return np.degrees(best_theta)

def get_filt_signal(receiver):
    receiver.set_responses(beta, sps, filter_span)
    receiver.filter_signal(sps, filter_span)
    filt_signal = receiver.filtered_signal[0]
    return filt_signal

def music_4ant(R, d, wavelength=cfg.wavelength, grid=1000):
    """
    R: (4,4) covariance matrix
    """

    M = R.shape[0]

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(R)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # 1 source → noise subspace is last 3 eigenvectors
    En = eigvecs[:, 1:]

    angles = np.linspace(-np.pi/2, np.pi/2, grid)

    P = np.zeros_like(angles, dtype=float)

    for i, theta in enumerate(angles):
        a = steering_vector(theta, d, wavelength, M)

        denom = np.conj(a) @ En @ En.conj().T @ a
        P[i] = 1 / np.abs(denom)

    return np.degrees(angles[np.argmax(P)])

def covariance(X):
    return X @ X.conj().T / X.shape[1]   # (4,4)

def conjugate_multiplication_2ant(rx1, rx2):
    wavelength = cfg.wavelength
    d = cfg.ANTENNAS_DISTANCE[0]

    phi = -np.angle(np.sum(rx2 * np.conj(rx1)))

    sin_theta = phi * wavelength / (2 * np.pi * d)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    return -np.degrees(np.arcsin(sin_theta))

def conjugate_multiplication_4ant(rx0, rx1, rx2, rx3):
    wavelength = cfg.wavelength
    d = cfg.ANTENNAS_DISTANCE[0]

    r01 = np.sum(rx0 * np.conj(rx1))
    r12 = np.sum(rx1 * np.conj(rx2))
    r23 = np.sum(rx2 * np.conj(rx3))

    phi = -np.angle(r01 + r12 + r23)

    sin_theta = phi * wavelength / (2 * np.pi * d)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    return np.degrees(np.arcsin(sin_theta))

def mrc_combine_phase_corrected(y1, y2):
    """
    Combines two pre-phase-corrected received signals using Maximum Ratio Combining (MRC).
    
    Parameters:
    y1 (numpy array): Phase-corrected complex or real signal from antenna 1.
    y2 (numpy array): Phase-corrected complex or real signal from antenna 2.
    
    Returns:
    numpy array: The optimally combined MRC signal.
    """
    # Since phase is corrected, the absolute value extracts the weighting factor
    w1 = np.abs(y1)
    w2 = np.abs(y2)
    
    # Apply weights and sum
    y_combined = (w1 * y1) + (w2 * y2)
    
    return y_combined

def estimate_channel_preamble(rx_preamble, tx_preamble):
    """
    Estimates the complex channel gain (h) using a known preamble sequence.
    
    Parameters:
    rx_preamble (numpy array): The received preamble symbols (contains noise and channel distortion).
    tx_preamble (numpy array): The ideal, originally transmitted preamble symbols.
    
    Returns:
    complex: The estimated complex channel coefficient h.
    """
    # Ensure inputs are numpy arrays
    rx_preamble = np.array(rx_preamble)
    tx_preamble = np.array(tx_preamble)
    
    # Element-wise Least Squares estimation: h_i = rx_i / tx_i
    # (Multiplying by the conjugate and dividing by magnitude squared handles complex division cleanly)
    h_estimates = rx_preamble / tx_preamble
    
    # Average the estimates across all preamble symbols to reduce noise variance
    h_estimated = np.mean(h_estimates)
    
    return h_estimated


if __name__ == "__main__":
    preamble = generate_zadoff_chu_preamble(preamble_length, preamble_root_index)
    
    sender = Client_Tx(number_of_bits, bit_mapping_QPSK, is_qpsk=True, farrow_degree=farrow_degree)
    

    # Dictionaries to store the final sweep data for the new assignment plots
    range_error_vs_snr_final = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    # Define the SPS values to sweep for Graphs 5 and 7
    sps_sweep_values = [1, 2, 4, 8, 16]
    range_error_vs_sps_final = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    ber_vs_sps_final = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    

    DOA_methods = {
                   "conjugate_2ant" : [0 for i in range(len(SNR))],
                   "conjugate_4ant" : [0 for i in range(len(SNR))],
                   "ESPRIT" : [0 for i in range(len(SNR))],
                   "MUSIC" : [0 for i in range(len(SNR))],
                   "ESPRIT_4ANT" : [0 for i in range(len(SNR))],
                   "mvdr_4ant" : [0 for i in range(len(SNR))],
                   "music_4ant" : [0 for i in range(len(SNR))]
                   }
    # =========================================================================
    # SWEEP 1: Varying SNR (Fixed SPS) - Graphs 1, 2, 3, 4, 6
    # =========================================================================
    for beta in betas:
        sender.set_responses(beta, sps, filter_span)
        for delay in delays:
            for iteration in range(cfg.ITERATIONS):
                estimations = {
                   "conjugate_2ant" : [],
                   "conjugate_4ant" : [],
                   "ESPRIT" : [],
                   "MUSIC" : [],
                   "ESPRIT_4ANT" : [],
                   "mvdr_4ant" : [],
                   "music_4ant" : []
                   }
                sender.generate_bit_array()
                original_bits_flat = np.array([bit for tuple_bits in sender.bit_array for bit in tuple_bits])
                num_data_symbols = len(sender.mapped_bits)
                print(f"ITERATION: {iteration} / {cfg.ITERATIONS}")
                print(f"\n=======================================================")
                print(f"Phase 1: Simulating SNR Sweep (Beta={beta}, Delay={delay} samples, SPS={sps})")
                print(f"=======================================================")

                clean_signal, Rx_signals = create_transmitted_signal(sender, preamble, delay, cfg.FREQ_OFFSET, sps)

                rx2 = np.array(clean_signal) * np.exp(1j *cfg.ANTENNA2_PHASE_SHIFT_VALUES[0])
                rx_antenna2 = []

                for ebno in SNR:
                    rx_antenna2.append(add_awgn(np.array(rx2), ebno, sps))
                
                rx3 = np.array(clean_signal) * np.exp(1j *cfg.ANTENNA2_PHASE_SHIFT_VALUES[0] * 2)
                rx_antenna3 = []

                for ebno in SNR:
                    rx_antenna3.append(add_awgn(np.array(rx3), ebno, sps))

                rx4 = np.array(clean_signal) * np.exp(1j *cfg.ANTENNA2_PHASE_SHIFT_VALUES[0] * 3)
                rx_antenna4 = []

                for ebno in SNR:
                    rx_antenna4.append(add_awgn(np.array(rx4), ebno, sps))
                
                ber_results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
                delay_results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

                ber_results_2 = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
                delay_results_2 = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
                
                for snr_idx, rx_sig in enumerate(Rx_signals):
                    current_snr = SNR[snr_idx]
                    rx1 = rx_sig
                    rx2 = rx_antenna2[snr_idx]
                    rx3 = rx_antenna3[snr_idx]
                    rx4 = rx_antenna4[snr_idx]

                    receiver = Client_Rx([rx1], is_qpsk=True)
                    receiver2 = Client_Rx([rx2], is_qpsk=True)
                    receiver3 = Client_Rx([rx3], is_qpsk=True)
                    receiver4 = Client_Rx([rx4], is_qpsk=True)

                    filt_signal = get_filt_signal(receiver)
                    filt_signal2 = get_filt_signal(receiver2)
                    filt_signal3 = get_filt_signal(receiver3)
                    filt_signal4 = get_filt_signal(receiver4)
                    
                    # --- APPROACH 1: Integer Correlation ---
                    detected_delays = receiver.detect_preamble(preamble, sps, filter_span)
                    coarse_delay = int(round(detected_delays[0]))
                    payload_start_idx = coarse_delay + (preamble_length * sps)

                    symbols_m1 = filt_signal[payload_start_idx : payload_start_idx + num_data_symbols * sps : sps]
                    ber_results[1].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m1)))
                    delay_results[1].append(to_scalar(coarse_delay))

                    detected_delays2 = receiver2.detect_preamble(preamble, sps, filter_span)
                    coarse_delay2 = int(round(detected_delays2[0]))
                    payload_start_idx2 = coarse_delay2 + (preamble_length * sps)

                    symbols_m1_2 = filt_signal2[payload_start_idx2 : payload_start_idx2 + num_data_symbols * sps : sps]
                    ber_results_2[1].append(get_BER(original_bits_flat, symbols_to_bits(symbols_m1_2)))
                    delay_results_2[1].append(to_scalar(coarse_delay2))

                    detected_delays3 = receiver3.detect_preamble(preamble, sps, filter_span)
                    coarse_delay3 = int(round(detected_delays3[0]))
                    payload_start_idx3 = coarse_delay3 + (preamble_length * sps)

                    symbols_m1_3 = filt_signal3[payload_start_idx3 : payload_start_idx3 + num_data_symbols * sps : sps]

                    detected_delays4 = receiver4.detect_preamble(preamble, sps, filter_span)
                    coarse_delay4 = int(round(detected_delays4[0]))
                    payload_start_idx4 = coarse_delay4 + (preamble_length * sps)

                    symbols_m1_4 = filt_signal4[payload_start_idx4 : payload_start_idx4 + num_data_symbols * sps : sps]

                    DOA_estimation2 = esprit_1d(symbols_m1, symbols_m1_2, 0)
                    estimations['ESPRIT'].append(np.abs(cfg.DOA - DOA_estimation2))

                    DOA_estimation3 = music_1d(symbols_m1, symbols_m1_2, 0, cfg.PREAMBLE_LENGTH, cfg.SPS)
                    estimations['MUSIC'].append(np.abs(cfg.DOA-DOA_estimation3))

                    DOA_estimation4 = esprit_4ant([symbols_m1, symbols_m1_2, symbols_m1_3, symbols_m1_4], cfg.ANTENNAS_DISTANCE[0], cfg.SPEED_OF_LIGHT / cfg.CARRIER_FREQ)
                    estimations['ESPRIT_4ANT'].append(np.abs(cfg.DOA - DOA_estimation4))

                    X = np.vstack([
                        symbols_m1,
                        symbols_m1_2,
                        symbols_m1_3,
                        symbols_m1_4
                    ])
                    R = covariance(X)
                    DOA_estimation5 = mvdr_4ant(R, 5e-2)
                    estimations['mvdr_4ant'].append(np.abs(cfg.DOA - DOA_estimation5))

                    DOA_estimation6 = music_4ant(R, 5e-2)
                    estimations['music_4ant'].append(np.abs(cfg.DOA - DOA_estimation6))

                    DOA_estimation7 = conjugate_multiplication_2ant(symbols_m1, symbols_m1_2)
                    estimations['conjugate_2ant'].append(np.abs(cfg.DOA - DOA_estimation7))

                    DOA_estimation8 = conjugate_multiplication_4ant(symbols_m1, symbols_m1_2, symbols_m1_3, symbols_m1_4)
                    estimations['conjugate_4ant'].append(np.abs(cfg.DOA - DOA_estimation8))

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
                    phi = (2 * np.pi * cfg.ANTENNAS_DISTANCE[0] * np.sin(np.deg2rad(DOA_estimation8)) / cfg.wavelength)
                    rx2 = np.array(rx2) * np.exp(-1j * phi)
                    rx3 = np.array(rx3) * np.exp(-1j * 2 * phi)
                    rx4 = np.array(rx4) * np.exp(-1j * 3 * phi)
                    rx_combined = (rx1 + rx2) / 2

                    # 5. Pass to your receiver pipeline
                    receiver_combined = Client_Rx([rx_combined], is_qpsk=True)
                    filt_signal_combined = get_filt_signal(receiver_combined)

                    # Extract data symbols using your existing payload start index
                    symbols_combined = filt_signal_combined[payload_start_idx : payload_start_idx + num_data_symbols * sps : sps]

                    ber_results[4].append(get_BER(original_bits_flat, symbols_to_bits(symbols_combined)))

                    
                    rx_combined_4ant = (rx1 + rx2 + rx3 + rx4) / 4

                    # 5. Pass to your receiver pipeline
                    receiver_combined_4ant = Client_Rx([rx_combined_4ant], is_qpsk=True)
                    filt_signal_combined_4ant = get_filt_signal(receiver_combined_4ant)

                    # Extract data symbols using your existing payload start index
                    symbols_combined_4ant = filt_signal_combined_4ant[payload_start_idx : payload_start_idx + num_data_symbols * sps : sps]

                    ber_results[5].append(get_BER(original_bits_flat, symbols_to_bits(symbols_combined_4ant)))
                    


                for method in DOA_methods.keys():
                    for i in range(len(SNR)):
                        DOA_methods[method][i] += estimations[method][i]
            for method in DOA_methods.keys():
                for i in range(len(SNR)):
                    DOA_methods[method][i] /= cfg.ITERATIONS
            plot_absolute_errors_comparison(DOA_methods)

            # Print console layout summary comparison
            print(f"\nSummary of BER results (%) for Corrected Delay = {delay}:")
            print(f"{'SNR (dB)':<10}{'Method 1':<12}{'Method 2':<12}{'Method 3':<12}{'Method 4':<12}{'Method 5':<12}{'Method 6':<12}")
            print("-" * 82)
            for i, snr_val in enumerate(SNR):
                print(f"{snr_val:<10}"
                      f"{ber_results[1][i]*100:<12.2f}"
                      f"{ber_results[2][i]*100:<12.2f}"
                      f"{ber_results[3][i]*100:<12.2f}"
                      f"{ber_results[4][i]*100:<12.2f}")
            
            # Map tracking datasets over to plotting modules (Legacy Plots)
            matrix_ber = np.array([ber_results[1], ber_results[2], ber_results[3], ber_results[4], ber_results[5]])
            matrix_delay = np.array([delay_results[1], delay_results[2], delay_results[3]])
            labels = ["Integer Correlation 1ant (M1)", "Parabolic Interp 1ant (M2)", "ML Grid Search 1ant (M3)", "EGC_2ant", "EGC_4ant"]
            
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
