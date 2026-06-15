"""
Main simulation runner for BPSK Communication System.

Orchestrates comprehensive timing synchronization analysis with multiple
recovery methods across various channel conditions and configurations.

This refactored version provides:
- Clear, modular structure
- Configuration centralization (config.py)
- Utility functions organization (utilities.py)
- Channel operations abstraction (channel.py)
- Signal processing utilities (signal_processing.py)
- Simulation orchestration (simulator.py)
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator import SimulationEngine
import plots
import config as cfg
from utilities import calculate_ber, to_scalar, symbols_to_bits, calculate_range_error

# Set random seed for reproducibility
np.random.seed(42)


def run_phase1_snr_sweep(engine, betas, delays, sps):
    """
    Phase 1: SNR/Rician K-Factor Sweep.
    
    Evaluates all 6 timing recovery methods across varying channel conditions
    at fixed SPS and delay configuration.
    
    Parameters
    ----------
    engine : SimulationEngine
        Initialized simulation engine
    betas : list
        RRC rolloff factors to test
    delays : list
        Delay values to test
    sps : int
        Samples per symbol (fixed for phase 1)
    """
    print(f"\n{'='*80}")
    print(f"PHASE 1: SNR SWEEP (Fixed SPS={sps})")
    print(f"{'='*80}\n")
    
    # Storage for plotting
    range_error_vs_snr_final = {i: [] for i in range(1, 7)}
    H_est_ant1 = None
    H_est_ant2 = None
    ber = None
    sum_delay = None

    lam = cfg.SPEED_OF_LIGHT / cfg.CARRIER_FREQ  # make sure this exists
    # Pre-allocate a 2D matrix for DOA: shape (num_methods, num_k_factors)
    # 1. Pre-allocate the master matrix: shape (6, num_k)
    num_methods = len(cfg.METHOD_LABELS)      # 6
    num_k_factors = len(cfg.RICIAN_K_FACTORS) # 6
    matrix_doa = np.zeros((num_methods, num_k_factors))
    
    iterations=5
    for beta in betas:
        for delay in delays:
            for i in range(iterations):
                print(f"Iteration {i+1}/{iterations} - Beta: {beta}, Delay: {delay}")
                for antenna in range(1, 3):  # Antenna 1 and Antenna 2
                    # Run the sweep
                    results = engine.run_snr_sweep(beta, delay, sps, antenna)
                    if antenna == 1:
                        if ber is None:
                            ber = results['ber'].copy()
                        else:
                            for k in ber:
                                ber[k] = [a + b for a, b in zip(ber[k], results['ber'][k])]

                        if sum_delay is None:
                            # Convert lists to NumPy arrays on the fly and compute the absolute difference
                            sum_delay = {k: np.abs(np.array(v) - delay) for k, v in results['delay'].items()}
                        else:
                            # Update the existing NumPy arrays in sum_delay by adding the new absolute differences
                            for k in sum_delay:
                                sum_delay[k] += np.abs(np.array(results['delay'][k]) - delay)
                        if H_est_ant1 is None:
                            H_est_ant1 = {k: v.copy() for k, v in results['H_est'].items()}
                        else:
                            for k in H_est_ant1:
                                H_est_ant1[k] = [
                                    a + b
                                    for a, b in zip(H_est_ant1[k], results['H_est'][k])
                                ]
                    else:
                        if H_est_ant2 is None:
                            H_est_ant2 = {k: v.copy() for k, v in results['H_est'].items()}
                        else:
                            for k in H_est_ant2:
                                H_est_ant2[k] = [
                                    a + b
                                    for a, b in zip(H_est_ant2[k], results['H_est'][k])
                                ]
                    
                    # Convert to arrays for plotting
            ber = {
                k: [x / iterations for x in v]
                for k, v in ber.items()
            }
            delay_error_avg = {
                k: [x / iterations for x in v]
                for k, v in sum_delay.items()
            }
            avg_H_est_ant1 = {
                k: [x / iterations for x in v]
                for k, v in H_est_ant1.items()
            }

            avg_H_est_ant2 = {
                k: [x / iterations for x in v]
                for k, v in H_est_ant2.items()
            }
            matrix_ber = np.array([
                ber[i] for i in range(1, 7)
            ])
            matrix_delay = np.array(
                [delay_error_avg[i]
                for i in range(1, 7)
            ])
            # Print summary table
            print(f"\nBER Results Summary (Delay={delay}):")
            print(f"{'K-Factor (dB)':<15}", end='')
            for i in range(1, 7):
                print(f"{'M'+str(i):<12}", end='')
            print()
            print("-" * 80)
            
            for k_idx, k_val in enumerate(cfg.RICIAN_K_FACTORS):
                print(f"{k_val:<15}", end='')
                for method_id in range(1, 7):
                    ber_val = results['ber'][method_id][k_idx]
                    print(f"{ber_val*100:<12.2f}", end='')
                print()
            
            # Calculate range errors
            for method_id in range(1, 7):
                range_errors = [
                    calculate_range_error(
                        est_delay, delay,
                        cfg.SAMPLING_FREQ,
                        cfg.SPEED_OF_LIGHT
                    )
                    for est_delay in delay_error_avg[method_id]
                ]
                range_error_vs_snr_final[method_id] = range_errors
                
            # Plot results
            if cfg.ENABLE_PLOTTING:
                plots.plot_ber_vs_k(
                    cfg.RICIAN_K_FACTORS,
                    matrix_ber,
                    cfg.TARGET_SNR,
                    series_labels=cfg.METHOD_LABELS
                )
                plots.plot_delay_tracking(
                    cfg.RICIAN_K_FACTORS,
                    matrix_delay,
                    cfg.METHOD_LABELS,
                    true_delay=delay
                )
                
                # Graph 4: Range Error vs K-Factor
                plt.figure(figsize=(12, 6))
                for method_id, style in cfg.METHOD_STYLES.items():
                    plt.plot(
                        cfg.RICIAN_K_FACTORS,
                        np.abs(range_error_vs_snr_final[method_id]),
                        color=style['color'],
                        marker=style['marker'],
                        linestyle=style['ls'],
                        label=style['label']
                    )
                plt.title(
                    f'Range Error vs. Rician K-Factor '
                    f'(SNR={cfg.TARGET_SNR} dB, SPS={sps})'
                )
                plt.xlabel('Rician K-Factor (dB)')
                plt.ylabel('Absolute Range Error (Meters)')
                plt.axhline(0, color='black', linestyle=':')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            # 2. Loop through both dimensions cleanly
            for k_idx, k_val in enumerate(cfg.RICIAN_K_FACTORS):
                for m_idx in range(num_methods):
                    
                    # Step A: Convert the specific method's nested list to a clean 2D NumPy array
                    # Shape becomes (num_k_factors, 256)
                    H1_method_matrix = np.array(avg_H_est_ant1[m_idx+1])  
                    H2_method_matrix = np.array(avg_H_est_ant2[m_idx+1])  

                    # Step B: Slice out ONLY the row corresponding to the current K-factor
                    # Shape becomes (256,)
                    H1 = H1_method_matrix[k_idx, :]
                    H2 = H2_method_matrix[k_idx, :]

                    # Step C: Compute spatial phase across the 256 subcarriers
                    phi = np.angle(np.mean(H2 * np.conj(H1)))

                    # Step D: Calculate DOA angle
                    sin_theta = (phi * lam) / (2 * np.pi * cfg.ANTENNAS_DISTANCE)
                    sin_theta = np.clip(sin_theta, -1.0, 1.0)
                    theta = np.degrees(np.arcsin(sin_theta))

                    # Step E: Store it perfectly at the intersection of this Method and this K-factor
                    matrix_doa[m_idx, k_idx] = np.abs(theta)
            
            plots.plot_doa_tracking(
            cfg.RICIAN_K_FACTORS,
            matrix_doa,
            cfg.METHOD_LABELS,
            true_doa=cfg.DOA
            )


def run_phase2_sps_sweep(engine, beta, delay, target_k_factor):
    """
    Phase 2: SPS Sweep.
    
    Evaluates all 6 timing recovery methods across varying samples per symbol
    at fixed SNR and K-factor.
    
    Parameters
    ----------
    engine : SimulationEngine
        Initialized simulation engine
    beta : float
        RRC rolloff factor
    delay : float
        Transmission delay
    target_k_factor : int
        Target Rician K-factor (dB)
    """
    print(f"\n{'='*80}")
    print(f"PHASE 2: SPS SWEEP (Fixed SNR={cfg.TARGET_SNR} dB, K={target_k_factor} dB)")
    print(f"{'='*80}\n")
    
    # Run sweep
    results = engine.run_sps_sweep(beta, delay, target_k_factor)
    
    # Convert to arrays
    matrix_ber = np.array([results['ber'][i] for i in range(1, 7)])
    matrix_delay = np.array([results['delay'][i] for i in range(1, 7)])
    sps_values = results['sps_values']
    
    # Print summary
    print(f"\nBER Results Summary (SNR={cfg.TARGET_SNR} dB, K={target_k_factor} dB):")
    print(f"{'SPS':<10}", end='')
    for i in range(1, 7):
        print(f"{'M'+str(i):<12}", end='')
    print()
    print("-" * 82)
    
    for sps_idx, sps_val in enumerate(sps_values):
        print(f"{sps_val:<10}", end='')
        for method_id in range(1, 7):
            ber_val = results['ber'][method_id][sps_idx]
            print(f"{ber_val*100:<12.2f}", end='')
        print()
    
    if cfg.ENABLE_PLOTTING:
        # Calculate range errors for each method and SPS
        range_errors = {i: [] for i in range(1, 7)}
        for method_id in range(1, 7):
            for sps_val in sps_values:
                sampling_freq = cfg.SAMPLE_RATE * sps_val
                for est_delay in results['delay'][method_id]:
                    r_error = calculate_range_error(
                        est_delay, delay,
                        sampling_freq,
                        cfg.SPEED_OF_LIGHT
                    )
                    range_errors[method_id].append(r_error)
                    break  # Just first value per SPS
        
        # Graph 5: Range Error vs SPS
        plt.figure(figsize=(12, 6))
        for method_id, style in cfg.METHOD_STYLES.items():
            method_range_errors = range_errors[method_id]
            plt.plot(
                sps_values,
                np.abs(method_range_errors),
                color=style['color'],
                marker=style['marker'],
                linestyle=style['ls'],
                label=style['label']
            )
        plt.title(
            f'Absolute Range Error vs. SPS '
            f'(SNR={cfg.TARGET_SNR} dB, K={target_k_factor} dB)'
        )
        plt.xlabel('Samples Per Symbol (SPS)')
        plt.ylabel('Absolute Range Error (Meters)')
        plt.xscale('log', base=2)
        plt.xticks(sps_values, labels=[str(s) for s in sps_values])
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Graph 7: BER vs SPS
        plt.figure(figsize=(12, 6))
        for method_id, style in cfg.METHOD_STYLES.items():
            clean_ber = [max(b, 1e-6) for b in results['ber'][method_id]]
            plt.semilogy(
                sps_values,
                clean_ber,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['ls'],
                label=style['label']
            )
        plt.title(
            f'BER vs. SPS '
            f'(SNR={cfg.TARGET_SNR} dB, K={target_k_factor} dB)'
        )
        plt.xlabel('Samples Per Symbol (SPS)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    """
    Main entry point for QPSK communication simulation.
    
    Orchestrates Phase 1 (SNR sweep) and Phase 2 (SPS sweep) analysis.
    """
    print("=" * 80)
    print("BPSK COMMUNICATION SYSTEM - TIMING SYNCHRONIZATION ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Samples Per Symbol (SPS): {cfg.SAMPLES_PER_SYMBOL}")
    print(f"  Number of Bits: {cfg.NUMBER_OF_BITS}")
    print(f"  RRC Rolloff Factor (Beta): {cfg.ROLLOFF_FACTOR}")
    print(f"  Filter Span: {cfg.FILTER_SPAN}")
    print(f"  Target SNR: {cfg.TARGET_SNR} dB")
    print(f"  Preamble Length: {cfg.PREAMBLE_LENGTH}")
    print(f"  CP Length: {cfg.CP_LENGTH}")
    print(f"  Data Block Size: {cfg.DATA_BLOCK_SIZE}")
    print()
    
    # Initialize simulation engine
    engine = SimulationEngine()
    engine.initialize_transmitter()
    engine.initialize_preamble()
    
    # =========================================================================
    # PHASE 1: SNR/K-FACTOR SWEEP
    # =========================================================================
    run_phase1_snr_sweep(
        engine,
        betas=[cfg.ROLLOFF_FACTOR],
        delays=cfg.ANTENNA1_DELAY_VALUES,
        sps=cfg.SAMPLES_PER_SYMBOL
    )
    
    # =========================================================================
    # PHASE 2: SPS SWEEP
    # =========================================================================
    run_phase2_sps_sweep(
        engine,
        beta=cfg.ROLLOFF_FACTOR,
        delay=cfg.ANTENNA1_DELAY_VALUES[0],
        target_k_factor=cfg.TARGET_K_FACTOR
    )
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
