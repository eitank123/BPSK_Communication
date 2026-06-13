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
np.random.seed(60)


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
    H_est = []
    
    for beta in betas:
        for delay in delays:
            for antenna in range(1, 3):  # Antenna 1 and Antenna 2
                # Run the sweep
                results = engine.run_snr_sweep(beta, delay, sps, antenna)
                H_est.append(results['H_est'])  # Store H_est
                
                # Convert to arrays for plotting
                matrix_ber = np.array([
                    results['ber'][i] for i in range(1, 7)
                ])
                matrix_delay = np.array([
                    results['delay'][i] for i in range(1, 7)
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
                        for est_delay in results['delay'][method_id]
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
            doa_vs_k = []

            lam = cfg.SPEED_OF_LIGHT / cfg.CARRIER_FREQ  # make sure this exists
            for k_idx, k_val in enumerate(cfg.RICIAN_K_FACTORS):


                # assume 2 antennas: H_k[0], H_k[1]
                H1 = np.array(H_est[0][k_idx+1])  # H_est from Antenna 1 for current K-factor
                H2 = np.array(H_est[1][k_idx+1])

                # if vector -> average over subcarriers / taps
                phi = np.angle(np.mean(H2 * np.conj(H1)))

                # DOA estimate
                sin_theta = (phi * lam) / (2 * np.pi * cfg.ANTENNAS_DISTANCE)

                sin_theta = np.clip(sin_theta, -1, 1)
                theta = np.degrees(np.arcsin(sin_theta))

                doa_vs_k.append(abs(theta))
            plt.figure(figsize=(10,5))

            plt.plot(
                cfg.RICIAN_K_FACTORS,
                doa_vs_k,
                marker='o',
                label='Estimated DOA'
            )

            plt.axhline(cfg.DOA, color='r', linestyle='--', label='True DOA')

            plt.title('DOA vs Rician K-Factor')
            plt.xlabel('Rician K-Factor (dB)')
            plt.ylabel('DOA (degrees)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()


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
