"""Quick test to verify all module imports work."""
try:
    import config
    import utilities
    import channel
    import signal_processing
    import RRC_Implementation
    import plots
    import client_Tx
    import client_Rx
    print("✓ SUCCESS: All imports successful!")
except ImportError as e:
    print(f"✗ ERROR: {e}")
    exit(1)
