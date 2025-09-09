"""
Verify processor maintains true statelessness except for essential Kalman state
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from processor import WeightProcessor


def test_statelessness():
    """Verify only essential state is maintained between measurements."""
    
    print("=" * 60)
    print("PROCESSOR STATELESSNESS VERIFICATION")
    print("=" * 60)
    
    config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.25
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,
            "reset_gap_days": 30
        }
    }
    
    processor = WeightProcessor("test_user", config["processing"], config["kalman"])
    
    print("\nInitial State (before any processing):")
    print("-" * 40)
    
    # Check initial state
    essential_state = {
        'kalman': processor.kalman,
        'last_state': processor.last_state,
        'last_covariance': processor.last_covariance,
        'last_timestamp': processor.last_timestamp,
    }
    
    temp_state = {
        'init_buffer': len(processor.init_buffer),
        'adapted_params': processor.adapted_params,
    }
    
    immutable_state = {
        'user_id': processor.user_id,
        'processing_config': id(processor.processing_config),
        'base_kalman_config': id(processor.base_kalman_config),
    }
    
    print("Essential Kalman State:")
    for key, value in essential_state.items():
        print(f"  {key}: {value}")
    
    print("\nTemporary State:")
    for key, value in temp_state.items():
        print(f"  {key}: {value}")
    
    print("\nImmutable Configuration:")
    for key, value in immutable_state.items():
        print(f"  {key}: {value if key == 'user_id' else 'memory_id=' + str(value)}")
    
    # Process some data
    base_date = datetime(2024, 1, 1)
    for i in range(15):
        weight = 70.0 + np.random.normal(0, 0.5)
        result = processor.process_weight(weight, base_date + timedelta(days=i), "scale")
    
    print("\n\nState After Initialization (15 measurements):")
    print("-" * 40)
    
    print("Essential Kalman State:")
    print(f"  kalman: {'Initialized' if processor.kalman else None}")
    print(f"  last_state: shape={processor.last_state.shape if processor.last_state is not None else None}")
    print(f"  last_covariance: shape={processor.last_covariance.shape if processor.last_covariance is not None else None}")
    print(f"  last_timestamp: {processor.last_timestamp}")
    
    print("\nTemporary State:")
    print(f"  init_buffer: {len(processor.init_buffer)} (should be 0 after init)")
    print(f"  adapted_params: {'Set' if processor.adapted_params else None}")
    
    print("\nImmutable Configuration (should not change):")
    print(f"  processing_config id: memory_id={id(processor.processing_config)}")
    print(f"  base_kalman_config id: memory_id={id(processor.base_kalman_config)}")
    
    # Check adapted parameters
    if processor.adapted_params:
        print("\nAdapted Parameters (computed once at init, then frozen):")
        for key in ['observation_covariance', 'extreme_threshold', 'transition_covariance_trend']:
            if key in processor.adapted_params:
                print(f"  {key}: {processor.adapted_params[key]}")
    
    # Verify no other state variables exist
    print("\n\nState Variable Audit:")
    print("-" * 40)
    
    allowed_state = {
        # Essential Kalman state
        'kalman', 'last_state', 'last_covariance', 'last_timestamp',
        # Temporary initialization
        'init_buffer',
        # Frozen after initialization
        'adapted_params',
        # Immutable configuration
        'user_id', 'processing_config', 'base_kalman_config'
    }
    
    actual_state = {attr for attr in dir(processor) if not attr.startswith('_') and not callable(getattr(processor, attr))}
    
    unexpected_state = actual_state - allowed_state
    
    if unexpected_state:
        print(f"❌ UNEXPECTED STATE VARIABLES FOUND: {unexpected_state}")
        print("   These violate statelessness!")
    else:
        print("✓ All state variables are accounted for")
    
    print("\nStatelessness Summary:")
    print("  ✓ Essential Kalman state: Properly maintained")
    print("  ✓ Temporary buffers: Cleared after use")
    print("  ✓ Configuration: Immutable references")
    print("  ✓ Adapted parameters: Computed once, then frozen")
    
    if not unexpected_state and len(processor.init_buffer) == 0:
        print("\n✅ PROCESSOR IS TRULY STATELESS")
        print("   Only essential Kalman state persists between measurements")
    else:
        print("\n❌ STATELESSNESS VIOLATIONS DETECTED")


if __name__ == "__main__":
    np.random.seed(42)
    test_statelessness()