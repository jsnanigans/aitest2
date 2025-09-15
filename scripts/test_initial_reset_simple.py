#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import toml
from src.reset_manager import ResetManager, ResetType

def test_initial_reset_config():
    """Test that initial reset config is properly loaded and used."""
    
    print("\n=== Testing Initial Reset Configuration ===\n")
    
    # Load config
    config = toml.load("config.toml")
    
    # Check initial reset config
    initial_config = config.get('kalman', {}).get('reset', {}).get('initial', {})
    print("Initial Reset Configuration from config.toml:")
    for key, value in initial_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Test getting parameters for initial reset
    params = ResetManager.get_reset_parameters(ResetType.INITIAL, config)
    print("Parameters returned by ResetManager for INITIAL reset:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Compare with defaults
    print("Checking if config values are being used:")
    print(f"  weight_boost_factor: config={initial_config.get('weight_boost_factor', 'NOT SET')}, actual={params['weight_boost_factor']}")
    print(f"  trend_boost_factor: config={initial_config.get('trend_boost_factor', 'NOT SET')}, actual={params['trend_boost_factor']}")
    print(f"  decay_rate: config={initial_config.get('decay_rate', 'NOT SET')}, actual={params['decay_rate']}")
    print(f"  warmup_measurements: config={initial_config.get('warmup_measurements', 'NOT SET')}, actual={params['warmup_measurements']}")
    print(f"  adaptive_days: config={initial_config.get('adaptive_days', 'NOT SET')}, actual={params['adaptive_days']}")
    print()
    
    # Test reset trigger for initial state
    print("Testing reset trigger for initial state (no kalman_params):")
    empty_state = {}
    reset_type = ResetManager.should_trigger_reset(
        empty_state, 
        weight=180.0, 
        timestamp=None,  # Will be handled
        source="scale",
        config=config
    )
    print(f"  Reset type for empty state: {reset_type.value if reset_type else 'None'}")
    
    # Test with state that has no kalman_params
    state_no_kalman = {'last_timestamp': '2024-01-01', 'last_weight': 180.0}
    reset_type = ResetManager.should_trigger_reset(
        state_no_kalman,
        weight=180.0,
        timestamp=None,
        source="scale", 
        config=config
    )
    print(f"  Reset type for state without kalman_params: {reset_type.value if reset_type else 'None'}")

if __name__ == "__main__":
    test_initial_reset_config()
