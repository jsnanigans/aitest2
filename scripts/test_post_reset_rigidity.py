#!/usr/bin/env python3
"""Test and fix Kalman filter rigidity after reset."""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kalman import KalmanFilterManager
from src.constants import KALMAN_DEFAULTS

def test_post_reset_behavior():
    """Test Kalman filter behavior after reset."""
    
    # Test data: weight drops from 120kg to 105kg after gap
    initial_weight = 120.0
    actual_weights = [105.0, 106.0, 105.5, 107.0, 105.0, 106.5]
    
    # Current configuration (too rigid)
    kalman_config = {
        'initial_variance': KALMAN_DEFAULTS['initial_variance'],  # 0.361
        'transition_covariance_weight': KALMAN_DEFAULTS['transition_covariance_weight'],  # 0.0160
        'transition_covariance_trend': KALMAN_DEFAULTS['transition_covariance_trend'],  # 0.0001
        'observation_covariance': KALMAN_DEFAULTS['observation_covariance'],  # 3.490
    }
    
    print("Current Configuration (Rigid):")
    print(f"  initial_variance: {kalman_config['initial_variance']}")
    print(f"  transition_covariance_weight: {kalman_config['transition_covariance_weight']}")
    print(f"  transition_covariance_trend: {kalman_config['transition_covariance_trend']}")
    print(f"  observation_covariance: {kalman_config['observation_covariance']}")
    print()
    
    # Initialize after reset with first measurement
    timestamp = datetime.now()
    state = KalmanFilterManager.initialize_immediate(
        actual_weights[0], timestamp, kalman_config
    )
    
    print(f"After reset initialization with {actual_weights[0]}kg:")
    print(f"  Initial state: weight={state['last_state'][0][0]:.1f}, trend={state['last_state'][0][1]:.4f}")
    print()
    
    # Process subsequent measurements
    print("Processing measurements with current config:")
    for i, weight in enumerate(actual_weights[1:], 1):
        timestamp += timedelta(days=1)
        state = KalmanFilterManager.update_state(
            state, weight, timestamp, 'test', {}, kalman_config['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(
            state, weight, timestamp, 'test', True, kalman_config['observation_covariance']
        )
        
        print(f"  Day {i}: raw={weight:.1f}, filtered={result['filtered_weight']:.1f}, "
              f"trend={result['trend']:.4f}, innovation={result['innovation']:.1f}, "
              f"norm_innovation={result['normalized_innovation']:.2f}")
    
    print("\n" + "="*80 + "\n")
    
    # Test with more adaptive configuration for post-reset period
    adaptive_config = {
        'initial_variance': 5.0,  # Higher initial uncertainty after reset
        'transition_covariance_weight': 0.5,  # Allow more weight variation
        'transition_covariance_trend': 0.01,  # Allow trend to adapt faster
        'observation_covariance': 2.0,  # Trust measurements more initially
    }
    
    print("Proposed Adaptive Configuration (Post-Reset):")
    print(f"  initial_variance: {adaptive_config['initial_variance']}")
    print(f"  transition_covariance_weight: {adaptive_config['transition_covariance_weight']}")
    print(f"  transition_covariance_trend: {adaptive_config['transition_covariance_trend']}")
    print(f"  observation_covariance: {adaptive_config['observation_covariance']}")
    print()
    
    # Re-initialize with adaptive config
    timestamp = datetime.now()
    state = KalmanFilterManager.initialize_immediate(
        actual_weights[0], timestamp, adaptive_config
    )
    
    print(f"After reset initialization with {actual_weights[0]}kg:")
    print(f"  Initial state: weight={state['last_state'][0][0]:.1f}, trend={state['last_state'][0][1]:.4f}")
    print()
    
    print("Processing measurements with adaptive config:")
    for i, weight in enumerate(actual_weights[1:], 1):
        timestamp += timedelta(days=1)
        state = KalmanFilterManager.update_state(
            state, weight, timestamp, 'test', {}, adaptive_config['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(
            state, weight, timestamp, 'test', True, adaptive_config['observation_covariance']
        )
        
        print(f"  Day {i}: raw={weight:.1f}, filtered={result['filtered_weight']:.1f}, "
              f"trend={result['trend']:.4f}, innovation={result['innovation']:.1f}, "
              f"norm_innovation={result['normalized_innovation']:.2f}")
    
    print("\n" + "="*80 + "\n")
    
    # Test with gradually tightening configuration
    print("Proposed Solution: Adaptive decay over time")
    print("Start with adaptive config, gradually tighten to normal config over 7 days\n")
    
    timestamp = datetime.now()
    reset_timestamp = timestamp
    state = KalmanFilterManager.initialize_immediate(
        actual_weights[0], timestamp, adaptive_config
    )
    
    print("Processing with adaptive decay:")
    for i, weight in enumerate(actual_weights[1:], 1):
        timestamp += timedelta(days=1)
        
        # Calculate decay factor (0 to 1 over 7 days)
        days_since_reset = (timestamp - reset_timestamp).days
        decay_factor = min(1.0, days_since_reset / 7.0)
        
        # Interpolate between adaptive and normal configs
        current_trans_weight = adaptive_config['transition_covariance_weight'] * (1 - decay_factor) + \
                               kalman_config['transition_covariance_weight'] * decay_factor
        current_trans_trend = adaptive_config['transition_covariance_trend'] * (1 - decay_factor) + \
                             kalman_config['transition_covariance_trend'] * decay_factor
        current_obs_cov = adaptive_config['observation_covariance'] * (1 - decay_factor) + \
                         kalman_config['observation_covariance'] * decay_factor
        
        # Update transition covariance in state
        state['kalman_params']['transition_covariance'] = [
            [current_trans_weight, 0],
            [0, current_trans_trend]
        ]
        
        state = KalmanFilterManager.update_state(
            state, weight, timestamp, 'test', {}, current_obs_cov
        )
        
        result = KalmanFilterManager.create_result(
            state, weight, timestamp, 'test', True, current_obs_cov
        )
        
        print(f"  Day {i} (decay={decay_factor:.2f}): raw={weight:.1f}, filtered={result['filtered_weight']:.1f}, "
              f"trend={result['trend']:.4f}, innovation={result['innovation']:.1f}, "
              f"norm_innovation={result['normalized_innovation']:.2f}")
        print(f"    Config: trans_weight={current_trans_weight:.4f}, trans_trend={current_trans_trend:.5f}, obs={current_obs_cov:.2f}")

if __name__ == "__main__":
    test_post_reset_behavior()