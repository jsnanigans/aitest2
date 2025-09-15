#!/usr/bin/env python3
"""Compare rigid vs adaptive Kalman behavior after reset."""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kalman import KalmanFilterManager
from src.kalman_adaptive import get_adaptive_kalman_params, get_reset_timestamp
from src.constants import KALMAN_DEFAULTS

def simulate_post_reset_scenario():
    """Simulate the exact scenario from the image: 120kg -> gap -> 105-107kg measurements."""
    
    print("="*80)
    print("Scenario: Weight was 120kg, 35-day gap, then measurements around 105-107kg")
    print("="*80)
    
    # Test data matching the image scenario
    pre_gap_weight = 120.0
    post_gap_weights = [105.0, 106.0, 105.5, 107.0, 105.0, 106.5, 105.8, 106.2, 107.5, 105.3]
    
    # Standard (rigid) configuration
    rigid_config = {
        'initial_variance': KALMAN_DEFAULTS['initial_variance'],
        'transition_covariance_weight': KALMAN_DEFAULTS['transition_covariance_weight'],
        'transition_covariance_trend': KALMAN_DEFAULTS['transition_covariance_trend'],
        'observation_covariance': KALMAN_DEFAULTS['observation_covariance'],
    }
    
    print("\n1. RIGID CONFIGURATION (Current Problem)")
    print("-" * 40)
    print(f"Config: trans_weight={rigid_config['transition_covariance_weight']:.4f}, "
          f"trans_trend={rigid_config['transition_covariance_trend']:.5f}, "
          f"obs={rigid_config['observation_covariance']:.2f}")
    print()
    
    # Initialize with first post-gap measurement
    timestamp = datetime.now()
    state_rigid = KalmanFilterManager.initialize_immediate(
        post_gap_weights[0], timestamp, rigid_config
    )
    
    print(f"Reset with {post_gap_weights[0]}kg (was {pre_gap_weight}kg before gap)")
    print()
    
    rejections_rigid = 0
    for i, weight in enumerate(post_gap_weights[1:], 1):
        timestamp += timedelta(days=1)
        state_rigid = KalmanFilterManager.update_state(
            state_rigid, weight, timestamp, 'test', {}, rigid_config['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(
            state_rigid, weight, timestamp, 'test', True, rigid_config['observation_covariance']
        )
        
        # Check if measurement would be rejected (normalized innovation > 3)
        rejected = result['normalized_innovation'] > 3.0
        if rejected:
            rejections_rigid += 1
        
        status = "REJECTED" if rejected else "accepted"
        print(f"Day {i:2}: raw={weight:6.1f}, filtered={result['filtered_weight']:6.1f}, "
              f"trend={result['trend']:7.4f}, norm_innov={result['normalized_innovation']:4.2f} [{status}]")
    
    print(f"\nTotal rejections: {rejections_rigid}/{len(post_gap_weights)-1}")
    
    print("\n2. ADAPTIVE CONFIGURATION (Proposed Solution)")
    print("-" * 40)
    
    # Simulate with adaptive decay
    timestamp = datetime.now()
    reset_timestamp = timestamp
    
    # Start with adaptive config
    adaptive_initial = {
        'initial_variance': 5.0,
        'transition_covariance_weight': 0.5,
        'transition_covariance_trend': 0.01,
        'observation_covariance': 2.0,
    }
    
    state_adaptive = KalmanFilterManager.initialize_immediate(
        post_gap_weights[0], timestamp, adaptive_initial
    )
    
    print(f"Reset with {post_gap_weights[0]}kg (was {pre_gap_weight}kg before gap)")
    print("Using adaptive parameters that decay over 7 days")
    print()
    
    rejections_adaptive = 0
    for i, weight in enumerate(post_gap_weights[1:], 1):
        timestamp += timedelta(days=1)
        
        # Get adaptive config for current day
        current_config = get_adaptive_kalman_params(
            reset_timestamp, timestamp, rigid_config, adaptive_days=7
        )
        
        # Update state's kalman_params with current adaptive values
        state_adaptive['kalman_params']['transition_covariance'] = [
            [current_config['transition_covariance_weight'], 0],
            [0, current_config['transition_covariance_trend']]
        ]
        
        state_adaptive = KalmanFilterManager.update_state(
            state_adaptive, weight, timestamp, 'test', {}, 
            current_config['observation_covariance']
        )
        
        result = KalmanFilterManager.create_result(
            state_adaptive, weight, timestamp, 'test', True, 
            current_config['observation_covariance']
        )
        
        # Check if measurement would be rejected
        rejected = result['normalized_innovation'] > 3.0
        if rejected:
            rejections_adaptive += 1
        
        status = "REJECTED" if rejected else "accepted"
        days_since = (timestamp - reset_timestamp).days
        decay = min(1.0, days_since / 7.0)
        
        print(f"Day {i:2}: raw={weight:6.1f}, filtered={result['filtered_weight']:6.1f}, "
              f"trend={result['trend']:7.4f}, norm_innov={result['normalized_innovation']:4.2f} "
              f"[{status}] (decay={decay:.1f})")
    
    print(f"\nTotal rejections: {rejections_adaptive}/{len(post_gap_weights)-1}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Rigid config:    {rejections_rigid} rejections  (filter too slow to adapt)")
    print(f"Adaptive config: {rejections_adaptive} rejections  (filter adapts quickly after reset)")
    print("\nThe adaptive approach allows the Kalman filter to quickly adjust to the new")
    print("weight level after a long gap, preventing false rejections of valid measurements.")

if __name__ == "__main__":
    simulate_post_reset_scenario()