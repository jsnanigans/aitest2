#!/usr/bin/env python3
"""
Test adaptive parameters for initial measurements.
Simulates user 01672f42-568b-4d49-abbc-eee60d87ccb2 scenario.
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_measurement
from src.database import get_state_db

def test_initial_adaptive():
    """Test that initial measurements use adaptive parameters."""
    
    db = get_state_db()
    user_id = "test_initial_user"
    db.delete_state(user_id)
    
    config = {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.0160,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.490,
            'reset': {
                'enabled': True,
                'gap_threshold_days': 30
            }
        },
        'processing': {
            'extreme_threshold': 0.20
        },
        'adaptive_noise': {
            'enabled': False
        },
        'quality_scoring': {
            'enabled': False
        }
    }
    
    print("Testing Initial Adaptive Parameters")
    print("=" * 60)
    print("Simulating user 01672f42-568b-4d49-abbc-eee60d87ccb2:")
    print("First measurement at 120kg, but actual weight is ~107kg\n")
    
    # First measurement at 120kg (like in the image)
    base_date = datetime(2025, 4, 20)  # Match the date in image
    result = process_measurement(
        user_id, 120.0, base_date, "initial-questionnaire", config
    )
    print(f"Initial: 120.0kg - Accepted: {result['accepted']}")
    
    # Check the Kalman parameters
    state = db.get_state(user_id)
    if state and state.get('kalman_params'):
        params = state['kalman_params']
        weight_cov = params['transition_covariance'][0][0]
        trend_cov = params['transition_covariance'][1][1]
        print(f"\nKalman parameters after initialization:")
        print(f"  Weight covariance: {weight_cov:.4f} (normal: 0.0160)")
        print(f"  Trend covariance: {trend_cov:.6f} (normal: 0.0001)")
        
        if weight_cov > 0.1:  # Should be ~0.5 with adaptive
            print("  ✓ Using ADAPTIVE parameters (flexible)")
        else:
            print("  ✗ Using NORMAL parameters (rigid)")
    
    # Now add measurements at actual weight (~107kg)
    print("\nSubsequent measurements at actual weight (~107kg):")
    weights = [108.0, 107.5, 107.0, 106.5, 107.2, 107.8, 106.9, 107.1, 105.5, 105.8]
    
    accepted_count = 0
    rejected_count = 0
    
    for i, weight in enumerate(weights):
        timestamp = base_date + timedelta(days=7*(i+1))  # Weekly measurements
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        accepted = result['accepted']
        if accepted:
            accepted_count += 1
            status = "✓ Accepted"
        else:
            rejected_count += 1
            reason = result.get('reason', 'unknown')
            status = f"✗ Rejected: {reason}"
        
        filtered = result.get('filtered_weight', 0)
        print(f"  Week {i+1}: {weight:.1f}kg → Kalman: {filtered:.1f}kg - {status}")
    
    print(f"\n" + "=" * 60)
    print(f"Results: {accepted_count} accepted, {rejected_count} rejected")
    
    if accepted_count >= 8:  # Should accept most measurements
        print("✓ SUCCESS: Adaptive initial parameters working!")
        print("  The Kalman filter quickly adapted to the actual weight level.")
    else:
        print("✗ ISSUE: Still rejecting too many valid measurements")
        print("  The Kalman filter may still be too rigid.")
    
    # Check final state
    final_state = db.get_state(user_id)
    if final_state and final_state.get('kalman_params'):
        params = final_state['kalman_params']
        print(f"\nFinal Kalman parameters:")
        print(f"  Weight covariance: {params['transition_covariance'][0][0]:.4f}")
        print(f"  Trend covariance: {params['transition_covariance'][1][1]:.6f}")

if __name__ == "__main__":
    test_initial_adaptive()
