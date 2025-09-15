#!/usr/bin/env python3
"""
Test the adaptive reset behavior with a scenario similar to the problematic user.
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_measurement
from src.database import get_state_db

def test_adaptive_reset():
    """Test adaptive behavior after reset."""
    
    db = get_state_db()
    user_id = "test_adaptive_user"
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
    
    print("Simulating problematic scenario:")
    print("=" * 50)
    
    # Phase 1: Establish baseline at 120kg
    print("\nPhase 1: Baseline at 120kg")
    base_date = datetime(2024, 1, 1)
    for i in range(5):
        timestamp = base_date + timedelta(days=i)
        weight = 120.0 + (i * 0.1)  # Small variations around 120
        result = process_measurement(
            user_id, weight, timestamp, "test-source", config
        )
        print(f"  Day {i}: {weight:.1f}kg - Accepted: {result['accepted']}")
    
    # Phase 2: 35-day gap (triggers reset)
    print("\nPhase 2: 35-day gap (triggers reset)")
    
    # Phase 3: New measurements at 107kg (like in the image)
    print("\nPhase 3: Measurements at 107kg after reset")
    new_base = base_date + timedelta(days=40)
    
    for i in range(10):
        timestamp = new_base + timedelta(days=i*2)  # Every 2 days
        weight = 107.0 + (i * 0.1)  # Around 107kg
        result = process_measurement(
            user_id, weight, timestamp, "test-source", config
        )
        
        # Check if it was accepted and what the Kalman prediction was
        accepted = result['accepted']
        filtered = result.get('filtered_weight', 0)
        confidence = result.get('confidence', 0)
        
        status = "✓ Accepted" if accepted else f"✗ Rejected ({result.get('reason', 'unknown')})"
        print(f"  Day {40+i*2}: {weight:.1f}kg → Kalman: {filtered:.1f}kg - {status}")
        
        if not accepted and 'deviation' in result.get('reason', '').lower():
            deviation = abs(weight - filtered) / filtered * 100
            print(f"    Deviation: {deviation:.1f}%")
    
    # Check final state
    state = db.get_state(user_id)
    if state and state.get('kalman_params'):
        params = state['kalman_params']
        print(f"\nFinal Kalman parameters:")
        print(f"  Weight covariance: {params['transition_covariance'][0][0]:.4f}")
        print(f"  Trend covariance: {params['transition_covariance'][1][1]:.6f}")

if __name__ == "__main__":
    test_adaptive_reset()
