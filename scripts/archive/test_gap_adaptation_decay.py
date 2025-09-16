#!/usr/bin/env python3
"""Test script for gap adaptation decay."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import numpy as np
from src.processor import process_measurement
from src.database import get_state_db

def create_test_config():
    """Create test configuration with gap handling enabled."""
    return {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.0160,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.490,
            'reset_gap_days': 30,
            'questionnaire_reset_days': 10,
            'gap_handling': {
                'enabled': True,
                'gap_threshold_days': 10,
                'warmup_size': 3,
                'max_warmup_days': 7,
                'gap_variance_multiplier': 2.0,
                'trend_variance_multiplier': 20.0,
                'adaptation_decay_rate': 5.0
            }
        },
        'processing': {
            'extreme_threshold': 0.20
        },
        'adaptive_noise': {
            'enabled': True,
            'default_multiplier': 1.5
        },
        'quality_scoring': {
            'enabled': False
        }
    }

def test_adaptation_decay():
    """Test that adaptation decay completes after enough measurements."""
    print("Testing Adaptation Decay Completion")
    print("=" * 50)
    
    db = get_state_db()
    config = create_test_config()
    user_id = "test_decay_user"
    
    db.delete_state(user_id)
    
    print("\n1. Creating initial measurements...")
    base_weight = 85.0
    base_time = datetime(2024, 1, 1)
    
    for i in range(5):
        weight = base_weight + np.random.normal(0, 0.2)
        timestamp = base_time + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
    
    print(f"   Initial state established")
    
    print("\n2. Creating 20-day gap...")
    gap_end = base_time + timedelta(days=25)
    
    print("\n3. Processing post-gap measurements to test decay...")
    adaptation_active = True
    measurements_after_complete = 0
    
    for i in range(20):
        weight = base_weight - 2.0 + np.random.normal(0, 0.2)
        timestamp = gap_end + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        stage = result.get('stage', 'unknown')
        gap_adaptation = result.get('gap_adaptation', {})
        
        if stage == 'gap_buffer_collecting':
            print(f"   Measurement {i}: Buffer collecting")
        elif stage == 'gap_buffer_complete':
            print(f"   Measurement {i}: Buffer complete, adaptation started")
        else:
            if gap_adaptation.get('active'):
                measurements_since = gap_adaptation.get('measurements_since_gap', 0)
                gap_factor = gap_adaptation.get('gap_factor', 0)
                decay_factor = np.exp(-measurements_since / 5.0)
                print(f"   Measurement {i}: Adaptation active (since_gap={measurements_since}, decay={decay_factor:.3f})")
            else:
                if adaptation_active:
                    print(f"   Measurement {i}: Adaptation COMPLETE!")
                    adaptation_active = False
                measurements_after_complete += 1
    
    print(f"\n4. Results:")
    print(f"   Adaptation completed: {'✓' if not adaptation_active else '✗'}")
    if not adaptation_active:
        print(f"   Measurements after completion: {measurements_after_complete}")
    
    state = db.get_state(user_id)
    if state and state.get('kalman_params'):
        trans_cov = state['kalman_params']['transition_covariance']
        print(f"\n   Final transition covariances:")
        print(f"   Weight: {trans_cov[0][0]:.6f} (should be ~0.0160)")
        print(f"   Trend:  {trans_cov[1][1]:.6f} (should be ~0.0001)")
    
    print("\n" + "=" * 50)
    print("Adaptation Decay Test Complete")

def test_no_gap_scenario():
    """Test that gap handling doesn't interfere with normal processing."""
    print("\n\nTesting Normal Processing (No Gap)")
    print("=" * 50)
    
    db = get_state_db()
    config = create_test_config()
    user_id = "test_normal_user"
    
    db.delete_state(user_id)
    
    print("\n1. Processing continuous measurements (no gaps)...")
    base_weight = 75.0
    base_time = datetime(2024, 1, 1)
    
    for i in range(20):
        weight = base_weight + 0.05 * i + np.random.normal(0, 0.2)
        timestamp = base_time + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        if i % 5 == 0:
            stage = result.get('stage', 'unknown')
            gap_buffer = result.get('gap_buffer_size')
            gap_adaptation = result.get('gap_adaptation')
            
            print(f"   Day {i}: Stage={stage}")
            if gap_buffer is not None:
                print(f"      ✗ Unexpected gap buffer!")
            if gap_adaptation is not None:
                print(f"      ✗ Unexpected gap adaptation!")
    
    state = db.get_state(user_id)
    if state:
        has_gap_buffer = state.get('gap_buffer') is not None
        has_gap_adaptation = state.get('gap_adaptation') is not None
        
        print(f"\n2. Final state check:")
        print(f"   No gap_buffer in state: {'✓' if not has_gap_buffer else '✗'}")
        print(f"   No gap_adaptation in state: {'✓' if not has_gap_adaptation else '✗'}")
    
    print("\n" + "=" * 50)
    print("Normal Processing Test Complete")

if __name__ == "__main__":
    test_adaptation_decay()
    test_no_gap_scenario()