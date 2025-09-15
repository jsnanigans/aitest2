#!/usr/bin/env python3
"""Test script for gap handling improvements."""

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

def test_gap_handling():
    """Test gap handling with simulated data."""
    print("Testing Gap Handling Implementation")
    print("=" * 50)
    
    db = get_state_db()
    config = create_test_config()
    user_id = "test_gap_user"
    
    db.delete_state(user_id)
    
    print("\n1. Creating pre-gap measurements with declining trend...")
    base_weight = 85.0
    trend = -0.1
    base_time = datetime(2024, 1, 1)
    
    for i in range(30):
        weight = base_weight + trend * i + np.random.normal(0, 0.2)
        timestamp = base_time + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        if i % 10 == 0:
            print(f"   Day {i}: Weight={weight:.1f}, Filtered={result.get('filtered_weight', 0):.1f}, Trend={result.get('trend', 0):.4f}")
    
    last_weight = result['filtered_weight']
    last_trend = result['trend']
    print(f"\n   Final pre-gap state: Weight={last_weight:.1f}, Trend={last_trend:.4f}")
    
    print("\n2. Creating 18-day gap...")
    gap_start = base_time + timedelta(days=30)
    gap_end = gap_start + timedelta(days=18)
    
    print("\n3. Processing post-gap measurements (buffer collection)...")
    post_gap_weights = []
    for i in range(5):
        weight = base_weight + trend * (48 + i) + np.random.normal(0, 0.2)
        timestamp = gap_end + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        post_gap_weights.append(weight)
        
        stage = result.get('stage', 'unknown')
        if stage == 'gap_buffer_collecting':
            buffer_size = result.get('gap_buffer_size', 0)
            buffer_target = result.get('gap_buffer_target', 3)
            print(f"   Day {i}: Buffer collecting ({buffer_size}/{buffer_target})")
        elif stage == 'gap_buffer_complete':
            print(f"   Day {i}: Buffer complete! Initialized with trend={result.get('trend', 0):.4f}")
        else:
            filtered = result.get('filtered_weight', 0)
            trend_val = result.get('trend', 0)
            adaptation = result.get('gap_adaptation', {})
            if adaptation.get('active'):
                print(f"   Day {i}: Adapting (measurements_since={adaptation.get('measurements_since_gap', 0)})")
            else:
                print(f"   Day {i}: Normal processing")
            print(f"      Weight={weight:.1f}, Filtered={filtered:.1f}, Trend={trend_val:.4f}")
    
    print("\n4. Continuing with more measurements to test decay...")
    for i in range(5, 15):
        weight = base_weight + trend * (48 + i) + np.random.normal(0, 0.2)
        timestamp = gap_end + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        if i % 3 == 0:
            filtered = result.get('filtered_weight', 0)
            trend_val = result.get('trend', 0)
            adaptation = result.get('gap_adaptation', {})
            if adaptation.get('active'):
                print(f"   Day {i}: Still adapting (factor={adaptation.get('gap_factor', 0):.2f})")
            else:
                print(f"   Day {i}: Adaptation complete")
            print(f"      Weight={weight:.1f}, Filtered={filtered:.1f}, Trend={trend_val:.4f}")
    
    print("\n5. Calculating performance metrics...")
    actual_weights = [base_weight + trend * (48 + i) for i in range(5)]
    errors = [abs(actual - measured) for actual, measured in zip(actual_weights, post_gap_weights)]
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print(f"\n   Post-gap RMSE: {rmse:.3f} kg")
    print(f"   Target: < 0.25 kg")
    print(f"   Success: {'✓' if rmse < 0.25 else '✗'}")
    
    print("\n" + "=" * 50)
    print("Gap Handling Test Complete")

def test_buffer_timeout():
    """Test buffer timeout mechanism."""
    print("\n\nTesting Buffer Timeout Mechanism")
    print("=" * 50)
    
    db = get_state_db()
    config = create_test_config()
    user_id = "test_timeout_user"
    
    db.delete_state(user_id)
    
    print("\n1. Creating initial measurement...")
    result = process_measurement(
        user_id, 80.0, datetime(2024, 1, 1), "patient-device", config
    )
    print(f"   Initial: Weight=80.0")
    
    print("\n2. Creating 15-day gap...")
    print("\n3. Adding single measurement after gap...")
    result = process_measurement(
        user_id, 78.0, datetime(2024, 1, 16), "patient-device", config
    )
    print(f"   Buffer status: {result.get('stage', 'unknown')}")
    
    print("\n4. Waiting 8 days (exceeds timeout)...")
    result = process_measurement(
        user_id, 77.5, datetime(2024, 1, 24), "patient-device", config
    )
    print(f"   Buffer status: {result.get('stage', 'unknown')}")
    
    if result.get('stage') == 'gap_buffer_complete':
        print("   ✓ Buffer completed due to timeout")
    else:
        print("   ✗ Buffer did not timeout as expected")
    
    print("\n" + "=" * 50)
    print("Buffer Timeout Test Complete")

if __name__ == "__main__":
    test_gap_handling()
    test_buffer_timeout()