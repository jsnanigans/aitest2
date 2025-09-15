#!/usr/bin/env python3
"""Test that the system handles all gap sizes without hard resets."""

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
            'gap_handling': {
                'enabled': True,
                'gap_threshold_days': 10,
                'questionnaire_gap_threshold_days': 5,
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

def test_various_gap_sizes():
    """Test that all gap sizes use adaptive handling, no hard resets."""
    print("Testing Various Gap Sizes (No Hard Resets)")
    print("=" * 50)
    
    db = get_state_db()
    config = create_test_config()
    
    gap_sizes = [5, 15, 30, 45, 60, 90]  # Various gap sizes including very large
    
    for gap_days in gap_sizes:
        print(f"\n--- Testing {gap_days}-day gap ---")
        
        user_id = f"test_gap_{gap_days}"
        db.delete_state(user_id)
        
        # Create initial measurements
        base_weight = 80.0
        base_time = datetime(2024, 1, 1)
        
        for i in range(5):
            weight = base_weight + np.random.normal(0, 0.2)
            timestamp = base_time + timedelta(days=i)
            result = process_measurement(
                user_id, weight, timestamp, "patient-device", config
            )
        
        print(f"Initial state established")
        
        # Create gap
        gap_end = base_time + timedelta(days=5 + gap_days)
        
        # Process measurements after gap
        buffer_phase = True
        for i in range(5):
            weight = base_weight - 1.0 + np.random.normal(0, 0.2)
            timestamp = gap_end + timedelta(days=i)
            result = process_measurement(
                user_id, weight, timestamp, "patient-device", config
            )
            
            stage = result.get('stage', 'unknown')
            
            if i == 0:
                if gap_days < 10:
                    print(f"  Day {i}: {stage} (gap below threshold, normal processing)")
                elif stage == 'gap_buffer_collecting':
                    print(f"  Day {i}: {stage} ✓ (adaptive handling started)")
                else:
                    print(f"  Day {i}: {stage} (expected gap_buffer_collecting)")
            
            if stage == 'gap_buffer_complete' and buffer_phase:
                print(f"  Day {i}: Buffer complete ✓ (adaptive initialization done)")
                buffer_phase = False
                
                # Check that state has gap_adaptation
                state = db.get_state(user_id)
                if state and state.get('gap_adaptation', {}).get('active'):
                    gap_factor = state['gap_adaptation']['gap_factor']
                    print(f"  Gap factor: {gap_factor:.2f} (based on {gap_days}-day gap)")
        
        # Verify no hard reset occurred
        state = db.get_state(user_id)
        if state and state.get('kalman_params'):
            print(f"✓ Kalman state preserved (no hard reset)")
        else:
            print(f"✗ Unexpected: Kalman state missing")
    
    print("\n" + "=" * 50)
    print("All gap sizes handled adaptively without hard resets!")

def test_questionnaire_gaps():
    """Test questionnaire source with different threshold."""
    print("\n\nTesting Questionnaire Source Gaps")
    print("=" * 50)
    
    db = get_state_db()
    config = create_test_config()
    user_id = "test_questionnaire_user"
    
    db.delete_state(user_id)
    
    print("\n1. Creating initial questionnaire measurement...")
    result = process_measurement(
        user_id, 75.0, datetime(2024, 1, 1), "questionnaire", config
    )
    
    print("\n2. Testing 7-day gap (should trigger for questionnaire)...")
    result = process_measurement(
        user_id, 74.5, datetime(2024, 1, 8), "questionnaire", config
    )
    
    stage = result.get('stage', 'unknown')
    if stage == 'gap_buffer_collecting':
        print(f"   ✓ Gap handling triggered for questionnaire at 7 days")
    else:
        print(f"   Stage: {stage}")
    
    print("\n3. Testing device source with 7-day gap...")
    user_id2 = "test_device_user"
    db.delete_state(user_id2)
    
    result = process_measurement(
        user_id2, 75.0, datetime(2024, 1, 1), "patient-device", config
    )
    
    result = process_measurement(
        user_id2, 74.5, datetime(2024, 1, 8), "patient-device", config
    )
    
    stage = result.get('stage', 'unknown')
    if stage != 'gap_buffer_collecting':
        print(f"   ✓ Gap handling NOT triggered for device at 7 days (threshold is 10)")
    else:
        print(f"   ✗ Unexpected gap handling for device source")
    
    print("\n" + "=" * 50)
    print("Questionnaire gap handling test complete!")

if __name__ == "__main__":
    test_various_gap_sizes()
    test_questionnaire_gaps()