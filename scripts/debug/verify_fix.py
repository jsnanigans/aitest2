#!/usr/bin/env python3
"""Verify the raw validation fix with comprehensive test."""

from datetime import datetime
from src.processor import WeightProcessor
from src.database import get_state_db

def verify_fix():
    """Comprehensive test of the raw validation fix."""
    
    # Test configurations
    processing_config = {
        'min_weight': 30,
        'max_weight': 400,
        'extreme_threshold': 0.2,
        'physiological': {
            'enable_physiological_limits': True,
            'max_change_1h_percent': 0.02,
            'max_change_1h_absolute': 3.0,
            'max_change_6h_percent': 0.025,
            'max_change_6h_absolute': 4.0,
            'max_change_24h_percent': 0.035,
            'max_change_24h_absolute': 5.0,
            'max_sustained_daily': 1.5,
            'session_timeout_minutes': 5,
            'session_variance_threshold': 5.0
        }
    }
    
    kalman_config = {
        'initial_variance': 1.0,
        'transition_covariance_weight': 0.5,
        'transition_covariance_trend': 0.001,
        'observation_covariance': 2.0,
        'reset_gap_days': 30
    }
    
    print("\n" + "="*70)
    print("COMPREHENSIVE RAW VALIDATION FIX VERIFICATION")
    print("="*70)
    
    # Test Case 1: Original bug scenario
    print("\n1. ORIGINAL BUG SCENARIO")
    print("-" * 40)
    
    user_id = "test_user_1"
    db = get_state_db()
    db.clear_state(user_id)  # Start fresh
    
    # Establish initial state where Kalman diverges from raw
    measurements = [
        ("2025-09-01 12:00:00", 104.0),  # Initial
        ("2025-09-02 12:00:00", 103.4),  # Kalman might filter to ~103.7
        ("2025-09-03 12:00:00", 103.1),  # Should be accepted (0.3kg change)
    ]
    
    for timestamp_str, weight in measurements:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="test",
            processing_config=processing_config,
            kalman_config=kalman_config
        )
        
        if result and result.get('accepted', True):
            print(f"  ✓ {timestamp_str}: {weight}kg → {result['filtered_weight']:.2f}kg")
        else:
            print(f"  ✗ {timestamp_str}: {weight}kg REJECTED - {result.get('reason')}")
    
    state = db.get_state(user_id)
    if state and 'last_raw_weight' in state:
        print(f"  State has last_raw_weight: {state['last_raw_weight']}kg ✓")
    
    # Test Case 2: Edge case - large filtered divergence
    print("\n2. LARGE FILTERED DIVERGENCE")
    print("-" * 40)
    
    user_id = "test_user_2"
    db.clear_state(user_id)
    
    # Simulate scenario where Kalman significantly diverges
    measurements = [
        ("2025-09-01 12:00:00", 100.0),  # Initial
        ("2025-09-02 12:00:00", 99.0),   # Small drop
        ("2025-09-03 12:00:00", 98.0),   # Another small drop
        ("2025-09-04 12:00:00", 101.0),  # Jump back up (3kg from yesterday's raw)
    ]
    
    for timestamp_str, weight in measurements:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="test",
            processing_config=processing_config,
            kalman_config=kalman_config
        )
        
        if result and result.get('accepted', True):
            print(f"  ✓ {timestamp_str}: {weight}kg → {result['filtered_weight']:.2f}kg")
        else:
            print(f"  ✗ {timestamp_str}: {weight}kg REJECTED - {result.get('reason')}")
    
    # Test Case 3: Verify migration path
    print("\n3. MIGRATION PATH (OLD STATE)")
    print("-" * 40)
    
    user_id = "test_user_3"
    
    # Manually create old-style state without last_raw_weight
    import numpy as np
    old_state = {
        'kalman_params': {
            'initial_state_mean': [100, 0],
            'initial_state_covariance': [[1.0, 0], [0, 0.001]],
            'transition_covariance': [[0.5, 0], [0, 0.001]],
            'observation_covariance': [[2.0]],
        },
        'last_state': np.array([[100.5, 0]]),  # Filtered weight
        'last_covariance': np.array([[[1.0, 0], [0, 0.001]]]),
        'last_timestamp': datetime(2025, 9, 1, 12, 0, 0),
        # Note: NO last_raw_weight field
    }
    
    db.save_state(user_id, old_state)
    
    # Process new measurement - should use filtered weight as fallback
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=99.0,  # 1.5kg drop from filtered 100.5
        timestamp=datetime(2025, 9, 2, 12, 0, 0),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    
    if result and result.get('accepted', True):
        print(f"  ✓ Migration worked: 99.0kg accepted (using filtered fallback)")
        print(f"    Filtered: {result['filtered_weight']:.2f}kg")
    else:
        print(f"  ✗ Migration issue: {result.get('reason')}")
    
    # Check state now has last_raw_weight
    state = db.get_state(user_id)
    if state and 'last_raw_weight' in state:
        print(f"  ✓ State upgraded with last_raw_weight: {state['last_raw_weight']}kg")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    
    return True

if __name__ == "__main__":
    verify_fix()
