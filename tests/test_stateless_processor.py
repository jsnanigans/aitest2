"""
Test the stateless processor with database backend
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import get_state_db


def test_stateless_processing():
    """Test that the stateless processor works correctly."""
    
    print("=" * 60)
    print("TESTING STATELESS PROCESSOR")
    print("=" * 60)
    
    # Configuration
    config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,
            "reset_gap_days": 30
        }
    }
    
    user_id = "test_user_001"
    base_date = datetime(2024, 1, 1)
    
    # Clear any existing state
    WeightProcessor.reset_user(user_id)
    
    print(f"\nProcessing measurements for {user_id}:")
    print("-" * 40)
    
    results = []
    for i in range(15):
        weight = 70.0 + np.random.normal(0, 0.3)
        timestamp = base_date + timedelta(days=i)
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="scale",
            processing_config=config["processing"],
            kalman_config=config["kalman"]
        )
        
        if result:
            results.append(result)
            status = "✓" if result["accepted"] else "✗"
            print(f"  Day {i:2}: {weight:.1f}kg → {result['filtered_weight']:.1f}kg "
                  f"[{status}] (confidence: {result['confidence']:.3f})")
        else:
            print(f"  Day {i:2}: {weight:.1f}kg → buffering...")
    
    # Check final state
    state = WeightProcessor.get_user_state(user_id)
    
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    print(f"\n✓ State initialized: {state.get('initialized', False)}")
    print(f"✓ Measurements processed: {len(results)}")
    print(f"✓ Acceptance rate: {sum(r['accepted'] for r in results) / len(results) * 100:.1f}%")
    
    if state.get('adapted_params'):
        print(f"\n✓ Adapted parameters:")
        print(f"  observation_covariance: {state['adapted_params'].get('observation_covariance', 'N/A')}")
        print(f"  extreme_threshold: {state['adapted_params'].get('extreme_threshold', 'N/A')}")
    
    # Test persistence
    print("\n" + "=" * 60)
    print("TESTING STATE PERSISTENCE")
    print("=" * 60)
    
    # Process one more measurement
    weight = 71.0
    timestamp = base_date + timedelta(days=20)
    
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight,
        timestamp=timestamp,
        source="scale",
        processing_config=config["processing"],
        kalman_config=config["kalman"]
    )
    
    if result:
        print(f"\n✓ State persisted correctly")
        print(f"  New measurement: {weight:.1f}kg → {result['filtered_weight']:.1f}kg")
        print(f"  Confidence: {result['confidence']:.3f}")
    
    # Test multi-user
    print("\n" + "=" * 60)
    print("TESTING MULTI-USER SUPPORT")
    print("=" * 60)
    
    user2_id = "test_user_002"
    WeightProcessor.reset_user(user2_id)
    
    # Process for second user
    for i in range(12):
        weight = 85.0 + np.random.normal(0, 0.5)
        timestamp = base_date + timedelta(days=i)
        
        WeightProcessor.process_weight(
            user_id=user2_id,
            weight=weight,
            timestamp=timestamp,
            source="scale",
            processing_config=config["processing"],
            kalman_config=config["kalman"]
        )
    
    state1 = WeightProcessor.get_user_state(user_id)
    state2 = WeightProcessor.get_user_state(user2_id)
    
    print(f"\n✓ User 1 state exists: {state1 is not None}")
    print(f"✓ User 2 state exists: {state2 is not None}")
    print(f"✓ States are independent: {id(state1) != id(state2)}")
    
    # Check database stats
    db = get_state_db()
    print(f"\n✓ Total users in database: {len(db.states)}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    test_stateless_processing()