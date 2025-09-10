#!/usr/bin/env python3
"""Test that physiological validation correctly uses raw-to-raw comparison."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.processor import WeightProcessor
from src.processor_database import ProcessorDatabase


def test_raw_validation_fix():
    """Test that the Sept 3 measurement for user 0093a653 is now accepted."""
    
    # Initialize database
    db = ProcessorDatabase()
    
    # Get configs - use defaults for test
    processing_config = {
        'min_weight': 30,
        'max_weight': 400,
        'extreme_threshold': 0.2,  # 20% deviation threshold
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
    
    # Test user ID
    user_id = "0093a653-476b-4401-bbec-33a89abc2b18"
    
    # Critical measurements from the user's history
    measurements = [
        # Initial measurements to establish state
        ("2025-08-15 13:13:47", 106.0),
        ("2025-08-17 13:20:26", 107.6),
        ("2025-08-18 21:46:31", 107.7),
        ("2025-08-20 22:33:45", 106.2),
        ("2025-08-22 23:08:37", 105.0),
        ("2025-08-25 20:03:52", 105.7),
        ("2025-08-27 18:03:03", 105.4),
        ("2025-08-28 19:28:23", 105.7),
        ("2025-08-29 19:48:33", 105.6),
        ("2025-08-30 19:35:16", 105.1),
        ("2025-08-31 22:11:06", 104.9),
        # The critical sequence
        ("2025-09-02 19:08:00", 103.4),  # This gets accepted, filtered to ~104.96
        ("2025-09-03 20:36:02", 103.1),  # This SHOULD be accepted (was wrongly rejected)
        ("2025-09-04 20:47:42", 102.3),  # This gets accepted
    ]
    
    results = []
    
    for timestamp_str, weight in measurements:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="https://api.iglucose.com",
            processing_config=processing_config,
            kalman_config=kalman_config
        )
        
        results.append({
            'timestamp': timestamp_str,
            'weight': weight,
            'accepted': result.get('accepted', True) if result else False,
            'filtered': result.get('filtered_weight') if result else None,
            'reason': result.get('reason') if result and not result.get('accepted') else None
        })
    
    # Check the critical Sept 3 measurement
    sept3_result = results[-2]  # Second to last
    
    print("\n" + "="*70)
    print("RAW VALIDATION FIX TEST RESULTS")
    print("="*70)
    
    # Show the critical sequence
    print("\nCritical sequence:")
    for r in results[-3:]:
        status = "✓ ACCEPTED" if r['accepted'] else f"✗ REJECTED: {r['reason']}"
        filtered_str = f" → filtered: {r['filtered']:.2f}kg" if r['filtered'] else ""
        print(f"  {r['timestamp']}: {r['weight']}kg{filtered_str} - {status}")
    
    print("\n" + "-"*70)
    
    # The key assertion
    if sept3_result['accepted']:
        print("✅ SUCCESS: Sept 3 measurement (103.1kg) is now ACCEPTED")
        print("   The fix correctly uses raw-to-raw comparison")
        print(f"   Raw change: |103.1 - 103.4| = 0.3kg (well within limits)")
    else:
        print("❌ FAILURE: Sept 3 measurement still rejected")
        print(f"   Rejection reason: {sept3_result['reason']}")
        print("   The fix may not be working correctly")
        return False
    
    # Additional validation - check state has last_raw_weight
    # Get the state from the SAME database instance used by WeightProcessor
    from src.processor_database import get_state_db
    actual_db = get_state_db()
    state = actual_db.get_state(user_id)
    
    if state and 'last_raw_weight' in state:
        print(f"\n✅ State correctly stores last_raw_weight: {state['last_raw_weight']}kg")
    else:
        print("\n⚠️  Warning: State does not contain last_raw_weight field")
        if state:
            print(f"   State keys: {list(state.keys())}")
    
    print("\n" + "="*70)
    return True


if __name__ == "__main__":
    success = test_raw_validation_fix()
    sys.exit(0 if success else 1)