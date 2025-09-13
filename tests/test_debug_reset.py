"""
Debug the reset behavior with rejections.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.processor import WeightProcessor
from src.database import ProcessorStateDB


def debug_reset_behavior():
    """Debug why reset is still happening."""
    
    print("\n" + "="*70)
    print("DEBUG: Reset Behavior with Rejections")
    print("="*70)
    
    db = ProcessorStateDB()
    user_id = "debug_user"
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 300.0,
        "max_daily_change": 0.02,
        "extreme_threshold": 0.10,
        "user_height_m": 1.75,
    }
    
    kalman_config = {
        "reset_gap_days": 30,
        "questionnaire_reset_days": 10,
    }
    
    # Initial measurement
    r1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"\n1. Initial: 100kg - {'✅' if r1['accepted'] else '❌'}")
    
    # Check state after first measurement
    state1 = db.get_state(user_id)
    print(f"   State after accept:")
    print(f"   - last_timestamp: {state1.get('last_timestamp')}")
    print(f"   - last_attempt_timestamp: {state1.get('last_attempt_timestamp')}")
    print(f"   - rejection_count: {state1.get('rejection_count_since_accept')}")
    
    # Rejected measurement
    r2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=120.0,
        timestamp=datetime(2024, 1, 16),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"\n2. Day 16: 120kg - {'✅' if r2['accepted'] else '❌'}")
    print(f"   Reason: {r2.get('reason')}")
    
    # Check state after rejection
    state2 = db.get_state(user_id)
    print(f"   State after reject:")
    print(f"   - last_timestamp: {state2.get('last_timestamp')}")
    print(f"   - last_attempt_timestamp: {state2.get('last_attempt_timestamp')}")
    print(f"   - rejection_count: {state2.get('rejection_count_since_accept')}")
    
    # Final measurement
    r3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=98.0,
        timestamp=datetime(2024, 2, 5),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"\n3. Day 36: 98kg - {'✅' if r3['accepted'] else '❌'}")
    print(f"   Was reset: {r3.get('was_reset')}")
    print(f"   Reset reason: {r3.get('reset_reason')}")
    print(f"   Gap days: {r3.get('gap_days')}")
    
    # Check final state
    state3 = db.get_state(user_id)
    print(f"   State after:")
    print(f"   - last_timestamp: {state3.get('last_timestamp')}")
    print(f"   - last_attempt_timestamp: {state3.get('last_attempt_timestamp')}")
    print(f"   - rejection_count: {state3.get('rejection_count_since_accept')}")
    
    # Calculate gaps manually
    t1 = datetime(2024, 1, 1)
    t2 = datetime(2024, 1, 16)
    t3 = datetime(2024, 2, 5)
    
    print(f"\n📊 Gap Analysis:")
    print(f"   Gap from last accept (Jan 1) to Feb 5: {(t3 - t1).days} days")
    print(f"   Gap from last attempt (Jan 16) to Feb 5: {(t3 - t2).days} days")
    print(f"   Reset threshold: {kalman_config['reset_gap_days']} days")
    
    return r3


if __name__ == "__main__":
    debug_reset_behavior()