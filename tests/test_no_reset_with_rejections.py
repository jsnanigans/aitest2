"""
Test that reset only happens when there's a true data gap (no measurements at all).
Rejected measurements should prevent reset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB


def test_no_reset_with_rejected_data():
    """Test that rejected measurements prevent reset."""
    
    print("\n" + "="*70)
    print("TEST: No Reset When Rejected Data Exists in Gap")
    print("="*70)
    
    db = ProcessorStateDB()
    user_id = "test_no_reset_rejections"
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 300.0,
        "max_daily_change": 0.02,  # 2% daily change limit
        "extreme_threshold": 0.10,  # 10% extreme threshold
        "user_height_m": 1.75,
        "min_valid_bmi": 15.0,
        "max_valid_bmi": 50.0,
    }
    
    kalman_config = {
        "process_noise": 0.01,
        "measurement_noise": 1.0,
        "reset_gap_days": 30,
        "questionnaire_reset_days": 10,
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "initial_variance": 1.0
    }
    
    print("\n1️⃣ SCENARIO: Rejected measurements within 30-day window")
    print("-" * 70)
    
    # Initial accepted measurement
    r1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Day 1: 100.0kg - {'✅ ACCEPTED' if r1['accepted'] else '❌ REJECTED'}")
    
    # 15 days later - extreme weight that gets rejected
    r2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=120.0,  # 20% increase - should be rejected
        timestamp=datetime(2024, 1, 16),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Day 16: 120.0kg - {'✅ ACCEPTED' if r2['accepted'] else '❌ REJECTED'}")
    if not r2['accepted']:
        print(f"        Rejection reason: {r2.get('reason')}")
    
    # 20 days after rejection (35 days total) - reasonable weight
    r3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=98.0,
        timestamp=datetime(2024, 2, 5),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Day 36: 98.0kg - {'✅ ACCEPTED' if r3['accepted'] else '❌ REJECTED'}")
    if r3.get('was_reset'):
        print(f"        ❌ INCORRECT: Reset triggered despite rejected data at day 16")
    else:
        print(f"        ✅ CORRECT: No reset because data existed (even if rejected)")
    
    # Verify the behavior
    assert r1['accepted'] == True, "First measurement should be accepted"
    assert r2['accepted'] == False, "Extreme measurement should be rejected"
    assert r3['accepted'] == True, "Reasonable measurement should be accepted"
    assert not r3.get('was_reset'), "Should NOT reset when rejected data exists in gap"
    
    print("\n2️⃣ SCENARIO: True data gap (no measurements at all for 30+ days)")
    print("-" * 70)
    
    db2 = ProcessorStateDB()
    user_id2 = "test_true_gap"
    
    # Initial measurement
    r1 = WeightProcessor.process_weight(
        user_id=user_id2,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db2
    )
    
    print(f"Day 1: 100.0kg - {'✅ ACCEPTED' if r1['accepted'] else '❌ REJECTED'}")
    
    # 35 days later with NO measurements in between
    r2 = WeightProcessor.process_weight(
        user_id=user_id2,
        weight=95.0,
        timestamp=datetime(2024, 2, 5),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db2
    )
    
    print(f"Day 36: 95.0kg - {'✅ ACCEPTED' if r2['accepted'] else '❌ REJECTED'}")
    if r2.get('was_reset'):
        print(f"        ✅ CORRECT: Reset triggered after true 35-day gap")
        print(f"        Reset reason: {r2.get('reset_reason')}")
    else:
        print(f"        ❌ INCORRECT: Should reset after 35 days with no data")
    
    assert r2.get('was_reset') == True, "Should reset after true 30+ day gap"
    
    print("\n3️⃣ SCENARIO: Multiple rejections prevent reset")
    print("-" * 70)
    
    db3 = ProcessorStateDB()
    user_id3 = "test_multiple_rejections"
    
    # Initial measurement
    r1 = WeightProcessor.process_weight(
        user_id=user_id3,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db3
    )
    
    print(f"Day 1: 100.0kg - {'✅ ACCEPTED' if r1['accepted'] else '❌ REJECTED'}")
    
    # Multiple rejected measurements over time
    rejection_count = 0
    for day in [10, 15, 20, 25, 30, 35]:
        r = WeightProcessor.process_weight(
            user_id=user_id3,
            weight=100.0 + day,  # Increasingly extreme weights
            timestamp=datetime(2024, 1, 1) + timedelta(days=day),
            source="patient-device",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db3
        )
        if not r['accepted']:
            rejection_count += 1
        print(f"Day {day}: {100.0 + day}kg - {'✅ ACCEPTED' if r['accepted'] else '❌ REJECTED'}")
    
    # Final reasonable measurement after all rejections
    r_final = WeightProcessor.process_weight(
        user_id=user_id3,
        weight=99.0,
        timestamp=datetime(2024, 2, 10),  # 40 days after start
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db3
    )
    
    print(f"Day 40: 99.0kg - {'✅ ACCEPTED' if r_final['accepted'] else '❌ REJECTED'}")
    print(f"        Rejections before this: {rejection_count}")
    
    if r_final.get('was_reset'):
        print(f"        ❌ INCORRECT: Reset despite {rejection_count} rejected measurements")
    else:
        print(f"        ✅ CORRECT: No reset because user was actively measuring")
    
    assert not r_final.get('was_reset'), "Should NOT reset when multiple rejections exist"
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
✅ Test Results:
1. Rejected measurements now prevent reset
2. True data gaps (no measurements) still trigger reset
3. Multiple rejections are tracked and prevent reset
4. System distinguishes between "no data" and "bad data"

This fixes the issue where users like 0040872d were getting
unnecessary resets despite having rejected measurements.
    """)
    
    return True


def test_questionnaire_gap_with_rejections():
    """Test questionnaire-specific gap behavior with rejections."""
    
    print("\n" + "="*70)
    print("TEST: Questionnaire Gap Reset with Rejections")
    print("="*70)
    
    db = ProcessorStateDB()
    user_id = "test_questionnaire_rejections"
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 300.0,
        "max_daily_change": 0.02,
        "extreme_threshold": 0.10,
        "user_height_m": 1.75,
    }
    
    kalman_config = {
        "process_noise": 0.01,
        "measurement_noise": 1.0,
        "reset_gap_days": 30,
        "questionnaire_reset_days": 10,
    }
    
    # Questionnaire entry
    r1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="internal-questionnaire",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Day 1: Questionnaire 100.0kg - {'✅' if r1['accepted'] else '❌'}")
    
    # Rejected measurement 8 days later
    r2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=120.0,
        timestamp=datetime(2024, 1, 9),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Day 9: Device 120.0kg - {'✅' if r2['accepted'] else '❌'}")
    
    # Reasonable measurement 5 days after rejection (13 days total)
    r3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=95.0,
        timestamp=datetime(2024, 1, 14),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Day 14: Device 95.0kg - {'✅' if r3['accepted'] else '❌'}")
    
    if r3.get('was_reset'):
        print("        ❌ Reset triggered despite rejection at day 9")
    else:
        print("        ✅ No reset: rejection at day 9 shows activity")
    
    assert not r3.get('was_reset'), "Should not reset with rejection in 10-day window"
    
    print("\n✅ Questionnaire gap logic respects rejection tracking")
    
    return True


if __name__ == "__main__":
    test_no_reset_with_rejected_data()
    print("\n" + "="*70 + "\n")
    test_questionnaire_gap_with_rejections()