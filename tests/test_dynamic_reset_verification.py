"""
Verify dynamic reset implementation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB

def test_dynamic_reset_implementation():
    """Test that dynamic reset is working in the actual processor."""
    
    print("\n" + "="*70)
    print("DYNAMIC RESET IMPLEMENTATION VERIFICATION")
    print("="*70)
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 300.0,
        "max_daily_change": 0.05,
        "extreme_threshold": 0.15,
        "physiological": {
            "enable_physiological_limits": True,
            "max_change_24h_percent": 0.035,
            "max_change_24h_absolute": 5.0,
        }
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
    
    test_cases = [
        {
            "name": "Test 1: Questionnaire → 12-day gap → Device",
            "user_id": "test_q_12d",
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "internal-questionnaire"),
                (datetime(2024, 1, 13), 92.0, "patient-device"),
            ],
            "expected": {
                "reset_on_second": True,
                "second_accepted": True
            }
        },
        {
            "name": "Test 2: Device → 12-day gap → Device",
            "user_id": "test_d_12d",
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "patient-device"),
                (datetime(2024, 1, 13), 92.0, "patient-device"),
            ],
            "expected": {
                "reset_on_second": False,
                "second_accepted": True
            }
        },
        {
            "name": "Test 3: Questionnaire → 8-day gap → Device",
            "user_id": "test_q_8d",
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "initial-questionnaire"),
                (datetime(2024, 1, 9), 95.0, "patient-device"),
            ],
            "expected": {
                "reset_on_second": False,
                "second_accepted": True
            }
        },
        {
            "name": "Test 4: Care-team-upload → 11-day gap → Device",
            "user_id": "test_ctu_11d",
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "care-team-upload"),
                (datetime(2024, 1, 12), 93.0, "patient-device"),
            ],
            "expected": {
                "reset_on_second": True,
                "second_accepted": True
            }
        },
        {
            "name": "Test 5: Device → 35-day gap → Device",
            "user_id": "test_d_35d",
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "patient-device"),
                (datetime(2024, 2, 5), 92.0, "patient-device"),
            ],
            "expected": {
                "reset_on_second": True,
                "second_accepted": True
            }
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"{test['name']}")
        print('-'*70)
        
        db = ProcessorStateDB()
        
        results = []
        for timestamp, weight, source in test['measurements']:
            result = WeightProcessor.process_weight(
                user_id=test['user_id'],
                weight=weight,
                timestamp=timestamp,
                source=source,
                processing_config=processing_config,
                kalman_config=kalman_config,
                db=db
            )
            results.append(result)
            
            gap_info = ""
            if len(results) > 1:
                prev_timestamp = test['measurements'][len(results)-2][0]
                gap_days = (timestamp - prev_timestamp).days
                gap_info = f" (Gap: {gap_days} days)"
            
            status = "✅ ACCEPTED" if result['accepted'] else "❌ REJECTED"
            reset_info = " [RESET]" if result.get('was_reset') else ""
            
            print(f"{timestamp.date()} | {source:25} | {weight:6.1f}kg | {status}{reset_info}{gap_info}")
            
            if not result['accepted']:
                print(f"         Reason: {result.get('reason', 'Unknown')}")
        
        second_result = results[1]
        
        test_passed = True
        
        if test['expected']['reset_on_second']:
            if second_result.get('was_reset'):
                print(f"✅ Reset occurred as expected")
            else:
                print(f"❌ FAIL: Expected reset but didn't occur")
                test_passed = False
        else:
            if not second_result.get('was_reset'):
                print(f"✅ No reset as expected")
            else:
                print(f"❌ FAIL: Unexpected reset occurred")
                test_passed = False
        
        if test['expected']['second_accepted']:
            if second_result['accepted']:
                print(f"✅ Second measurement accepted as expected")
            else:
                print(f"❌ FAIL: Expected acceptance but was rejected")
                test_passed = False
        else:
            if not second_result['accepted']:
                print(f"✅ Second measurement rejected as expected")
            else:
                print(f"❌ FAIL: Expected rejection but was accepted")
                test_passed = False
        
        if test_passed:
            print(f"\n✅ TEST PASSED")
        else:
            print(f"\n❌ TEST FAILED")
            all_passed = False
    
    print("\n" + "="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Dynamic reset is working correctly!")
        print("\nKey behaviors verified:")
        print("• 12-day gap after questionnaire triggers reset (10-day threshold)")
        print("• 12-day gap after device does NOT trigger reset (30-day threshold)")
        print("• 8-day gap after questionnaire does NOT trigger reset")
        print("• Care-team-upload treated as questionnaire source")
        print("• 35-day gap always triggers reset regardless of source")
    else:
        print("❌ SOME TESTS FAILED - Implementation needs adjustment")
    
    return all_passed

def test_state_persistence():
    """Test that last_source is properly persisted in state."""
    
    print("\n" + "="*70)
    print("STATE PERSISTENCE TEST")
    print("="*70)
    
    db = ProcessorStateDB()
    user_id = "test_persistence"
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 300.0,
        "max_daily_change": 0.05,
        "extreme_threshold": 0.15,
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
    
    result1 = WeightProcessor.process_weight(
        user_id, 100.0, datetime(2024, 1, 1),
        "internal-questionnaire",
        processing_config, kalman_config, db
    )
    
    state = db.get_state(user_id)
    
    if state and state.get('last_source') == 'internal-questionnaire':
        print("✅ last_source correctly stored as 'internal-questionnaire'")
    else:
        print(f"❌ last_source not stored correctly: {state.get('last_source') if state else 'No state'}")
        return False
    
    result2 = WeightProcessor.process_weight(
        user_id, 99.0, datetime(2024, 1, 2),
        "patient-device",
        processing_config, kalman_config, db
    )
    
    state = db.get_state(user_id)
    
    if state and state.get('last_source') == 'patient-device':
        print("✅ last_source correctly updated to 'patient-device'")
    else:
        print(f"❌ last_source not updated correctly: {state.get('last_source') if state else 'No state'}")
        return False
    
    print("\n✅ STATE PERSISTENCE TEST PASSED")
    return True

if __name__ == "__main__":
    test1_passed = test_dynamic_reset_implementation()
    test2_passed = test_state_persistence()
    
    print("\n" + "="*70)
    print("OVERALL VERIFICATION RESULT")
    print("="*70)
    
    if test1_passed and test2_passed:
        print("✅ IMPLEMENTATION VERIFIED - Dynamic reset is fully functional!")
    else:
        print("❌ IMPLEMENTATION ISSUES DETECTED - Review needed")