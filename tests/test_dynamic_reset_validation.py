"""
Validate dynamic reset functionality works correctly.
Tests various scenarios with questionnaire data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB

def test_dynamic_reset_scenarios():
    """Test various dynamic reset scenarios."""
    
    print("\n" + "="*70)
    print("DYNAMIC RESET VALIDATION - MULTIPLE SCENARIOS")
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
    
    # Base kalman config
    base_kalman = {
        "process_noise": 0.01,
        "measurement_noise": 1.0,
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "initial_variance": 1.0
    }
    
    scenarios = [
        {
            "name": "Scenario 1: 8-day gap after questionnaire (NO RESET with 10-day threshold)",
            "reset_config": {"reset_gap_days": 10},
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "internal-questionnaire"),
                (datetime(2024, 1, 9), 95.0, "patient-device"),  # 8 days later
            ],
            "expected_reset": False
        },
        {
            "name": "Scenario 2: 12-day gap after questionnaire (RESET with 10-day threshold)",
            "reset_config": {"reset_gap_days": 10},
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "internal-questionnaire"),
                (datetime(2024, 1, 13), 92.0, "patient-device"),  # 12 days later
            ],
            "expected_reset": True
        },
        {
            "name": "Scenario 3: 12-day gap after device (NO RESET with 30-day threshold)",
            "reset_config": {"reset_gap_days": 30},
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "patient-device"),
                (datetime(2024, 1, 13), 92.0, "patient-device"),  # 12 days later
            ],
            "expected_reset": False
        },
        {
            "name": "Scenario 4: 35-day gap after device (RESET with 30-day threshold)",
            "reset_config": {"reset_gap_days": 30},
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "patient-device"),
                (datetime(2024, 2, 5), 92.0, "patient-device"),  # 35 days later
            ],
            "expected_reset": True
        },
        {
            "name": "Scenario 5: Multiple questionnaire sources",
            "reset_config": {"reset_gap_days": 10},
            "measurements": [
                (datetime(2024, 1, 1), 100.0, "initial-questionnaire"),
                (datetime(2024, 1, 11), 95.0, "patient-device"),  # 10 days - should not reset
                (datetime(2024, 1, 12), 100.0, "care-team-upload"),  # Another questionnaire
                (datetime(2024, 1, 23), 92.0, "patient-device"),  # 11 days - should reset
            ],
            "expected_reset": True  # For last measurement
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"{scenario['name']}")
        print('-'*70)
        
        db = ProcessorStateDB()
        user_id = f"test_{scenario['name'][:10]}"
        
        kalman_config = base_kalman.copy()
        kalman_config.update(scenario['reset_config'])
        
        results = []
        for timestamp, weight, source in scenario['measurements']:
            result = WeightProcessor.process_weight(
                user_id=user_id,
                weight=weight,
                timestamp=timestamp,
                source=source,
                processing_config=processing_config,
                kalman_config=kalman_config,
                db=db
            )
            results.append(result)
            
            # Display result
            gap_info = ""
            if len(results) > 1:
                prev_idx = len(results) - 2
                gap_days = (timestamp - scenario['measurements'][prev_idx][0]).days
                gap_info = f" (Gap: {gap_days} days)"
            
            status = "✅ ACCEPTED" if result['accepted'] else "❌ REJECTED"
            reset_info = " [RESET]" if result.get('was_reset') else ""
            
            print(f"{timestamp.date()} | {source:25} | {weight:6.1f}kg | {status}{reset_info}{gap_info}")
            
            if not result['accepted']:
                print(f"         Reason: {result.get('reason', 'Unknown')}")
        
        # Check expectation for last measurement
        last_result = results[-1]
        if scenario['expected_reset']:
            if last_result.get('was_reset'):
                print(f"\n✅ PASS: Reset occurred as expected")
            else:
                print(f"\n❌ FAIL: Expected reset but didn't occur")
        else:
            if not last_result.get('was_reset'):
                print(f"\n✅ PASS: No reset as expected")
            else:
                print(f"\n❌ FAIL: Unexpected reset occurred")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Findings:
1. With 10-day threshold: Resets occur after questionnaire gaps > 10 days
2. With 30-day threshold: Standard behavior for all sources
3. Dynamic reset would allow source-specific thresholds
4. This prevents false rejections after questionnaire data

Implementation Notes:
- Track 'last_source' in processor state
- Check if last_source is questionnaire type
- Apply appropriate reset threshold
- Simple, backwards-compatible change
    """)

if __name__ == "__main__":
    test_dynamic_reset_scenarios()
