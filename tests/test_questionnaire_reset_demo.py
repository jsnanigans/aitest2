"""
Demonstrate dynamic reset after questionnaire data.
Shows how 10-day gap after questionnaire allows faster recovery.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.database import ProcessorStateDB

def test_questionnaire_dynamic_reset():
    """Test that we can reset after just 10 days following questionnaire data."""
    
    print("\n" + "="*60)
    print("DYNAMIC RESET AFTER QUESTIONNAIRE DEMONSTRATION")
    print("="*60)
    
    user_id = "test_user_questionnaire"
    
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
    
    print("\n📊 CURRENT BEHAVIOR (30-day reset):")
    print("-" * 50)
    
    # Test with standard 30-day gap
    db1 = ProcessorStateDB()
    kalman_config = {
        "process_noise": 0.01,
        "measurement_noise": 1.0,
        "reset_gap_days": 30,  # Standard gap
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "initial_variance": 1.0
    }
    
    # Process questionnaire entry
    r1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="internal-questionnaire",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db1
    )
    
    print(f"Day 1: Questionnaire entry - 100.0kg - {'✅ ACCEPTED' if r1['accepted'] else '❌ REJECTED'}")
    
    # 12 days later with weight drop
    r2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=92.0,
        timestamp=datetime(2024, 1, 13),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db1
    )
    
    print(f"Day 13: Device measurement - 92.0kg - {'✅ ACCEPTED' if r2['accepted'] else '❌ REJECTED'}")
    if not r2['accepted']:
        print(f"        ❌ Rejection reason: {r2.get('reason')}")
    if r2.get('was_reset'):
        print(f"        Note: State was reset (gap > {kalman_config['reset_gap_days']} days)")
    
    print("\n📊 PROPOSED BEHAVIOR (10-day reset after questionnaire):")
    print("-" * 50)
    
    # Now test with dynamic 10-day gap
    db2 = ProcessorStateDB()
    kalman_config_dynamic = {
        "process_noise": 0.01,
        "measurement_noise": 1.0,
        "reset_gap_days": 10,  # Shorter gap for demonstration
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "initial_variance": 1.0
    }
    
    # Process questionnaire entry
    r1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=100.0,
        timestamp=datetime(2024, 1, 1),
        source="internal-questionnaire",
        processing_config=processing_config,
        kalman_config=kalman_config_dynamic,
        db=db2
    )
    
    print(f"Day 1: Questionnaire entry - 100.0kg - {'✅ ACCEPTED' if r1['accepted'] else '❌ REJECTED'}")
    
    # 12 days later with weight drop
    r2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=92.0,
        timestamp=datetime(2024, 1, 13),
        source="patient-device",
        processing_config=processing_config,
        kalman_config=kalman_config_dynamic,
        db=db2
    )
    
    print(f"Day 13: Device measurement - 92.0kg - {'✅ ACCEPTED' if r2['accepted'] else '❌ REJECTED'}")
    if r2.get('was_reset'):
        print(f"        ✅ State reset triggered (gap > {kalman_config_dynamic['reset_gap_days']} days)")
        print(f"        ✅ Large weight change accepted due to reset")
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print(f"""
Current system (30-day reset):
- 12-day gap is NOT enough to trigger reset
- Result: 8kg drop likely REJECTED as extreme deviation

With dynamic reset (10-day after questionnaire):
- 12-day gap DOES trigger reset
- Result: 8kg drop ACCEPTED due to state reset

This matches the real-world scenario where:
- Questionnaire data is often inaccurate or aspirational
- Users may not weigh themselves for 10-15 days after questionnaire
- When they do, the real weight can be quite different
- Current system rejects this as "extreme deviation"
- Dynamic reset would accept it by resetting state
    """)
    
    print("\n" + "="*60)
    print("IMPLEMENTATION APPROACH")
    print("="*60)
    
    print("""
To implement dynamic reset in processor.py (around line 112):

1. Track last source in state:
   - Add 'last_source' to state when saving
   - Check it when processing new measurements

2. Modify reset logic:
```python
# Determine reset threshold based on last source
reset_gap_days = kalman_config.get("reset_gap_days", 30)

# Check if last measurement was from questionnaire
questionnaire_sources = ['internal-questionnaire', 
                         'initial-questionnaire',
                         'care-team-upload']
                         
if state.get('last_source') in questionnaire_sources:
    # Use shorter gap after questionnaire
    reset_gap_days = kalman_config.get("questionnaire_reset_days", 10)

if delta > reset_gap_days:
    should_reset = True
    # ... rest of reset logic
```

3. Benefits:
   • 3x faster recovery after questionnaires
   • Reduces false rejections
   • Maintains strict validation for device data
   • Simple to implement - just track last_source
    """)
    
    return True

if __name__ == "__main__":
    test_questionnaire_dynamic_reset()
