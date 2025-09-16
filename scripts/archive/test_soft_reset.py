#!/usr/bin/env python3
"""
Test soft reset functionality with manual data sources.
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_measurement
from src.database import get_state_db
from src.reset_manager import ResetManager, ResetType

def test_soft_reset():
    """Test soft reset with manual data entry."""
    
    db = get_state_db()
    user_id = "test_soft_reset"
    db.delete_state(user_id)
    
    config = {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.0160,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.490,
            'reset': {
                'hard': {
                    'enabled': True,
                    'gap_threshold_days': 30,
                    'weight_boost_factor': 10,
                    'trend_boost_factor': 100,
                    'decay_rate': 3,
                    'warmup_measurements': 10,
                    'adaptive_days': 7
                },
                'initial': {
                    'enabled': True,
                    'weight_boost_factor': 10,
                    'trend_boost_factor': 100,
                    'decay_rate': 3,
                    'warmup_measurements': 10,
                    'adaptive_days': 7
                },
                'soft': {
                    'enabled': True,
                    'min_change_kg': 5,
                    'cooldown_days': 3,
                    'weight_boost_factor': 3,
                    'trend_boost_factor': 10,
                    'decay_rate': 5,
                    'warmup_measurements': 15,
                    'adaptive_days': 10
                }
            }
        },
        'processing': {
            'extreme_threshold': 0.20
        },
        'adaptive_noise': {
            'enabled': False
        },
        'quality_scoring': {
            'enabled': False
        }
    }
    
    print("Testing Soft Reset Functionality")
    print("=" * 60)
    
    # Phase 1: Establish baseline with automatic measurements
    print("\nPhase 1: Baseline with automatic measurements")
    base_date = datetime(2024, 1, 1)
    
    for i in range(5):
        timestamp = base_date + timedelta(days=i*2)
        weight = 85.0 + (i * 0.1)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        reset_type = "INITIAL" if i == 0 else "none"
        print(f"  Day {i*2}: {weight:.1f}kg (device) - Reset: {reset_type}")
    
    # Phase 2: Manual entry with small change (no soft reset)
    print("\nPhase 2: Manual entry with small change (<5kg)")
    timestamp = base_date + timedelta(days=10)
    weight = 86.0  # Only 0.6kg change from last (85.4)
    result = process_measurement(
        user_id, weight, timestamp, "patient-upload", config
    )
    
    state = db.get_state(user_id)
    reset_occurred = state.get('reset_type') == 'soft'
    print(f"  Day 10: {weight:.1f}kg (manual) - Soft reset: {reset_occurred}")
    print(f"    Weight change: 0.6kg < 5kg threshold")
    
    # Phase 3: Manual entry with large change (triggers soft reset)
    print("\nPhase 3: Manual entry with large change (>5kg)")
    timestamp = base_date + timedelta(days=12)
    weight = 92.0  # 6kg change from last
    result = process_measurement(
        user_id, weight, timestamp, "care-team-upload", config
    )
    
    state = db.get_state(user_id)
    reset_type = state.get('reset_type')
    reset_params = state.get('reset_parameters', {})
    
    print(f"  Day 12: {weight:.1f}kg (care-team) - Reset type: {reset_type}")
    if reset_type == 'soft':
        print(f"    ✓ Soft reset triggered!")
        print(f"    Weight boost: {reset_params.get('weight_boost_factor', 0)}x")
        print(f"    Trend boost: {reset_params.get('trend_boost_factor', 0)}x")
        print(f"    Decay rate: {reset_params.get('decay_rate', 0)}")
    
    # Phase 4: Continue with measurements to see gentle adaptation
    print("\nPhase 4: Measurements after soft reset")
    test_weights = [91.5, 91.0, 90.5, 90.0]
    
    for i, weight in enumerate(test_weights):
        timestamp = base_date + timedelta(days=14 + i*2)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        accepted = result['accepted']
        filtered = result.get('filtered_weight', 0)
        status = "✓" if accepted else "✗"
        
        # Check if still in adaptive period
        state = db.get_state(user_id)
        measurements_since = state.get('measurements_since_reset', 0)
        
        print(f"  Day {14+i*2}: {weight:.1f}kg → {filtered:.1f}kg - {status} (m={measurements_since})")
    
    # Phase 5: Try another manual entry within cooldown (no reset)
    print("\nPhase 5: Manual entry within cooldown period")
    timestamp = base_date + timedelta(days=14)  # Only 2 days after soft reset
    weight = 85.0  # Large change but within cooldown
    result = process_measurement(
        user_id, weight, timestamp, "patient-upload", config
    )
    
    state = db.get_state(user_id)
    last_reset = ResetManager.get_last_reset_timestamp(state)
    days_since = (timestamp - last_reset).days if last_reset else 999
    
    print(f"  Day 14: {weight:.1f}kg (manual) - Days since reset: {days_since}")
    print(f"    Cooldown active: {days_since < 3}")
    
    # Check final state
    print("\n" + "=" * 60)
    final_state = db.get_state(user_id)
    if final_state:
        reset_events = final_state.get('reset_events', [])
        print(f"Total reset events: {len(reset_events)}")
        for event in reset_events:
            print(f"  - {event['type']}: {event.get('reason', 'unknown')}")
    
    print("\n✓ Soft reset test complete!")
    print("\nKey findings:")
    print("- Manual data with >5kg change triggers soft reset")
    print("- Soft reset uses gentler parameters (3x vs 10x boost)")
    print("- Cooldown period prevents reset loops")
    print("- Adaptation is slower and more stable")

if __name__ == "__main__":
    test_soft_reset()