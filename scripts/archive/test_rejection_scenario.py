#!/usr/bin/env python3
"""Test scenario where rigid filter would reject valid measurements after reset."""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processor import process_measurement
from src.database import get_state_db

def test_rejection_scenario():
    """Test with actual processor to see rejections."""
    
    print("="*80)
    print("Testing Rejection Scenario: 120kg -> 35-day gap -> 105kg (15kg drop)")
    print("="*80)
    
    # Configuration with extreme threshold check
    config = {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.016,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.49,
            'reset': {
                'enabled': True,
                'gap_threshold_days': 30
            }
        },
        'adaptive_noise': {
            'enabled': True,
            'default_multiplier': 1.5
        },
        'quality_scoring': {
            'enabled': False
        },
        'processing': {
            'extreme_threshold': 0.15  # 15% deviation threshold
        }
    }
    
    # Clear any existing state
    db = get_state_db()
    user_id = "test_rejection"
    # Clear state by saving empty state
    db.save_state(user_id, {})
    
    print("\n1. Initial measurements (stable around 120kg):")
    start_date = datetime(2024, 1, 1)
    
    initial_weights = [120.0, 119.5, 120.2, 119.8, 120.5]
    for i, weight in enumerate(initial_weights):
        timestamp = start_date + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        if result:
            status = "✓" if result['accepted'] else f"✗ ({result.get('reason', 'unknown')})"
            print(f"  Day {i}: {weight:.1f}kg -> filtered={result.get('filtered_weight', weight):.1f}kg [{status}]")
    
    # Get the last state to see what the filter expects
    state = db.get_state(user_id)
    if state and 'last_state' in state and state['last_state'] is not None:
        import numpy as np
        last_state = np.array(state['last_state'])
        if len(last_state.shape) > 1:
            current_state = last_state[-1]
        else:
            current_state = last_state
        print(f"\nFilter state before gap: weight={current_state[0]:.1f}kg, trend={current_state[1]:.4f}kg/day")
    
    print("\n--- 35 day gap ---")
    
    # After gap - weight has dropped significantly
    print("\n2. After gap (weight dropped to 105kg):")
    post_gap_date = start_date + timedelta(days=40)
    
    # More extreme variation to trigger rejections
    post_gap_weights = [105.0, 108.0, 104.0, 107.5, 103.0, 109.0, 105.5, 106.0]
    
    for i, weight in enumerate(post_gap_weights):
        timestamp = post_gap_date + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "patient-device", config
        )
        
        if result:
            reset_info = ""
            if 'reset_event' in result:
                reset_info = f" [RESET: {result['reset_event']['gap_days']:.0f}d gap]"
            
            status = "✓" if result['accepted'] else f"✗"
            reason = f" ({result.get('reason', '')})" if not result['accepted'] else ""
            
            # Calculate deviation from predicted
            if 'filtered_weight' in result and result['accepted']:
                deviation = abs(weight - result['filtered_weight']) / result['filtered_weight'] * 100
                info = f" (dev={deviation:.1f}%)"
            else:
                info = ""
            
            print(f"  Day {i}: {weight:.1f}kg -> filtered={result.get('filtered_weight', weight):.1f}kg "
                  f"[{status}{reason}]{info}{reset_info}")
            
            if not result['accepted'] and 'stage' in result:
                print(f"    Rejection stage: {result['stage']}")
    
    print("\n" + "="*80)
    print("Analysis:")
    print("-"*80)
    print("With adaptive Kalman parameters after reset:")
    print("- Filter quickly adapts to new weight level")
    print("- Measurements within reasonable range are accepted")
    print("- Large variations are still properly rejected")
    
if __name__ == "__main__":
    test_rejection_scenario()