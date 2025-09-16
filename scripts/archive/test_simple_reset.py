#!/usr/bin/env python3
"""
Test script for the simplified 30-day reset system.
Creates synthetic data with gaps and verifies reset behavior.
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import process_measurement
from src.database import get_state_db
from src.visualization import create_weight_timeline

def test_simple_reset():
    """Test the simplified reset system with various gap scenarios."""
    
    db = get_state_db()
    user_id = "test_reset_user"
    
    # Clear any existing state
    db.delete_state(user_id)
    
    config = {
        'kalman': {
            'initial_variance': 0.361,
            'transition_covariance_weight': 0.0160,
            'transition_covariance_trend': 0.0001,
            'observation_covariance': 3.490,
            'reset': {
                'enabled': True,
                'gap_threshold_days': 30
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
        },
        'visualization': {
            'reset': {
                'show_reset_lines': True,
                'show_gap_regions': True,
                'show_gap_labels': True
            }
        }
    }
    
    results = []
    
    # Phase 1: Initial measurements
    print("Phase 1: Initial measurements")
    base_date = datetime(2024, 1, 1)
    weights = [85.0, 84.8, 84.9, 85.1, 84.7]
    
    for i, weight in enumerate(weights):
        timestamp = base_date + timedelta(days=i)
        result = process_measurement(
            user_id, weight, timestamp, "test-source", config
        )
        results.append(result)
        print(f"  Day {i}: {weight}kg - Accepted: {result['accepted']}")
    
    # Phase 2: 29-day gap (no reset)
    print("\nPhase 2: 29-day gap (should NOT reset)")
    timestamp = base_date + timedelta(days=33)  # 29 days after day 4
    result = process_measurement(
        user_id, 84.5, timestamp, "test-source", config
    )
    results.append(result)
    print(f"  Day 33: 84.5kg - Reset occurred: {result.get('reset_event') is not None}")
    
    # Phase 3: More measurements
    print("\nPhase 3: Continue without reset")
    for i in range(3):
        timestamp = base_date + timedelta(days=34+i)
        weight = 84.5 + i * 0.1
        result = process_measurement(
            user_id, weight, timestamp, "test-source", config
        )
        results.append(result)
        print(f"  Day {34+i}: {weight}kg - Accepted: {result['accepted']}")
    
    # Phase 4: 30-day gap (should reset)
    print("\nPhase 4: 30-day gap (SHOULD reset)")
    timestamp = base_date + timedelta(days=66)  # 30 days after day 36
    result = process_measurement(
        user_id, 86.0, timestamp, "test-source", config
    )
    results.append(result)
    print(f"  Day 68: 86.0kg - Reset occurred: {result.get('reset_event') is not None}")
    if result.get('reset_event'):
        print(f"    Gap days: {result['reset_event']['gap_days']:.1f}")
        print(f"    Reason: {result['reset_event']['reason']}")
    
    # Phase 5: Continue after reset
    print("\nPhase 5: Continue after reset")
    for i in range(3):
        timestamp = base_date + timedelta(days=67+i)
        weight = 86.0 - i * 0.2
        result = process_measurement(
            user_id, weight, timestamp, "test-source", config
        )
        results.append(result)
        print(f"  Day {69+i}: {weight}kg - Accepted: {result['accepted']}")
    
    # Phase 6: Large gap (100+ days)
    print("\nPhase 6: Large gap (100+ days)")
    timestamp = base_date + timedelta(days=200)
    result = process_measurement(
        user_id, 82.0, timestamp, "test-source", config
    )
    results.append(result)
    print(f"  Day 200: 82.0kg - Reset occurred: {result.get('reset_event') is not None}")
    if result.get('reset_event'):
        print(f"    Gap days: {result['reset_event']['gap_days']:.1f}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    reset_count = sum(1 for r in results if r.get('reset_event'))
    print(f"Total measurements: {len(results)}")
    print(f"Total resets: {reset_count}")
    
    print("\nReset events:")
    for r in results:
        if r.get('reset_event'):
            event = r['reset_event']
            print(f"  - {r['timestamp']}: {event['gap_days']:.0f} day gap")
    
    # Check state
    final_state = db.get_state(user_id)
    print(f"\nFinal state has Kalman params: {final_state.get('kalman_params') is not None}")
    print(f"Reset events tracked: {len(final_state.get('reset_events', []))}")
    
    # Create visualization
    print("\nCreating visualization...")
    try:
        fig = create_weight_timeline(results, user_id, config)
        output_file = "test_simple_reset.html"
        fig.write_html(output_file)
        print(f"Visualization saved to: {output_file}")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    return results

if __name__ == "__main__":
    results = test_simple_reset()
    print("\nTest completed successfully!")