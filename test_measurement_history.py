#!/usr/bin/env python3
"""Test measurement history tracking."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database import ProcessorStateDB
from src.processor import process_measurement

def test_measurement_history():
    """Test that measurement history is properly tracked."""
    
    db = ProcessorStateDB()
    
    config = {
        'kalman': {
            'process_noise': 0.016,
            'measurement_noise': 8.725,
            'initial_uncertainty': 0.361,
        },
        'thresholds': {
            'max_weight_change_per_day': 1.0,
            'min_weight_change_per_day': -1.0,
        },
        'quality_scoring': {
            'enabled': True
        }
    }
    
    # Process multiple measurements
    base_time = datetime.now()
    weights = [75.0, 75.2, 75.1, 74.9, 75.3]
    
    for i, weight in enumerate(weights):
        result = process_measurement(
            weight=weight,
            timestamp=base_time + timedelta(hours=i*12),
            source="scale",
            user_id="history_test",
            unit="kg",
            config=config,
            db=db
        )
        print(f"Measurement {i+1}: {weight:.1f} kg → {result['filtered_weight']:.2f} kg")
    
    # Check the state
    state = db.get_state("history_test")
    
    if 'measurement_history' in state:
        history = state['measurement_history']
        print(f"\n✓ Measurement history has {len(history)} entries")
        for i, entry in enumerate(history):
            print(f"  Entry {i+1}: {entry.get('weight', 'N/A'):.2f} kg at {entry.get('timestamp', 'N/A')[:19]}")
    else:
        print("\n✗ No measurement_history in state!")
        print(f"State keys: {list(state.keys())}")
    
    print("\n✓ Test completed!")

if __name__ == "__main__":
    test_measurement_history()
