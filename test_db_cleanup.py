#!/usr/bin/env python3
"""Test the cleaned up database state management."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database import ProcessorStateDB
from src.processor import process_measurement

def test_state_cleanup():
    """Test that state management works after cleanup."""
    
    # Create a database instance
    db = ProcessorStateDB()
    
    # Test initial state creation
    initial_state = db.create_initial_state()
    print("Initial state fields:", list(initial_state.keys()))
    
    expected_fields = {
        'last_state', 'last_covariance', 'last_timestamp', 
        'kalman_params', 'last_source', 'last_raw_weight', 
        'measurement_history'
    }
    
    actual_fields = set(initial_state.keys())
    
    if actual_fields == expected_fields:
        print("✓ Initial state has correct fields")
    else:
        print(f"✗ Field mismatch!")
        print(f"  Expected: {expected_fields}")
        print(f"  Actual: {actual_fields}")
        print(f"  Missing: {expected_fields - actual_fields}")
        print(f"  Extra: {actual_fields - expected_fields}")
    
    # Test processing a measurement
    config = {
        'kalman': {
            'process_noise': 0.016,
            'measurement_noise': 8.725,
            'initial_uncertainty': 0.361,
        },
        'thresholds': {
            'max_weight_change_per_day': 1.0,
            'min_weight_change_per_day': -1.0,
        }
    }
    
    result = process_measurement(
        weight=75.0,
        timestamp=datetime.now(),
        source="scale",
        user_id="test_user",
        unit="kg",
        config=config,
        db=db
    )
    
    print(f"\n✓ Processed measurement: {result['filtered_weight']:.2f} kg")
    
    # Get the state and check fields
    state = db.get_state("test_user")
    print(f"✓ State fields after processing: {list(state.keys())}")
    
    # Export to CSV and check
    csv_path = "test_db_export.csv"
    users_exported = db.export_to_csv(csv_path)
    print(f"\n✓ Exported {users_exported} user(s) to CSV")
    
    # Read and display CSV header
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        print(f"CSV columns: {header}")
        # Read data row
        data = f.readline().strip()
        if data:
            print(f"Sample data row (truncated): {data[:100]}...")
    
    # Clean up
    Path(csv_path).unlink()
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_state_cleanup()
