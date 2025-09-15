#!/usr/bin/env python3
"""Generate a clean database dump with proper state management."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent))

from src.database import ProcessorStateDB
from src.processor import process_measurement

def generate_test_data():
    """Generate test data and export clean DB dump."""
    
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
    
    # Create data for multiple users
    users = [
        {'id': 'user001', 'base_weight': 70.0, 'variation': 0.5},
        {'id': 'user002', 'base_weight': 85.0, 'variation': 0.8},
        {'id': 'user003', 'base_weight': 65.0, 'variation': 0.3},
        {'id': 'user004', 'base_weight': 95.0, 'variation': 1.0},
        {'id': 'user005', 'base_weight': 75.0, 'variation': 0.6},
    ]
    
    base_time = datetime.now() - timedelta(days=10)
    
    print("Generating test data for users...")
    for user in users:
        print(f"\nProcessing {user['id']}:")
        
        # Generate 20 measurements over 10 days
        for day in range(10):
            for measurement in range(2):  # 2 measurements per day
                timestamp = base_time + timedelta(days=day, hours=measurement*12)
                
                # Add some realistic variation
                weight = user['base_weight'] + random.gauss(0, user['variation'])
                source = random.choice(['scale', 'manual', 'questionnaire'])
                
                result = process_measurement(
                    weight=weight,
                    timestamp=timestamp,
                    source=source,
                    user_id=user['id'],
                    unit="kg",
                    config=config,
                    db=db
                )
                
                if day == 0 and measurement == 0:
                    print(f"  First: {weight:.2f} kg → {result['filtered_weight']:.2f} kg")
                elif day == 9 and measurement == 1:
                    print(f"  Last:  {weight:.2f} kg → {result['filtered_weight']:.2f} kg")
    
    # Export to CSV
    output_dir = Path("clean_db_export")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "db_dump_clean.csv"
    users_exported = db.export_to_csv(str(csv_path))
    
    print(f"\n✓ Exported {users_exported} users to {csv_path}")
    
    # Display the CSV content
    print("\nCSV Preview:")
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        print(lines[0].strip())  # Header
        for line in lines[1:min(4, len(lines))]:  # First 3 data rows
            parts = line.strip().split(',')
            # Format for readability
            print(f"{parts[0]}: weight={parts[2]}, trend={parts[3]}, source={parts[4]}, raw={parts[5]}")
    
    # Check state consistency
    print("\nState Consistency Check:")
    for user in users[:2]:  # Check first 2 users
        state = db.get_state(user['id'])
        print(f"\n{user['id']} state fields: {list(state.keys())}")
        if 'measurement_history' in state:
            print(f"  - measurement_history: {len(state['measurement_history'])} entries")
        if 'last_source' in state:
            print(f"  - last_source: {state['last_source']}")
        if 'last_raw_weight' in state:
            print(f"  - last_raw_weight: {state['last_raw_weight']:.2f}")

if __name__ == "__main__":
    generate_test_data()
