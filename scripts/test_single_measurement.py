#!/usr/bin/env python3
"""
Test a single measurement to see what's happening
"""

from datetime import datetime
from src.processor import process_measurement
from src.database import get_state_db
import toml
import json

# Load config
config = toml.load('config.toml')

# Test user data
user_id = "09105245-67b3-48eb-a637-425412e9d93a"

# Clear any existing state
db = get_state_db()
db.states.pop(user_id, None)

# Process first 8 measurements
measurements = [
    (112.037224, "2025-05-15 00:00:00", "internal-questionnaire"),
    (109.6, "2025-06-13 17:07:35", "https://api.iglucose.com"),
    (110.9, "2025-06-13 18:14:19", "https://api.iglucose.com"),
    (104.1, "2025-06-16 02:05:51", "https://api.iglucose.com"),
    (104.1, "2025-06-16 02:06:11", "https://api.iglucose.com"),
    (90.6, "2025-06-25 20:59:45", "https://api.iglucose.com"),
    (90.5, "2025-06-25 21:38:05", "https://api.iglucose.com"),
    (110.4, "2025-06-26 12:08:53", "https://api.iglucose.com"),  # This one gets rejected
]

for i, (weight, timestamp_str, source) in enumerate(measurements):
    timestamp = datetime.fromisoformat(timestamp_str)
    
    # Get state before
    state_before = db.get_state(user_id)
    
    # Process measurement
    result = process_measurement(
        user_id, weight, timestamp, source, config, db=db
    )
    
    # For measurement 7, show full details
    if i == 7:
        print(f"\n=== Measurement {i} (the rejected one): {weight}kg ===")
        print(f"State before processing:")
        if state_before:
            print(f"  measurements_since_reset: {state_before.get('measurements_since_reset', 0)}")
            print(f"  reset_type: {state_before.get('reset_type', 'none')}")
            reset_params = state_before.get('reset_parameters', {})
            if reset_params:
                print(f"  reset_parameters:")
                print(f"    adaptation_measurements: {reset_params.get('adaptation_measurements')}")
                print(f"    quality_acceptance_threshold: {reset_params.get('quality_acceptance_threshold')}")
        
        print(f"\nFull result:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Measurement {i}: {weight}kg - Accepted: {result.get('accepted')}")
