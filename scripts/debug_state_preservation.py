#!/usr/bin/env python3
"""
Debug state preservation issue
"""

import sys
import json
from datetime import datetime
from src.processor import process_measurement
from src.database import get_state_db
import toml

# Load config
config = toml.load('config.toml')

# Test user data
user_id = "test_user_debug"
measurements = [
    (112.0, "2025-05-15 00:00:00", "internal-questionnaire"),
    (109.6, "2025-06-13 17:07:35", "https://api.iglucose.com"),
    (110.9, "2025-06-13 18:14:19", "https://api.iglucose.com"),
]

# Clear any existing state
db = get_state_db()
db.states.pop(user_id, None)

print("Processing measurements with debug output:\n")

for i, (weight, timestamp_str, source) in enumerate(measurements):
    timestamp = datetime.fromisoformat(timestamp_str)
    
    print(f"=== Measurement {i}: {weight}kg ===")
    
    # Get state before processing
    state_before = db.get_state(user_id)
    if state_before:
        print(f"State before:")
        print(f"  reset_type: {state_before.get('reset_type', 'none')}")
        print(f"  measurements_since_reset: {state_before.get('measurements_since_reset', 0)}")
        reset_params = state_before.get('reset_parameters', {})
        if reset_params:
            print(f"  reset_parameters found: adaptation_measurements={reset_params.get('adaptation_measurements')}")
    else:
        print("No state before (first measurement)")
    
    # Process measurement
    result = process_measurement(
        user_id, weight, timestamp, source, config, db=db
    )
    
    # Get state after processing
    state_after = db.get_state(user_id)
    print(f"State after:")
    print(f"  reset_type: {state_after.get('reset_type', 'none')}")
    print(f"  measurements_since_reset: {state_after.get('measurements_since_reset', 0)}")
    reset_params = state_after.get('reset_parameters', {})
    if reset_params:
        print(f"  reset_parameters found: adaptation_measurements={reset_params.get('adaptation_measurements')}")
    else:
        print(f"  NO reset_parameters in state!")
    
    print(f"Result: accepted={result['accepted']}")
    if 'reset_event' in result:
        print(f"Reset occurred: {result['reset_event']['type']}")
    print()
