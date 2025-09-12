#!/usr/bin/env python3
"""
Debug the processor for user 1e87d3ab-20b1-479d-ad4d-8986e1af38da
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.processor import WeightProcessor
from src.database import ProcessorStateDB
import tomllib

# Load config
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Create database
db = ProcessorStateDB()

# Test measurements
user_id = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
measurements = [
    (datetime(2013, 3, 30), 56.7),
    (datetime(2013, 3, 30), 56.7),
    (datetime(2017, 2, 27), 88.0),  # 1430 days later - should reset!
]

print("Processing measurements:")
for i, (timestamp, weight) in enumerate(measurements):
    print(f"\n{i+1}. Processing {weight}kg at {timestamp}")
    
    # Check state before processing
    state_before = db.get_state(user_id)
    if state_before:
        print(f"   State before: last_timestamp={state_before.get('last_timestamp')}")
        if state_before.get('last_state') is not None:
            print(f"   Last weight: {state_before['last_state'][0]}")
    else:
        print("   No state yet")
    
    # Process measurement
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight,
        timestamp=timestamp,
        source="test",
        processing_config=config["processing"],
        kalman_config=config["kalman"],
        db=db
    )
    
    if result:
        print(f"   Result: accepted={result.get('accepted', False)}")
        if 'filtered_weight' in result:
            print(f"   Filtered: {result['filtered_weight']:.1f}kg")
        if 'reason' in result:
            print(f"   Reason: {result['reason']}")
    
    # Check state after
    state_after = db.get_state(user_id)
    if state_after and state_after.get('last_timestamp'):
        # Calculate gap if there was a previous timestamp
        if state_before and state_before.get('last_timestamp'):
            prev_ts = state_before['last_timestamp']
            if isinstance(prev_ts, str):
                prev_ts = datetime.fromisoformat(prev_ts)
            gap_days = (timestamp - prev_ts).days
            print(f"   Gap from previous: {gap_days} days")
            if gap_days > 30:
                print(f"   *** SHOULD HAVE RESET! ***")
