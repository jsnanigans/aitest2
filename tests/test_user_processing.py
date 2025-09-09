#!/usr/bin/env python3
"""Test processing for user 0093a653 to see what's accepted/rejected."""

import pandas as pd
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.processor_database import ProcessorDatabase
import toml

# Load config
config = toml.load('config.toml')
processing_config = config['processing']
kalman_config = config['kalman']

# Initialize database
db = ProcessorDatabase()

# Load data for specific user
df = pd.read_csv('data/2025-09-05_optimized.csv')
user_id = '0093a653-476b-4401-bbec-33a89abc2b18'
user_data = df[df['user_id'] == user_id].sort_values('effectivDateTime')

print(f"Processing user {user_id}")
print(f"Total measurements: {len(user_data)}")
print("\nProcessing each measurement:")
print("-" * 80)

results = []
for idx, row in user_data.iterrows():
    timestamp = datetime.fromisoformat(row['effectivDateTime'])
    weight = row['weight']
    source = row['source_type']
    
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight,
        timestamp=timestamp,
        source=source,
        processing_config=processing_config,
        kalman_config=kalman_config
    )
    
    if result is None:
        status = "BUFFERING"
        print(f"{idx+1:2}. {row['effectivDateTime']}: {weight:6.2f} kg -> {status}")
    elif result.get('accepted'):
        status = "ACCEPTED"
        filtered = result.get('filtered_weight', 0)
        innovation = result.get('innovation', 0)
        print(f"{idx+1:2}. {row['effectivDateTime']}: {weight:6.2f} kg -> {status} (filtered={filtered:.2f}, innovation={innovation:+.2f})")
    else:
        status = "REJECTED"
        reason = result.get('rejection_reason', 'unknown')
        print(f"{idx+1:2}. {row['effectivDateTime']}: {weight:6.2f} kg -> {status} ({reason})")
    
    if result:
        results.append(result)

print("\n" + "=" * 80)
print("SUMMARY:")
accepted = sum(1 for r in results if r.get('accepted'))
rejected = sum(1 for r in results if not r.get('accepted'))
print(f"  Accepted: {accepted}/{len(results)}")
print(f"  Rejected: {rejected}/{len(results)}")

if rejected > 0:
    print("\nRejected measurements:")
    for idx, (_, row) in enumerate(user_data.iterrows()):
        if idx < len(results) and not results[idx].get('accepted'):
            print(f"  - {row['effectivDateTime']}: {row['weight']:.2f} kg ({results[idx].get('rejection_reason', 'unknown')})")