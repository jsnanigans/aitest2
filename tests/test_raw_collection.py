#!/usr/bin/env python3
"""Test that raw measurements are being collected properly."""

import pandas as pd
from collections import defaultdict

# Load data
df = pd.read_csv('data/2025-09-05_optimized.csv')
user_id = '0093a653-476b-4401-bbec-33a89abc2b18'

# Simulate what main.py does
raw_measurements = defaultdict(list)

# Process the CSV like main.py does
for idx, row in df.iterrows():
    if row['user_id'] == user_id:
        raw_measurements[user_id].append({
            'weight': row['weight'],
            'timestamp': row['effectiveDateTime'],
            'source': row['source_type']
        })

print(f"Collected {len(raw_measurements[user_id])} raw measurements for user {user_id}")
print("\nAll measurements:")
for i, m in enumerate(raw_measurements[user_id], 1):
    print(f"  {i:2}. {m['timestamp']}: {m['weight']:.2f} kg")