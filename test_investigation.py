#!/usr/bin/env python3
"""
Investigation script for user 1e87d3ab-20b1-479d-ad4d-8986e1af38da
"""

import csv
from datetime import datetime
from collections import defaultdict

# Read the CSV and track this specific user
user_id = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
measurements = []

with open("data/2025-09-05_optimized.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("user_id") == user_id:
            measurements.append({
                'timestamp': row.get('effectiveDateTime'),
                'weight': row.get('weight'),
                'unit': row.get('unit', 'kg')
            })

print(f"Found {len(measurements)} measurements for user {user_id}")
print("\nFirst 10 measurements:")
for i, m in enumerate(measurements[:10]):
    print(f"  {i+1}. {m['timestamp']}: {m['weight']} {m['unit']}")

# Check for gaps
if len(measurements) > 1:
    print("\nChecking gaps between measurements:")
    prev_date = datetime.strptime(measurements[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
    for i, m in enumerate(measurements[1:6], 1):
        curr_date = datetime.strptime(m['timestamp'], "%Y-%m-%d %H:%M:%S")
        gap_days = (curr_date - prev_date).days
        print(f"  Gap {i}: {prev_date.date()} to {curr_date.date()} = {gap_days} days")
        if gap_days > 30:
            print(f"    ^^^ SHOULD TRIGGER RESET!")
        prev_date = curr_date
