#!/usr/bin/env python3
import json
from pathlib import Path

with open('output/results_yes.json', 'r') as f:
    data = json.load(f)

users = ['0040872d-333a-4ace-8c5a-b2fcd056e65a', '0675ed39-53be-480b-baa6-fa53fc33709f', '055b0c48-d5b4-44cb-8772-48154999e6c3']

print("Rejected Readings Visualization Check:")
print("-" * 50)

for user_id in users:
    if user_id in data['users']:
        user_data = data['users'][user_id]
        if 'all_readings_with_validation' in user_data:
            all_readings = user_data['all_readings_with_validation']
            total = len(all_readings)
            rejected = sum(1 for r in all_readings if not r.get('validated', False))
            print(f"\nUser {user_id[:8]}...")
            print(f"  Total: {total}, Rejected (red): {rejected}")
            viz_file = Path(f"output/dashboards_yes/{user_id}.png")
            print(f"  Viz: {'✓' if viz_file.exists() else '✗'}")
