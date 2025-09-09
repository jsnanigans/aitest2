#!/usr/bin/env python3
"""Check if predictions were added to the output."""

import json

with open('output/results_yes.json', 'r') as f:
    data = json.load(f)

for user_id in list(data.keys())[:3]:  # Check first 3 users
    user_data = data[user_id]
    print(f"\nUser: {user_id[:8]}...")
    print(f"  Total time series points: {len(user_data.get('time_series', []))}")
    print(f"  Predicted points added: {user_data.get('predicted_points_added', 0)}")
    
    time_series = user_data.get('time_series', [])
    predicted_points = [ts for ts in time_series if ts.get('is_predicted', False)]
    
    if predicted_points:
        print(f"  Found {len(predicted_points)} predicted points")
        print(f"  Sample predicted point:")
        sample = predicted_points[0]
        for key, value in sample.items():
            print(f"    {key}: {value}")
    else:
        print("  No predicted points found")
        
    # Check for gaps
    if len(time_series) > 1:
        from datetime import datetime
        dates = [datetime.fromisoformat(ts['date'].replace('Z', '+00:00')) for ts in time_series]
        max_gap = 0
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > max_gap:
                max_gap = gap
        print(f"  Maximum gap in days: {max_gap}")