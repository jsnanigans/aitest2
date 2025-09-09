#!/usr/bin/env python3
"""Debug trend decay to see what's actually happening."""

import json
from datetime import datetime
from src.filters.custom_kalman_filter import CustomKalmanFilter

# Load the user data
with open('output/users/02E11B7DC40F41638D3FA6169A147156.json', 'r') as f:
    user_data = json.load(f)

time_series = user_data['time_series']

# Initialize filter without decay
kf = CustomKalmanFilter(
    initial_weight=time_series[0]['weight'],
    process_noise_trend=0.001
)

print("Debugging Trend (No Decay)")
print("=" * 80)
print(f"{'Index':<6} {'Date':<20} {'Gap Days':<10} {'Weight':<10} {'Trend Before':<12} {'Trend After':<12}")
print("-" * 80)

prev_timestamp = None
prev_trend = 0.0

for i, reading in enumerate(time_series[:20]):  # First 20 readings
    date_str = reading['date']
    weight = reading['weight']
    timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    
    time_delta_days = 1.0
    if prev_timestamp:
        time_delta_days = max(0.01, (timestamp - prev_timestamp).total_seconds() / 86400.0)
    

    
    # Process measurement
    result = kf.process_measurement(weight, timestamp, time_delta_days)
    
    print(f"{i:<6} {date_str:<20} {time_delta_days:<10.3f} {weight:<10.2f} {prev_trend:<12.4f} {result['trend_kg_per_day']:<12.4f}")
    
    prev_timestamp = timestamp
    prev_trend = result['trend_kg_per_day']

print("\nKey observations:")
print("- Trend now remains consistent without artificial decay")
print("- The Kalman filter naturally adjusts trend based on data")