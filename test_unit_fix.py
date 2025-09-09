#!/usr/bin/env python3
"""
Test that the unit conversion fix works correctly
"""

import sys
sys.path.insert(0, '.')

import csv
from datetime import datetime
import subprocess
import os
import json

def check_user_output(user_id):
    """Check the output for a specific user"""
    output_file = f"output/user_{user_id}_result_test_no_date.json"
    
    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        return None
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Calculate statistics
    measurements = data.get('measurements', [])
    if not measurements:
        return None
    
    filtered_weights = [m['filtered_weight'] for m in measurements if 'filtered_weight' in m]
    raw_weights = [m['raw_weight'] for m in measurements if 'raw_weight' in m]
    
    if filtered_weights:
        import numpy as np
        
        # Calculate volatility as std dev of differences between consecutive filtered values
        if len(filtered_weights) > 1:
            diffs = [abs(filtered_weights[i] - filtered_weights[i-1]) for i in range(1, len(filtered_weights))]
            volatility = np.std(diffs)
        else:
            volatility = 0
        
        return {
            'user_id': user_id,
            'num_measurements': len(measurements),
            'num_filtered': len(filtered_weights),
            'raw_std': np.std(raw_weights),
            'filtered_std': np.std(filtered_weights),
            'volatility': volatility,
            'raw_min': np.min(raw_weights),
            'raw_max': np.max(raw_weights),
            'filtered_min': np.min(filtered_weights),
            'filtered_max': np.max(filtered_weights),
        }
    
    return None

def main():
    target_user = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
    
    # Create config that only processes our target user
    test_config = f"""
[data]
csv_file = "./data/2025-09-05_optimized.csv"
output_dir = "output"
max_users = 0
user_offset = 0
min_readings = 20
test_users = ["{target_user}"]

[processing]
min_weight = 30.0
max_weight = 400.0
max_daily_change = 0.05
extreme_threshold = 0.20
kalman_cleanup_threshold = 4.0

[kalman]
initial_variance = 0.5
transition_covariance_weight = 0.05
transition_covariance_trend = 0.0005
observation_covariance = 1.5
reset_gap_days = 30

[visualization]
dashboard_dpi = 100
dashboard_figsize = [16, 10]
moving_average_window = 7
cropped_months = 9

[logging]
progress_interval = 10000
timestamp_format = "test_no_date"
"""
    
    # Write test config
    with open('test_config.toml', 'w') as f:
        f.write(test_config)
    
    print(f"Running processor with unit conversion fix for user {target_user[:8]}...")
    
    # Run the processor
    result = subprocess.run(
        ["uv", "run", "python", "main.py", "--config", "test_config.toml"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running processor: {result.stderr}")
        return
    
    # Check the output
    stats = check_user_output(target_user)
    
    if stats:
        print(f"\n=== Results for user {target_user[:8]}... ===")
        print(f"Measurements processed: {stats['num_measurements']}")
        print(f"Measurements filtered: {stats['num_filtered']}")
        print(f"\nRaw data:")
        print(f"  Std Dev: {stats['raw_std']:.2f} kg")
        print(f"  Range: {stats['raw_min']:.1f} - {stats['raw_max']:.1f} kg")
        print(f"\nFiltered data:")
        print(f"  Std Dev: {stats['filtered_std']:.2f} kg")
        print(f"  Range: {stats['filtered_min']:.1f} - {stats['filtered_max']:.1f} kg")
        print(f"  Volatility (std of changes): {stats['volatility']:.2f} kg")
        
        if stats['raw_std'] > 20:
            print("\n⚠️  WARNING: Raw data still has high std dev - unit conversion may not be working!")
        else:
            print("\n✅ SUCCESS: Raw data std dev is reasonable - unit conversion is working!")
    
    # Clean up
    os.remove('test_config.toml')

if __name__ == "__main__":
    main()