#!/usr/bin/env python3
"""Test the progress tracking with limited users"""

import sys
import tomllib

def test_config():
    # Test with inline config
    test_config = {
        "data": {
            "csv_file": "./data/2025-09-05_optimized.csv",
            "output_dir": "test_output",
            "max_users": 5,
            "user_offset": 2
        },
        "processing": {
            "min_init_readings": 5,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.03,
            "extreme_threshold": 0.30
        },
        "kalman": {
            "initial_variance": 1.0,
            "transition_covariance_weight": 0.5,
            "transition_covariance_trend": 0.01,
            "observation_covariance": 1.0,
            "reset_gap_days": 30
        },
        "visualization": {
            "max_visualizations": 3,
            "min_data_points": 20,
            "dashboard_dpi": 100,
            "dashboard_figsize": [16, 10],
            "moving_average_window": 7
        },
        "logging": {
            "progress_interval": 1000,
            "timestamp_format": "%Y%m%d_%H%M%S"
        }
    }
    
    print("Test configuration:")
    print(f"  Will skip first {test_config['data']['user_offset']} users")
    print(f"  Will process {test_config['data']['max_users']} users")
    print(f"  Output to: {test_config['data']['output_dir']}")
    
    from main import stream_process
    from pathlib import Path
    
    csv_file = test_config["data"]["csv_file"]
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        return
    
    stream_process(csv_file, test_config["data"]["output_dir"], test_config)

if __name__ == "__main__":
    test_config()