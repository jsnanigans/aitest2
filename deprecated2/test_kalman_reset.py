#!/usr/bin/env python3
"""
Test Kalman filter state reset mechanism on problematic user data
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.filters.robust_kalman import RobustKalmanFilter
from src.core.types import WeightMeasurement


def test_problematic_user():
    """Test the Kalman reset logic on user 0157fbc2-e4c2-4bd1-96c9-96d3962cacc7 data"""
    
    # Readings from the problematic user (dates and weights)
    readings = [
        ("2019-10-30", 85.73),
        ("2020-01-21", 84.46),  # 83 days gap
        ("2020-06-12", 89.81),  # 143 days gap
        ("2021-01-27", 90.99),  # 229 days gap  
        ("2021-09-07", 87.45),  # 223 days gap
        ("2021-10-12", 84.82),  # 35 days gap
        ("2022-05-11", 82.56),  # 211 days gap
        ("2022-05-12", 87.00),  # 1 day gap
        ("2022-09-09", 83.73),  # 120 days gap - THIS IS WHERE IT BROKE
        ("2022-11-16", 80.74),  # 68 days gap
        ("2022-12-13", 78.47),  # 27 days gap
        ("2023-03-16", 78.83),  # 93 days gap
    ]
    
    # Initialize Kalman filter with reset parameters
    kalman = RobustKalmanFilter(
        initial_weight=85.73,
        initial_variance=1.0,
        process_noise_weight=0.5,
        process_noise_trend=0.01,
        measurement_noise=1.0,
        outlier_threshold=3.0,
        extreme_outlier_threshold=5.0,
        innovation_window_size=20,
        reset_gap_days=30,  # Reset after 30 days
        reset_deviation_threshold=0.5,  # Reset if prediction is 50% off
        physiological_min=40.0,
        physiological_max=300.0
    )
    
    print("Testing Kalman filter with state reset on problematic user data")
    print("=" * 70)
    print(f"Reset triggers: gap > 30 days OR deviation > 50%")
    print("=" * 70)
    
    results = []
    prev_date = None
    
    for date_str, weight in readings:
        # Create measurement
        timestamp = datetime.strptime(date_str, "%Y-%m-%d")
        measurement = WeightMeasurement(
            weight=weight,
            timestamp=timestamp,
            source_type="test"
        )
        
        # Calculate gap
        gap_days = 0
        if prev_date:
            gap_days = (timestamp - prev_date).days
        
        # Process measurement
        result = kalman.process_measurement(measurement)
        
        # Extract key values
        filtered_weight = result['filtered_weight']
        state_reset = result.get('state_reset', False)
        reset_reason = result.get('reset_reason', '')
        
        # Display results
        status = "RESET" if state_reset else "OK"
        print(f"{date_str} | Weight: {weight:6.2f} | Filtered: {filtered_weight:6.2f} | "
              f"Gap: {gap_days:3d}d | {status:5s}")
        
        if state_reset:
            print(f"  └─ Reset reason: {reset_reason}")
        
        # Check for corruption
        if filtered_weight > 500:  # Clearly corrupted
            print(f"  ⚠️  CORRUPTION DETECTED: Filtered weight = {filtered_weight:.1f} kg")
            return False
        
        prev_date = timestamp
        results.append({
            'date': date_str,
            'weight': weight,
            'filtered': filtered_weight,
            'gap_days': gap_days,
            'reset': state_reset
        })
    
    print("=" * 70)
    print(f"✅ Test PASSED: No corruption detected")
    print(f"Total resets: {kalman.reset_count}")
    
    # Verify the critical point (2022-09-09)
    critical_idx = 8  # Index of 2022-09-09
    critical_result = results[critical_idx]
    if critical_result['filtered'] < 100:  # Should be around 83.73, not 578
        print(f"✅ Critical point handled correctly: {critical_result['filtered']:.2f} kg")
    else:
        print(f"❌ Critical point still corrupted: {critical_result['filtered']:.2f} kg")
        return False
    
    return True


if __name__ == "__main__":
    success = test_problematic_user()
    sys.exit(0 if success else 1)