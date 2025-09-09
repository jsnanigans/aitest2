#!/usr/bin/env python3
"""
Demo: Robust vs Standard Kalman Filter Performance
Shows how the robust implementation handles extreme outliers better
"""

import numpy as np
from datetime import datetime, timedelta
from src.filters.robust_kalman import RobustKalmanFilter
from src.filters.layer3_kalman import PureKalmanFilter
from src.core.types import WeightMeasurement


def demonstrate_outlier_handling():
    """
    Demonstrate the difference between standard and robust Kalman filters
    when handling extreme outliers.
    """
    
    print("=" * 60)
    print("ROBUST KALMAN FILTER DEMONSTRATION")
    print("=" * 60)
    
    baseline_weight = 115.0
    baseline_variance = 0.5
    
    print(f"\n📊 Initial Setup:")
    print(f"   Baseline weight: {baseline_weight:.1f} kg")
    print(f"   Baseline variance: {baseline_variance:.3f}")
    
    standard_kalman = PureKalmanFilter(
        initial_weight=baseline_weight,
        initial_variance=baseline_variance,
        process_noise_weight=0.5,
        process_noise_trend=0.01,
        measurement_noise=1.0
    )
    
    robust_kalman = RobustKalmanFilter(
        initial_weight=baseline_weight,
        initial_variance=baseline_variance,
        process_noise_weight=0.5,
        process_noise_trend=0.01,
        measurement_noise=1.0,
        outlier_threshold=3.0,
        extreme_outlier_threshold=5.0
    )
    
    test_sequence = [
        ("Normal", 115.5),
        ("Normal", 115.2),
        ("Normal", 114.8),
        ("EXTREME OUTLIER", 45.0),
        ("EXTREME OUTLIER", 200.0),
        ("Normal", 115.0),
        ("Normal", 114.7),
    ]
    
    print("\n📈 Processing Weight Sequence:")
    print("-" * 60)
    
    timestamp = datetime.now()
    
    for i, (label, weight) in enumerate(test_sequence):
        timestamp += timedelta(days=1)
        measurement = WeightMeasurement(
            user_id="demo_user",
            weight=weight,
            timestamp=timestamp,
            source_type="patient-upload"
        )
        
        print(f"\nMeasurement {i+1}: {weight:.1f} kg ({label})")
        
        std_pred, _ = standard_kalman.predict(1.0)
        std_update = standard_kalman.update(weight)
        standard_kalman.last_timestamp = timestamp
        
        rob_result = robust_kalman.process_measurement(measurement, robust_mode=True)
        
        print(f"  Standard Kalman:")
        print(f"    - Predicted: {std_pred:.1f} kg")
        print(f"    - Filtered: {std_update['filtered_weight']:.1f} kg")
        print(f"    - Innovation: {std_update['innovation']:.1f} kg")
        
        print(f"  Robust Kalman:")
        print(f"    - Predicted: {rob_result['predicted_weight']:.1f} kg")
        print(f"    - Filtered: {rob_result['filtered_weight']:.1f} kg")
        print(f"    - Outlier Type: {rob_result.get('outlier_type', 'none')}")
        
        if rob_result.get('outlier_type') == 'extreme':
            print(f"    ⚠️  EXTREME OUTLIER REJECTED ({rob_result['normalized_innovation']:.1f}σ)")
        elif rob_result.get('outlier_type') == 'moderate':
            print(f"    ⚡ Moderate outlier - dampened update applied")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    std_final = standard_kalman.state[0]
    rob_final = robust_kalman.state[0]
    
    print(f"\n📊 Final State Estimates:")
    print(f"   Standard Kalman: {std_final:.1f} kg (error: {abs(std_final - baseline_weight):.1f} kg)")
    print(f"   Robust Kalman:   {rob_final:.1f} kg (error: {abs(rob_final - baseline_weight):.1f} kg)")
    
    print(f"\n📈 Outlier Statistics (Robust):")
    print(f"   Total measurements: {robust_kalman.measurement_count}")
    print(f"   Extreme outliers rejected: {robust_kalman.extreme_outlier_count}")
    print(f"   Moderate outliers dampened: {robust_kalman.outlier_count}")
    
    improvement = (abs(std_final - baseline_weight) - abs(rob_final - baseline_weight)) / abs(std_final - baseline_weight) * 100
    
    print(f"\n✅ Result: Robust Kalman reduced error by {improvement:.1f}%")
    print(f"   The robust filter successfully maintained accurate state estimates")
    print(f"   despite extreme outliers that corrupted the standard filter.\n")


if __name__ == "__main__":
    demonstrate_outlier_handling()