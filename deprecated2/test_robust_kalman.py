#!/usr/bin/env python3
"""
Test script for Robust Kalman Filter with extreme outlier handling
Simulates the problematic user data pattern from the screenshot
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple

from src.filters.robust_kalman import RobustKalmanFilter
from src.filters.layer3_kalman import PureKalmanFilter
from src.core.types import WeightMeasurement


def generate_problematic_data() -> List[WeightMeasurement]:
    """
    Generate weight data similar to the problematic user in the screenshot.
    Includes normal data, then extreme outliers, then normal data again.
    """
    measurements = []
    base_weight = 115.0
    start_date = datetime(2024, 10, 1)
    
    for i in range(180):
        date = start_date + timedelta(days=i)
        
        if i < 120:
            weight = base_weight + np.random.normal(0, 0.5)
            weight = max(110, min(120, weight))
        elif 120 <= i < 140:
            if np.random.random() < 0.3:
                weight = np.random.choice([40, 50, 180, 200])
            else:
                weight = base_weight + np.random.normal(0, 0.5)
        else:
            weight = base_weight + np.random.normal(0, 0.5)
            weight = max(110, min(120, weight))
        
        measurements.append(WeightMeasurement(
            user_id="test_user",
            weight=weight,
            timestamp=date,
            source_type="patient-upload"
        ))
    
    return measurements


def run_comparison():
    """
    Compare standard Kalman filter vs Robust Kalman filter on problematic data.
    """
    measurements = generate_problematic_data()
    
    initial_weights = [m.weight for m in measurements[:7] 
                      if 100 < m.weight < 130]
    baseline_weight = np.median(initial_weights)
    baseline_variance = np.var(initial_weights)
    
    print(f"Baseline: {baseline_weight:.1f}kg (variance={baseline_variance:.3f})")
    
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
    
    standard_results = []
    robust_results = []
    
    for i, measurement in enumerate(measurements):
        time_delta = 1.0
        if i > 0:
            prev_time = measurements[i-1].timestamp
            time_delta = (measurement.timestamp - prev_time).total_seconds() / 86400.0
        
        std_pred, std_var = standard_kalman.predict(time_delta)
        std_update = standard_kalman.update(measurement.weight)
        standard_kalman.last_timestamp = measurement.timestamp
        standard_results.append({
            'measurement': measurement.weight,
            'filtered': std_update['filtered_weight'],
            'predicted': std_pred,
            'confidence': calculate_confidence(std_update['innovation'], std_update['innovation_variance'])
        })
        
        rob_result = robust_kalman.process_measurement(measurement, robust_mode=True)
        robust_results.append({
            'measurement': measurement.weight,
            'filtered': rob_result['filtered_weight'],
            'predicted': rob_result['predicted_weight'],
            'confidence': calculate_confidence_robust(rob_result),
            'outlier_type': rob_result.get('outlier_type', 'none')
        })
    
    return measurements, standard_results, robust_results


def calculate_confidence(innovation: float, innovation_variance: float) -> float:
    """Calculate confidence for standard Kalman."""
    if innovation_variance <= 0:
        return 0.95
    
    normalized = abs(innovation) / np.sqrt(innovation_variance)
    
    if normalized <= 1.0:
        return 0.95
    elif normalized <= 2.0:
        return 0.8
    elif normalized <= 3.0:
        return 0.5
    else:
        return 0.1


def calculate_confidence_robust(result: dict) -> float:
    """Calculate confidence for robust Kalman based on outlier type."""
    outlier_type = result.get('outlier_type', 'none')
    normalized = result.get('normalized_innovation', 0)
    
    if outlier_type == 'extreme':
        return 0.05
    elif outlier_type == 'moderate':
        return 0.4
    else:
        if normalized <= 1.0:
            return 0.95
        elif normalized <= 2.0:
            return 0.8
        else:
            return 0.5


def plot_results(measurements, standard_results, robust_results):
    """Create comparison plots."""
    dates = [m.timestamp for m in measurements]
    weights = [m.weight for m in measurements]
    
    standard_filtered = [r['filtered'] for r in standard_results]
    standard_confidence = [r['confidence'] for r in standard_results]
    
    robust_filtered = [r['filtered'] for r in robust_results]
    robust_confidence = [r['confidence'] for r in robust_results]
    robust_outliers = [i for i, r in enumerate(robust_results) 
                       if r['outlier_type'] in ['moderate', 'extreme']]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Standard vs Robust Kalman Filter Comparison', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.scatter(dates, weights, alpha=0.3, s=20, c='gray', label='Raw measurements')
    ax.plot(dates, standard_filtered, 'b-', linewidth=2, label='Standard Kalman')
    ax.axhline(y=115, color='g', linestyle='--', alpha=0.3, label='True baseline')
    ax.set_title('Standard Kalman Filter')
    ax.set_ylabel('Weight (kg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.scatter(dates, weights, alpha=0.3, s=20, c='gray', label='Raw measurements')
    ax.plot(dates, robust_filtered, 'r-', linewidth=2, label='Robust Kalman')
    for idx in robust_outliers:
        ax.scatter(dates[idx], weights[idx], color='orange', s=50, marker='x', zorder=5)
    ax.axhline(y=115, color='g', linestyle='--', alpha=0.3, label='True baseline')
    ax.set_title('Robust Kalman Filter (with outlier detection)')
    ax.set_ylabel('Weight (kg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    colors = ['red' if c < 0.5 else 'yellow' if c < 0.8 else 'green' 
              for c in standard_confidence]
    ax.scatter(dates, standard_confidence, c=colors, s=10, alpha=0.7)
    ax.set_title('Standard Kalman Confidence Scores')
    ax.set_ylabel('Confidence')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    colors = ['red' if c < 0.5 else 'yellow' if c < 0.8 else 'green' 
              for c in robust_confidence]
    ax.scatter(dates, robust_confidence, c=colors, s=10, alpha=0.7)
    ax.set_title('Robust Kalman Confidence Scores')
    ax.set_ylabel('Confidence')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    errors_standard = [abs(w - f) for w, f in zip(weights, standard_filtered)]
    ax.plot(dates, errors_standard, 'b-', alpha=0.7)
    ax.fill_between(dates, 0, errors_standard, alpha=0.3, color='blue')
    ax.set_title('Standard Kalman Absolute Errors')
    ax.set_ylabel('Absolute Error (kg)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    errors_robust = [abs(w - f) for w, f in zip(weights, robust_filtered)]
    ax.plot(dates, errors_robust, 'r-', alpha=0.7)
    ax.fill_between(dates, 0, errors_robust, alpha=0.3, color='red')
    ax.set_title('Robust Kalman Absolute Errors')
    ax.set_ylabel('Absolute Error (kg)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    mean_error_std = np.mean(errors_standard[140:])
    mean_error_rob = np.mean(errors_robust[140:])
    max_error_std = np.max(errors_standard[140:])
    max_error_rob = np.max(errors_robust[140:])
    
    print("\n=== Performance Comparison (after outlier period) ===")
    print(f"Standard Kalman - Mean Error: {mean_error_std:.2f}kg, Max Error: {max_error_std:.2f}kg")
    print(f"Robust Kalman   - Mean Error: {mean_error_rob:.2f}kg, Max Error: {max_error_rob:.2f}kg")
    print(f"Improvement: {(1 - mean_error_rob/mean_error_std)*100:.1f}% reduction in mean error")
    
    outlier_count = len(robust_outliers)
    extreme_count = sum(1 for r in robust_results if r['outlier_type'] == 'extreme')
    moderate_count = sum(1 for r in robust_results if r['outlier_type'] == 'moderate')
    
    print(f"\n=== Outlier Detection Stats ===")
    print(f"Total outliers detected: {outlier_count}")
    print(f"  - Extreme outliers (rejected): {extreme_count}")
    print(f"  - Moderate outliers (dampened): {moderate_count}")
    
    plt.savefig('robust_kalman_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Testing Robust Kalman Filter with extreme outliers...")
    print("=" * 50)
    
    measurements, standard_results, robust_results = run_comparison()
    plot_results(measurements, standard_results, robust_results)
    
    print("\n✅ Test complete! See robust_kalman_comparison.png for visual results.")