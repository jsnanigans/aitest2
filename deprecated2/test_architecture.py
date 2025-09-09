#!/usr/bin/env python3
"""
Test script for the new clean architecture.
Tests each layer independently and then the full pipeline.
"""

import numpy as np
from datetime import datetime, timedelta
import json

from src.core.types import WeightMeasurement
from src.filters.layer1_heuristic import Layer1Pipeline
from src.filters.layer3_kalman import PureKalmanFilter, ValidationGate
from src.processing.robust_baseline import RobustBaselineEstimator
from src.processing.weight_pipeline import WeightProcessingPipeline


def generate_test_data():
    """Generate synthetic test data with known patterns."""
    measurements = []
    base_weight = 70.0
    start_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        timestamp = start_date + timedelta(days=day)
        
        # Normal daily variation (±1-2%)
        daily_variation = np.random.normal(0, 0.5)
        
        # Weekly pattern (higher on weekends)
        weekly_effect = 0.5 if timestamp.weekday() in [5, 6] else 0
        
        # Trend (slight loss)
        trend_effect = -0.02 * day
        
        # Add some outliers
        if day in [5, 15, 25]:  # Inject outliers
            weight = base_weight + np.random.choice([-10, 10, 15])
        else:
            weight = base_weight + daily_variation + weekly_effect + trend_effect
            
        measurements.append(WeightMeasurement(
            weight=weight,
            timestamp=timestamp,
            source_type="test"
        ))
        
    return measurements


def test_layer1():
    """Test Layer 1: Heuristic filters."""
    print("\n=== Testing Layer 1: Heuristic Filters ===")
    
    layer1 = Layer1Pipeline()
    test_cases = [
        (25.0, False, "Below physiological minimum"),
        (70.0, True, "Normal weight"),
        (450.0, False, "Above physiological maximum"),
        (71.0, True, "Small change from 70kg"),
        (85.0, False, "Large jump from 70kg (if previous was 70)")
    ]
    
    for weight, expected_valid, description in test_cases:
        measurement = WeightMeasurement(
            weight=weight,
            timestamp=datetime.now()
        )
        is_valid, outlier_type, metadata = layer1.process(measurement)
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"{status} {description}: weight={weight}kg, valid={is_valid}, type={outlier_type}")
        
        
def test_kalman():
    """Test Layer 3: Pure Kalman filter."""
    print("\n=== Testing Layer 3: Kalman Filter ===")
    
    # Initialize with baseline
    kalman = PureKalmanFilter(
        initial_weight=70.0,
        initial_variance=1.0,
        process_noise_weight=0.5,
        process_noise_trend=0.01,
        measurement_noise=0.5
    )
    
    # Test measurements
    measurements = [70.5, 70.3, 70.8, 70.2, 69.9]
    
    for i, weight in enumerate(measurements):
        measurement = WeightMeasurement(
            weight=weight,
            timestamp=datetime.now() + timedelta(days=i)
        )
        
        results = kalman.process_measurement(measurement)
        print(f"Day {i}: measured={weight:.1f}, "
              f"filtered={results['filtered_weight']:.1f}, "
              f"trend={results['trend_kg_per_day']:.3f} kg/day")
        

def test_baseline():
    """Test baseline establishment."""
    print("\n=== Testing Baseline Establishment ===")
    
    estimator = RobustBaselineEstimator()
    measurements = generate_test_data()[:7]  # Use first week
    
    result = estimator.establish_baseline(measurements)
    
    if result.success:
        print(f"✓ Baseline established: {result.baseline_weight:.1f}kg")
        print(f"  Confidence: {result.confidence}")
        print(f"  Variance: {result.measurement_variance:.3f}")
        print(f"  Readings used: {result.readings_used}")
    else:
        print(f"✗ Baseline failed: {result.error}")
        

def test_full_pipeline():
    """Test complete pipeline integration."""
    print("\n=== Testing Full Pipeline ===")
    
    # Create pipeline
    pipeline = WeightProcessingPipeline({
        'validation_gamma': 3.0,
        'layer1': {'mad_threshold': 3.0},
        'kalman': {'process_noise_weight': 0.5}
    })
    
    # Generate test data
    measurements = generate_test_data()
    
    # Initialize with first week
    baseline_result = pipeline.initialize_user(measurements[:7])
    print(f"Initialization: {baseline_result.success}")
    
    if baseline_result.success:
        print(f"Baseline: {baseline_result.baseline_weight:.1f}kg "
              f"(confidence={baseline_result.confidence})")
        
        # Process remaining measurements
        accepted = 0
        rejected = 0
        
        for m in measurements[7:]:
            result = pipeline.process_measurement(m)
            
            if result.is_valid:
                accepted += 1
                print(f"  Day {m.timestamp.day}: {m.weight:.1f}kg → "
                      f"{result.filtered_weight:.1f}kg (confidence={result.confidence:.2f})")
            else:
                rejected += 1
                print(f"  Day {m.timestamp.day}: {m.weight:.1f}kg REJECTED "
                      f"({result.rejection_reason})")
                
        print(f"\nAccepted: {accepted}, Rejected: {rejected}")
        print(f"Acceptance rate: {accepted/(accepted+rejected):.1%}")
        
        # Get final state
        state = pipeline.get_state_summary()
        print(f"\nFinal state:")
        print(f"  Weight: {state['current_state']['weight']:.1f}kg")
        print(f"  Trend: {state['current_state']['trend_kg_per_week']:.2f}kg/week")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING NEW CLEAN ARCHITECTURE")
    print("=" * 60)
    
    test_layer1()
    test_kalman()
    test_baseline()
    test_full_pipeline()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()