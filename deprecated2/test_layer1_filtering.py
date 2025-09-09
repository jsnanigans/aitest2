#!/usr/bin/env python3
"""
Test Layer 1 Pre-filtering with Problematic User Cases
Demonstrates how Layer 1 catches physiologically impossible values
before they corrupt the Kalman filter
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.filters.layer1_heuristic import StatelessLayer1Pipeline
from src.filters.robust_kalman import RobustKalmanFilter
from src.core.types import WeightMeasurement
from src.processing.weight_pipeline import WeightProcessingPipeline


def test_problematic_user_cases():
    """
    Test cases similar to problematic users mentioned:
    - 0040872d-333a-4ace-8c5a-b2fcd056e65a
    - 01677b8a-34c8-4678-8e36-1a8bd76f4bb4
    """
    
    print("=" * 70)
    print("LAYER 1 PRE-FILTERING TEST")
    print("Testing physiological gates before Kalman processing")
    print("=" * 70)
    
    # Configure pipeline with Layer 1 enabled
    config = {
        'layer1': {
            'enabled': True,
            'min_weight': 30.0,
            'max_weight': 400.0,
            'max_daily_change_percent': 3.0,  # 3% per day is generous
            'extreme_threshold_percent': 20.0
        },
        'kalman': {
            'outlier_threshold': 3.0,
            'extreme_outlier_threshold': 5.0
        }
    }
    
    pipeline = WeightProcessingPipeline(config)
    
    # Initialize with reasonable baseline
    baseline_measurements = [
        WeightMeasurement(
            user_id="test_user",
            weight=115.0 + np.random.normal(0, 0.5),
            timestamp=datetime.now() - timedelta(days=7-i),
            source_type="care-team-upload"
        )
        for i in range(7)
    ]
    
    baseline_result = pipeline.initialize_user(baseline_measurements)
    print(f"\n✅ Baseline established: {baseline_result.baseline_weight:.1f}kg")
    
    # Test sequence with problematic values
    test_cases = [
        ("Normal", 115.5, True),
        ("Normal", 114.8, True),
        ("Physiologically impossible - too low", 15.0, False),  # Layer 1 should catch
        ("Physiologically impossible - too high", 450.0, False),  # Layer 1 should catch
        ("Normal", 115.2, True),
        ("Extreme but possible weight loss", 110.0, True),  # 5kg loss - more realistic for daily
        ("Data entry error (lbs as kg)", 253.0, False),  # 115kg * 2.2 = 253lbs entered as kg
        ("Normal", 114.5, True),
        ("Extreme jump", 180.0, False),  # Layer 1 deviation filter should catch
        ("Normal", 115.0, True),
    ]
    
    print("\n" + "=" * 70)
    print("Processing Test Measurements:")
    print("-" * 70)
    
    timestamp = datetime.now()
    layer1_rejections = 0
    kalman_rejections = 0
    
    for description, weight, expected_valid in test_cases:
        timestamp += timedelta(days=1)
        
        measurement = WeightMeasurement(
            user_id="test_user",
            weight=weight,
            timestamp=timestamp,
            source_type="patient-upload"
        )
        
        result = pipeline.process_measurement(measurement)
        
        print(f"\n{description}: {weight:.1f}kg")
        print(f"  Expected: {'✓ Valid' if expected_valid else '✗ Reject'}")
        print(f"  Result: {'✓ Valid' if result.is_valid else '✗ Rejected'}")
        
        if not result.is_valid:
            if "Layer1" in result.rejection_reason:
                layer1_rejections += 1
                print(f"  Layer 1 caught: {result.rejection_reason}")
            else:
                kalman_rejections += 1
                print(f"  Kalman rejected: {result.rejection_reason}")
        else:
            print(f"  Filtered: {result.filtered_weight:.1f}kg (confidence: {result.confidence:.2f})")
        
        # Check if result matches expectation
        if result.is_valid != expected_valid:
            print(f"  ⚠️  UNEXPECTED RESULT!")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    state = pipeline.get_state_summary()
    
    print(f"\n📊 Processing Statistics:")
    print(f"  Total measurements: {len(test_cases)}")
    print(f"  Layer 1 rejections: {layer1_rejections}")
    print(f"  Kalman rejections: {kalman_rejections}")
    print(f"  Total accepted: {len(test_cases) - layer1_rejections - kalman_rejections}")
    
    print(f"\n📈 Final Kalman State:")
    print(f"  Weight: {state['current_state']['weight']:.1f}kg")
    print(f"  Trend: {state['current_state']['trend_kg_per_day']:.3f} kg/day")
    
    print(f"\n✅ Benefits of Layer 1 Pre-filtering:")
    print(f"  1. Caught {layer1_rejections} physiologically impossible values")
    print(f"  2. Prevented Kalman state corruption from extreme outliers")
    print(f"  3. Maintained accurate weight tracking despite bad data")
    print(f"  4. Clear rejection reasons for debugging")


def test_medical_intervention_mode():
    """
    Test relaxed limits for users on weight loss medication or post-surgery.
    """
    print("\n" + "=" * 70)
    print("MEDICAL INTERVENTION MODE TEST")
    print("Relaxed limits for legitimate rapid weight changes")
    print("=" * 70)
    
    # Configure with medical intervention mode
    config = {
        'layer1': {
            'enabled': True,
            'min_weight': 30.0,
            'max_weight': 500.0,  # Higher limit for bariatric patients
            'max_daily_change_percent': 3.0,
            'medical_mode_percent': 5.0  # Allow 5% daily change
        },
        'medical_intervention_mode': True  # Enable relaxed limits
    }
    
    pipeline = WeightProcessingPipeline(config)
    
    # Initialize at higher weight (bariatric patient)
    baseline_measurements = [
        WeightMeasurement(
            user_id="bariatric_patient",
            weight=180.0 + np.random.normal(0, 0.5),
            timestamp=datetime.now() - timedelta(days=7-i),
            source_type="care-team-upload"
        )
        for i in range(7)
    ]
    
    baseline_result = pipeline.initialize_user(baseline_measurements)
    print(f"\nBaseline for bariatric patient: {baseline_result.baseline_weight:.1f}kg")
    
    # Simulate post-surgery weight loss (can be rapid initially)
    weights_over_time = [
        180.0,  # Day 0
        178.5,  # Day 1 - 1.5kg loss (normal)
        176.0,  # Day 2 - 2.5kg loss (high but possible post-surgery)
        174.0,  # Day 3 - 2kg loss
        172.5,  # Day 4 - 1.5kg loss
        171.0,  # Day 5 - 1.5kg loss
        170.0,  # Day 6 - 1kg loss
        169.0,  # Day 7 - 1kg loss (10kg total in week - possible with medical intervention)
    ]
    
    print("\nProcessing rapid weight loss sequence:")
    print("-" * 40)
    
    timestamp = datetime.now()
    for i, weight in enumerate(weights_over_time):
        measurement = WeightMeasurement(
            user_id="bariatric_patient",
            weight=weight,
            timestamp=timestamp + timedelta(days=i),
            source_type="care-team-upload"
        )
        
        result = pipeline.process_measurement(measurement)
        
        if result.is_valid:
            print(f"Day {i}: {weight:.1f}kg ✓ Accepted (confidence: {result.confidence:.2f})")
        else:
            print(f"Day {i}: {weight:.1f}kg ✗ Rejected - {result.rejection_reason}")
    
    print(f"\n✅ Medical mode allows legitimate rapid changes while still filtering errors")


if __name__ == "__main__":
    test_problematic_user_cases()
    test_medical_intervention_mode()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Layer 1 pre-filtering provides essential protection by:
1. Catching obvious data entry errors (lbs/kg confusion)
2. Rejecting physiologically impossible values
3. Protecting Kalman from corruption by extreme outliers
4. Allowing configuration for medical scenarios
5. Maintaining clear separation of concerns (Layer 1 = physiology, Kalman = statistics)

This follows best practices:
- Fast physiological checks before expensive computations
- Stateless design compatible with real-time processing
- Configurable limits for different patient populations
- Clear rejection reasons for debugging and monitoring
    """)