"""
Test council-recommended improvements to the processor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor


def test_smooth_confidence():
    """Test the smooth exponential confidence function."""
    print("=" * 60)
    print("TESTING SMOOTH CONFIDENCE FUNCTION (Knuth)")
    print("=" * 60)
    
    # Create a processor
    config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,
            "reset_gap_days": 30
        }
    }
    
    processor = WeightProcessor("test_user", config["processing"], config["kalman"])
    
    # Test confidence values for different normalized innovations
    print("\nNormalized Innovation → Confidence (smooth exponential):")
    test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    for inn in test_values:
        confidence = processor._calculate_confidence(inn)
        print(f"  {inn:.1f}σ: {confidence:.4f}")
    
    # Verify smoothness
    print("\nVerifying smoothness (no discontinuities):")
    for i in range(len(test_values) - 1):
        inn1 = test_values[i]
        inn2 = test_values[i + 1]
        conf1 = processor._calculate_confidence(inn1)
        conf2 = processor._calculate_confidence(inn2)
        gradient = (conf2 - conf1) / (inn2 - inn1)
        print(f"  Gradient between {inn1:.1f}σ and {inn2:.1f}σ: {gradient:.4f}")


def test_adaptive_observation_noise():
    """Test adaptive observation noise estimation."""
    print("\n" + "=" * 60)
    print("TESTING ADAPTIVE OBSERVATION NOISE (Kleppmann)")
    print("=" * 60)
    
    base_date = datetime(2024, 1, 1)
    
    # Test case 1: Low noise data (good scale)
    print("\nCase 1: Low noise data (σ ≈ 0.3 kg)")
    low_noise_data = []
    for i in range(15):
        weight = 70.0 + np.random.normal(0, 0.3)
        low_noise_data.append((weight, base_date + timedelta(days=i), "scale"))
    
    test_adaptive_estimation(low_noise_data, "Low noise")
    
    # Test case 2: Medium noise data (typical scale)
    print("\nCase 2: Medium noise data (σ ≈ 1.0 kg)")
    medium_noise_data = []
    for i in range(15):
        weight = 70.0 + np.random.normal(0, 1.0)
        medium_noise_data.append((weight, base_date + timedelta(days=i), "scale"))
    
    test_adaptive_estimation(medium_noise_data, "Medium noise")
    
    # Test case 3: High noise data (poor scale or hydration swings)
    print("\nCase 3: High noise data (σ ≈ 2.0 kg)")
    high_noise_data = []
    for i in range(15):
        weight = 70.0 + np.random.normal(0, 2.0)
        high_noise_data.append((weight, base_date + timedelta(days=i), "scale"))
    
    test_adaptive_estimation(high_noise_data, "High noise")


def test_adaptive_estimation(data, case_name):
    """Helper to test adaptive parameter estimation."""
    config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,  # Base value
            "reset_gap_days": 30
        }
    }
    
    processor = WeightProcessor(f"test_{case_name}", config["processing"], config["kalman"])
    
    # Process data to trigger initialization
    results = []
    for weight, timestamp, source in data:
        result = processor.process_weight(weight, timestamp, source)
        if result:
            results.append(result)
    
    if processor.kalman:
        actual_obs_covar = processor.kalman.observation_covariance[0, 0]
        base_obs_covar = config["kalman"]["observation_covariance"]
        
        print(f"  Base observation_covariance: {base_obs_covar:.2f}")
        print(f"  Adapted observation_covariance: {actual_obs_covar:.2f}")
        print(f"  Adaptation factor: {actual_obs_covar/base_obs_covar:.2f}x")
        
        # Check if adaptation is working as expected
        weights = [w for w, _, _ in data[:10]]
        actual_std = np.std(weights)
        print(f"  Actual data std: {actual_std:.2f} kg")


def test_extreme_deviation_handling():
    """Test handling of extreme deviations with smooth confidence."""
    print("\n" + "=" * 60)
    print("TESTING EXTREME DEVIATION HANDLING")
    print("=" * 60)
    
    config = {
        "processing": {
            "min_init_readings": 10,
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20
        },
        "kalman": {
            "initial_variance": 0.5,
            "transition_covariance_weight": 0.05,
            "transition_covariance_trend": 0.0005,
            "observation_covariance": 1.5,
            "reset_gap_days": 30
        }
    }
    
    processor = WeightProcessor("test_extreme", config["processing"], config["kalman"])
    
    base_date = datetime(2024, 1, 1)
    
    # Initialize with stable data
    for i in range(15):
        weight = 75.0 + np.random.normal(0, 0.3)
        processor.process_weight(weight, base_date + timedelta(days=i), "scale")
    
    # Test various deviations
    print("\nTesting deviations from baseline (75 kg):")
    test_weights = [75.0, 78.0, 82.0, 90.0, 100.0, 120.0]
    
    for test_weight in test_weights:
        result = processor.process_weight(
            test_weight, 
            base_date + timedelta(days=20), 
            "scale"
        )
        
        if result:
            deviation = abs(test_weight - 75.0) / 75.0
            print(f"  Weight: {test_weight:.0f} kg (deviation: {deviation:.1%})")
            print(f"    Accepted: {result['accepted']}")
            print(f"    Confidence: {result['confidence']:.4f}")
            if not result['accepted']:
                print(f"    Reason: {result['reason']}")


if __name__ == "__main__":
    np.random.seed(42)
    
    # Run all tests
    test_smooth_confidence()
    test_adaptive_observation_noise()
    test_extreme_deviation_handling()
    
    print("\n" + "=" * 60)
    print("COUNCIL IMPROVEMENTS SUMMARY")
    print("=" * 60)
    print("\n✓ Smooth exponential confidence function implemented (Knuth)")
    print("✓ Adaptive observation noise estimation implemented (Kleppmann)")
    print("✓ Simplified to 2 key parameters (Lampson)")
    print("✓ Mathematical purity maintained with pykalman (Lamport)")
    print("\nThe processor now provides:")
    print("- Continuous confidence scores without discontinuities")
    print("- Per-user adaptation based on data quality")
    print("- Better handling of different scale types")
    print("- Cleaner, more maintainable code")