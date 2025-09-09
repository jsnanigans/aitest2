#!/usr/bin/env python3
"""
Test harness to validate pipeline rejection logic with known data patterns.
Tests the full pipeline with realistic weight scenarios.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Tuple

from src.core.types import WeightMeasurement
from src.processing.weight_pipeline import WeightProcessingPipeline


def create_measurement(date: datetime, weight: float, source: str = "care-team-upload") -> WeightMeasurement:
    """Helper to create test measurements."""
    return WeightMeasurement(
        user_id="test_user",
        timestamp=date,
        weight=weight,
        source_type=source
    )


def create_timeline(start_date: datetime, weights: List[float], days_apart: int = 1) -> List[WeightMeasurement]:
    """Create a timeline of measurements."""
    measurements = []
    current_date = start_date
    
    for weight in weights:
        measurements.append(create_measurement(current_date, weight))
        current_date += timedelta(days=days_apart)
    
    return measurements


class TestPipelineValidation:
    """Test suite for pipeline validation logic."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = {
            'baseline': {
                'collection_days': 7,
                'min_readings': 3,
                'iqr_multiplier': 1.5
            },
            'kalman': {
                'process_noise_weight': 0.5,
                'process_noise_trend': 0.01
            },
            'validation_gamma': 3.0
        }
    
    def test_stable_weight_high_acceptance(self):
        """Test that stable weight patterns have high acceptance rate."""
        start_date = datetime(2025, 1, 1)
        
        # Stable weight around 80kg with small variations (±0.5kg)
        weights = [80.0, 80.2, 79.8, 80.1, 79.9, 80.3, 80.0, 79.7, 80.2, 80.1,
                  80.0, 79.8, 80.2, 80.1, 79.9, 80.0, 80.3, 79.8, 80.1, 80.0]
        
        measurements = create_timeline(start_date, weights)
        
        pipeline = WeightProcessingPipeline(self.config)
        
        # Initialize with first 7 days
        baseline_result = pipeline.initialize_user(measurements[:7])
        assert baseline_result.success, f"Baseline failed: {baseline_result.error}"
        
        # Process remaining measurements
        accepted = 0
        rejected = 0
        
        for measurement in measurements[7:]:
            result = pipeline.process_measurement(measurement)
            if result.is_valid:
                accepted += 1
            else:
                rejected += 1
        
        acceptance_rate = accepted / (accepted + rejected)
        
        assert acceptance_rate > 0.85, f"Stable weight acceptance rate too low: {acceptance_rate:.1%}"
        print(f"✓ Stable weight test: {acceptance_rate:.1%} acceptance")
    
    def test_gradual_weight_loss(self):
        """Test gradual weight loss pattern (diet scenario)."""
        start_date = datetime(2025, 1, 1)
        
        # Start at 90kg, lose ~0.2kg per day (realistic diet)
        weights = []
        current = 90.0
        for i in range(30):
            weights.append(current)
            current -= 0.2 + (0.1 if i % 3 == 0 else -0.05)  # Some variation
        
        measurements = create_timeline(start_date, weights)
        
        pipeline = WeightProcessingPipeline(self.config)
        
        # Initialize
        baseline_result = pipeline.initialize_user(measurements[:7])
        assert baseline_result.success
        
        # Process remaining
        accepted = 0
        rejected = 0
        
        for measurement in measurements[7:]:
            result = pipeline.process_measurement(measurement)
            if result.is_valid:
                accepted += 1
            else:
                rejected += 1
        
        acceptance_rate = accepted / (accepted + rejected)
        
        assert acceptance_rate > 0.75, f"Gradual loss acceptance rate too low: {acceptance_rate:.1%}"
        print(f"✓ Gradual weight loss test: {acceptance_rate:.1%} acceptance")
    
    def test_outlier_rejection(self):
        """Test that true outliers are rejected."""
        start_date = datetime(2025, 1, 1)
        
        # Normal weight with occasional outliers
        weights = [75.0, 75.2, 74.8, 75.1,  # Normal
                  40.0,  # Outlier (scale error)
                  75.0, 74.9,  # Normal
                  150.0,  # Outlier (wrong person?)
                  75.1, 75.3, 74.8,  # Normal
                  85.0,  # Outlier (too much change)
                  75.0, 75.2]  # Normal
        
        measurements = create_timeline(start_date, weights)
        
        pipeline = WeightProcessingPipeline(self.config)
        
        # Initialize with first few normal readings
        baseline_result = pipeline.initialize_user(measurements[:4])
        assert baseline_result.success
        
        # Track specific outliers
        outlier_indices = [4, 7, 11]  # Positions of outliers
        
        for i, measurement in enumerate(measurements[4:], start=4):
            result = pipeline.process_measurement(measurement)
            
            if i in outlier_indices:
                assert not result.is_valid, f"Failed to reject outlier at index {i}: {measurement.weight}kg"
            
        print(f"✓ Outlier rejection test: All outliers correctly rejected")
    
    def test_daily_fluctuations(self):
        """Test normal daily weight fluctuations (morning vs evening)."""
        start_date = datetime(2025, 1, 1)
        
        # Simulate morning (lower) and evening (higher) weights
        weights = []
        base = 70.0
        for day in range(14):
            # Morning weight
            weights.append(base + (day * 0.05))  # Slight trend
            # Evening weight (1-2kg higher)
            weights.append(base + (day * 0.05) + 1.5)
        
        # Create measurements with alternating times
        measurements = []
        current_date = start_date
        for i, weight in enumerate(weights):
            hour = 7 if i % 2 == 0 else 19  # 7am or 7pm
            timestamp = current_date.replace(hour=hour)
            measurements.append(create_measurement(timestamp, weight))
            if i % 2 == 1:  # After evening measurement, move to next day
                current_date += timedelta(days=1)
        
        pipeline = WeightProcessingPipeline(self.config)
        
        # Initialize
        baseline_result = pipeline.initialize_user(measurements[:7])
        assert baseline_result.success
        
        # Process remaining
        accepted = 0
        rejected = 0
        
        for measurement in measurements[7:]:
            result = pipeline.process_measurement(measurement)
            if result.is_valid:
                accepted += 1
            else:
                rejected += 1
        
        acceptance_rate = accepted / (accepted + rejected)
        
        assert acceptance_rate > 0.70, f"Daily fluctuation acceptance too low: {acceptance_rate:.1%}"
        print(f"✓ Daily fluctuation test: {acceptance_rate:.1%} acceptance")
    
    def test_data_gap_handling(self):
        """Test handling of data gaps (vacation, missed readings)."""
        start_date = datetime(2025, 1, 1)
        
        # Initial stable period
        weights_before = [80.0, 80.2, 79.8, 80.1, 79.9, 80.0, 80.3]
        
        # After 30-day gap (slight weight gain during vacation)
        weights_after = [82.0, 82.2, 81.8, 82.1, 81.9, 82.0]
        
        measurements_before = create_timeline(start_date, weights_before)
        
        # Create gap
        gap_start = start_date + timedelta(days=7)
        measurements_after = create_timeline(
            gap_start + timedelta(days=35),  # 35-day gap
            weights_after
        )
        
        all_measurements = measurements_before + measurements_after
        
        pipeline = WeightProcessingPipeline(self.config)
        
        # Initialize with pre-gap data
        baseline_result = pipeline.initialize_user(measurements_before)
        assert baseline_result.success
        
        # Process post-gap measurements
        accepted = 0
        rejected = 0
        
        for measurement in measurements_after:
            result = pipeline.process_measurement(measurement)
            if result.is_valid:
                accepted += 1
            else:
                rejected += 1
        
        # Should re-establish baseline after gap
        acceptance_rate = accepted / (accepted + rejected)
        
        assert acceptance_rate > 0.60, f"Post-gap acceptance too low: {acceptance_rate:.1%}"
        print(f"✓ Data gap handling test: {acceptance_rate:.1%} acceptance after gap")


def run_validation_tests():
    """Run all validation tests and report results."""
    print("\n" + "=" * 60)
    print("PIPELINE VALIDATION TEST SUITE")
    print("=" * 60)
    
    test_suite = TestPipelineValidation()
    
    # Run each test
    tests = [
        ("Stable Weight Pattern", test_suite.test_stable_weight_high_acceptance),
        ("Gradual Weight Loss", test_suite.test_gradual_weight_loss),
        ("Outlier Rejection", test_suite.test_outlier_rejection),
        ("Daily Fluctuations", test_suite.test_daily_fluctuations),
        ("Data Gap Handling", test_suite.test_data_gap_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        test_suite.setup_method()
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: Unexpected error: {str(e)}")
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Configuration is suitable for production.")
    else:
        print("⚠️  Some tests failed. Review configuration settings.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_validation_tests()