"""
Test threshold consistency and unit handling.
Verifies the fix for threshold unit mismatch issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import numpy as np

from src.threshold_calculator import ThresholdCalculator, ThresholdResult
from src.processor_enhanced import process_weight_enhanced
from src.processor_database import get_state_db


def test_threshold_calculator_units():
    """Test that threshold calculator handles units correctly."""
    
    print("\n=== Testing Threshold Calculator Units ===")
    
    # Test case 1: Percentage request
    result_pct = ThresholdCalculator.get_extreme_deviation_threshold(
        source='patient-upload',
        time_gap_days=1,
        current_weight=75.0,
        unit='percentage'
    )
    
    assert result_pct.unit == 'percentage'
    assert 0.03 <= result_pct.value <= 0.30  # Should be bounded
    print(f"✓ Percentage threshold: {result_pct.value:.3f} ({result_pct.value*100:.1f}%)")
    
    # Test case 2: Kg request
    result_kg = ThresholdCalculator.get_extreme_deviation_threshold(
        source='patient-upload',
        time_gap_days=1,
        current_weight=75.0,
        unit='kg'
    )
    
    assert result_kg.unit == 'kg'
    assert 3.0 <= result_kg.value <= 20.0  # Should be bounded
    print(f"✓ Kg threshold: {result_kg.value:.1f}kg")
    
    # Test case 3: Conversion consistency
    manual_pct = result_kg.value / 75.0
    # Account for bounds
    bounded_manual = max(0.03, min(manual_pct, 0.30))
    assert abs(bounded_manual - result_pct.value) < 0.001
    print(f"✓ Conversion consistent: {result_kg.value:.1f}kg = {bounded_manual:.3f} of 75kg")


def test_source_reliability_adaptation():
    """Test that different sources get appropriate thresholds."""
    
    print("\n=== Testing Source Reliability Adaptation ===")
    
    sources = [
        ('care-team-upload', 'excellent'),
        ('patient-upload', 'excellent'),
        ('patient-device', 'good'),
        ('https://connectivehealth.io', 'moderate'),
        ('https://api.iglucose.com', 'poor'),
        ('unknown-source', 'unknown')
    ]
    
    for source, expected_reliability in sources:
        # Get threshold in kg for comparison
        result = ThresholdCalculator.get_extreme_deviation_threshold(
            source=source,
            time_gap_days=1,
            current_weight=75.0,
            unit='kg'
        )
        
        reliability = ThresholdCalculator.get_source_reliability(source)
        assert reliability == expected_reliability
        
        print(f"✓ {source:30s} -> {reliability:10s} -> {result.value:5.1f}kg threshold")
    
    # Verify ordering: excellent sources should have higher thresholds (more lenient)
    excellent_threshold = ThresholdCalculator.get_extreme_deviation_threshold(
        'care-team-upload', 1, 75.0, 'kg'
    ).value
    
    poor_threshold = ThresholdCalculator.get_extreme_deviation_threshold(
        'https://api.iglucose.com', 1, 75.0, 'kg'
    ).value
    
    assert excellent_threshold > poor_threshold
    print(f"✓ Excellent source more lenient: {excellent_threshold:.1f}kg > {poor_threshold:.1f}kg")


def test_weight_scaling():
    """Test that thresholds scale appropriately with weight."""
    
    print("\n=== Testing Weight Scaling ===")
    
    weights = [40, 75, 150, 200]
    
    for weight in weights:
        result_pct = ThresholdCalculator.get_extreme_deviation_threshold(
            source='patient-device',
            time_gap_days=1,
            current_weight=float(weight),
            unit='percentage'
        )
        
        result_kg = ThresholdCalculator.get_extreme_deviation_threshold(
            source='patient-device',
            time_gap_days=1,
            current_weight=float(weight),
            unit='kg'
        )
        
        # Percentage should be relatively stable
        assert 0.03 <= result_pct.value <= 0.30
        
        # Kg threshold should be reasonable for the weight
        expected_kg = result_pct.value * weight
        # Account for independent kg calculation
        assert abs(result_kg.value - expected_kg) < weight * 0.1  # Within 10%
        
        print(f"✓ Weight {weight:3d}kg: {result_pct.value*100:5.1f}% = {result_kg.value:5.1f}kg")


def test_time_gap_scaling():
    """Test that thresholds scale with time gaps."""
    
    print("\n=== Testing Time Gap Scaling ===")
    
    time_gaps = [0.5, 1, 7, 14, 30]
    
    for gap in time_gaps:
        result = ThresholdCalculator.get_extreme_deviation_threshold(
            source='patient-device',
            time_gap_days=gap,
            current_weight=75.0,
            unit='kg'
        )
        
        print(f"✓ Gap {gap:4.1f} days: {result.value:5.1f}kg threshold")
    
    # Longer gaps should allow more change
    short_gap = ThresholdCalculator.get_extreme_deviation_threshold(
        'patient-device', 1, 75.0, 'kg'
    ).value
    
    long_gap = ThresholdCalculator.get_extreme_deviation_threshold(
        'patient-device', 30, 75.0, 'kg'
    ).value
    
    assert long_gap >= short_gap
    print(f"✓ Longer gaps more lenient: {long_gap:.1f}kg >= {short_gap:.1f}kg")


def test_physiological_limits():
    """Test physiological limit calculations."""
    
    print("\n=== Testing Physiological Limits ===")
    
    config = {
        'physiological': {
            'enable_physiological_limits': True,
            'max_change_1h_pct': 0.02,
            'max_change_1h_kg': 3.0,
            'max_change_24h_pct': 0.035,
            'max_change_24h_kg': 5.0,
            'max_sustained_kg_per_day': 1.5,
            'limit_tolerance': 0.10,
            'sustained_tolerance': 0.25
        }
    }
    
    test_cases = [
        (0.5, "hydration/bathroom"),
        (6, "daily fluctuation"),
        (24, "daily fluctuation"),
        (48, "sustained")
    ]
    
    for hours, expected_reason_part in test_cases:
        result = ThresholdCalculator.get_physiological_limit(
            time_delta_hours=hours,
            last_weight=75.0,
            config=config,
            unit='kg'
        )
        
        assert result.unit == 'kg'
        assert expected_reason_part in result.metadata['reason']
        print(f"✓ {hours:3.0f}h gap: {result.value:5.1f}kg - {result.metadata['reason']}")


def test_enhanced_processor_integration():
    """Test that enhanced processor correctly uses threshold calculator."""
    
    print("\n=== Testing Enhanced Processor Integration ===")
    
    user_id = "test_threshold_user"
    
    # Clear any existing state
    db = get_state_db()
    db.delete_state(user_id)
    
    # Process measurements with different sources
    sources = [
        ('patient-upload', 75.0),
        ('https://api.iglucose.com', 85.0),  # 10kg jump from unreliable source
        ('care-team-upload', 76.0),  # Back to normal from reliable source
    ]
    
    timestamp = datetime.now()
    
    config = {
        'min_weight': 30,
        'max_weight': 400,
        'extreme_threshold': 0.15,  # This should be overridden
        'physiological': {
            'enable_physiological_limits': True,
            'max_change_24h_pct': 0.035,
            'max_change_24h_kg': 5.0,
        }
    }
    
    kalman_config = {
        'initial_variance': 1.0,
        'transition_covariance_weight': 0.01,
        'transition_covariance_trend': 0.0001,
        'observation_covariance': 1.0,
        'reset_gap_days': 30
    }
    
    for i, (source, weight) in enumerate(sources):
        result = process_weight_enhanced(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp + timedelta(hours=i*24),
            source=source,
            processing_config=config,
            kalman_config=kalman_config,
            unit='kg'
        )
        
        if result:
            # Check that threshold info is present
            assert 'threshold_info' in result
            assert 'extreme_threshold_pct' in result['threshold_info']
            assert 'extreme_threshold_kg' in result['threshold_info']
            assert 'source_reliability' in result['threshold_info']
            
            reliability = result['threshold_info']['source_reliability']
            threshold_pct = result['threshold_info']['extreme_threshold_pct']
            threshold_kg = result['threshold_info']['extreme_threshold_kg']
            accepted = result.get('accepted', False)
            
            print(f"✓ {source:30s} {weight:5.1f}kg -> "
                  f"Reliability: {reliability:10s} "
                  f"Threshold: {threshold_pct*100:4.1f}% ({threshold_kg:.1f}kg) "
                  f"Accepted: {accepted}")


def test_problematic_user_case():
    """Test the specific user case that revealed the bug."""
    
    print("\n=== Testing Problematic User Case ===")
    
    # Simulate the scenario for user 0040872d-333a-4ace-8c5a-b2fcd056e65a
    # This user had noisy data that was incorrectly rejected
    
    user_id = "0040872d-333a-4ace-8c5a-b2fcd056e65a"
    
    db = get_state_db()
    db.delete_state(user_id)
    
    # Simulate noisy measurements around 75kg
    measurements = [
        (75.0, 'patient-device'),
        (85.0, 'https://api.iglucose.com'),  # Noisy but possible
        (73.0, 'patient-device'),
        (88.0, 'https://api.iglucose.com'),  # Another noisy reading
        (74.0, 'patient-upload'),
    ]
    
    config = {
        'min_weight': 30,
        'max_weight': 400,
        'extreme_threshold': 0.15,
        'physiological': {
            'enable_physiological_limits': True,
        }
    }
    
    kalman_config = {
        'initial_variance': 1.0,
        'transition_covariance_weight': 0.01,
        'transition_covariance_trend': 0.0001,
        'observation_covariance': 1.0,
        'reset_gap_days': 30
    }
    
    timestamp = datetime.now()
    accepted_count = 0
    rejected_count = 0
    
    for i, (weight, source) in enumerate(measurements):
        result = process_weight_enhanced(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp + timedelta(hours=i*6),
            source=source,
            processing_config=config,
            kalman_config=kalman_config,
            unit='kg'
        )
        
        if result:
            accepted = result.get('accepted', False)
            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1
            
            filtered = result.get('filtered_weight', weight)
            print(f"  Measurement {i+1}: {weight:5.1f}kg from {source:30s} -> "
                  f"{'Accepted' if accepted else 'Rejected':8s} "
                  f"(filtered: {filtered:.1f}kg)")
    
    # Should not reject all noisy measurements
    assert accepted_count > 0, "All measurements were rejected!"
    print(f"\n✓ Accepted {accepted_count}/{len(measurements)} measurements")
    print(f"✓ System handles noisy data appropriately")


def main():
    """Run all tests."""
    
    print("=" * 60)
    print("THRESHOLD CONSISTENCY TESTS")
    print("=" * 60)
    
    test_threshold_calculator_units()
    test_source_reliability_adaptation()
    test_weight_scaling()
    test_time_gap_scaling()
    test_physiological_limits()
    test_enhanced_processor_integration()
    test_problematic_user_case()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()