"""
Test strict unit validation - no BMI detection, no assumptions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from src.processing.validation import DataQualityPreprocessor
from src.constants import SUPPORTED_WEIGHT_UNITS


class TestStrictUnitValidation:
    """Test that unit validation is strict and rejects unsupported units."""
    
    def test_supported_units_are_processed(self):
        """Test that all supported units are correctly processed."""
        test_cases = [
            ('kg', 70.0, 70.0),
            ('kilogram', 70.0, 70.0),
            ('kilograms', 70.0, 70.0),
            ('lb', 154.0, 69.85),  # 154 lb ≈ 69.85 kg
            ('lbs', 154.0, 69.85),
            ('pound', 154.0, 69.85),
            ('pounds', 154.0, 69.85),
            ('st', 11.0, 69.85),  # 11 stone ≈ 69.85 kg
            ('stone', 11.0, 69.85),
            ('stones', 11.0, 69.85),
            ('g', 70000.0, 70.0),  # 70000 g = 70 kg
            ('gram', 70000.0, 70.0),
            ('grams', 70000.0, 70.0),
        ]
        
        for unit, input_weight, expected_kg in test_cases:
            result, metadata = DataQualityPreprocessor.preprocess(
                weight=input_weight,
                source='test',
                timestamp=datetime.now(),
                user_id='test_user',
                unit=unit
            )
            
            assert result is not None, f"Unit '{unit}' should be supported but was rejected: {metadata.get('rejected')}"
            assert abs(result - expected_kg) < 0.1, f"Unit '{unit}' conversion incorrect: got {result}, expected {expected_kg}"
            assert 'unit_validation' in metadata.get('checks_passed', [])
    
    def test_unsupported_units_are_rejected(self):
        """Test that unsupported units are rejected."""
        unsupported_units = [
            'oz', 'ounce', 'ounces',
            't', 'ton', 'tons',
            'mg', 'milligram', 'milligrams',
            'μg', 'microgram', 'micrograms',
            'invalid', 'unknown', 'test',
            'kgs',  # Common typo - not in whitelist
            'lbm',  # Pound-mass variant
            'kilo',  # Abbreviation not in whitelist
        ]
        
        for unit in unsupported_units:
            result, metadata = DataQualityPreprocessor.preprocess(
                weight=70.0,
                source='test',
                timestamp=datetime.now(),
                user_id='test_user',
                unit=unit
            )
            
            assert result is None, f"Unsupported unit '{unit}' should be rejected"
            assert 'rejected' in metadata
            assert 'Unsupported unit' in metadata['rejected'] or 'not supported' in metadata['rejected']
    
    def test_missing_unit_is_rejected(self):
        """Test that missing unit is rejected (no default to kg)."""
        test_cases = [
            None,
            '',
            '   ',  # Whitespace only
        ]
        
        for unit in test_cases:
            result, metadata = DataQualityPreprocessor.preprocess(
                weight=70.0,
                source='test',
                timestamp=datetime.now(),
                user_id='test_user',
                unit=unit
            )
            
            assert result is None, f"Missing unit '{unit}' should be rejected, not defaulted"
            assert 'rejected' in metadata, f"Missing 'rejected' key for unit '{unit}'"
            rejected_msg = metadata['rejected'].lower()
            assert 'missing unit' in rejected_msg or 'unsupported unit' in rejected_msg or 'explicit unit' in rejected_msg, \
                f"Unexpected rejection message for unit '{unit}': {metadata['rejected']}"
    
    def test_no_bmi_detection_or_conversion(self):
        """Test that BMI values are NOT detected or converted to weight."""
        # Use weight values that would be valid weights but could be mistaken for BMI
        # These are in the range where old system might have detected them as BMI
        # but are still physiologically valid as weights
        test_values = [
            (50.0, 'kg'),  # 50kg is a valid weight, could be mistaken for BMI
            (60.0, 'kg'),  # 60kg is a valid weight
            (70.0, 'kg'),  # 70kg is a valid weight
            (80.0, 'kg'),  # 80kg is a valid weight
        ]
        
        for value, unit in test_values:
            result, metadata = DataQualityPreprocessor.preprocess(
                weight=value,
                source='test',
                timestamp=datetime.now(),
                user_id='test_user',
                unit=unit
            )
            
            # Should return the value unchanged (no BMI conversion)
            if result is not None:  # Only check if not rejected for other reasons
                assert result == value, f"Value {value} should not be converted from BMI"
            
            # Check that no BMI conversion happened
            assert 'Converted BMI' not in str(metadata.get('corrections', [])), \
                f"Should not convert BMI for {value}"
            assert 'likely BMI' not in str(metadata.get('warnings', [])), \
                f"Should not warn about BMI for {value}"
    
    def test_case_insensitive_unit_matching(self):
        """Test that unit matching is case-insensitive."""
        test_cases = [
            ('KG', 70.0),
            ('Kg', 70.0),
            ('KILOGRAM', 70.0),
            ('LB', 154.0),
            ('Pound', 154.0),
            ('STONE', 11.0),
        ]
        
        for unit, weight in test_cases:
            result, metadata = DataQualityPreprocessor.preprocess(
                weight=weight,
                source='test',
                timestamp=datetime.now(),
                user_id='test_user',
                unit=unit
            )
            
            assert result is not None, f"Unit '{unit}' should be case-insensitive"
    
    def test_unit_with_whitespace(self):
        """Test that units with leading/trailing whitespace are handled."""
        test_cases = [
            ('  kg  ', 70.0),
            ('\tpound\t', 154.0),
            (' stone ', 11.0),
        ]
        
        for unit, weight in test_cases:
            result, metadata = DataQualityPreprocessor.preprocess(
                weight=weight,
                source='test',
                timestamp=datetime.now(),
                user_id='test_user',
                unit=unit
            )
            
            assert result is not None, f"Unit '{unit}' with whitespace should be handled"
    
    def test_physiological_limits_still_enforced(self):
        """Test that physiological BMI limits still cause rejection (but no conversion)."""
        # Extremely low weight that would create impossible BMI
        result, metadata = DataQualityPreprocessor.preprocess(
            weight=5.0,  # 5kg would be impossibly low BMI
            source='test',
            timestamp=datetime.now(),
            user_id='test_user',
            unit='kg'
        )
        
        assert result is None, "Physiologically impossible weights should still be rejected"
        assert 'physiologically impossible' in metadata['rejected'].lower()
        
        # Extremely high weight
        result, metadata = DataQualityPreprocessor.preprocess(
            weight=500.0,  # 500kg would be impossibly high BMI
            source='test',
            timestamp=datetime.now(),
            user_id='test_user',
            unit='kg'
        )
        
        assert result is None, "Physiologically impossible weights should still be rejected"
        assert 'physiologically impossible' in metadata['rejected'].lower()


if __name__ == "__main__":
    # Run tests
    test = TestStrictUnitValidation()
    
    print("Testing supported units...")
    test.test_supported_units_are_processed()
    print("✓ Supported units work correctly")
    
    print("\nTesting unsupported units...")
    test.test_unsupported_units_are_rejected()
    print("✓ Unsupported units are rejected")
    
    print("\nTesting missing units...")
    test.test_missing_unit_is_rejected()
    print("✓ Missing units are rejected (no default)")
    
    print("\nTesting BMI detection is disabled...")
    test.test_no_bmi_detection_or_conversion()
    print("✓ BMI detection/conversion is disabled")
    
    print("\nTesting case sensitivity...")
    test.test_case_insensitive_unit_matching()
    print("✓ Unit matching is case-insensitive")
    
    print("\nTesting whitespace handling...")
    test.test_unit_with_whitespace()
    print("✓ Whitespace in units is handled")
    
    print("\nTesting physiological limits...")
    test.test_physiological_limits_still_enforced()
    print("✓ Physiological limits still enforced")
    
    print("\n✅ All unit validation tests passed!")