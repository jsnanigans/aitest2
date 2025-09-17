"""
Tests for BMIValidator focusing on key functionality.
"""

import pytest
from datetime import datetime, timedelta
from src.processing.validation import BMIValidator
from src.constants import BMI_LIMITS, PHYSIOLOGICAL_LIMITS


class TestBMIValidator:

    def test_calculate_bmi_normal(self):
        """Test BMI calculation with normal values."""
        bmi = BMIValidator.calculate_bmi(70.0, 1.75)
        assert pytest.approx(bmi, 0.01) == 22.86

    def test_calculate_bmi_zero_height(self):
        """Test BMI calculation handles zero height."""
        bmi = BMIValidator.calculate_bmi(70.0, 0)
        assert bmi == 0

    def test_is_likely_bmi_by_unit(self):
        """Test BMI detection by unit."""
        assert BMIValidator.is_likely_bmi(25.0, 'bmi') is True
        assert BMIValidator.is_likely_bmi(25.0, 'kg/m2') is True
        assert BMIValidator.is_likely_bmi(25.0, 'kg/m^2') is True

    def test_is_likely_bmi_by_value_range(self):
        """Test BMI detection by value range for kg unit."""
        assert BMIValidator.is_likely_bmi(25.0, 'kg') is True
        assert BMIValidator.is_likely_bmi(35.0, 'kg') is True
        assert BMIValidator.is_likely_bmi(70.0, 'kg') is False
        assert BMIValidator.is_likely_bmi(10.0, 'kg') is False

    def test_convert_bmi_to_weight(self):
        """Test BMI to weight conversion."""
        weight = BMIValidator.convert_bmi_to_weight(25.0, 1.75)
        assert pytest.approx(weight, 0.01) == 76.56

    def test_validate_bmi_valid(self):
        """Test valid BMI passes validation."""
        valid, reason = BMIValidator.validate_bmi(22.0)
        assert valid is True
        assert reason is None

    def test_validate_bmi_too_low(self):
        """Test extremely low BMI is rejected."""
        valid, reason = BMIValidator.validate_bmi(8.0)
        assert valid is False
        assert "below physiological minimum" in reason

    def test_validate_bmi_too_high(self):
        """Test extremely high BMI is rejected."""
        valid, reason = BMIValidator.validate_bmi(105.0)  # Above IMPOSSIBLE_HIGH (100)
        assert valid is False
        assert "above physiological maximum" in reason

    def test_categorize_bmi(self):
        """Test BMI categorization."""
        assert BMIValidator.categorize_bmi(17.0) == 'underweight'
        assert BMIValidator.categorize_bmi(22.0) == 'normal'
        assert BMIValidator.categorize_bmi(27.0) == 'overweight'
        assert BMIValidator.categorize_bmi(35.0) == 'obese'

    def test_detect_and_convert_pounds(self):
        """Test pound to kg conversion."""
        weight, converted, metadata = BMIValidator.detect_and_convert(
            154.0, 'lb', 1.75
        )
        assert pytest.approx(weight, 0.1) == 69.9
        assert converted is False
        assert 'conversion' in metadata
        assert 'lb to' in metadata['conversion']

    def test_detect_and_convert_stones(self):
        """Test stone to kg conversion."""
        weight, converted, metadata = BMIValidator.detect_and_convert(
            11.0, 'st', 1.75
        )
        assert pytest.approx(weight, 0.1) == 69.9
        assert converted is False
        assert 'conversion' in metadata
        assert 'st to' in metadata['conversion']

    def test_detect_and_convert_bmi_to_weight(self):
        """Test BMI detection and conversion to weight."""
        weight, converted, metadata = BMIValidator.detect_and_convert(
            25.0, 'kg', 1.75
        )
        assert pytest.approx(weight, 0.1) == 76.6
        assert converted is True
        assert metadata['detected_as_bmi'] is True
        assert 'BMI' in metadata['conversion']

    def test_detect_and_convert_normal_weight(self):
        """Test normal weight passes through unchanged."""
        weight, converted, metadata = BMIValidator.detect_and_convert(
            70.0, 'kg', 1.75
        )
        assert weight == 70.0
        assert converted is False
        assert metadata['original_value'] == 70.0

    def test_validate_weight_bmi_consistency_valid(self):
        """Test weight-BMI consistency validation for valid measurement."""
        result = BMIValidator.validate_weight_bmi_consistency(
            70.0, 1.75
        )
        assert result['valid'] is True
        assert pytest.approx(result['bmi'], 0.1) == 22.9
        assert result['bmi_category'] == 'normal'
        assert len(result['warnings']) == 0

    def test_validate_weight_bmi_consistency_suspicious_low(self):
        """Test suspicious low BMI generates warning."""
        result = BMIValidator.validate_weight_bmi_consistency(
            38.0, 1.75  # BMI ~12.4, below SUSPICIOUS_LOW (13)
        )
        assert result['valid'] is True
        assert len(result['warnings']) > 0
        assert 'suspiciously low' in result['warnings'][0]

    def test_validate_weight_bmi_consistency_suspicious_high(self):
        """Test suspicious high BMI generates warning."""
        result = BMIValidator.validate_weight_bmi_consistency(
            190.0, 1.75  # BMI ~62, above SUSPICIOUS_HIGH (60)
        )
        assert result['valid'] is True
        assert len(result['warnings']) > 0
        assert 'suspiciously high' in result['warnings'][0]

    def test_validate_weight_bmi_consistency_impossible_bmi(self):
        """Test impossible BMI is rejected."""
        result = BMIValidator.validate_weight_bmi_consistency(
            250.0, 1.0  # BMI = 250
        )
        assert result['valid'] is False
        assert 'rejection_reason' in result

    def test_validate_weight_bmi_consistency_iglucose_source(self):
        """Test iglucose source triggers high-risk flag."""
        result = BMIValidator.validate_weight_bmi_consistency(
            70.0, 1.75, source='iglucose.com'
        )
        assert result['valid'] is True
        assert 'high_risk' in result
        assert result['high_risk'] is True
        assert 'High-outlier source' in result['warnings'][0]

    def test_estimate_height_from_weights_and_bmis(self):
        """Test height estimation from weight-BMI pairs."""
        pairs = [
            (70.0, 22.86),  # Implies ~1.75m
            (71.0, 23.18),  # Implies ~1.75m
            (69.0, 22.53),  # Implies ~1.75m
        ]
        height = BMIValidator.estimate_height_from_weights_and_bmis(pairs)
        assert height is not None
        assert pytest.approx(height, 0.05) == 1.75

    def test_estimate_height_insufficient_data(self):
        """Test height estimation with insufficient data."""
        pairs = [(70.0, 22.86)]
        height = BMIValidator.estimate_height_from_weights_and_bmis(pairs)
        assert height is None

    def test_detect_unit_confusion_insufficient_data(self):
        """Test unit confusion detection with insufficient data."""
        measurements = [
            (datetime.now(), 70.0, 'kg'),
            (datetime.now(), 71.0, 'kg')
        ]
        result = BMIValidator.detect_unit_confusion(measurements, 1.75)
        assert result['sufficient_data'] is False

    def test_detect_unit_confusion_frequent_bmi_values(self):
        """Test detection of frequent BMI-range values."""
        now = datetime.now()
        measurements = [
            (now - timedelta(hours=24), 25.0, 'kg'),
            (now - timedelta(hours=12), 24.5, 'kg'),
            (now - timedelta(hours=6), 25.2, 'kg'),
            (now, 24.8, 'kg')
        ]
        result = BMIValidator.detect_unit_confusion(measurements, 1.75)
        assert result['sufficient_data'] is True
        assert result['bmi_ratio'] == 1.0  # All values in BMI range
        assert 'likely_confusion' in result
        assert result['likely_confusion'] == 'frequent_bmi_values'

    def test_detect_unit_confusion_mixed_values(self):
        """Test detection with mixed weight and BMI-range values."""
        now = datetime.now()
        measurements = [
            (now - timedelta(hours=24), 70.0, 'kg'),  # Weight
            (now - timedelta(hours=12), 25.0, 'kg'),  # BMI
            (now - timedelta(hours=6), 71.0, 'kg'),   # Weight
            (now, 24.5, 'kg')                          # BMI
        ]
        result = BMIValidator.detect_unit_confusion(measurements, 1.75)
        assert result['sufficient_data'] is True
        assert result['bmi_ratio'] == 0.5  # Half are BMI values
        assert result['weight_ratio'] == 0.5  # Half are weight values