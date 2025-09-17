"""
Tests for PhysiologicalValidator focusing on key functionality.
"""

import pytest
from datetime import datetime, timedelta
from src.processing.validation import PhysiologicalValidator
from src.constants import PHYSIOLOGICAL_LIMITS


class TestPhysiologicalValidator:

    def test_validate_absolute_limits_valid_weight(self):
        """Test normal weight passes absolute limits."""
        valid, reason = PhysiologicalValidator.validate_absolute_limits(70.0)
        assert valid is True
        assert reason is None

    def test_validate_absolute_limits_too_low(self):
        """Test weight below absolute minimum is rejected."""
        valid, reason = PhysiologicalValidator.validate_absolute_limits(5.0)
        assert valid is False
        assert "below absolute minimum" in reason

    def test_validate_absolute_limits_too_high(self):
        """Test weight above absolute maximum is rejected."""
        valid, reason = PhysiologicalValidator.validate_absolute_limits(700.0)
        assert valid is False
        assert "above absolute maximum" in reason

    def test_check_suspicious_range_normal(self):
        """Test normal weight has no warnings."""
        warning = PhysiologicalValidator.check_suspicious_range(70.0)
        assert warning is None

    def test_check_suspicious_range_low(self):
        """Test suspiciously low weight generates warning."""
        warning = PhysiologicalValidator.check_suspicious_range(35.0)
        assert warning is not None
        assert "suspiciously low" in warning

    def test_check_suspicious_range_high(self):
        """Test suspiciously high weight generates warning."""
        warning = PhysiologicalValidator.check_suspicious_range(310.0)
        assert warning is not None
        assert "suspiciously high" in warning

    def test_validate_rate_of_change_normal(self):
        """Test normal weight change rate passes validation."""
        valid, reason, rate = PhysiologicalValidator.validate_rate_of_change(
            current_weight=70.0,
            previous_weight=69.5,
            time_diff_hours=24
        )
        assert valid is True
        assert reason is None
        assert rate == 0.5  # 0.5 kg per day

    def test_validate_rate_of_change_too_fast(self):
        """Test excessive weight change rate is rejected."""
        valid, reason, rate = PhysiologicalValidator.validate_rate_of_change(
            current_weight=75.0,
            previous_weight=70.0,
            time_diff_hours=12  # 5kg in 12 hours = 10kg/day
        )
        assert valid is False
        assert "exceeds max rate" in reason
        assert rate == 10.0

    def test_validate_rate_of_change_zero_time(self):
        """Test zero time difference is handled gracefully."""
        valid, reason, rate = PhysiologicalValidator.validate_rate_of_change(
            current_weight=70.0,
            previous_weight=69.0,
            time_diff_hours=0
        )
        assert valid is True
        assert reason is None
        assert rate == 0.0

    def test_check_measurement_pattern_insufficient_data(self):
        """Test pattern analysis with insufficient data."""
        measurements = [(datetime.now(), 70.0)]
        result = PhysiologicalValidator.check_measurement_pattern(measurements)
        assert result['sufficient_data'] is False

    def test_check_measurement_pattern_normal(self):
        """Test pattern analysis with normal variations."""
        now = datetime.now()
        measurements = [
            (now - timedelta(hours=12), 69.5),
            (now - timedelta(hours=6), 69.8),
            (now, 70.0)
        ]
        result = PhysiologicalValidator.check_measurement_pattern(measurements)
        assert result['sufficient_data'] is True
        assert 'mean' in result
        assert 'std' in result
        assert result['suspicious_pattern'] == False

    def test_check_measurement_pattern_high_variation(self):
        """Test pattern analysis detects suspicious variations."""
        now = datetime.now()
        measurements = [
            (now - timedelta(hours=12), 65.0),
            (now - timedelta(hours=6), 75.0),
            (now, 68.0)
        ]
        result = PhysiologicalValidator.check_measurement_pattern(measurements, window_hours=24)
        assert result['sufficient_data'] is True
        assert result['suspicious_pattern'] == True

    def test_validate_comprehensive_valid_measurement(self):
        """Test comprehensive validation passes for valid measurement."""
        result = PhysiologicalValidator.validate_comprehensive(
            weight=70.0,
            previous_weight=69.5,
            time_diff_hours=24,
            source='patient-upload'
        )
        assert result['valid'] is True
        assert result['rejection_reason'] is None
        assert 'absolute_limits' in result['checks']
        assert 'rate_of_change' in result['checks']

    def test_validate_comprehensive_invalid_weight(self):
        """Test comprehensive validation rejects invalid weight."""
        result = PhysiologicalValidator.validate_comprehensive(
            weight=5.0
        )
        assert result['valid'] is False
        assert result['rejection_reason'] is not None
        assert "below absolute minimum" in result['rejection_reason']

    def test_validate_comprehensive_excessive_rate(self):
        """Test comprehensive validation rejects excessive rate of change."""
        result = PhysiologicalValidator.validate_comprehensive(
            weight=80.0,
            previous_weight=70.0,
            time_diff_hours=12  # 10kg in 12 hours
        )
        assert result['valid'] is False
        assert "exceeds max rate" in result['rejection_reason']
        assert result['daily_change_rate'] == 20.0