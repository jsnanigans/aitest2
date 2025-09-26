"""
Comprehensive validation tests for weight measurements.

These tests verify that the API correctly validates input data,
rejects invalid measurements, and handles edge cases appropriately.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

from .conftest import APIClient, TestUser, create_measurement


class TestWeightRangeValidation:
    """Test weight range validation for different units."""

    @pytest.mark.parametrize("weight,unit,should_accept", [
        # Kilograms - valid range: 10-500 kg
        (10.0, "kg", True),      # Minimum valid
        (9.9, "kg", False),      # Below minimum
        (500.0, "kg", True),     # Maximum valid
        (500.1, "kg", False),    # Above maximum
        (75.0, "kg", True),      # Normal weight
        (150.0, "kg", True),     # Heavy but valid
        (0.0, "kg", False),      # Zero weight
        (-75.0, "kg", False),    # Negative weight

        # Pounds - valid range: 22-1102 lbs
        (22.0, "lbs", True),     # Minimum valid
        (21.9, "lbs", False),    # Below minimum
        (1102.0, "lbs", True),   # Maximum valid
        (1102.1, "lbs", False),  # Above maximum
        (165.0, "lbs", True),    # Normal weight
        (330.0, "lbs", True),    # Heavy but valid

        # Stones - valid range: 1.57-78.7 st
        (1.57, "st", True),      # Minimum valid
        (1.56, "st", False),     # Below minimum
        (78.7, "st", True),      # Maximum valid
        (78.8, "st", False),     # Above maximum
        (11.8, "st", True),      # Normal weight

        # Ounces - valid range: 352-17637 oz
        (352, "oz", True),       # Minimum valid
        (351, "oz", False),      # Below minimum
        (17637, "oz", True),     # Maximum valid
        (17638, "oz", False),    # Above maximum
        (2640, "oz", True),      # Normal weight (165 lbs)

        # Grams - valid range: 10000-500000 g
        (10000, "g", True),      # Minimum valid
        (9999, "g", False),      # Below minimum
        (500000, "g", True),     # Maximum valid
        (500001, "g", False),    # Above maximum
        (75000, "g", True),      # Normal weight (75 kg)
    ])
    def test_weight_range_by_unit(self, api_client, test_user, weight, unit, should_accept):
        """Test weight validation for each unit type."""
        measurement = create_measurement(weight=weight, unit=unit)

        response = api_client.process_measurements(
            test_user.user_id,
            [measurement]
        )

        if should_accept:
            assert response.is_success
            assert response.data["accepted_count"] == 1
            assert response.data["rejected_count"] == 0
        else:
            # Invalid weights should be rejected
            if response.is_success:
                assert response.data["rejected_count"] == 1
                assert response.data["accepted_count"] == 0
            else:
                # Or return an error
                assert not response.is_success

    def test_extreme_weight_values(self, api_client, test_user):
        """Test handling of extreme weight values."""
        extreme_values = [
            float('inf'),     # Infinity
            float('-inf'),    # Negative infinity
            1e10,            # Very large number
            -1e10,           # Very large negative
            1e-10,           # Very small positive
            -1e-10,          # Very small negative
        ]

        for value in extreme_values:
            try:
                measurement = create_measurement(weight=value, unit="kg")
                response = api_client.process_measurements(
                    test_user.user_id,
                    [measurement]
                )

                # Should reject or error
                if response.is_success:
                    assert response.data["rejected_count"] == 1
                else:
                    assert not response.is_success
            except (ValueError, TypeError):
                # Some values might fail at serialization
                pass


class TestRateOfChangeValidation:
    """Test validation of weight change rates."""

    def test_rapid_weight_loss_detection(self, api_client, test_user):
        """Detect and handle rapid weight loss."""
        measurements = [
            create_measurement(weight=80.0, days_ago=7),
            create_measurement(weight=79.0, days_ago=6),
            create_measurement(weight=78.0, days_ago=5),
            create_measurement(weight=77.0, days_ago=4),
            create_measurement(weight=76.0, days_ago=3),
            create_measurement(weight=70.0, days_ago=2),  # 6kg in one day - unrealistic
            create_measurement(weight=69.0, days_ago=1),
            create_measurement(weight=68.0, days_ago=0),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # The 70kg measurement should be flagged as outlier
        assert response.data["rejected_count"] >= 1 or \
               any(m["status"] == "outlier" for m in response.data["measurements"]
                   if m["weight"] == 70.0)

    def test_rapid_weight_gain_detection(self, api_client, test_user):
        """Detect and handle rapid weight gain."""
        measurements = [
            create_measurement(weight=70.0, days_ago=3),
            create_measurement(weight=70.5, days_ago=2),
            create_measurement(weight=80.0, days_ago=1),  # 9.5kg gain in one day
            create_measurement(weight=70.8, days_ago=0),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        # The 80kg measurement should be flagged
        measurement_results = response.data["measurements"]
        outlier_weights = [m["weight"] for m in measurement_results if m["status"] == "outlier"]
        assert 80.0 in outlier_weights or response.data["rejected_count"] >= 1

    def test_acceptable_daily_variation(self, api_client, test_user):
        """Normal daily weight variation should be accepted."""
        # 0.5-1kg daily variation is normal
        measurements = [
            create_measurement(weight=75.0, days_ago=6),
            create_measurement(weight=74.7, days_ago=5),
            create_measurement(weight=75.2, days_ago=4),
            create_measurement(weight=74.9, days_ago=3),
            create_measurement(weight=74.5, days_ago=2),
            create_measurement(weight=74.8, days_ago=1),
            create_measurement(weight=74.4, days_ago=0),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        # All should be accepted as normal variation
        assert response.data["accepted_count"] >= 6
        assert response.data["rejected_count"] == 0

    def test_maximum_acceptable_weekly_change(self, api_client, test_user):
        """Test maximum acceptable weekly weight change."""
        # 2-3kg per week is aggressive but possible
        measurements = [
            create_measurement(weight=80.0, days_ago=7),
            create_measurement(weight=77.5, days_ago=0),  # 2.5kg in a week
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        assert response.is_success
        assert response.data["accepted_count"] == 2  # Should accept realistic weekly change


class TestTemporalValidation:
    """Test temporal validation of measurements."""

    def test_reject_future_dates(self, api_client, test_user):
        """Measurements with future dates should be rejected."""
        future_measurement = create_measurement(
            weight=75.0,
            days_ago=-1  # Tomorrow
        )

        response = api_client.process_measurements(
            test_user.user_id,
            [future_measurement]
        )

        # Should reject or handle future dates
        if response.is_success:
            assert response.data["rejected_count"] == 1
        else:
            assert response.status_code == 400

    def test_handle_very_old_dates(self, api_client, test_user):
        """Handle measurements with very old dates."""
        # 10 years ago
        old_measurement = create_measurement(
            weight=75.0,
            days_ago=3650
        )

        response = api_client.process_measurements(
            test_user.user_id,
            [old_measurement]
        )

        # Should either accept with warning or reject based on config
        assert response.status_code in [200, 400]

    def test_duplicate_timestamps(self, api_client, test_user):
        """Handle measurements with identical timestamps."""
        timestamp = datetime.utcnow()
        measurements = [
            {
                "uuid": str(uuid.uuid4()),
                "weight": 75.0,
                "unit": "kg",
                "effectiveDateTime": timestamp.isoformat() + "Z",
                "source": "scale"
            },
            {
                "uuid": str(uuid.uuid4()),
                "weight": 76.0,
                "unit": "kg",
                "effectiveDateTime": timestamp.isoformat() + "Z",  # Same timestamp
                "source": "manual"
            }
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        # Should handle duplicate timestamps appropriately
        if response.is_success:
            # May accept both with conflict resolution or reject one
            assert response.data["processed_count"] == 2
        else:
            assert "timestamp" in str(response.error).lower()

    def test_out_of_order_measurements(self, api_client, test_user):
        """Test processing of out-of-order measurements."""
        # Send measurements in reverse chronological order
        measurements = [
            create_measurement(weight=74.0, days_ago=0),
            create_measurement(weight=74.5, days_ago=1),
            create_measurement(weight=75.0, days_ago=2),
            create_measurement(weight=75.5, days_ago=3),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        # System should handle out-of-order appropriately
        if response.is_success:
            # May reorder internally or process as-is
            assert response.data["processed_count"] == 4
        else:
            # Or reject out-of-order
            assert response.status_code == 409  # Conflict


class TestDataTypeValidation:
    """Test validation of data types and formats."""

    def test_invalid_weight_types(self, api_client, test_user):
        """Test handling of invalid weight data types."""
        invalid_weights = [
            "seventy-five",       # String weight
            [75.0],              # Array
            {"value": 75.0},     # Object
            None,                # Null
            "",                  # Empty string
            "75.0kg",           # String with unit
            True,               # Boolean
        ]

        for invalid_weight in invalid_weights:
            measurement = {
                "uuid": str(uuid.uuid4()),
                "weight": invalid_weight,
                "unit": "kg",
                "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
                "source": "scale"
            }

            response = api_client.process_measurements(
                test_user.user_id,
                [measurement]
            )

            assert not response.is_success or response.data["rejected_count"] == 1

    def test_invalid_unit_values(self, api_client, test_user):
        """Test handling of invalid unit values."""
        invalid_units = [
            "kilograms",    # Full name instead of abbreviation
            "KG",           # Wrong case
            "pound",        # Singular instead of "lbs"
            "kgs",          # Plural
            "stone",        # Singular instead of "st"
            "gr",           # Wrong abbreviation for grams
            "",             # Empty
            None,           # Null
            123,            # Number
        ]

        for invalid_unit in invalid_units:
            measurement = {
                "uuid": str(uuid.uuid4()),
                "weight": 75.0,
                "unit": invalid_unit,
                "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
                "source": "scale"
            }

            response = api_client.process_measurements(
                test_user.user_id,
                [measurement]
            )

            assert not response.is_success or response.data["rejected_count"] == 1

    def test_invalid_datetime_formats(self, api_client, test_user):
        """Test handling of invalid datetime formats."""
        invalid_datetimes = [
            "2024-01-15",                    # Date only
            "15:30:00",                      # Time only
            "2024/01/15T15:30:00Z",         # Wrong date separator
            "2024-01-15 15:30:00",           # Space instead of T
            "January 15, 2024",              # Human readable
            "1705334400",                    # Unix timestamp
            "",                              # Empty
            None,                            # Null
        ]

        for invalid_datetime in invalid_datetimes:
            measurement = {
                "uuid": str(uuid.uuid4()),
                "weight": 75.0,
                "unit": "kg",
                "effectiveDateTime": invalid_datetime,
                "source": "scale"
            }

            response = api_client.process_measurements(
                test_user.user_id,
                [measurement]
            )

            assert not response.is_success or response.data["rejected_count"] == 1

    def test_missing_required_fields(self, api_client, test_user):
        """Test handling of missing required fields."""
        # Test each missing field
        complete_measurement = {
            "uuid": str(uuid.uuid4()),
            "weight": 75.0,
            "unit": "kg",
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "source": "scale"
        }

        required_fields = ["weight", "unit", "effectiveDateTime"]

        for field in required_fields:
            incomplete_measurement = complete_measurement.copy()
            del incomplete_measurement[field]

            response = api_client.process_measurements(
                test_user.user_id,
                [incomplete_measurement]
            )

            assert not response.is_success or response.data["rejected_count"] == 1


class TestSourceValidation:
    """Test validation of measurement sources."""

    def test_valid_source_values(self, api_client, test_user):
        """Test processing with all valid source values."""
        valid_sources = [
            "scale", "doctor", "app", "manual", "fitbit",
            "withings", "garmin", "apple_health", "google_fit",
            "patient-device", "care-team-upload", "clinic-scale"
        ]

        for source in valid_sources:
            measurement = create_measurement(weight=75.0, source=source)
            response = api_client.process_measurements(
                f"{test_user.user_id}_{source}",
                [measurement]
            )

            assert response.is_success
            assert response.data["accepted_count"] == 1

    def test_source_affects_quality_score(self, api_client, test_user):
        """Different sources should have different quality scores."""
        source_quality_map = [
            ("doctor", 0.95),
            ("scale", 0.90),
            ("fitbit", 0.80),
            ("app", 0.70),
            ("manual", 0.60),
            ("unknown_source", 0.50),
        ]

        for source, expected_min_quality in source_quality_map:
            measurement = create_measurement(weight=75.0, source=source)
            response = api_client.process_measurements(
                f"{test_user.user_id}_{source}",
                [measurement]
            )

            assert response.is_success
            result = response.data["measurements"][0]
            assert result["quality_score"] >= expected_min_quality - 0.1


class TestBMIValidation:
    """Test BMI-based validation if height is provided."""

    def test_bmi_based_weight_validation(self, api_client, test_user):
        """Test weight validation considering BMI if height is available."""
        # Assuming user height is 175cm (from TestUser default)
        height_cm = 175

        # Test various BMI scenarios
        test_cases = [
            (40.0, "Severely underweight"),   # BMI ~13
            (55.0, "Underweight"),            # BMI ~18
            (70.0, "Normal"),                 # BMI ~23
            (95.0, "Overweight"),            # BMI ~31
            (120.0, "Obese"),                # BMI ~39
            (160.0, "Severely obese"),       # BMI ~52
        ]

        for weight, category in test_cases:
            measurement = create_measurement(weight=weight, unit="kg")
            response = api_client.process_measurements(
                f"{test_user.user_id}_{category}",
                [measurement]
            )

            # All should be accepted as they're within physiological bounds
            # but may have different quality scores
            assert response.is_success
            assert response.data["accepted_count"] == 1


class TestQualityScoreValidation:
    """Test quality score impact on validation."""

    def test_high_quality_prevents_outlier_marking(self, api_client, test_user):
        """High quality measurements should be protected from outlier marking."""
        measurements = [
            create_measurement(weight=75.0, days_ago=3, source="scale"),
            create_measurement(weight=74.8, days_ago=2, source="scale"),
            create_measurement(weight=78.0, days_ago=1, source="doctor"),  # Jump but from doctor
            create_measurement(weight=74.5, days_ago=0, source="scale"),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        # Doctor measurement should not be marked as outlier despite jump
        doctor_result = next(
            (m for m in response.data["measurements"] if m["source"] == "doctor"),
            None
        )

        if doctor_result:
            assert doctor_result["status"] != "outlier"
            assert doctor_result["quality_score"] > 0.9

    def test_low_quality_more_likely_outlier(self, api_client, test_user):
        """Low quality measurements more likely to be marked as outliers."""
        measurements = [
            create_measurement(weight=75.0, days_ago=3, source="scale"),
            create_measurement(weight=74.8, days_ago=2, source="scale"),
            create_measurement(weight=77.5, days_ago=1, source="manual"),  # Small jump, low quality
            create_measurement(weight=74.7, days_ago=0, source="scale"),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        # Manual entry more likely to be marked as outlier
        manual_result = next(
            (m for m in response.data["measurements"] if m["source"] == "manual"),
            None
        )

        if manual_result:
            # Lower quality score
            assert manual_result["quality_score"] < 0.7