"""Unit tests for DataQualityPreprocessor.

Tests data validation and preprocessing which is CRITICAL for safety.
Bad data must be rejected before Kalman processing.
"""

import pytest
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

from weight_processor_lib.core.processing.validation import DataQualityPreprocessor
from weight_processor_lib.core.constants import PHYSIOLOGICAL_LIMITS, BMI_LIMITS


class TestDataValidation:
    """Critical tests for data validation and preprocessing."""

    @pytest.fixture
    def user_height(self) -> float:
        """Standard test user height (1.75m)."""
        return 1.75

    def test_absolute_minimum_weight_rejection(self, base_timestamp, user_height):
        """Test values below absolute minimum (< 30kg) are rejected.

        This is CRITICAL: prevents data entry errors (kg/lb confusion).
        Absolute minimum: 30kg (from PHYSIOLOGICAL_LIMITS).

        Expected behavior:
        - Weight < 30kg rejected with clear reason
        - Weight at 30kg accepted (boundary)
        """
        # Test 1: 25kg (below minimum)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=25.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is None, "25kg should be rejected (below minimum)"
        assert "rejected" in metadata, "Should have rejection reason"
        assert "bmi" in metadata["rejected"].lower(), "Should mention BMI in rejection"

        # Test 2: 19.9kg (well below minimum)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=19.9,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is None, "19.9kg should be rejected"

        # Test 3: 15kg (very low)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=15.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is None, "15kg should be rejected"

    def test_absolute_maximum_weight_rejection(self, base_timestamp, user_height):
        """Test values above absolute maximum (> 400kg) are rejected.

        This is CRITICAL: prevents data entry errors.
        Absolute maximum: 400kg (from PHYSIOLOGICAL_LIMITS).

        Expected behavior:
        - Weight > 400kg rejected with clear reason
        - Weight at 400kg accepted (boundary)
        """
        # Test 1: 450kg (above maximum)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=450.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is None, "450kg should be rejected (above maximum)"
        assert "rejected" in metadata, "Should have rejection reason"

        # Test 2: 500kg (well above maximum)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=500.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is None, "500kg should be rejected"

        # Test 3: 401kg (just above maximum of 400kg)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=401.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is None, "401kg should be rejected (above 400kg maximum)"

    def test_unit_conversion_accuracy(self, base_timestamp, user_height):
        """Test all supported units convert correctly to kg.

        This is CRITICAL: incorrect conversion corrupts all downstream logic.

        Supported units:
        - kg/kilogram/kilograms (no conversion)
        - lb/lbs/pound/pounds (× 0.453592)
        - st/stone/stones (× 6.35029)
        - g/gram/grams (÷ 1000)

        Expected behavior:
        - All conversions within 0.1kg accuracy
        - Original and converted values in metadata
        """
        # Test 1: Pounds to kg
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=154.0,  # 154 lbs ≈ 69.85kg
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="lbs",
            user_height_m=user_height,
        )
        assert weight is not None, "Valid lbs should be accepted"
        assert 69.7 < weight < 70.0, f"154 lbs should convert to ~69.85kg, got {weight}"
        assert metadata["original_weight"] == 154.0
        assert metadata["original_unit"] == "lbs"

        # Test 2: Stone to kg
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=11.0,  # 11 stone ≈ 69.85kg
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="st",
            user_height_m=user_height,
        )
        assert weight is not None, "Valid stone should be accepted"
        assert 69.7 < weight < 70.0, f"11 st should convert to ~69.85kg, got {weight}"
        assert "corrections" in metadata
        assert any("st" in c.lower() for c in metadata["corrections"])

        # Test 3: Grams to kg
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=70000.0,  # 70000g = 70kg
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="g",
            user_height_m=user_height,
        )
        assert weight is not None, "Valid grams should be accepted"
        assert abs(weight - 70.0) < 0.1, f"70000g should convert to 70kg, got {weight}"

        # Test 4: kg (no conversion needed)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=70.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=user_height,
        )
        assert weight is not None, "Valid kg should be accepted"
        assert weight == 70.0, "kg should not be converted"

    def test_bmi_validation_rejects_impossible_values(self, base_timestamp):
        """Test BMI confusion detection rejects impossible values.

        This is CRITICAL: common user error in manual entry.
        BMI range: 17-100 (IMPOSSIBLE_LOW to IMPOSSIBLE_HIGH).

        Expected behavior:
        - Value that looks like BMI (with height) is rejected if BMI < 17 or > 100
        - Rejection mentions BMI in reason
        """
        # Test 1: Value of 22.5 with height 1.75m
        # If this is weight: BMI = 22.5 / (1.75^2) = 7.35 (impossible)
        # Should be rejected as too low BMI
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=22.5,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=1.75,
        )
        assert weight is None, "22.5kg with height 1.75m should be rejected (BMI too low)"
        assert "bmi" in metadata["rejected"].lower(), "Should mention BMI"

        # Test 2: Value of 5.0 with height 1.75m (extreme case)
        # BMI = 5.0 / (1.75^2) = 1.63 (way too low)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=5.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=1.75,
        )
        assert weight is None, "5.0kg should be rejected (BMI impossible)"

        # Test 3: Valid weight with normal BMI should pass
        # 70kg with height 1.75m → BMI = 22.9 (normal)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=70.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="kg",
            user_height_m=1.75,
        )
        assert weight is not None, "70kg with height 1.75m should be accepted (normal BMI)"
        assert metadata["implied_bmi"] == pytest.approx(22.9, abs=0.1)

    def test_unsupported_unit_rejection(self, base_timestamp, user_height):
        """Test units other than kg/lb/lbs/g/st are rejected.

        This is CRITICAL: clear error messages for invalid input.

        Expected behavior:
        - Unsupported units rejected with clear message
        - Null/missing unit rejected
        - Only whitelisted units accepted
        """
        # Test 1: "bmi" unit (not supported)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=22.5,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="bmi",
            user_height_m=user_height,
        )
        assert weight is None, "BMI unit should be rejected"
        assert "rejected" in metadata
        assert "unsupported" in metadata["rejected"].lower()

        # Test 2: "oz" unit (ounces - not supported)
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=2464.0,  # ~70kg in ounces
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="oz",
            user_height_m=user_height,
        )
        assert weight is None, "Ounces unit should be rejected"

        # Test 3: None/missing unit
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=70.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit=None,
            user_height_m=user_height,
        )
        assert weight is None, "Missing unit should be rejected"
        assert "missing unit" in metadata["rejected"].lower()

        # Test 4: Empty string unit
        weight, metadata = DataQualityPreprocessor.preprocess(
            weight=70.0,
            source="manual",
            timestamp=base_timestamp,
            user_id="test-user",
            unit="",
            user_height_m=user_height,
        )
        assert weight is None, "Empty string unit should be rejected"
