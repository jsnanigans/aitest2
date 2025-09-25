"""Test that measurements_since_reset increments even for rejected measurements during adaptive period."""

import pytest
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from src.processing.processor import process_measurement
from src.database.database import ProcessorStateDB
from src.processing.reset_manager import ResetType

# Load config directly from main.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import load_config


def test_adaptive_period_counter_increments_on_rejection():
    """Test that the measurements_since_reset counter increments even when measurements are rejected during adaptive period."""

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Load base config and enable state persistence
        config = {
            "features": {
                "state_persistence": True,
                "quality_scoring": True,
                "unified_quality_scoring": False,  # Don't use unified scoring
                "kalman_filtering": True,  # Ensure Kalman is enabled
                "outlier_detection": True,
            },
            "quality_scoring": {
                "enabled": True,
                "threshold": 0.25,
                "use_harmonic_mean": True,
                "component_weights": {
                    "safety": 0.35,
                    "plausibility": 0.25,
                    "consistency": 0.25,
                    "reliability": 0.15,
                },
            },
            "outlier_detection": {
                "enabled": True,
                "threshold": 0.05,
                "extreme_threshold": 0.15,
            },
            "quality_override": {"enabled": True, "threshold": 0.8},
            "kalman": {
                "initial_variance": 1.0,
                "transition_covariance": 0.1,
                "observation_covariance": 1.0,
                "reset": {
                    "gap_threshold_days": 30,
                    "soft_reset_sources": [
                        "questionnaire",
                        "patient-upload",
                        "manual-entry",
                    ],
                },
            },
        }

        # Create database
        db = ProcessorStateDB(db_path)

        # User ID for testing
        user_id = "test_user_adaptive_fix"

        # Patch get_state_db to return our test database
        with patch("src.processing.processor.get_state_db", return_value=db):
            # Initialize with a reasonable weight
            timestamp1 = datetime(2024, 1, 1, 8, 0, 0)
            result1 = process_measurement(
                user_id=user_id,
                weight=82.0,
                timestamp=timestamp1,
                source="patient-device",
                config=config,
            )
            if not result1["accepted"]:
                print(f"First measurement rejected: {result1}")
            assert result1["accepted"] is True, (
                f"First measurement should be accepted: {result1}"
            )

            # Trigger a hard reset by creating a large gap
            timestamp2 = datetime(2024, 2, 15, 8, 0, 0)  # 45 days later
            result2 = process_measurement(
                user_id=user_id,
                weight=116.0,  # Bad measurement that gets accepted due to reset
                timestamp=timestamp2,
                source="patient-device",
                config=config,
            )
            assert result2["accepted"] is True
            assert (
                result2.get("reset_type") == "hard"
                or result2.get("reset_type") == ResetType.HARD
            )

            # Load the state to check initial counter
            state_after_reset = db.get_state(user_id)
            assert (
                state_after_reset["measurements_since_reset"] == 1
            )  # First accepted measurement after reset

            # Now send good measurements that will be rejected due to large deviation
            # These should still increment the counter due to our fix
            rejected_measurements = []
            for i in range(5):
                timestamp = timestamp2 + timedelta(hours=i + 1)
                result = process_measurement(
                    user_id=user_id,
                    weight=82.5 + i * 0.1,  # Good weights around 82-83 kg
                    timestamp=timestamp,
                    source="patient-device",
                    config=config,
                )
                rejected_measurements.append(result)

                # Check that counter increments even though measurement was rejected
                state = db.get_state(user_id)
                expected_count = 2 + i  # 1 initial + 1 bad accepted + i rejected
                assert state["measurements_since_reset"] == expected_count, (
                    f"After {i + 1} rejected measurements, counter should be {expected_count}, but got {state['measurements_since_reset']}"
                )

            # Verify measurements were actually rejected
            for result in rejected_measurements:
                assert result["accepted"] is False, (
                    "Measurement should have been rejected"
                )

            # Continue sending measurements until we exit adaptive period
            # Default adaptation_measurements is 10
            for i in range(5, 10):
                timestamp = timestamp2 + timedelta(hours=i + 1)
                result = process_measurement(
                    user_id=user_id,
                    weight=82.5,  # Good weight
                    timestamp=timestamp,
                    source="patient-device",
                    config=config,
                )
                state = db.get_state(user_id)
                # Counter should keep incrementing
                assert state["measurements_since_reset"] >= 7

            # After 10+ measurements, we should be out of adaptive period
            final_state = db.get_state(user_id)
            assert final_state["measurements_since_reset"] >= 10, (
                f"Should have at least 10 measurements, got {final_state['measurements_since_reset']}"
            )

            # Verify we're no longer in adaptive period by checking if good measurements are accepted
            timestamp_final = timestamp2 + timedelta(hours=12)
            result_final = process_measurement(
                user_id=user_id,
                weight=82.5,  # Good weight that should now be accepted
                timestamp=timestamp_final,
                source="patient-device",
                config=config,
            )

            # This might still be rejected if Kalman hasn't converged, but counter should be > 10
            final_state = db.get_state(user_id)
            assert final_state["measurements_since_reset"] > 10, (
                "Counter should be well past adaptive period threshold"
            )

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_adaptive_period_counter_with_unified_scoring():
    """Test counter increment with unified quality scoring enabled."""

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Load base config and configure
        config = {
            "features": {
                "state_persistence": True,
                "unified_quality_scoring": True,
                "quality_scoring": False,
                "kalman_filtering": True,
                "outlier_detection": True,
            },
            "quality_scoring": {
                "enabled": True,
                "threshold": 0.25,
                "use_harmonic_mean": True,
                "component_weights": {
                    "kalman_fit": 0.6,
                    "temporal_consistency": 0.2,
                    "anomaly_detection": 0.0,
                    "source_reliability": 0.0,
                },
            },
            "outlier_detection": {
                "enabled": True,
                "threshold": 0.05,
                "extreme_threshold": 0.15,
            },
            "quality_override": {"enabled": True, "threshold": 0.8},
            "kalman": {
                "initial_variance": 1.0,
                "transition_covariance": 0.1,
                "observation_covariance": 1.0,
                "reset": {
                    "gap_threshold_days": 30,
                    "soft_reset_sources": [
                        "questionnaire",
                        "patient-upload",
                        "manual-entry",
                    ],
                },
            },
        }

        # Create database
        db = ProcessorStateDB(db_path)

        user_id = "test_user_unified"

        # Patch get_state_db to return our test database
        with patch("src.processing.processor.get_state_db", return_value=db):
            # Initialize with a reasonable weight
            timestamp1 = datetime(2024, 1, 1, 8, 0, 0)
            result1 = process_measurement(
                user_id=user_id,
                weight=75.0,
                timestamp=timestamp1,
                source="patient-device",
                config=config,
            )
            assert result1["accepted"] is True

            # Trigger a hard reset
            timestamp2 = datetime(2024, 2, 20, 8, 0, 0)  # 50 days later
            result2 = process_measurement(
                user_id=user_id,
                weight=105.0,  # Bad measurement
                timestamp=timestamp2,
                source="patient-device",
                config=config,
            )
            assert result2["accepted"] is True
            assert (
                result2.get("reset_type") == "hard"
                or result2.get("reset_type") == ResetType.HARD
            )

            # Send good measurements that will be rejected
            for i in range(3):
                timestamp = timestamp2 + timedelta(hours=i + 1)
                result = process_measurement(
                    user_id=user_id,
                    weight=75.0,  # Good weight
                    timestamp=timestamp,
                    source="patient-device",
                    config=config,
                )

                # Check counter increments
                state = db.get_state(user_id)
                expected = 2 + i  # 1 initial + 1 bad + i rejected
                assert state["measurements_since_reset"] == expected, (
                    f"With unified scoring, counter should be {expected}, got {state['measurements_since_reset']}"
                )

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_adaptive_period_counter_with_legacy_validation():
    """Test counter increment with legacy validation path."""

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Load base config and explicitly disable quality scoring
        config = {
            "features": {
                "state_persistence": True,
                "quality_scoring": False,  # Use legacy validation by setting to False
                "unified_quality_scoring": False,
                "kalman_filtering": True,
                "outlier_detection": True,
            },
            "outlier_detection": {
                "enabled": True,
                "threshold": 0.05,
                "extreme_threshold": 0.15,
            },
            "kalman": {
                "initial_variance": 1.0,
                "transition_covariance": 0.1,
                "observation_covariance": 1.0,
                "reset": {
                    "gap_threshold_days": 30,
                    "soft_reset_sources": [
                        "questionnaire",
                        "patient-upload",
                        "manual-entry",
                    ],
                },
            },
        }

        # Create database
        db = ProcessorStateDB(db_path)

        user_id = "test_user_legacy"

        # Patch get_state_db to return our test database
        with patch("src.processing.processor.get_state_db", return_value=db):
            # Initialize with a reasonable weight
            timestamp1 = datetime(2024, 1, 1, 8, 0, 0)
            result1 = process_measurement(
                user_id=user_id,
                weight=70.0,
                timestamp=timestamp1,
                source="patient-device",
                config=config,
            )
            assert result1["accepted"] is True

            # Trigger a hard reset
            timestamp2 = datetime(2024, 3, 1, 8, 0, 0)  # 60 days later
            result2 = process_measurement(
                user_id=user_id,
                weight=100.0,  # Bad measurement
                timestamp=timestamp2,
                source="patient-device",
                config=config,
            )
            assert result2["accepted"] is True
            assert (
                result2.get("reset_type") == "hard"
                or result2.get("reset_type") == ResetType.HARD
            )

            # Send good measurements that will be rejected
            for i in range(3):
                timestamp = timestamp2 + timedelta(hours=i + 1)
                result = process_measurement(
                    user_id=user_id,
                    weight=70.0,  # Good weight
                    timestamp=timestamp,
                    source="patient-device",
                    config=config,
                )

                # Check counter increments with legacy validation too
                state = db.get_state(user_id)
                expected = 2 + i  # 1 initial + 1 bad + i rejected
                assert state["measurements_since_reset"] == expected, (
                    f"With legacy validation, counter should be {expected}, got {state['measurements_since_reset']}"
                )

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_adaptive_period_counter_increments_on_rejection()
    test_adaptive_period_counter_with_unified_scoring()
    test_adaptive_period_counter_with_legacy_validation()
    print("All tests passed!")
