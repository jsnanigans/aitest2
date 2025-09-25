"""Simple test to verify adaptive period counter increments for rejected measurements."""

from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import patch
from src.processing.processor import process_measurement
from src.database.database import ProcessorStateDB


def test_counter_increments_during_adaptive_period():
    """Verify the core fix: counter increments even when measurements are rejected during adaptive period."""

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Minimal config - use defaults where possible
        config = {
            "features": {
                "state_persistence": True,
                "quality_scoring": True,
                "unified_quality_scoring": False,  # Use regular quality scoring
                "kalman_filtering": True,
                "outlier_detection": True,
            },
            "quality_scoring": {"threshold": 0.25},
            "kalman": {"reset": {"gap_threshold_days": 30}},
        }

        db = ProcessorStateDB(db_path)
        user_id = "test_user"

        with patch("src.processing.processor.get_state_db", return_value=db):
            # Step 1: Initial measurement
            result1 = process_measurement(
                user_id=user_id,
                weight=80.0,
                timestamp=datetime(2024, 1, 1, 10, 0),
                source="patient-device",
                config=config,
            )
            assert result1["accepted"] is True, "Initial measurement should be accepted"

            # Step 2: Trigger hard reset with 35-day gap
            result2 = process_measurement(
                user_id=user_id,
                weight=110.0,  # Bad weight that gets accepted due to reset
                timestamp=datetime(2024, 2, 5, 10, 0),  # 35 days later
                source="patient-device",
                config=config,
            )
            assert result2["accepted"] is True, (
                "Post-reset measurement should be accepted"
            )
            assert result2.get("reset_type") in ["hard", "HARD"], (
                "Should trigger hard reset"
            )

            # Verify counter after accepting bad measurement
            state = db.get_state(user_id)
            assert state["measurements_since_reset"] == 1, (
                "Counter should be 1 after first accepted measurement"
            )

            # Step 3: Send good measurements that get rejected
            # These should still increment the counter due to our fix
            print("\nTesting rejected measurements during adaptive period:")
            for i in range(3):
                timestamp = datetime(2024, 2, 5, 10 + i + 1, 0)
                result = process_measurement(
                    user_id=user_id,
                    weight=80.0,  # Good weight that will be rejected
                    timestamp=timestamp,
                    source="patient-device",
                    config=config,
                )

                # Verify measurement was rejected
                assert result["accepted"] is False, (
                    f"Measurement {i + 1} should be rejected"
                )
                print(
                    f"  Measurement {i + 1}: Rejected (reason: {result.get('reason', 'unknown')[:50]}...)"
                )

                # Verify counter still incremented
                state = db.get_state(user_id)
                expected_count = 2 + i  # 1 initial + 1 bad accepted + i rejected
                actual_count = state["measurements_since_reset"]
                print(
                    f"  Counter after rejection {i + 1}: {actual_count} (expected: {expected_count})"
                )
                assert actual_count == expected_count, (
                    f"Counter should increment even for rejected measurements during adaptive period"
                )

            print(
                "\n✓ Test passed: Counter increments correctly during adaptive period"
            )

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_counter_increments_during_adaptive_period()
    print("\nAll tests passed!")
