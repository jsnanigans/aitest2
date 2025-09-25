"""
Test that replay works correctly after resets.

This test verifies that when a reset occurs, a snapshot is saved immediately
so that replay can later restore to the correct state and re-evaluate measurements.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.processing.processor import process_measurement
from src.replay.replay_buffer import ReplayBuffer
from src.replay.replay_manager import ReplayManager
from src.processing.outlier_detection import OutlierDetector


class TestReplayAfterReset:
    """Test replay functionality after resets."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user_id = "e751ebe4-3e13-423d-bf50-88a9dd13f132"
        self.base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Mock database with snapshot functionality
        self.mock_db = Mock()
        self.snapshots = {}
        self.states = {}

        def save_state(user_id, state):
            self.states[user_id] = state.copy()

        def get_state(user_id):
            return self.states.get(user_id, None)

        def create_initial_state():
            # Note: processor calls this without user_id
            return {
                "kalman_params": None,
                "last_state": None,
                "last_timestamp": None,
                "last_accepted_timestamp": None,
                "measurements_since_reset": 0,
            }

        def save_state_snapshot(user_id, timestamp):
            if user_id in self.states:
                self.snapshots[f"{user_id}_{timestamp.isoformat()}"] = {
                    "timestamp": timestamp,
                    "state": self.states[user_id].copy(),
                }

        def check_and_restore_snapshot(user_id, before_time):
            # Find the most recent snapshot before the given time
            matching_snapshots = []
            for key, snapshot in self.snapshots.items():
                if key.startswith(user_id) and snapshot["timestamp"] < before_time:
                    matching_snapshots.append(snapshot)

            if matching_snapshots:
                # Get the most recent one
                latest = max(matching_snapshots, key=lambda x: x["timestamp"])
                self.states[user_id] = latest["state"].copy()
                return {
                    "success": True,
                    "snapshot": latest["state"],
                    "snapshot_timestamp": latest["timestamp"],
                }
            else:
                return {
                    "success": False,
                    "error": f"No snapshot found for {user_id} before {before_time}",
                }

        self.mock_db.save_state = save_state
        self.mock_db.get_state = get_state
        self.mock_db.create_initial_state = create_initial_state
        self.mock_db.save_state_snapshot = save_state_snapshot
        self.mock_db.check_and_restore_snapshot = check_and_restore_snapshot

        # Configuration
        self.config = {
            "kalman": {
                "initial_variance": 0.361,
                "transition_covariance_weight": 0.016,
                "transition_covariance_trend": 0.0001,
                "observation_covariance": 3.4,
                "reset": {
                    "soft": {
                        "enabled": True,
                        "min_weight_change_kg": 5,
                        "trigger_sources": ["questionnaire"],
                        "cooldown_days": 3,
                    }
                },
            },
            "quality_scoring": {"threshold": 0.6, "use_harmonic_mean": True},
            "processing": {"extreme_threshold": 0.15},
            "replay": {
                "buffer_hours": 1,
                "trigger_mode": "time_based",
                "outlier_detection": {"min_measurements_for_analysis": 2},
            },
            "features": {
                "kalman_filtering": True,
                "quality_scoring": True,
                "state_persistence": True,
                "resets": {"soft": True},
            },
        }

    @patch("src.processing.processor.get_state_db")
    @patch("src.processing.processor.logger")
    def test_snapshot_saved_after_reset(self, mock_logger, mock_get_db):
        """Test that a snapshot is saved immediately after a reset occurs."""
        mock_get_db.return_value = self.mock_db

        # Process initial measurements to establish baseline
        # These should be around 95kg
        initial_weights = [95.0, 94.8, 95.2, 94.9]
        for i, weight in enumerate(initial_weights):
            result = process_measurement(
                user_id=self.user_id,
                weight=weight,
                timestamp=self.base_time + timedelta(days=i),
                source="patient-device",
                config=self.config,
                unit="kg",
                db=self.mock_db,
            )
            assert result["accepted"]

        # Verify we have a state established
        assert self.user_id in self.states
        state_before_reset = self.states[self.user_id].copy()

        # Now process a measurement that should trigger a soft reset
        # This simulates the 100.1kg value from questionnaire
        reset_time = self.base_time + timedelta(days=10)
        reset_result = process_measurement(
            user_id=self.user_id,
            weight=100.1,  # >5kg change from ~95kg
            timestamp=reset_time,
            source="questionnaire",  # Trigger source for soft reset
            config=self.config,
            unit="kg",
            db=self.mock_db,
        )

        # Verify the reset occurred
        assert reset_result["accepted"]
        assert "reset_event" in reset_result
        assert reset_result["reset_event"]["type"] == "soft"

        # Debug: Print all snapshots to understand what's happening
        print(f"\nAll snapshots: {list(self.snapshots.keys())}")
        for key, snap in self.snapshots.items():
            print(
                f"  {key}: reset_type={snap['state'].get('reset_type')}, measurements={snap['state'].get('measurements_since_reset')}"
            )

        # CRITICAL: Verify a snapshot was saved at the reset time
        snapshot_key = f"{self.user_id}_{reset_time.isoformat()}"
        assert snapshot_key in self.snapshots, (
            f"Snapshot should be saved immediately after reset. Available: {list(self.snapshots.keys())}"
        )

        # Verify the snapshot contains the post-reset state
        snapshot = self.snapshots[snapshot_key]
        assert snapshot["timestamp"] == reset_time
        assert "reset_type" in snapshot["state"]
        # The snapshot should contain the soft reset type that just occurred
        assert snapshot["state"]["reset_type"] == "soft", (
            f"Expected 'soft' reset type, got '{snapshot['state']['reset_type']}'"
        )

    @patch("src.processing.processor.get_state_db")
    def test_replay_can_restore_after_reset(self, mock_get_db):
        """Test that replay can restore to the correct state after a reset."""
        mock_get_db.return_value = self.mock_db

        # Setup initial state
        initial_weights = [95.0, 94.8, 95.2]
        for i, weight in enumerate(initial_weights):
            result = process_measurement(
                user_id=self.user_id,
                weight=weight,
                timestamp=self.base_time + timedelta(days=i),
                source="patient-device",
                config=self.config,
                unit="kg",
                db=self.mock_db,
            )

        # Trigger a reset (need >5kg change from ~95kg)
        reset_time = self.base_time + timedelta(days=5)
        reset_result = process_measurement(
            user_id=self.user_id,
            weight=101.0,  # 6kg change from ~95kg
            timestamp=reset_time,
            source="questionnaire",
            config=self.config,
            unit="kg",
            db=self.mock_db,
        )

        # Verify the reset occurred
        assert "reset_event" in reset_result, "Reset should have occurred"
        assert reset_result["reset_event"]["type"] == "soft", (
            f"Expected soft reset, got {reset_result.get('reset_event')}"
        )

        # Add more measurements after reset
        post_reset_measurements = []
        for i in range(3):
            timestamp = reset_time + timedelta(hours=i + 1)
            weight = 95.0 + i * 0.1  # Back to normal range
            post_reset_measurements.append(
                {
                    "weight": weight,
                    "timestamp": timestamp,
                    "source": "patient-device",
                    "unit": "kg",
                }
            )

            process_measurement(
                user_id=self.user_id,
                weight=weight,
                timestamp=timestamp,
                source="patient-device",
                config=self.config,
                unit="kg",
                db=self.mock_db,
            )

        # Now simulate replay processing
        buffer_start_time = reset_time + timedelta(minutes=30)  # After reset
        replay_manager = ReplayManager(self.mock_db, self.config.get("replay", {}))

        # Debug: print available snapshots
        print(f"\nAvailable snapshots before restore:")
        for key in self.snapshots:
            print(f"  {key}: timestamp={self.snapshots[key]['timestamp']}")
        print(f"Looking for snapshot before: {buffer_start_time}")

        # Attempt to restore state to before buffer start
        restore_result = self.mock_db.check_and_restore_snapshot(
            self.user_id, buffer_start_time
        )

        # Should successfully restore to the post-reset snapshot
        assert restore_result["success"], (
            "Should find and restore from post-reset snapshot"
        )
        # The snapshot should be from the soft reset time (most recent before buffer_start_time)
        assert restore_result["snapshot_timestamp"] == reset_time, (
            f"Expected snapshot from {reset_time}, got {restore_result['snapshot_timestamp']}"
        )

        # Verify the restored state has the reset information
        restored_state = self.mock_db.get_state(self.user_id)
        assert restored_state["reset_type"] == "soft"
        # Handle both datetime and string formats
        reset_ts = restored_state["reset_timestamp"]
        if isinstance(reset_ts, str):
            assert reset_ts == reset_time.isoformat()
        else:
            assert reset_ts == reset_time

    @patch("src.processing.processor.get_state_db")
    def test_replay_fails_gracefully_without_snapshot(self, mock_get_db):
        """Test that replay handles missing snapshots gracefully."""
        mock_get_db.return_value = self.mock_db

        # Setup some state
        process_measurement(
            user_id=self.user_id,
            weight=95.0,
            timestamp=self.base_time,
            source="patient-device",
            config=self.config,
            unit="kg",
            db=self.mock_db,
        )

        # Clear snapshots AFTER processing to simulate missing snapshot
        self.snapshots.clear()

        # Try to restore to a time with no snapshot
        restore_result = self.mock_db.check_and_restore_snapshot(
            self.user_id, self.base_time + timedelta(hours=1)
        )

        # Should fail but not crash
        assert not restore_result["success"]
        assert "No snapshot found" in restore_result["error"]

        # Original state should remain unchanged
        assert self.user_id in self.states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
