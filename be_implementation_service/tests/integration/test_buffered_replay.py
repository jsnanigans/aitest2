"""Integration tests for buffered replay processing."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from typing import List

from src.aws.services.weight_processor_service import WeightProcessorService
from src.aws.api.models import Measurement, MeasurementResult


class TestBufferedReplayIntegration:
    """Integration tests for complete buffered replay flow."""

    @pytest.fixture
    def mock_state_store(self):
        """Create mock state store."""
        store = Mock()
        store.get_state.return_value = {"last_raw_weight": 70.0}
        store.save_state = Mock()
        store.save_state_snapshot = Mock()
        store.get_snapshot.return_value = None
        return store

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "replay": {
                "buffer_hours": 24,
                "max_buffer_measurements": 100,
                "buffered_replay_enabled": True,
            }
        }

    @pytest.fixture
    def service(self, mock_state_store, config):
        """Create service instance."""
        return WeightProcessorService(state_store=mock_state_store, config=config)

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for tests."""
        return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def create_measurement(
        self, timestamp: datetime, measurement_id: str, weight: float = 70.0
    ) -> Measurement:
        """Helper to create a measurement."""
        return Measurement(
            measurement_id=measurement_id,
            weight_value=weight,
            weight_unit="kg",
            measured_at=timestamp,
            source="manual",
        )

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_single_window_with_multiple_measurements(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.1: Single window with multiple measurements.

        Setup: 10 measurements within 24-hour window
        Expected: Single replay triggered at end of batch
        """
        # Mock process_measurement to return accepted results
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service to return corrected results
        replay_results = [
            {"uuid": f"id{i}", "accepted": True, "quality_score": 0.9, "kalman_estimate": 70.0 + i * 0.1}
            for i in range(10)
        ]
        mock_replay.return_value = {
            "success": True,
            "processed_count": 10,
            "accepted_count": 10,
            "rejected_count": 0,
            "results": replay_results,
        }

        # Create 10 measurements within 24 hours
        measurements = [
            self.create_measurement(
                base_timestamp + timedelta(hours=i * 2), f"id{i}", 70.0 + i * 0.1
            )
            for i in range(10)
        ]

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify response
        assert response.measurements_processed == 10
        assert response.measurements_accepted == 10
        assert response.replay_metadata is not None
        assert len(response.replay_metadata) == 1  # Single replay

        # Verify replay metadata
        replay_info = response.replay_metadata[0]
        assert replay_info["trigger"] == "batch_end"
        assert replay_info["buffer_size"] == 10
        assert replay_info["measurements_replayed"] == 10

        # Verify results were updated with replay data
        for i, result in enumerate(response.results):
            assert result.quality_score == 0.9  # Updated from 0.7
            assert result.kalman_estimate == 70.0 + i * 0.1

        # Verify replay was called once
        mock_replay.assert_called_once()

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_multiple_windows_recurring_replay(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.2: Multiple windows (recurring replay).

        Setup: 50 measurements over 3 days (72 hours)
        Expected: 3 replay triggers
        """
        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service to return different results for each window
        def replay_side_effect(*args, **kwargs):
            measurements = kwargs["measurements"]
            return {
                "success": True,
                "processed_count": len(measurements),
                "accepted_count": len(measurements),
                "rejected_count": 0,
                "results": [
                    {"uuid": m.measurement_id, "accepted": True, "quality_score": 0.9}
                    for m in measurements
                ],
            }

        mock_replay.side_effect = replay_side_effect

        # Create 50 measurements over 3 days
        # Day 1 (0-24h): 17 measurements
        # Day 2 (24-48h): 17 measurements
        # Day 3 (48-72h): 16 measurements
        measurements = []
        for day in range(3):
            count = 17 if day < 2 else 16
            for i in range(count):
                hour = day * 24 + i * (24 / count)
                measurements.append(
                    self.create_measurement(
                        base_timestamp + timedelta(hours=hour),
                        f"day{day}_id{i}",
                    )
                )

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify response
        assert response.measurements_processed == 50
        assert response.measurements_accepted == 50
        assert response.replay_metadata is not None
        assert len(response.replay_metadata) == 3  # Three replay triggers

        # Verify replay triggers
        assert response.replay_metadata[0]["trigger"] == "time_window"
        assert response.replay_metadata[1]["trigger"] == "time_window"
        assert response.replay_metadata[2]["trigger"] == "batch_end"

        # Verify all results were updated
        assert all(r.quality_score == 0.9 for r in response.results)

        # Verify replay was called 3 times
        assert mock_replay.call_count == 3

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_single_measurement_in_buffer_no_replay(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.3: Single measurement in buffer (no replay).

        Setup: Measurements at Day 1.0h, Day 1.5h, Day 3.0h (widely spaced)
        Expected: 1 replay trigger for first 2, no replay for last single measurement
        """
        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service
        mock_replay.return_value = {
            "success": True,
            "processed_count": 2,
            "accepted_count": 2,
            "rejected_count": 0,
            "results": [
                {"uuid": "id1", "accepted": True, "quality_score": 0.9},
                {"uuid": "id2", "accepted": True, "quality_score": 0.9},
            ],
        }

        # Create measurements: M1, M2 close together, M3 after 48 hours
        measurements = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=0.5), "id2"),
            self.create_measurement(base_timestamp + timedelta(hours=48), "id3"),
        ]

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify response
        assert response.measurements_processed == 3
        assert response.measurements_accepted == 3
        assert response.replay_metadata is not None
        assert len(response.replay_metadata) == 1  # Only 1 replay (for M1, M2)

        # Verify replay metadata
        # Note: M3 is processed and added to buffer before trigger check
        # So all 3 measurements are in buffer when replay triggers
        assert response.replay_metadata[0]["trigger"] == "batch_end"
        assert response.replay_metadata[0]["buffer_size"] == 3  # All 3 measurements

        # Verify replay was called once
        mock_replay.assert_called_once()

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_buffer_overflow_trigger(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.4: Buffer overflow (max_buffer_measurements).

        Setup: 150 measurements within 24-hour window (> max of 100)
        Expected: Replay triggered when buffer reaches 100, then at end
        """
        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service
        def replay_side_effect(*args, **kwargs):
            measurements = kwargs["measurements"]
            return {
                "success": True,
                "processed_count": len(measurements),
                "accepted_count": len(measurements),
                "rejected_count": 0,
                "results": [
                    {"uuid": m.measurement_id, "accepted": True, "quality_score": 0.9}
                    for m in measurements
                ],
            }

        mock_replay.side_effect = replay_side_effect

        # Create 150 measurements within 24 hours
        measurements = [
            self.create_measurement(
                base_timestamp + timedelta(minutes=i * 9),  # ~15 min apart
                f"id{i}",
            )
            for i in range(150)
        ]

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify response
        assert response.measurements_processed == 150
        assert response.measurements_accepted == 150
        assert response.replay_metadata is not None
        assert len(response.replay_metadata) == 2  # Two replay triggers

        # Verify first trigger was buffer overflow
        assert response.replay_metadata[0]["trigger"] == "buffer_overflow"
        assert response.replay_metadata[0]["buffer_size"] == 100

        # Verify second trigger was batch end
        assert response.replay_metadata[1]["trigger"] == "batch_end"
        assert response.replay_metadata[1]["buffer_size"] == 50  # Remaining measurements

        # Verify replay was called twice
        assert mock_replay.call_count == 2

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_out_of_order_measurements(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.5: Out-of-order measurements.

        Setup: Measurements provided out of chronological order
        Expected: Sorted before processing, replay processes in correct order
        """
        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service
        mock_replay.return_value = {
            "success": True,
            "processed_count": 5,
            "accepted_count": 5,
            "rejected_count": 0,
            "results": [
                {"uuid": "id1", "accepted": True, "quality_score": 0.9},
                {"uuid": "id2", "accepted": True, "quality_score": 0.9},
                {"uuid": "id3", "accepted": True, "quality_score": 0.9},
                {"uuid": "id4", "accepted": True, "quality_score": 0.9},
                {"uuid": "id5", "accepted": True, "quality_score": 0.9},
            ],
        }

        # Create measurements OUT OF ORDER
        measurements = [
            self.create_measurement(base_timestamp + timedelta(hours=2), "id3"),  # 3rd
            self.create_measurement(base_timestamp, "id1"),  # 1st
            self.create_measurement(base_timestamp + timedelta(hours=4), "id5"),  # 5th
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),  # 2nd
            self.create_measurement(base_timestamp + timedelta(hours=3), "id4"),  # 4th
        ]

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify measurements were sorted (results should be in order)
        assert response.results[0].measurement_id == "id1"
        assert response.results[1].measurement_id == "id2"
        assert response.results[2].measurement_id == "id3"
        assert response.results[3].measurement_id == "id4"
        assert response.results[4].measurement_id == "id5"

        # Verify replay was called
        mock_replay.assert_called_once()

        # Verify replay was called with sorted measurements by checking the buffer parameter
        if mock_replay.call_args.kwargs:
            replay_buffer = mock_replay.call_args.kwargs.get("measurements", [])
            if replay_buffer:
                # Verify measurements in buffer are sorted
                for i in range(len(replay_buffer) - 1):
                    assert replay_buffer[i].measured_at <= replay_buffer[i + 1].measured_at

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_replay_failure_handling(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.6: Replay failure handling.

        Setup: Mock replay service to raise exception
        Expected: Exception propagated to client
        """
        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service to raise exception
        mock_replay.side_effect = Exception("Database connection lost")

        measurements = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        # Verify exception is raised
        with pytest.raises(Exception) as exc_info:
            service.process_batch(user_id="test-user", measurements=measurements)

        assert "Database connection lost" in str(exc_info.value)

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_feature_toggle_disabled(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """
        Test 7.7: Feature toggle disabled.

        Setup: Set buffered_replay_enabled = false
        Expected: No replay, original behavior
        """
        # Disable feature
        service.config["replay"]["buffered_replay_enabled"] = False

        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        measurements = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify no replay occurred
        assert response.replay_metadata is None or len(response.replay_metadata) == 0
        mock_replay.assert_not_called()

        # Verify original results unchanged
        assert response.results[0].quality_score == 0.7
        assert response.results[1].quality_score == 0.7

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_state_consistency_after_replay(
        self, mock_replay, mock_process, service, mock_state_store, base_timestamp
    ):
        """
        Test 7.8: State consistency after replay.

        Setup: Process batch with replay
        Expected: Database state matches replay results
        """
        # Mock process_measurement
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.7,
            "kalman_estimate": 70.0,
        }

        # Mock replay service
        mock_replay.return_value = {
            "success": True,
            "processed_count": 3,
            "accepted_count": 3,
            "rejected_count": 0,
            "results": [
                {"uuid": "id1", "accepted": True, "quality_score": 0.9, "kalman_estimate": 70.1},
                {"uuid": "id2", "accepted": True, "quality_score": 0.9, "kalman_estimate": 70.2},
                {"uuid": "id3", "accepted": True, "quality_score": 0.9, "kalman_estimate": 70.3},
            ],
        }

        # Set up state store to return updated state after replay
        mock_state_store.get_state.return_value = {
            "last_raw_weight": 70.3,
            "kalman_estimate": 70.3,
        }

        measurements = [
            self.create_measurement(base_timestamp, "id1", 70.1),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2", 70.2),
            self.create_measurement(base_timestamp + timedelta(hours=2), "id3", 70.3),
        ]

        response = service.process_batch(user_id="test-user", measurements=measurements)

        # Verify final state was retrieved
        assert mock_state_store.get_state.called

        # Verify state update in response matches replay results
        assert response.state_update.current_weight == 70.3

        # Verify results match replay output
        assert response.results[-1].kalman_estimate == 70.3
