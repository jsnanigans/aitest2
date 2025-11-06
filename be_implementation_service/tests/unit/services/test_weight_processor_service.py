"""Unit tests for WeightProcessorService buffered replay functionality.

Tests the buffered replay system which enables efficient batch reprocessing of
weight measurements to maintain eventual consistency. Critical for handling
scale and data quality improvements.

Key functionality tested:
- Replay trigger conditions (buffer size, time window, is_last flag)
- Replay execution with error handling
- Result merging after replay completes
- Snapshot creation for replay checkpoints
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

from src.aws.services.weight_processor_service import WeightProcessorService
from src.aws.api.models import Measurement, MeasurementResult


class TestShouldTriggerReplay:
    """Unit tests for _should_trigger_replay() method."""

    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies."""
        mock_state_store = Mock()
        config = {
            "replay": {
                "buffer_hours": 24,
                "max_buffer_measurements": 100,
                "buffered_replay_enabled": True,
            }
        }
        return WeightProcessorService(state_store=mock_state_store, config=config)

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for tests."""
        return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def create_measurement(self, timestamp: datetime, measurement_id: str = "test-id") -> Measurement:
        """Helper to create a measurement."""
        return Measurement(
            measurement_id=measurement_id,
            weight_value=70.0,
            weight_unit="kg",
            measured_at=timestamp,
            source="manual",
        )

    def test_replay_not_triggered_when_buffer_is_empty(self, service, base_timestamp):
        """Test replay not triggered when buffer is empty.

        Empty buffer means no measurements to replay, so should return False.
        """
        buffer = []
        result = service._should_trigger_replay(buffer, base_timestamp, is_last=False)
        assert result is False

    def test_single_measurement_buffer_returns_false_even_if_last(self, service, base_timestamp):
        """Test 6.1.1 & 6.1.7: Buffer with < 2 measurements returns False (even if is_last=True)."""
        buffer = [self.create_measurement(base_timestamp)]

        # Test with is_last=False
        result = service._should_trigger_replay(buffer, base_timestamp, is_last=False)
        assert result is False

        # Test with is_last=True (should still be False due to minimum buffer size)
        result = service._should_trigger_replay(buffer, base_timestamp, is_last=True)
        assert result is False

    def test_replay_triggered_when_is_last_flag_true_regardless_of_buffer_size(self, service, base_timestamp):
        """Test replay triggered when is_last flag is True, regardless of buffer size.

        When processing the last measurement in a batch, trigger replay to ensure
        all buffered measurements are processed before batch completion.
        """
        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        result = service._should_trigger_replay(
            buffer, base_timestamp + timedelta(hours=1), is_last=True
        )
        assert result is True

    def test_replay_triggered_when_time_window_exceeds_24_hours(self, service, base_timestamp):
        """Test replay triggered when time window exceeds configured buffer_hours (24h).

        After 24 hours, buffered measurements should be replayed to avoid
        indefinite buffering and maintain eventual consistency.
        """
        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=12), "id2"),
        ]

        # Current timestamp is 25 hours after first measurement (exceeds 24 hour window)
        current_timestamp = base_timestamp + timedelta(hours=25)

        result = service._should_trigger_replay(buffer, current_timestamp, is_last=False)
        assert result is True

    def test_replay_triggered_when_buffer_reaches_100_measurements(self, service, base_timestamp):
        """Test replay triggered when buffer reaches max_buffer_measurements (100).

        Prevents unbounded buffer growth by triggering replay when buffer size limit
        is reached, ensuring measurements are processed in reasonable batches.
        """
        # Create buffer with exactly max_buffer_measurements (100)
        buffer = [
            self.create_measurement(base_timestamp + timedelta(minutes=i), f"id{i}")
            for i in range(100)
        ]

        current_timestamp = base_timestamp + timedelta(minutes=100)

        result = service._should_trigger_replay(buffer, current_timestamp, is_last=False)
        assert result is True

    def test_within_time_window_not_last_returns_false(self, service, base_timestamp):
        """Test 6.1.5: Buffer with 2+ measurements, within time window, not last, returns False."""
        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=12), "id2"),
        ]

        # Current timestamp is 23 hours after first measurement (within 24 hour window)
        current_timestamp = base_timestamp + timedelta(hours=23)

        result = service._should_trigger_replay(buffer, current_timestamp, is_last=False)
        assert result is False

    def test_exactly_at_buffer_hours_triggers(self, service, base_timestamp):
        """Test edge case: Exactly at buffer_hours should trigger."""
        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        # Exactly 24 hours after first measurement
        current_timestamp = base_timestamp + timedelta(hours=24)

        result = service._should_trigger_replay(buffer, current_timestamp, is_last=False)
        assert result is True


class TestExecuteBufferedReplay:
    """Unit tests for _execute_buffered_replay() method."""

    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies."""
        mock_state_store = Mock()
        config = {
            "replay": {
                "buffer_hours": 24,
                "max_buffer_measurements": 100,
            }
        }
        return WeightProcessorService(state_store=mock_state_store, config=config)

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for tests."""
        return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def create_measurement(self, timestamp: datetime, measurement_id: str = "test-id") -> Measurement:
        """Helper to create a measurement."""
        return Measurement(
            measurement_id=measurement_id,
            weight_value=70.0,
            weight_unit="kg",
            measured_at=timestamp,
            source="manual",
        )

    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_replay_execution_returns_result_dict_with_success_status(self, mock_replay, service, base_timestamp):
        """Test replay execution returns result dictionary with success status and metadata.

        Validates the structure and completeness of the replay result, including
        success flag, counts, and timing information for monitoring.
        """
        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        # Mock replay service response
        mock_replay.return_value = {
            "success": True,
            "processed_count": 2,
            "accepted_count": 2,
            "rejected_count": 0,
            "results": [
                {"uuid": "id1", "accepted": True, "quality_score": 0.9},
                {"uuid": "id2", "accepted": True, "quality_score": 0.85},
            ],
        }

        result = service._execute_buffered_replay(
            user_id="test-user",
            buffer=buffer,
            buffer_start_time=base_timestamp,
            user_height_m=1.75,
        )

        assert result["success"] is True
        assert result["processed_count"] == 2
        assert result["accepted_count"] == 2
        assert "duration_seconds" in result
        assert isinstance(result["duration_seconds"], (int, float))

    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_replay_execution_propagates_replay_service_exceptions(self, mock_replay, service, base_timestamp):
        """Test replay execution propagates exceptions from replay service.

        Ensures errors during replay are not silently caught but instead propagated
        to caller for proper error handling and alerting.
        """
        buffer = [self.create_measurement(base_timestamp, "id1")]

        # Mock replay service to raise exception
        mock_replay.side_effect = Exception("Replay failed: database error")

        with pytest.raises(Exception) as exc_info:
            service._execute_buffered_replay(
                user_id="test-user",
                buffer=buffer,
                buffer_start_time=base_timestamp,
            )

        assert "Replay failed: database error" in str(exc_info.value)

    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_correct_parameters_passed_to_replay_measurements(self, mock_replay, service, base_timestamp):
        """Test 6.2.3: Correct parameters passed to replay_measurements."""
        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        mock_replay.return_value = {
            "success": True,
            "processed_count": 2,
            "accepted_count": 2,
            "rejected_count": 0,
            "results": [],
        }

        service._execute_buffered_replay(
            user_id="test-user-123",
            buffer=buffer,
            buffer_start_time=base_timestamp,
            user_height_m=1.80,
        )

        # Verify replay_measurements was called with correct parameters
        mock_replay.assert_called_once()
        call_args = mock_replay.call_args

        assert call_args.kwargs["user_id"] == "test-user-123"
        assert call_args.kwargs["measurements"] == buffer
        assert call_args.kwargs["replay_from"] == base_timestamp
        assert call_args.kwargs["user_height_m"] == 1.80
        assert call_args.kwargs["state_store"] == service.state_store
        assert call_args.kwargs["config"] == service.config

    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_replay_failure_status_raises_exception(self, mock_replay, service, base_timestamp):
        """Test that replay service returning success=False raises exception."""
        buffer = [self.create_measurement(base_timestamp, "id1")]

        # Mock replay service to return failure status
        mock_replay.return_value = {
            "success": False,
            "error": "State snapshot not found",
        }

        with pytest.raises(Exception) as exc_info:
            service._execute_buffered_replay(
                user_id="test-user",
                buffer=buffer,
                buffer_start_time=base_timestamp,
            )

        assert "Replay failed: State snapshot not found" in str(exc_info.value)


class TestMergeReplayResults:
    """Unit tests for _merge_replay_results() method."""

    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies."""
        mock_state_store = Mock()
        config = {"replay": {"buffered_replay_enabled": True}}
        return WeightProcessorService(state_store=mock_state_store, config=config)

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for tests."""
        return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def create_measurement(self, timestamp: datetime, measurement_id: str) -> Measurement:
        """Helper to create a measurement."""
        return Measurement(
            measurement_id=measurement_id,
            weight_value=70.0,
            weight_unit="kg",
            measured_at=timestamp,
            source="manual",
        )

    def test_buffered_measurements_updated_with_replay_data(self, service, base_timestamp):
        """Test 6.3.1: Buffered measurements updated with replay data."""
        # Original results
        original_results = [
            MeasurementResult(
                measurement_id="id1",
                accepted=True,
                quality_score=0.7,
                kalman_estimate=69.5,
            ),
            MeasurementResult(
                measurement_id="id2",
                accepted=True,
                quality_score=0.6,
                kalman_estimate=70.2,
            ),
        ]

        # Replay output with corrected values
        replay_output = {
            "results": [
                {"uuid": "id1", "accepted": True, "quality_score": 0.9, "kalman_estimate": 69.8},
                {"uuid": "id2", "accepted": True, "quality_score": 0.85, "kalman_estimate": 70.0},
            ]
        }

        buffer = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
        ]

        updated_results = service._merge_replay_results(original_results, replay_output, buffer)

        assert len(updated_results) == 2
        assert updated_results[0].measurement_id == "id1"
        assert updated_results[0].quality_score == 0.9
        assert updated_results[0].kalman_estimate == 69.8
        assert updated_results[1].measurement_id == "id2"
        assert updated_results[1].quality_score == 0.85
        assert updated_results[1].kalman_estimate == 70.0

    def test_non_buffered_measurements_remain_unchanged(self, service, base_timestamp):
        """Test 6.3.2: Non-buffered measurements remain unchanged."""
        # Original results (3 measurements, but only 2 in buffer)
        original_results = [
            MeasurementResult(measurement_id="id1", accepted=False, rejection_reason="outlier"),
            MeasurementResult(measurement_id="id2", accepted=True, quality_score=0.6),
            MeasurementResult(measurement_id="id3", accepted=True, quality_score=0.7),
        ]

        # Replay output only for id2 and id3
        replay_output = {
            "results": [
                {"uuid": "id2", "accepted": True, "quality_score": 0.85},
                {"uuid": "id3", "accepted": True, "quality_score": 0.9},
            ]
        }

        # Buffer only contains id2 and id3
        buffer = [
            self.create_measurement(base_timestamp, "id2"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id3"),
        ]

        updated_results = service._merge_replay_results(original_results, replay_output, buffer)

        # id1 should remain unchanged (not in buffer)
        assert updated_results[0].measurement_id == "id1"
        assert updated_results[0].accepted is False
        assert updated_results[0].rejection_reason == "outlier"

        # id2 and id3 should be updated
        assert updated_results[1].quality_score == 0.85
        assert updated_results[2].quality_score == 0.9

    def test_measurement_id_matching_works_correctly(self, service, base_timestamp):
        """Test 6.3.3: Measurement ID matching works correctly."""
        original_results = [
            MeasurementResult(measurement_id="abc-123", accepted=True, quality_score=0.5),
            MeasurementResult(measurement_id="def-456", accepted=True, quality_score=0.6),
        ]

        replay_output = {
            "results": [
                {"uuid": "abc-123", "accepted": True, "quality_score": 0.95},
                {"uuid": "def-456", "accepted": True, "quality_score": 0.88},
            ]
        }

        buffer = [
            self.create_measurement(base_timestamp, "abc-123"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "def-456"),
        ]

        updated_results = service._merge_replay_results(original_results, replay_output, buffer)

        # Verify correct matching by ID
        result_map = {r.measurement_id: r for r in updated_results}
        assert result_map["abc-123"].quality_score == 0.95
        assert result_map["def-456"].quality_score == 0.88

    def test_all_result_fields_updated_from_replay(self, service, base_timestamp):
        """Test 6.3.4: All result fields updated from replay (quality_score, kalman_estimate, etc.)."""
        original_results = [
            MeasurementResult(
                measurement_id="id1",
                accepted=True,
                quality_score=0.5,
                kalman_estimate=69.0,
                kalman_uncertainty=2.0,
            ),
        ]

        replay_output = {
            "results": [
                {
                    "uuid": "id1",
                    "accepted": True,
                    "quality_score": 0.95,
                    "kalman_estimate": 70.5,
                    # Note: replay service doesn't return kalman_uncertainty
                }
            ]
        }

        buffer = [self.create_measurement(base_timestamp, "id1")]

        updated_results = service._merge_replay_results(original_results, replay_output, buffer)

        result = updated_results[0]
        assert result.accepted is True
        assert result.quality_score == 0.95
        assert result.kalman_estimate == 70.5
        # Fields not in replay output should retain original values
        assert result.kalman_uncertainty == 2.0

    def test_rejected_measurements_in_replay_handled_correctly(self, service, base_timestamp):
        """Test 6.3.5: Rejected measurements in replay are handled correctly."""
        original_results = [
            MeasurementResult(
                measurement_id="id1",
                accepted=True,  # Originally accepted
                quality_score=0.7,
            ),
        ]

        # Replay rejects the measurement
        replay_output = {
            "results": [
                {"uuid": "id1", "accepted": False, "quality_score": 0.3}
            ]
        }

        buffer = [self.create_measurement(base_timestamp, "id1")]

        updated_results = service._merge_replay_results(original_results, replay_output, buffer)

        result = updated_results[0]
        assert result.accepted is False  # Should now be rejected
        assert result.quality_score == 0.3

    def test_empty_replay_results_handled_gracefully(self, service, base_timestamp):
        """Test edge case: Empty replay results."""
        original_results = [
            MeasurementResult(measurement_id="id1", accepted=True, quality_score=0.7),
        ]

        replay_output = {"results": []}
        buffer = [self.create_measurement(base_timestamp, "id1")]

        updated_results = service._merge_replay_results(original_results, replay_output, buffer)

        # Should return original results unchanged
        assert len(updated_results) == 1
        assert updated_results[0].quality_score == 0.7


class TestSnapshotCreation:
    """Unit tests for snapshot creation logic."""

    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies."""
        mock_state_store = Mock()
        config = {
            "replay": {
                "buffer_hours": 24,
                "max_buffer_measurements": 100,
                "buffered_replay_enabled": True,
            }
        }
        return WeightProcessorService(state_store=mock_state_store, config=config)

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for tests."""
        return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def create_measurement(self, timestamp: datetime, measurement_id: str = "test-id") -> Measurement:
        """Helper to create a measurement."""
        return Measurement(
            measurement_id=measurement_id,
            weight_value=70.0,
            weight_unit="kg",
            measured_at=timestamp,
            source="manual",
        )

    @patch('src.aws.services.weight_processor_service.process_measurement')
    def test_snapshot_created_before_first_buffered_measurement(
        self, mock_process, service, base_timestamp
    ):
        """Test 6.4.1: Snapshot created before first buffered measurement."""
        # Mock process_measurement to return accepted result
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.9,
            "kalman_estimate": 70.0,
        }

        # Setup mocks
        service.state_store.get_state.return_value = {"last_raw_weight": 69.0}
        service.state_store.save_state_snapshot = Mock()

        measurements = [self.create_measurement(base_timestamp, "id1")]

        service.process_batch(user_id="test-user", measurements=measurements)

        # Verify snapshot was created
        service.state_store.save_state_snapshot.assert_called_once_with("test-user", base_timestamp)

    @patch('src.aws.services.weight_processor_service.process_measurement')
    def test_snapshot_not_created_for_rejected_measurements(
        self, mock_process, service, base_timestamp
    ):
        """Test 6.4.2: Snapshot not created for rejected measurements."""
        # Mock process_measurement to return rejected result
        mock_process.return_value = {
            "accepted": False,
            "reason": "outlier",
        }

        # Setup mocks
        service.state_store.get_state.return_value = {"last_raw_weight": 69.0}
        service.state_store.save_state_snapshot = Mock()

        measurements = [self.create_measurement(base_timestamp, "id1")]

        service.process_batch(user_id="test-user", measurements=measurements)

        # Verify snapshot was NOT created (measurement was rejected)
        service.state_store.save_state_snapshot.assert_not_called()

    @patch('src.aws.services.weight_processor_service.process_measurement')
    @patch('src.aws.services.weight_processor_service.replay_measurements')
    def test_snapshot_created_once_per_buffer_window(
        self, mock_replay, mock_process, service, base_timestamp
    ):
        """Test 6.4.3: Snapshot created once per buffer window."""
        # Mock process_measurement to return accepted results
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.9,
            "kalman_estimate": 70.0,
        }

        # Mock replay service
        mock_replay.return_value = {
            "success": True,
            "processed_count": 3,
            "accepted_count": 3,
            "rejected_count": 0,
            "results": [
                {"uuid": "id1", "accepted": True, "quality_score": 0.9},
                {"uuid": "id2", "accepted": True, "quality_score": 0.9},
                {"uuid": "id3", "accepted": True, "quality_score": 0.9},
            ],
        }

        # Setup mocks
        service.state_store.get_state.return_value = {"last_raw_weight": 69.0}
        service.state_store.get_snapshot.return_value = None
        service.state_store.save_state_snapshot = Mock()

        # Three measurements: first creates snapshot, second and third use same buffer
        measurements = [
            self.create_measurement(base_timestamp, "id1"),
            self.create_measurement(base_timestamp + timedelta(hours=1), "id2"),
            self.create_measurement(base_timestamp + timedelta(hours=2), "id3"),
        ]

        service.process_batch(user_id="test-user", measurements=measurements)

        # Verify snapshot was created exactly once (before first measurement)
        assert service.state_store.save_state_snapshot.call_count == 1
        service.state_store.save_state_snapshot.assert_called_with("test-user", base_timestamp)
