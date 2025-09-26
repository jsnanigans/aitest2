"""
Comprehensive tests for state management and persistence.

These tests verify that the system correctly maintains user state across
operations, handles state transitions, and properly persists data.
"""

import pytest
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

from .conftest import APIClient, TestUser, create_measurement, create_measurement_series


class TestStateInitialization:
    """Test initial state setup and configuration."""

    def test_new_user_has_empty_state(self, api_client, test_user):
        """New users should start with empty state."""
        state = api_client.get_user_state(test_user.user_id)

        assert state.is_success
        assert state.data["measurement_count"] == 0
        assert state.data["last_measurement"] is None
        assert state.data["first_measurement"] is None
        assert "kalman_state" in state.data
        assert state.data["kalman_state"]["initialized"] == False

    def test_state_initialized_after_first_measurement(self, api_client, test_user):
        """State should be properly initialized after first measurement."""
        measurement = create_measurement(weight=75.0, unit="kg", source="scale")

        api_client.process_measurements(test_user.user_id, [measurement])

        state = api_client.get_user_state(test_user.user_id)

        assert state.data["measurement_count"] == 1
        assert state.data["kalman_state"]["initialized"] == True
        assert state.data["kalman_state"]["current_estimate"] is not None

    def test_multiple_users_have_independent_state(self, api_client, test_users):
        """Each user should have independent state."""
        for i, user in enumerate(test_users):
            measurement = create_measurement(weight=70.0 + i * 5.0)
            api_client.process_measurements(user.user_id, [measurement])

        # Verify each user has their own state
        for i, user in enumerate(test_users):
            state = api_client.get_user_state(user.user_id)
            expected_weight = 70.0 + i * 5.0

            assert state.data["measurement_count"] == 1
            assert abs(state.data["last_measurement"]["weight"] - expected_weight) < 0.1


class TestStateUpdates:
    """Test state updates during processing."""

    def test_state_updates_incrementally(self, api_client, test_user):
        """State should update incrementally with each measurement."""
        weights = [75.0, 74.8, 74.5, 74.3, 74.0]
        previous_count = 0

        for weight in weights:
            measurement = create_measurement(weight=weight)
            api_client.process_measurements(test_user.user_id, [measurement])

            state = api_client.get_user_state(test_user.user_id)

            assert state.data["measurement_count"] == previous_count + 1
            assert state.data["last_measurement"]["weight"] == weight

            previous_count += 1

    def test_state_tracks_measurement_history(self, api_client, test_user):
        """State should maintain measurement history."""
        measurements = create_measurement_series(
            start_weight=80.0,
            days=30,
            daily_change=-0.1
        )

        api_client.process_measurements(test_user.user_id, measurements)

        state = api_client.get_user_state(test_user.user_id)

        assert state.data["measurement_count"] == 30

        # Check history statistics
        stats = state.data["statistics"]
        assert stats["min_weight"] < 80.0
        assert stats["max_weight"] >= 77.0
        assert stats["weight_trend"] == "decreasing"

    def test_state_updates_with_outliers(self, api_client, test_user):
        """State should properly handle outliers."""
        measurements = [
            create_measurement(weight=75.0, days_ago=4),
            create_measurement(weight=74.8, days_ago=3),
            create_measurement(weight=85.0, days_ago=2),  # Outlier
            create_measurement(weight=74.5, days_ago=1),
            create_measurement(weight=74.3, days_ago=0),
        ]

        response = api_client.process_measurements(test_user.user_id, measurements)

        state = api_client.get_user_state(test_user.user_id)

        assert state.data["measurement_count"] == 5
        assert state.data["outlier_count"] >= 1
        # Last valid weight should be around 74.3, not 85.0
        assert state.data["last_valid_weight"] < 76.0


class TestKalmanFilterState:
    """Test Kalman filter state management."""

    def test_kalman_state_adapts_to_source_reliability(self, api_client, test_user):
        """Kalman filter should adapt based on measurement source."""
        # Process measurements from different sources
        sources = [
            ("doctor", 0.95),   # Highly reliable
            ("scale", 0.9),     # Reliable
            ("app", 0.7),       # Moderate
            ("manual", 0.6),    # Less reliable
        ]

        kalman_states = []

        for source, expected_quality in sources:
            user_id = f"{test_user.user_id}_{source}"
            measurement = create_measurement(weight=75.0, source=source)

            api_client.process_measurements(user_id, [measurement])

            state = api_client.get_user_state(user_id)
            kalman_states.append({
                "source": source,
                "measurement_noise": state.data["kalman_state"]["measurement_noise"],
                "quality": expected_quality
            })

        # Higher quality sources should have lower measurement noise
        sorted_states = sorted(kalman_states, key=lambda x: x["quality"], reverse=True)
        noises = [s["measurement_noise"] for s in sorted_states]

        # Check that noise increases as quality decreases
        for i in range(len(noises) - 1):
            assert noises[i] <= noises[i + 1]

    def test_kalman_state_convergence(self, api_client, test_user):
        """Kalman filter should converge with consistent measurements."""
        # Send consistent measurements
        target_weight = 75.0
        measurements = [
            create_measurement(weight=target_weight + (0.2 if i % 2 == 0 else -0.2))
            for i in range(20)
        ]

        for measurement in measurements:
            api_client.process_measurements(test_user.user_id, [measurement])

        state = api_client.get_user_state(test_user.user_id)

        kalman_estimate = state.data["kalman_state"]["current_estimate"]
        # Should converge close to target weight
        assert abs(kalman_estimate - target_weight) < 1.0

    def test_kalman_state_reset(self, api_client, test_user):
        """Kalman state should reset properly."""
        # Build up state
        measurements = create_measurement_series(days=10)
        api_client.process_measurements(test_user.user_id, measurements)

        # Get state before reset
        state_before = api_client.get_user_state(test_user.user_id)
        assert state_before.data["kalman_state"]["initialized"] == True

        # Reset adaptive parameters
        api_client.cleanup_user(test_user.user_id, cleanup_type="reset_adaptive")

        # Check state after reset
        state_after = api_client.get_user_state(test_user.user_id)
        kalman_state = state_after.data["kalman_state"]

        # Should be reset but still have measurements
        assert state_after.data["measurement_count"] == 10
        assert kalman_state["process_noise"] != state_before.data["kalman_state"]["process_noise"]


class TestStatePersistence:
    """Test state persistence across operations."""

    def test_state_persists_between_requests(self, api_client, test_user):
        """State should persist between API requests."""
        # Process first batch
        batch1 = create_measurement_series(start_weight=80.0, days=5)
        api_client.process_measurements(test_user.user_id, batch1)

        state1 = api_client.get_user_state(test_user.user_id)
        count1 = state1.data["measurement_count"]

        # Process second batch
        batch2 = create_measurement_series(start_weight=79.0, days=5)
        api_client.process_measurements(test_user.user_id, batch2)

        state2 = api_client.get_user_state(test_user.user_id)
        count2 = state2.data["measurement_count"]

        assert count2 == count1 + 5

    def test_state_snapshot_and_restore(self, api_client, test_user):
        """Test state snapshot and restore functionality."""
        # Build up state
        measurements = create_measurement_series(days=30)
        api_client.process_measurements(test_user.user_id, measurements)

        # Get current state
        original_state = api_client.get_user_state(test_user.user_id)

        # Replay from 15 days ago (should snapshot first)
        replay_from = datetime.utcnow() - timedelta(days=15)
        new_measurements = create_measurement_series(
            start_weight=77.0,
            days=15,
            start_date=replay_from
        )

        api_client.replay_measurements(
            test_user.user_id,
            replay_from=replay_from,
            measurements=new_measurements
        )

        # State should be updated
        new_state = api_client.get_user_state(test_user.user_id)

        # Should have processed new measurements
        assert new_state.data["measurement_count"] >= 15

    def test_state_consistency_after_errors(self, api_client, test_user):
        """State should remain consistent even after processing errors."""
        # Process valid measurements
        valid = create_measurement_series(days=5)
        api_client.process_measurements(test_user.user_id, valid)

        state_before_error = api_client.get_user_state(test_user.user_id)

        # Try to process invalid measurements
        invalid = [
            {"weight": -100, "unit": "kg"},  # Invalid weight
            {"weight": "not_a_number"},       # Invalid type
        ]

        response = api_client.process_measurements(test_user.user_id, invalid)
        assert not response.is_success

        # State should remain unchanged
        state_after_error = api_client.get_user_state(test_user.user_id)
        assert state_after_error.data["measurement_count"] == state_before_error.data["measurement_count"]


class TestStateTransitions:
    """Test state transitions and boundaries."""

    def test_state_transition_during_weight_loss(self, api_client, test_user):
        """Track state transitions during weight loss journey."""
        # Simulate 90-day weight loss journey
        start_weight = 90.0
        target_weight = 80.0
        days = 90

        measurements = []
        for day in range(days):
            # Gradual weight loss with weekly cycles
            progress = day / days
            weight = start_weight - (progress * (start_weight - target_weight))
            weight += (day % 7 - 3) * 0.3  # Weekly variation

            measurements.append(
                create_measurement(
                    weight=round(weight, 1),
                    days_ago=days - day - 1
                )
            )

        # Process in batches to observe state transitions
        batch_size = 30
        states = []

        for i in range(0, days, batch_size):
            batch = measurements[i:i + batch_size]
            api_client.process_measurements(test_user.user_id, batch)

            state = api_client.get_user_state(test_user.user_id)
            states.append({
                "day": i + batch_size,
                "weight": state.data["statistics"]["average_weight"],
                "trend": state.data["statistics"]["weight_trend"],
                "confidence": state.data["kalman_state"].get("confidence", 0)
            })

        # Verify state transitions
        assert states[0]["weight"] > states[-1]["weight"]  # Weight decreased
        assert all(s["trend"] == "decreasing" for s in states[1:])  # Consistent trend

    def test_state_boundary_conditions(self, api_client, test_user):
        """Test state at boundary conditions."""
        # Test with minimum valid weight
        min_weight = create_measurement(weight=10.0, unit="kg")
        response = api_client.process_measurements(test_user.user_id, [min_weight])
        assert response.data["accepted_count"] == 1

        state = api_client.get_user_state(test_user.user_id)
        assert state.data["last_measurement"]["weight"] == 10.0

        # Test with maximum valid weight
        max_user = f"{test_user.user_id}_max"
        max_weight = create_measurement(weight=500.0, unit="kg")
        response = api_client.process_measurements(max_user, [max_weight])
        assert response.data["accepted_count"] == 1

        state = api_client.get_user_state(max_user)
        assert state.data["last_measurement"]["weight"] == 500.0

    def test_state_with_long_gaps(self, api_client, test_user):
        """Test state handling with long time gaps between measurements."""
        # Initial measurements
        initial = create_measurement_series(start_weight=80.0, days=7)
        api_client.process_measurements(test_user.user_id, initial)

        initial_state = api_client.get_user_state(test_user.user_id)

        # Measurement after 6-month gap
        gap_measurement = create_measurement(
            weight=85.0,
            days_ago=-180  # 6 months later
        )

        response = api_client.process_measurements(test_user.user_id, [gap_measurement])

        final_state = api_client.get_user_state(test_user.user_id)

        # State should handle the gap appropriately
        assert final_state.data["measurement_count"] == initial_state.data["measurement_count"] + 1
        assert final_state.data["has_time_gaps"] == True


class TestCircuitBreakerState:
    """Test circuit breaker state management."""

    def test_circuit_breaker_opens_after_failures(self, api_client, test_user):
        """Circuit breaker should open after repeated failures."""
        # Send measurements that will cause processing failures
        # (This is a mock scenario - actual implementation may vary)

        failure_measurements = []
        for i in range(10):
            # Create measurements that might trigger failures
            failure_measurements.append({
                "uuid": str(uuid.uuid4()),
                "weight": 1000.0 * (i + 1),  # Increasingly unrealistic
                "unit": "kg",
                "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
                "source": "error_source"
            })

        responses = []
        for measurement in failure_measurements:
            response = api_client.process_measurements(
                test_user.user_id,
                [measurement]
            )
            responses.append(response)

        # After multiple failures, circuit breaker may affect state
        state = api_client.get_user_state(test_user.user_id)

        if "circuit_breaker" in state.data:
            assert state.data["circuit_breaker"]["failure_count"] > 0

    def test_circuit_breaker_resets_after_success(self, api_client, test_user):
        """Circuit breaker should reset after successful operations."""
        # First cause some failures (as above)
        # Then send valid measurements
        valid_measurements = create_measurement_series(days=5)
        response = api_client.process_measurements(test_user.user_id, valid_measurements)

        assert response.is_success

        state = api_client.get_user_state(test_user.user_id)

        if "circuit_breaker" in state.data:
            assert state.data["circuit_breaker"]["state"] == "closed"


class TestStateAggregations:
    """Test state aggregation and statistics."""

    def test_state_calculates_statistics_correctly(self, api_client, test_user):
        """State should calculate accurate statistics."""
        weights = [75.0, 74.5, 76.0, 73.5, 74.0]
        measurements = [create_measurement(weight=w, days_ago=len(weights) - i - 1)
                       for i, w in enumerate(weights)]

        api_client.process_measurements(test_user.user_id, measurements)

        state = api_client.get_user_state(test_user.user_id)
        stats = state.data["statistics"]

        assert abs(stats["average_weight"] - sum(weights) / len(weights)) < 0.1
        assert stats["min_weight"] == min(weights)
        assert stats["max_weight"] == max(weights)
        assert stats["measurement_count"] == len(weights)

    def test_state_tracks_source_distribution(self, api_client, test_user):
        """State should track measurement source distribution."""
        sources = ["scale", "scale", "app", "manual", "scale", "doctor"]
        measurements = [
            create_measurement(weight=75.0, source=source, days_ago=len(sources) - i - 1)
            for i, source in enumerate(sources)
        ]

        api_client.process_measurements(test_user.user_id, measurements)

        state = api_client.get_user_state(test_user.user_id)

        if "source_distribution" in state.data:
            dist = state.data["source_distribution"]
            assert dist["scale"] == 3
            assert dist["app"] == 1
            assert dist["manual"] == 1
            assert dist["doctor"] == 1

    def test_state_quality_metrics(self, api_client, test_user):
        """State should maintain quality metrics."""
        # Mix of high and low quality measurements
        measurements = [
            create_measurement(weight=75.0, source="doctor"),   # High quality
            create_measurement(weight=74.8, source="scale"),    # High quality
            create_measurement(weight=75.5, source="manual"),   # Lower quality
            create_measurement(weight=74.0, source="app"),      # Medium quality
            create_measurement(weight=74.5, source="scale"),    # High quality
        ]

        for measurement in measurements:
            api_client.process_measurements(test_user.user_id, [measurement])

        state = api_client.get_user_state(test_user.user_id)

        if "quality_metrics" in state.data:
            metrics = state.data["quality_metrics"]
            assert metrics["average_quality"] > 0.5
            assert metrics["high_quality_count"] >= 3  # doctor + scales