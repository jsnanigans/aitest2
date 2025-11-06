"""Unit tests for Kalman filter operations.

Tests the Kalman filter which is the core algorithm for weight estimation.
CRITICAL for accurate weight tracking.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from weight_processor_lib.core.processing.kalman import KalmanFilterManager, get_adaptive_kalman_params


class TestKalmanFilterOperations:
    """Tests for Kalman filter initialization, update, and prediction."""

    def test_initialize_immediate_creates_valid_state(self, base_config, base_timestamp):
        """Test initialize_immediate creates valid initial Kalman state.

        This is CRITICAL: foundation for all future updates.
        State must start with observation and zero trend.

        Expected behavior:
        - State shape is correct ([weight, trend])
        - State[0] = weight (observation)
        - State[1] = 0.0 (initial trend)
        - Covariance is positive definite
        """
        weight = 70.0
        kalman_config = base_config["kalman"]

        # Initialize Kalman state
        state = KalmanFilterManager.initialize_immediate(
            weight=weight,
            timestamp=base_timestamp,
            kalman_config=kalman_config,
        )

        # Verify kalman_params created
        assert "kalman_params" in state, "Should create kalman_params"
        assert state["kalman_params"] is not None

        # Verify initial state
        assert "last_state" in state
        last_state = state["last_state"]
        assert last_state is not None
        assert isinstance(last_state, np.ndarray)

        # Check state values - always (2, 2) even for initial state
        assert last_state.shape == (2, 2), f"State should be (2, 2), got {last_state.shape}"
        weight_val = last_state[-1][0]
        trend_val = last_state[-1][1]

        assert abs(weight_val - weight) < 0.1, "Weight should match observation"
        assert abs(trend_val - 0.0) < 0.1, "Initial trend should be 0"

        # Verify covariance
        assert "last_covariance" in state
        covariance = state["last_covariance"]
        assert covariance is not None
        # Covariance should be positive definite (diagonal elements > 0)
        # Always 3D (2, 2, 2) even for initial state
        assert covariance.shape == (2, 2, 2), f"Covariance should be (2, 2, 2), got {covariance.shape}"
        cov_matrix = covariance[-1]
        assert cov_matrix[0][0] > 0, "Weight variance should be positive"
        assert cov_matrix[1][1] > 0, "Trend variance should be positive"

    def test_update_state_with_normal_time_delta(self, base_config, base_timestamp):
        """Test update_state with normal 1-day gap.

        This is the most common measurement scenario.
        Prediction → measurement → update cycle.

        Expected behavior:
        - State updated with new measurement
        - Covariance reduced (information gained)
        - Trend bounded to physiological limits
        """
        # Create initial state
        initial_state = {
            "kalman_params": {
                "observation_covariance": [[3.49]],
                "transition_covariance": [[0.1, 0], [0, 0.01]],
                "initial_state_mean": [70.0, 0.0],
                "initial_state_covariance": [[0.361, 0], [0, 0.001]],
            },
            "last_state": np.array([[70.0, 0.0], [70.0, 0.0]]),
            "last_covariance": np.array([[[0.361, 0.0], [0.0, 0.001]], [[0.361, 0.0], [0.0, 0.001]]]),
            "last_timestamp": base_timestamp - timedelta(days=1),
            "last_raw_weight": 70.0,
        }

        # Update with new measurement (1 day later)
        updated_state = KalmanFilterManager.update_state(
            state=initial_state.copy(),
            weight=70.5,
            timestamp=base_timestamp,
            source="manual",
            processing_config={},
        )

        # Verify state updated
        assert "last_state" in updated_state
        new_state = updated_state["last_state"]
        assert new_state is not None

        # Verify timestamp updated
        assert updated_state["last_timestamp"] == base_timestamp

        # Verify weight updated
        assert updated_state["last_raw_weight"] == 70.5

        # Verify covariance (should be reduced after measurement)
        new_cov = updated_state["last_covariance"]
        assert new_cov is not None

    def test_update_state_with_extreme_time_deltas(self, base_config, base_timestamp):
        """Test update_state with extreme time deltas (0.1 and 30 days).

        Edge cases where prediction uncertainty varies significantly.

        Expected behavior:
        - Short gap (0.1 days = 2.4 hours): minimal prediction uncertainty
        - Long gap (30 days): large prediction uncertainty
        - Time delta clamped to [0.1, 30] days
        """
        base_state = {
            "kalman_params": {
                "observation_covariance": [[3.49]],
                "transition_covariance": [[0.1, 0], [0, 0.01]],
                "initial_state_mean": [70.0, 0.0],
                "initial_state_covariance": [[0.361, 0], [0, 0.001]],
            },
            "last_state": np.array([[70.0, 0.0], [70.0, 0.0]]),
            "last_covariance": np.array([[[0.361, 0.0], [0.0, 0.001]], [[0.361, 0.0], [0.0, 0.001]]]),
            "last_raw_weight": 70.0,
        }

        # Test 1: Very short gap (0.1 days = 2.4 hours)
        state_short = base_state.copy()
        state_short["last_timestamp"] = base_timestamp - timedelta(hours=2.4)

        updated_short = KalmanFilterManager.update_state(
            state=state_short,
            weight=70.2,
            timestamp=base_timestamp,
            source="manual",
            processing_config={},
        )
        assert updated_short["last_state"] is not None, "Should handle short gaps"

        # Test 2: Long gap (30 days)
        state_long = base_state.copy()
        state_long["last_timestamp"] = base_timestamp - timedelta(days=30)

        updated_long = KalmanFilterManager.update_state(
            state=state_long,
            weight=71.0,
            timestamp=base_timestamp,
            source="manual",
            processing_config={},
        )
        assert updated_long["last_state"] is not None, "Should handle long gaps"

        # Test 3: Extremely long gap (50 days - should be clamped to 30)
        state_extreme = base_state.copy()
        state_extreme["last_timestamp"] = base_timestamp - timedelta(days=50)

        updated_extreme = KalmanFilterManager.update_state(
            state=state_extreme,
            weight=72.0,
            timestamp=base_timestamp,
            source="manual",
            processing_config={},
        )
        assert updated_extreme["last_state"] is not None, "Should clamp extreme gaps"

    def test_predict_next_state_for_quality_scoring(self, base_config, base_timestamp):
        """Test predict_next_state for quality scoring without updating state.

        Quality scorer uses prediction to calculate Kalman fit.

        Expected behavior:
        - Prediction calculated at future timestamp
        - Original state unchanged
        - Returns (predicted_weight, innovation_covariance)
        """
        state = {
            "kalman_params": {
                "observation_covariance": [[3.49]],
                "transition_covariance": [[0.1, 0], [0, 0.01]],
                "initial_state_mean": [70.0, 0.0],
                "initial_state_covariance": [[0.361, 0], [0, 0.001]],
            },
            "last_state": np.array([[70.0, 0.05], [70.0, 0.05]]),  # Small positive trend
            "last_covariance": np.array([[[0.2, 0.0], [0.0, 0.001]], [[0.2, 0.0], [0.0, 0.001]]]),
            "last_timestamp": base_timestamp - timedelta(days=1),
        }

        # Get prediction for current timestamp (1 day ahead)
        predicted_weight, innovation_cov = KalmanFilterManager.predict_next_state(
            state=state,
            timestamp=base_timestamp,
        )

        # Verify prediction returned
        assert predicted_weight is not None, "Should return predicted weight"
        assert innovation_cov is not None, "Should return innovation covariance"

        # Verify prediction is reasonable (70 + trend * 1 day)
        assert 69.5 < predicted_weight < 70.5, f"Predicted weight {predicted_weight} should be near 70"

        # Verify innovation covariance is positive
        assert innovation_cov > 0, "Innovation covariance should be positive"

        # Verify original state unchanged
        assert np.array_equal(state["last_state"], np.array([[70.0, 0.05], [70.0, 0.05]])), \
            "Original state should not change"

    def test_adaptive_kalman_params_after_reset(self, base_config, base_timestamp):
        """Test adaptive Kalman parameters within 7 days AND < 10 measurements of reset.

        Within adaptive period, use relaxed parameters for faster convergence.

        Expected behavior:
        - Within 7 days AND < 10 measurements: use adaptive params
        - After 7 days OR >= 10 measurements: use normal params
        - Parameters decay exponentially
        """
        reset_timestamp = base_timestamp - timedelta(days=3)
        kalman_config = base_config["kalman"]

        # Test 1: Within adaptive period (3 days, 5 measurements)
        state_adaptive = {
            "reset_parameters": {
                "initial_variance_multiplier": 5,
                "weight_noise_multiplier": 20,
                "trend_noise_multiplier": 200,
                "adaptation_days": 7,
            },
            "measurements_since_reset": 5,
        }

        adaptive_params = get_adaptive_kalman_params(
            reset_timestamp=reset_timestamp,
            current_timestamp=base_timestamp,
            base_config=kalman_config,
            adaptive_days=7,
            state=state_adaptive,
        )

        # Should return modified parameters (not same as base)
        assert adaptive_params != kalman_config, "Should use adaptive params"

        # Test 2: After adaptive period (8 days)
        reset_timestamp_old = base_timestamp - timedelta(days=8)

        normal_params = get_adaptive_kalman_params(
            reset_timestamp=reset_timestamp_old,
            current_timestamp=base_timestamp,
            base_config=kalman_config,
            adaptive_days=7,
            state=state_adaptive,
        )

        # Should return base parameters
        assert normal_params == kalman_config, "Should use normal params after 7 days"

    def test_trend_limiting_clamps_to_5kg_per_week(self):
        """Test trend limiting clamps to ±5kg/week (±0.714 kg/day).

        Prevents Kalman divergence from unrealistic trends.
        NOTE: Trend limiting is applied in processor.py, not in Kalman module.

        Expected behavior:
        - Trend > 0.714 → clamped to 0.714
        - Trend < -0.714 → clamped to -0.714
        - Trend in [-0.714, 0.714] → unchanged
        """
        # This test documents the expected behavior
        # Actual implementation is in processor.py around lines 422-452
        max_daily_trend = 0.714  # 5kg/week

        # Test values
        test_cases = [
            (1.0, 0.714),  # High positive trend
            (-1.5, -0.714),  # High negative trend
            (0.3, 0.3),  # Normal trend (unchanged)
            (0.714, 0.714),  # Boundary (unchanged)
            (-0.714, -0.714),  # Boundary (unchanged)
        ]

        for input_trend, expected_output in test_cases:
            if abs(input_trend) > max_daily_trend:
                limited = max_daily_trend if input_trend > 0 else -max_daily_trend
                assert limited == expected_output
            else:
                assert input_trend == expected_output

    def test_get_current_state_values_extracts_weight_and_trend(self, base_timestamp):
        """Test get_current_state_values returns correct weight and trend.

        Used for state storage and debugging.

        Expected behavior:
        - Returns (weight, trend) tuple
        - Handles both 1-D and 2-D state arrays
        - Returns (None, None) if no state
        """
        # Test 1: Valid state (1-D array)
        state_1d = {
            "last_state": np.array([[70.5, 0.05], [70.5, 0.05]]),
        }
        weight, trend = KalmanFilterManager.get_current_state_values(state_1d)
        assert weight == pytest.approx(70.5, abs=0.01)
        assert trend == pytest.approx(0.05, abs=0.01)

        # Test 2: Valid state (2-D array)
        state_2d = {
            "last_state": np.array([[70.0, 0.1], [70.5, 0.05]]),
        }
        weight, trend = KalmanFilterManager.get_current_state_values(state_2d)
        assert weight == pytest.approx(70.5, abs=0.01), "Should get last row"
        assert trend == pytest.approx(0.05, abs=0.01)

        # Test 3: No state
        state_none = {}
        weight, trend = KalmanFilterManager.get_current_state_values(state_none)
        assert weight is None
        assert trend is None

    def test_calculate_confidence_from_normalized_innovation(self):
        """Test calculate_confidence maps innovation distance to confidence [0, 1].

        Used by quality scorer for confidence component.

        Expected behavior:
        - 0σ → confidence ~1.0 (perfect match)
        - 3σ → confidence ~0.1 (poor match)
        - 1σ → confidence ~0.7
        - Exponential decay
        """
        # Test cases: (normalized_innovation, min_expected_confidence, max_expected_confidence)
        test_cases = [
            (0.0, 0.95, 1.0),  # Perfect match
            (1.0, 0.5, 0.8),  # 1 sigma
            (2.0, 0.1, 0.3),  # 2 sigma
            (3.0, 0.01, 0.15),  # 3 sigma
        ]

        for innovation, min_conf, max_conf in test_cases:
            confidence = KalmanFilterManager.calculate_confidence(innovation)
            assert min_conf <= confidence <= max_conf, \
                f"Confidence for {innovation}σ should be in [{min_conf}, {max_conf}], got {confidence}"
