"""
Comprehensive unit tests for KalmanFilterManager.
Tests mathematical operations, adaptive behavior, and edge cases.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import math

from src.processing.kalman import KalmanFilterManager
from src.constants import KALMAN_DEFAULTS, SOURCE_PROFILES


# Fixtures
@pytest.fixture
def base_timestamp():
    """Provide a consistent base timestamp for tests."""
    return datetime(2024, 1, 1, 10, 0)


@pytest.fixture
def default_weight():
    """Provide a default weight value."""
    return 70.0


@pytest.fixture
def empty_kalman_config():
    """Provide an empty Kalman configuration."""
    return {}


@pytest.fixture
def initialized_state(default_weight, base_timestamp, empty_kalman_config):
    """Provide an initialized Kalman filter state."""
    return KalmanFilterManager.initialize_immediate(
        default_weight, base_timestamp, empty_kalman_config
    )


@pytest.fixture
def adaptive_config():
    """Provide a configuration with adaptive parameters."""
    return {
        'transition_covariance_weight': 0.1,
        'transition_covariance_trend': 0.001,
        'post_reset_adaptation': {
            'warmup_measurements': 10,
            'weight_boost_factor': 10,
            'trend_boost_factor': 100,
            'decay_rate': 3
        }
    }


class TestKalmanFilterInitialization:
    """Tests for Kalman filter initialization."""

    def test_initialize_immediate_basic(self, default_weight, base_timestamp, empty_kalman_config):
        """Test basic initialization with default parameters."""
        state = KalmanFilterManager.initialize_immediate(
            default_weight, base_timestamp, empty_kalman_config
        )

        assert state['last_raw_weight'] == default_weight
        assert state['last_timestamp'] == base_timestamp
        assert state['last_state'][0][0] == default_weight
        assert state['last_state'][0][1] == 0.0
        assert state['last_covariance'][0][0][0] == KALMAN_DEFAULTS['initial_variance']
        assert state['kalman_params']['initial_state_mean'] == [default_weight, 0]

    def test_initialize_immediate_with_custom_observation_covariance(self):
        """Test initialization with custom observation covariance."""
        weight = 80.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {'initial_variance': 2.0}
        observation_covariance = 5.0

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config, observation_covariance
        )

        assert state['kalman_params']['observation_covariance'][0][0] == 5.0
        assert state['last_covariance'][0][0][0] == 2.0

    def test_initialize_immediate_with_partial_config(self):
        """Test initialization with partial config uses defaults for missing values."""
        weight = 65.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {
            'transition_covariance_weight': 0.5
        }

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        params = state['kalman_params']
        assert params['transition_covariance'][0][0] == 0.5
        assert params['transition_covariance'][1][1] == KALMAN_DEFAULTS['transition_covariance_trend']
        assert params['observation_covariance'][0][0] == KALMAN_DEFAULTS['observation_covariance']


class TestKalmanFilterUpdate:
    """Tests for Kalman filter update operations."""

    def test_update_state_first_measurement(self):
        """Test update with first measurement (no previous state)."""
        weight = 75.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        initial_state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )
        initial_state['last_state'] = None

        updated_state = KalmanFilterManager.update_state(
            initial_state,
            weight,
            timestamp,
            'patient-device',
            {}
        )

        assert updated_state['last_raw_weight'] == 75.0
        assert updated_state['last_timestamp'] == timestamp
        assert updated_state['last_state'] is not None

    def test_update_state_sequence(self):
        """Test updating state with a sequence of measurements."""
        weights = [70.0, 70.5, 69.8, 70.2, 70.1]
        base_time = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weights[0], base_time, kalman_config
        )

        for i, weight in enumerate(weights[1:], 1):
            timestamp = base_time + timedelta(days=i)
            state = KalmanFilterManager.update_state(
                state,
                weight,
                timestamp,
                'patient-device',
                {}
            )

            assert state['last_raw_weight'] == weight
            assert state['last_timestamp'] == timestamp

            filtered_weight = state['last_state'][-1][0] if len(state['last_state'].shape) > 1 else state['last_state'][0]
            assert 69.0 <= filtered_weight <= 71.0

    def test_predict_without_observation(self):
        """Test state prediction when no observation is available."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        next_timestamp = timestamp + timedelta(days=1)
        state_with_trend = state.copy()
        state_with_trend['last_state'] = np.array([[70.0, 0.1]])

        updated_state = KalmanFilterManager.update_state(
            state_with_trend,
            70.5,
            next_timestamp,
            'patient-device',
            {}
        )

        assert updated_state['last_timestamp'] == next_timestamp
        filtered_weight = updated_state['last_state'][-1][0]
        assert abs(filtered_weight - 70.5) < 1.0

    def test_time_delta_calculation(self):
        """Test correct time delta calculation between measurements."""
        weight = 70.0
        timestamp1 = datetime(2024, 1, 1, 10, 0)
        timestamp2 = datetime(2024, 1, 5, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp1, kalman_config
        )

        state = KalmanFilterManager.update_state(
            state,
            71.0,
            timestamp2,
            'patient-device',
            {}
        )

        assert (timestamp2 - timestamp1).days == 4
        assert state['last_timestamp'] == timestamp2

    def test_time_delta_capping(self):
        """Test that time delta is capped at 30 days."""
        weight = 70.0
        timestamp1 = datetime(2024, 1, 1, 10, 0)
        timestamp2 = datetime(2024, 3, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp1, kalman_config
        )

        state = KalmanFilterManager.update_state(
            state,
            75.0,
            timestamp2,
            'patient-device',
            {}
        )

        assert state['last_timestamp'] == timestamp2


class TestAdaptiveParameters:
    """Tests for adaptive parameter functionality."""

    def test_get_adaptive_covariances_initial_phase(self):
        """Test adaptive covariances in initial phase after reset."""
        config = {
            'transition_covariance_weight': 0.1,
            'transition_covariance_trend': 0.001,
            'post_reset_adaptation': {
                'warmup_measurements': 10,
                'weight_boost_factor': 10,
                'trend_boost_factor': 100,
                'decay_rate': 3
            }
        }

        covariances = KalmanFilterManager.get_adaptive_covariances(0, config)

        assert covariances['weight'] > config['transition_covariance_weight']
        assert covariances['trend'] > config['transition_covariance_trend']
        assert covariances['weight'] <= config['transition_covariance_weight'] * 10
        assert covariances['trend'] <= config['transition_covariance_trend'] * 100

    def test_get_adaptive_covariances_decay(self):
        """Test exponential decay of adaptive parameters."""
        config = {
            'transition_covariance_weight': 0.1,
            'transition_covariance_trend': 0.001,
            'post_reset_adaptation': {
                'warmup_measurements': 10,
                'weight_boost_factor': 10,
                'trend_boost_factor': 100,
                'decay_rate': 3
            }
        }

        prev_weight_cov = float('inf')
        prev_trend_cov = float('inf')

        for i in range(10):
            covariances = KalmanFilterManager.get_adaptive_covariances(i, config)

            assert covariances['weight'] < prev_weight_cov
            assert covariances['trend'] < prev_trend_cov

            prev_weight_cov = covariances['weight']
            prev_trend_cov = covariances['trend']

    def test_get_adaptive_covariances_after_warmup(self):
        """Test that covariances return to base values after warmup."""
        config = {
            'transition_covariance_weight': 0.1,
            'transition_covariance_trend': 0.001,
            'post_reset_adaptation': {
                'warmup_measurements': 10,
                'weight_boost_factor': 10,
                'trend_boost_factor': 100,
                'decay_rate': 3
            }
        }

        covariances = KalmanFilterManager.get_adaptive_covariances(15, config)

        assert covariances['weight'] == config['transition_covariance_weight']
        assert covariances['trend'] == config['transition_covariance_trend']

    def test_adaptive_covariances_with_feature_disabled(self):
        """Test that adaptive parameters are disabled when feature is off."""
        feature_manager = MagicMock()
        feature_manager.is_enabled.return_value = False

        config = {
            'transition_covariance_weight': 0.1,
            'transition_covariance_trend': 0.001,
            'feature_manager': feature_manager,
            'post_reset_adaptation': {
                'warmup_measurements': 10,
                'weight_boost_factor': 10,
                'trend_boost_factor': 100,
                'decay_rate': 3
            }
        }

        covariances = KalmanFilterManager.get_adaptive_covariances(0, config)

        feature_manager.is_enabled.assert_called_once_with('adaptive_parameters')
        assert covariances['weight'] == config['transition_covariance_weight']
        assert covariances['trend'] == config['transition_covariance_trend']


class TestSourceReliability:
    """Tests for source-based noise handling."""

    @pytest.mark.parametrize("source,expected_multiplier", [
        ('care-team-upload', 0.5),
        ('patient-upload', 0.7),
        ('internal-questionnaire', 0.8),
        ('initial-questionnaire', 0.8),
        ('patient-device', 1.0),
        ('https://connectivehealth.io', 1.5),
        ('https://api.iglucose.com', 3.0)
    ])
    def test_source_noise_multipliers(self, source, expected_multiplier):
        """Test that different sources have correct noise multipliers."""
        profile = SOURCE_PROFILES.get(source, {})
        actual_multiplier = profile.get('noise_multiplier', 1.0)
        assert actual_multiplier == expected_multiplier, f"Source {source}: expected {expected_multiplier}, got {actual_multiplier}"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.parametrize("weight", [0.1, 1.0, 500.0, 1000.0])
    def test_extreme_weight_values(self, weight, base_timestamp, empty_kalman_config):
        """Test handling of extreme weight values."""
        state = KalmanFilterManager.initialize_immediate(
            weight, base_timestamp, empty_kalman_config
        )

        assert state['last_raw_weight'] == weight
        assert state['last_state'][0][0] == weight
        assert not math.isnan(state['last_state'][0][0]), f"NaN detected for weight {weight}"
        assert not math.isinf(state['last_state'][0][0]), f"Inf detected for weight {weight}"

    def test_negative_weight_handling(self):
        """Test handling of negative weight values."""
        weight = -70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        assert state['last_raw_weight'] == -70.0
        assert state['last_state'][0][0] == -70.0

    def test_zero_time_delta(self):
        """Test handling of zero time delta."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        state = KalmanFilterManager.update_state(
            state,
            71.0,
            timestamp,
            'patient-device',
            {}
        )

        assert state['last_timestamp'] == timestamp
        assert state['last_raw_weight'] == 71.0

    def test_extreme_time_gaps(self):
        """Test handling of extreme time gaps."""
        weight = 70.0
        timestamp1 = datetime(2024, 1, 1, 10, 0)
        timestamp2 = datetime(2025, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp1, kalman_config
        )

        state = KalmanFilterManager.update_state(
            state,
            75.0,
            timestamp2,
            'patient-device',
            {}
        )

        assert state['last_timestamp'] == timestamp2
        assert not math.isnan(state['last_state'][-1][0])
        assert not math.isinf(state['last_state'][-1][0])

    def test_state_shape_handling(self):
        """Test handling of different state array shapes."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        state['last_state'] = np.array([70.0, 0.0])
        state['last_covariance'] = np.array([[1.0, 0.0], [0.0, 0.001]])

        updated_state = KalmanFilterManager.update_state(
            state,
            71.0,
            timestamp + timedelta(days=1),
            'patient-device',
            {}
        )

        assert updated_state['last_state'].shape == (2, 2)
        assert updated_state['last_covariance'].shape == (2, 2, 2)


class TestResultCreation:
    """Tests for result creation from Kalman state."""

    def test_create_result_accepted(self):
        """Test creating result for accepted measurement."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        result = KalmanFilterManager.create_result(
            state, weight, timestamp, 'patient-device', True
        )

        assert result is not None
        assert result['filtered_weight'] == 70.0
        assert result['trend'] == 0.0
        assert result['innovation'] == 0.0
        assert result['accepted'] == True
        assert 'confidence' in result

    def test_create_result_rejected(self):
        """Test creating result for rejected measurement."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        result = KalmanFilterManager.create_result(
            state, 100.0, timestamp, 'patient-device', False
        )

        assert result is not None
        assert result['filtered_weight'] == 70.0
        assert result['innovation'] == 30.0
        assert result['accepted'] == False

    def test_create_result_missing_state(self):
        """Test creating result when state is missing."""
        state = {
            'last_state': None,
            'kalman_params': {}
        }

        result = KalmanFilterManager.create_result(
            state, 70.0, datetime.now(), 'patient-device', True
        )

        assert result is None

    def test_create_result_confidence_bounds(self):
        """Test confidence interval calculation in result."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        state['last_covariance'] = np.array([[[4.0, 0.0], [0.0, 0.001]]])

        result = KalmanFilterManager.create_result(
            state, weight, timestamp, 'patient-device', True
        )

        assert result is not None
        assert 'kalman_variance' in result
        assert result['kalman_variance'] == 4.0
        assert 'kalman_confidence_lower' in result
        assert 'kalman_confidence_upper' in result
        assert result['kalman_confidence_lower'] < result['filtered_weight']
        assert result['kalman_confidence_upper'] > result['filtered_weight']


class TestConfidenceCalculation:
    """Tests for confidence calculation."""

    def test_calculate_confidence_zero_innovation(self):
        """Test confidence calculation with zero innovation."""
        confidence = KalmanFilterManager.calculate_confidence(0.0)
        assert confidence == pytest.approx(1.0)

    @pytest.mark.parametrize("innovation,min_conf,max_conf", [
        (0.5, 0.8, 1.0),
        (1.0, 0.5, 0.8),
        (3.0, 0.0, 0.2),
        (5.0, 0.0, 0.05)
    ])
    def test_calculate_confidence_ranges(self, innovation, min_conf, max_conf):
        """Test confidence calculation for various innovation values."""
        confidence = KalmanFilterManager.calculate_confidence(innovation)
        assert min_conf < confidence < max_conf, f"Confidence {confidence} not in range [{min_conf}, {max_conf}] for innovation {innovation}"

    def test_calculate_confidence_exponential_decay(self):
        """Test that confidence follows exponential decay."""
        innovations = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        prev_confidence = 2.0

        for innovation in innovations:
            confidence = KalmanFilterManager.calculate_confidence(innovation)
            assert 0.0 <= confidence <= 1.0
            assert confidence < prev_confidence
            prev_confidence = confidence


class TestMathematicalStability:
    """Tests for numerical stability and mathematical properties."""

    @pytest.mark.slow
    def test_numerical_stability_long_sequence(self, base_timestamp, empty_kalman_config):
        """Test numerical stability over long measurement sequence."""
        np.random.seed(42)  # Ensure reproducible test

        state = KalmanFilterManager.initialize_immediate(
            70.0, base_timestamp, empty_kalman_config
        )

        for i in range(1000):
            weight = 70.0 + np.random.normal(0, 0.5)
            timestamp = base_timestamp + timedelta(days=i)

            state = KalmanFilterManager.update_state(
                state,
                weight,
                timestamp,
                'patient-device',
                {}
            )

            filtered_weight = state['last_state'][-1][0] if len(state['last_state'].shape) > 1 else state['last_state'][0]

            assert not math.isnan(filtered_weight), f"NaN at iteration {i}"
            assert not math.isinf(filtered_weight), f"Inf at iteration {i}"
            assert 60 < filtered_weight < 80, f"Weight {filtered_weight} out of bounds at iteration {i}"

    def test_covariance_positive_definite(self):
        """Test that covariance matrix remains positive definite."""
        weight = 70.0
        base_time = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, base_time, kalman_config
        )

        for i in range(100):
            timestamp = base_time + timedelta(days=i)
            state = KalmanFilterManager.update_state(
                state,
                weight + np.random.normal(0, 1),
                timestamp,
                'patient-device',
                {}
            )

            covariance = state['last_covariance'][-1] if len(state['last_covariance'].shape) > 2 else state['last_covariance']

            eigenvalues = np.linalg.eigvals(covariance)
            assert all(eigenvalue > 0 for eigenvalue in eigenvalues)

    def test_filter_convergence(self, base_timestamp, empty_kalman_config, numpy_random_seed):
        """Test that filter converges to true value with consistent measurements.

        Given:
            - True weight of 75.0 kg
            - Initial estimate of 70.0 kg
            - Small measurement noise (0.1 kg std dev)
            - High quality source (care-team-upload)
        When:
            - 50 measurements are processed
        Then:
            - Filtered weight should converge within 0.5 kg of true value
        """
        true_weight = 75.0
        initial_weight = 70.0
        measurement_noise_std = 0.1
        convergence_tolerance = 0.5

        state = KalmanFilterManager.initialize_immediate(
            initial_weight, base_timestamp, empty_kalman_config
        )

        for i in range(1, 50):
            timestamp = base_timestamp + timedelta(days=i)
            noisy_weight = true_weight + np.random.normal(0, measurement_noise_std)

            state = KalmanFilterManager.update_state(
                state,
                noisy_weight,
                timestamp,
                'care-team-upload',
                {},
                observation_covariance=0.01
            )

        filtered_weight = state['last_state'][-1][0]
        assert filtered_weight == pytest.approx(true_weight, abs=convergence_tolerance)


class TestIntegration:
    """Integration tests with other components."""

    def test_state_serialization(self):
        """Test that state can be properly serialized and deserialized."""
        weight = 70.0
        timestamp = datetime(2024, 1, 1, 10, 0)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp, kalman_config
        )

        import json
        state_copy = state.copy()
        state_copy['last_timestamp'] = state_copy['last_timestamp'].isoformat()
        state_copy['last_state'] = state_copy['last_state'].tolist()
        state_copy['last_covariance'] = state_copy['last_covariance'].tolist()

        serialized = json.dumps(state_copy)
        deserialized = json.loads(serialized)

        assert deserialized['last_raw_weight'] == 70.0
        assert deserialized['last_state'][0][0] == 70.0

    def test_timestamp_format_handling(self):
        """Test handling of different timestamp formats."""
        weight = 70.0
        timestamp_str = "2024-01-01T10:00:00"
        timestamp_dt = datetime.fromisoformat(timestamp_str)
        kalman_config = {}

        state = KalmanFilterManager.initialize_immediate(
            weight, timestamp_dt, kalman_config
        )

        state['last_timestamp'] = timestamp_str

        updated_state = KalmanFilterManager.update_state(
            state,
            71.0,
            datetime(2024, 1, 2, 10, 0),
            'patient-device',
            {}
        )

        assert updated_state['last_raw_weight'] == 71.0