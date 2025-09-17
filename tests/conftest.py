"""
Shared test configuration and fixtures for all tests.
Provides common test fixtures and marker definitions for the entire test suite.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    # Test speed markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "fast: marks tests as fast-running unit tests"
    )

    # Test type markers
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )

    # Priority markers
    config.addinivalue_line(
        "markers", "critical: marks tests as critical for safety"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests for basic functionality"
    )


# =============================================================================
# COMMON FIXTURES
# =============================================================================

@pytest.fixture
def numpy_random_seed():
    """Set a consistent numpy random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset after test
    np.random.seed()


@pytest.fixture
def base_timestamp():
    """Provide a consistent base timestamp for all tests."""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_weights():
    """Provide a realistic weight sequence for testing."""
    return [70.0, 70.5, 69.8, 70.2, 70.1, 69.9, 70.3]


@pytest.fixture
def sample_timestamps(base_timestamp):
    """Provide a sequence of timestamps at daily intervals."""
    return [base_timestamp + timedelta(days=i) for i in range(7)]


@pytest.fixture
def weight_config():
    """Standard configuration for weight processing."""
    return {
        'min_weight': 30.0,
        'max_weight': 400.0,
        'max_daily_change': 5.0,
        'outlier_threshold': 0.15,
        'quality_threshold': 0.6
    }


@pytest.fixture
def kalman_config():
    """Standard Kalman filter configuration."""
    return {
        'initial_variance': 1.0,
        'transition_covariance_weight': 0.1,
        'transition_covariance_trend': 0.001,
        'observation_covariance': 1.0,
        'reset_gap_days': 30
    }


@pytest.fixture
def mock_feature_manager():
    """Mock FeatureManager for testing feature flags."""
    manager = MagicMock()
    manager.is_enabled.return_value = True  # Default all features to enabled
    manager.get_config.return_value = {}
    return manager


@pytest.fixture
def sample_measurement(base_timestamp):
    """Single weight measurement for testing."""
    return {
        'weight': 70.0,
        'timestamp': base_timestamp,
        'source': 'patient-device',
        'user_id': 'test_user'
    }


@pytest.fixture
def measurement_batch(base_timestamp, sample_weights):
    """Batch of measurements for testing."""
    return [
        {
            'weight': weight,
            'timestamp': base_timestamp + timedelta(days=i),
            'source': 'patient-device',
            'user_id': 'test_user'
        }
        for i, weight in enumerate(sample_weights)
    ]


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def mock_database():
    """Provide a mock database for testing."""
    db = MagicMock()
    db.get_state.return_value = None
    db.save_state.return_value = None
    db.states = {}

    # Make get_state and save_state actually store/retrieve
    def get_state(user_id):
        return db.states.get(user_id)

    def save_state(user_id, state):
        db.states[user_id] = state

    db.get_state.side_effect = get_state
    db.save_state.side_effect = save_state

    return db


# =============================================================================
# STATE FIXTURES
# =============================================================================

@pytest.fixture
def initial_kalman_state(base_timestamp):
    """Initial Kalman filter state for testing."""
    return {
        'last_state': np.array([70.0, 0.0]),  # Weight and trend
        'last_covariance': np.array([[1.0, 0.0], [0.0, 0.001]]),
        'last_timestamp': base_timestamp,
        'kalman_params': {
            'initial_state_mean': [70.0, 0.0],
            'initial_state_covariance': [[1.0, 0.0], [0.0, 0.001]],
            'transition_covariance': [[0.1, 0.0], [0.0, 0.001]],
            'observation_covariance': [[1.0]]
        },
        'last_raw_weight': 70.0,
        'measurement_history': []
    }


@pytest.fixture
def user_state_with_history(initial_kalman_state, measurement_batch):
    """User state with measurement history."""
    state = initial_kalman_state.copy()
    state['measurement_history'] = measurement_batch
    state['buffer'] = measurement_batch[-5:]  # Last 5 measurements in buffer
    return state


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

@pytest.fixture
def weight_generator():
    """Generate realistic weight sequences."""
    def generate(start_weight=70.0, days=30, daily_variation=0.5, trend=0.0):
        """Generate weight sequence with optional trend."""
        weights = []
        current_weight = start_weight

        for day in range(days):
            # Add trend
            current_weight += trend
            # Add random variation
            variation = np.random.normal(0, daily_variation)
            weight = current_weight + variation
            weights.append(weight)

        return weights

    return generate


@pytest.fixture
def outlier_generator():
    """Generate measurements with controlled outliers."""
    def generate(base_weights, outlier_indices, outlier_magnitude=10.0):
        """Add outliers to a weight sequence."""
        weights = base_weights.copy()
        for idx in outlier_indices:
            if idx < len(weights):
                # Add or subtract outlier magnitude
                direction = 1 if np.random.random() > 0.5 else -1
                weights[idx] += direction * outlier_magnitude
        return weights

    return generate


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

@pytest.fixture
def assert_weight_in_range():
    """Helper to assert weight is in valid range."""
    def check(weight, min_weight=30.0, max_weight=400.0):
        assert min_weight <= weight <= max_weight, \
            f"Weight {weight}kg outside valid range [{min_weight}, {max_weight}]"
    return check


@pytest.fixture
def assert_quality_score():
    """Helper to assert quality score is valid."""
    def check(score):
        assert 0.0 <= score <= 1.0, \
            f"Quality score {score} outside valid range [0.0, 1.0]"
    return check