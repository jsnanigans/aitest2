"""Shared fixtures for unit tests."""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock
from typing import Dict, Any


@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Standard config for tests.

    Returns default configuration that matches production config structure.
    Can be overridden in individual tests as needed.
    """
    return {
        "kalman": {
            "observation_covariance": 3.49,
            "process_noise_weight": 0.1,
            "process_noise_trend": 0.01,
            "initial_variance_multiplier": 1.0,
        },
        "quality": {
            "acceptance_threshold": 0.46,
            "weights": {
                "kalman_fit": 0.40,
                "temporal_consistency": 0.30,
                "anomaly_detection": 0.20,
                "source_reliability": 0.05,
                "trend_alignment": 0.05,
            },
        },
        "reset": {
            "hard_reset_days": 30,
            "soft_reset_min_change_kg": 5.0,
            "soft_reset_cooldown_days": 3,
            "adaptive_period_days": 7,
            "adaptive_period_measurements": 10,
        },
        "validation": {
            "absolute_min_weight_kg": 20.0,
            "absolute_max_weight_kg": 300.0,
            "bmi_impossible_low": 10.0,
            "bmi_impossible_high": 80.0,
        },
        "replay": {
            "buffer_hours": 24,
            "max_buffer_measurements": 100,
            "buffered_replay_enabled": True,
        },
        "snapshots": {
            "enabled": True,
            "interval_hours": 24,
        },
    }


@pytest.fixture
def base_timestamp() -> datetime:
    """Standard timestamp for tests (2025-10-01 12:00:00 UTC).

    Using a consistent timestamp across tests makes it easier to reason
    about time-based logic and compare results.
    """
    return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_state_store():
    """Mock state store with common methods.

    Provides a pre-configured mock that matches the StateStore interface.
    Default behavior returns None for load_state (simulating new user).
    """
    store = Mock()
    store.load_state.return_value = None
    store.save_state.return_value = True
    store.get_snapshot.return_value = None
    store.save_state_snapshot.return_value = True
    return store


@pytest.fixture
def clean_state() -> Dict[str, Any]:
    """Fresh state for first measurement (no prior history).

    Represents state of a user who has never had a measurement processed.
    All critical fields are None or empty.
    """
    return {
        "kalman_params": None,
        "last_state": None,
        "last_covariance": None,
        "last_timestamp": None,
        "last_raw_weight": None,
        "measurements_since_reset": 0,
        "measurement_history": [],
        "last_reset_timestamp": None,
        "last_reset_type": None,
    }


@pytest.fixture
def initialized_state(base_timestamp) -> Dict[str, Any]:
    """State after first measurement processed.

    Represents a typical user who has had at least one measurement accepted.
    Kalman filter is initialized with realistic values for a 70kg person.
    """
    return {
        "kalman_params": {
            "observation_covariance": 3.49,
            "process_noise_weight": 0.1,
            "process_noise_trend": 0.01,
        },
        "last_state": np.array([[70.0], [0.0]]),  # weight=70kg, trend=0
        "last_covariance": np.array([
            [[0.361, 0.0], [0.0, 0.001]]
        ]),  # Small variance after first measurement
        "last_timestamp": base_timestamp,
        "last_raw_weight": 70.0,
        "measurements_since_reset": 1,
        "measurement_history": [
            {
                "timestamp": base_timestamp.isoformat(),
                "weight": 70.0,
                "quality_score": 0.95,
                "accepted": True,
            }
        ],
        "last_reset_timestamp": base_timestamp,
        "last_reset_type": "INITIAL",
    }


@pytest.fixture
def state_with_history(base_timestamp) -> Dict[str, Any]:
    """State with multiple measurements in history.

    Represents an established user with 5+ measurements.
    Useful for testing trend analysis and quality scoring.
    """
    from datetime import timedelta

    history = []
    for i in range(5):
        history.append({
            "timestamp": (base_timestamp - timedelta(days=5-i)).isoformat(),
            "weight": 70.0 + (i * 0.1),  # Gradually increasing weight
            "quality_score": 0.9,
            "accepted": True,
        })

    return {
        "kalman_params": {
            "observation_covariance": 3.49,
            "process_noise_weight": 0.1,
            "process_noise_trend": 0.01,
        },
        "last_state": np.array([[70.4], [0.02]]),  # Slight positive trend
        "last_covariance": np.array([
            [[0.1, 0.0], [0.0, 0.0001]]
        ]),  # Lower variance with more measurements
        "last_timestamp": base_timestamp,
        "last_raw_weight": 70.4,
        "measurements_since_reset": 5,
        "measurement_history": history,
        "last_reset_timestamp": base_timestamp - timedelta(days=5),
        "last_reset_type": "INITIAL",
    }
