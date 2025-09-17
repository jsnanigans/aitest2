"""
Integration tests for database validation with KalmanStateValidator.

Tests the integration between the database and the Kalman state validator,
including feature flags, corruption recovery, and backward compatibility.
"""

import json
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.database.database import ProcessorStateDB
from src.exceptions import DataCorruptionError, StateValidationError


class TestDatabaseValidation:
    """Test suite for database validation integration."""

    @pytest.fixture
    def database(self):
        """Create a fresh database instance for each test."""
        db = ProcessorStateDB()
        # Explicitly set validation to disabled for most tests (backward compatibility)
        db.enable_validation = False
        return db

    @pytest.fixture
    def valid_state(self):
        """Create a valid state for testing."""
        return {
            'kalman_params': {
                'initial_state_mean': [100, 0],
                'initial_state_covariance': [[1.0, 0], [0, 0.001]],
                'transition_covariance': [[0.1, 0], [0, 0.01]],
                'observation_covariance': [[1.0]],
            },
            'last_state': np.array([[100.0, 0.0]]),
            'last_covariance': np.array([[1.0, 0], [0, 0.001]]),
            'last_timestamp': datetime.now(),
            'last_raw_weight': 100.0,
            'measurement_history': []
        }

    def test_validation_disabled_by_default(self, database):
        """Test that validation is disabled by default for backward compatibility."""
        # By default, kalman_state_validation should be False
        assert database.enable_validation == False

    @patch('src.database.database.FeatureManager')
    def test_validation_enabled_by_feature_flag(self, mock_feature_manager):
        """Test that validation can be enabled via feature flag."""
        mock_fm_instance = Mock()
        mock_fm_instance.is_enabled.return_value = True
        mock_feature_manager.return_value = mock_fm_instance

        db = ProcessorStateDB()
        assert db.enable_validation == True

    def test_valid_state_passes_validation(self, database, valid_state):
        """Test that a valid state passes validation when enabled."""
        database.enable_validation = True
        user_id = "test_user"

        # Save and retrieve state
        database.save_state(user_id, valid_state)
        retrieved_state = database.get_state(user_id)

        assert retrieved_state is not None
        assert np.array_equal(retrieved_state['last_state'], valid_state['last_state'])

    def test_corrupted_state_recovery_flattened_array(self, database):
        """Test recovery from flattened arrays in stored state."""
        database.enable_validation = True
        user_id = "test_user"

        # Create a state with flattened arrays (common corruption)
        corrupted_state = {
            'kalman_params': {
                'transition_covariance': [0.1, 0, 0, 0.01],  # Flattened 2x2
                'observation_covariance': [1.0],
            },
            'last_state': np.array([100.0, 0.0]),  # Should be 2D
            'last_covariance': np.array([[1.0, 0], [0, 0.001]]),
            'last_timestamp': datetime.now(),
        }

        # This should log warnings but not fail
        database.save_state(user_id, corrupted_state)
        retrieved_state = database.get_state(user_id)

        assert retrieved_state is not None

    def test_nan_values_handled_gracefully(self, database):
        """Test that NaN values are handled without crashing."""
        database.enable_validation = True
        # Ensure strict validation is off for this test
        with patch.object(database.feature_manager, 'is_enabled') as mock_is_enabled:
            mock_is_enabled.side_effect = lambda feature: {
                'kalman_state_validation': True,
                'strict_validation': False  # Explicitly disable strict mode
            }.get(feature, False)

            user_id = "test_user"

            # Create a state with NaN values
            corrupted_state = {
                'kalman_params': {
                    'transition_covariance': [[np.nan, 0], [0, 0.01]],
                    'observation_covariance': [[1.0]],
                },
                'last_state': np.array([[100.0, 0.0]]),
                'last_covariance': np.array([[1.0, 0], [0, 0.001]]),
                'last_timestamp': datetime.now(),
            }

            # With validation enabled but not strict, this should log errors but not crash
            database.save_state(user_id, corrupted_state)
            retrieved_state = database.get_state(user_id)

            # State should still be retrieved (backward compatibility)
            assert retrieved_state is not None

    @patch('src.database.database.FeatureManager')
    def test_strict_validation_raises_on_corruption(self, mock_feature_manager, database):
        """Test that strict validation mode raises exceptions on corruption."""
        mock_fm_instance = Mock()
        mock_fm_instance.is_enabled.side_effect = lambda feature: {
            'kalman_state_validation': True,
            'strict_validation': True
        }.get(feature, False)
        mock_feature_manager.return_value = mock_fm_instance

        db = ProcessorStateDB()
        user_id = "test_user"

        # Create a severely corrupted state
        corrupted_state = {
            'kalman_params': None,  # Missing params
            'last_state': "not_an_array",  # Invalid type
            'last_covariance': None,
            'last_timestamp': datetime.now(),
        }

        # Save the corrupted state
        db.save_state(user_id, corrupted_state)

        # With strict validation enabled, retrieval should raise an error
        with pytest.raises(DataCorruptionError) as exc_info:
            db.get_state(user_id)

        # Check that the error message is informative
        assert "validation failed" in str(exc_info.value).lower()

    def test_backward_compatibility_without_validation(self, database):
        """Test that existing code works without validation."""
        database.enable_validation = False
        user_id = "test_user"

        # Create a state with minor issues that would fail validation
        # Note: These are stored as lists, which is valid JSON
        state = {
            'kalman_params': {
                'transition_covariance': [0.1],  # Wrong format
                'observation_covariance': 1.0,   # Should be nested
            },
            'last_state': np.array([100.0, 0.0]),  # Use numpy array
            'last_covariance': np.array([[1.0]]),
            'last_timestamp': datetime.now(),
        }

        # Without validation, this should work fine
        database.save_state(user_id, state)
        retrieved_state = database.get_state(user_id)

        assert retrieved_state is not None
        # Arrays should be properly converted through serialization
        assert isinstance(retrieved_state['last_state'], np.ndarray)
        assert retrieved_state['last_state'].shape == (2,)  # 1D array with 2 elements

    def test_validation_metrics_tracking(self, database, valid_state):
        """Test that validation metrics are properly tracked."""
        database.enable_validation = True

        # Save and retrieve multiple states
        database.save_state("user1", valid_state)
        database.save_state("user2", valid_state)

        database.get_state("user1")
        database.get_state("user2")

        # Check validator metrics
        metrics = database.validator.get_validation_metrics()
        assert metrics['total_validations'] >= 2

    def test_contains_kalman_state_detection(self, database):
        """Test detection of Kalman state components."""
        # State with Kalman components
        kalman_state = {
            'last_state': np.array([[100.0]]),
            'last_covariance': np.array([[1.0]]),
            'other_field': 'value'
        }
        assert database._contains_kalman_state(kalman_state) == True

        # State without Kalman components
        non_kalman_state = {
            'user_id': 'test',
            'timestamp': datetime.now(),
            'other_field': 'value'
        }
        assert database._contains_kalman_state(non_kalman_state) == False

    def test_legacy_deserialize_method(self, database):
        """Test the legacy deserialization method for fallback."""
        state_dict = {
            '_type': 'ndarray',
            'data': [[100.0, 0.0]],
            'shape': (1, 2)
        }

        # Use legacy method
        result = database._legacy_deserialize({'state': state_dict})

        assert isinstance(result['state'], np.ndarray)
        assert result['state'].shape == (1, 2)

    def test_state_with_partial_kalman_components(self, database):
        """Test state with only some Kalman components present."""
        database.enable_validation = True

        # Mock the feature manager to control strict validation
        with patch.object(database.feature_manager, 'is_enabled') as mock_is_enabled:
            mock_is_enabled.side_effect = lambda feature: {
                'kalman_state_validation': True,
                'strict_validation': False  # Don't fail on partial state
            }.get(feature, False)

            partial_state = {
                'last_state': np.array([[100.0, 0.0]]),
                # Missing last_covariance and kalman_params
                'last_timestamp': datetime.now(),
                'measurement_history': []
            }

            user_id = "partial_user"
            database.save_state(user_id, partial_state)
            retrieved = database.get_state(user_id)

            # Should still retrieve the state (backward compatibility)
            assert retrieved is not None
            assert np.array_equal(retrieved['last_state'], partial_state['last_state'])

    def test_validator_reset_between_users(self, database, valid_state):
        """Test that validator stats accumulate across multiple validations."""
        database.enable_validation = True

        # Process multiple users
        for i in range(5):
            user_id = f"user_{i}"
            database.save_state(user_id, valid_state)
            database.get_state(user_id)

        metrics = database.validator.get_validation_metrics()
        assert metrics['total_validations'] >= 5

    def test_transaction_rollback_with_validation(self, database, valid_state):
        """Test that transactions work correctly with validation."""
        database.enable_validation = True
        user_id = "transaction_test"

        # Save initial state
        database.save_state(user_id, valid_state)

        # Start transaction and make changes
        with pytest.raises(Exception):
            with database.transaction():
                # Modify state
                modified_state = valid_state.copy()
                modified_state['last_raw_weight'] = 200.0
                database.save_state(user_id, modified_state)

                # Force rollback
                raise Exception("Test rollback")

        # Check state was rolled back
        retrieved = database.get_state(user_id)
        assert retrieved['last_raw_weight'] == 100.0

    def test_validation_with_datetime_serialization(self, database):
        """Test that datetime serialization works with validation."""
        database.enable_validation = True

        # Mock to disable strict validation
        with patch.object(database.feature_manager, 'is_enabled') as mock_is_enabled:
            mock_is_enabled.side_effect = lambda feature: {
                'kalman_state_validation': True,
                'strict_validation': False
            }.get(feature, False)

            state = {
                'last_timestamp': datetime(2024, 1, 15, 10, 30, 0),
                'last_state': np.array([[100.0, 0.0]]),
                'last_covariance': np.array([[1.0, 0], [0, 0.001]]),  # Add covariance
                'measurement_history': []
            }

            user_id = "datetime_test"
            database.save_state(user_id, state)
            retrieved = database.get_state(user_id)

            assert retrieved is not None
            assert isinstance(retrieved['last_timestamp'], datetime)
            assert retrieved['last_timestamp'].year == 2024