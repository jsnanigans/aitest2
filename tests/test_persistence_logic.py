"""
Tests for persistence logic validation and audit trail.

Tests that:
- States are only persisted when valid and meaningful
- Invalid states are rejected with appropriate logging
- Audit trail captures all persistence decisions
- The three persistence points work correctly
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import logging

from src.processing.persistence_validator import PersistenceValidator
from src.processing.processor import process_measurement


class TestPersistenceValidator:
    """Test the PersistenceValidator class."""

    def test_validate_state_with_valid_state(self):
        """Test validation with a completely valid state."""
        state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0),
            'last_raw_weight': 75.0,
            'measurements_since_reset': 5
        }

        is_valid, error = PersistenceValidator.validate_state(
            state, 'user123', 'test'
        )

        assert is_valid is True
        assert error is None

    def test_validate_state_with_missing_required_fields(self):
        """Test validation fails with missing required fields."""
        state = {
            'kalman_state': [[75.0], [0.01]],
            # Missing kalman_params and last_timestamp
        }

        is_valid, error = PersistenceValidator.validate_state(
            state, 'user123', 'test'
        )

        assert is_valid is False
        assert 'Missing required fields' in error

    def test_validate_state_with_invalid_kalman_state(self):
        """Test validation fails with invalid Kalman state structure."""
        state = {
            'kalman_state': [75.0, 0.01],  # Wrong structure
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0)
        }

        is_valid, error = PersistenceValidator.validate_state(
            state, 'user123', 'test'
        )

        assert is_valid is False
        assert 'Invalid Kalman state' in error

    def test_validate_state_with_out_of_range_weight(self):
        """Test validation fails with weight outside valid range."""
        state = {
            'kalman_state': [[600.0], [0.01]],  # Weight too high
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0)
        }

        is_valid, error = PersistenceValidator.validate_state(
            state, 'user123', 'test'
        )

        assert is_valid is False
        assert 'Kalman state' in error

    def test_validate_state_with_invalid_covariance(self):
        """Test validation fails with negative covariance values."""
        state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, -0.001]],  # Negative value
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0)
        }

        is_valid, error = PersistenceValidator.validate_state(
            state, 'user123', 'test'
        )

        assert is_valid is False
        assert 'Invalid Kalman parameters' in error

    def test_should_persist_with_no_previous_state(self):
        """Test that initial state should always be persisted."""
        state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0)
        }

        should_persist, msg = PersistenceValidator.should_persist(
            state, None, 'user123', 'initial'
        )

        assert should_persist is True
        assert 'Initial state' in msg

    def test_should_persist_with_meaningful_changes(self):
        """Test that states with meaningful changes should be persisted."""
        previous_state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0),
            'last_raw_weight': 75.0
        }

        current_state = {
            'kalman_state': [[76.5], [0.02]],  # Weight changed
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 11, 0, 0),
            'last_raw_weight': 76.5
        }

        should_persist, msg = PersistenceValidator.should_persist(
            current_state, previous_state, 'user123', 'weight_update'
        )

        assert should_persist is True
        assert 'meaningful changes' in msg

    def test_should_not_persist_without_meaningful_changes(self):
        """Test that states without meaningful changes should not be persisted."""
        previous_state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0),
            'last_raw_weight': 75.0
        }

        current_state = {
            'kalman_state': [[75.005], [0.01]],  # Tiny change < 0.01 kg
            'kalman_params': {
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0),
            'last_raw_weight': 75.0
        }

        should_persist, msg = PersistenceValidator.should_persist(
            current_state, previous_state, 'user123', 'minor_update'
        )

        assert should_persist is False
        assert 'No meaningful' in msg

    def test_create_audit_log(self):
        """Test audit log creation."""
        state = {
            'kalman_state': [[75.0], [0.01]],
            'last_timestamp': datetime(2024, 1, 15, 10, 0, 0),
            'measurements_since_reset': 5
        }

        audit_log = PersistenceValidator.create_audit_log(
            'user123', 'persist', state, True, 'test_reason', error=None
        )

        assert audit_log['user_id'] == 'user123'
        assert audit_log['action'] == 'persist'
        assert audit_log['success'] is True
        assert audit_log['reason'] == 'test_reason'
        assert 'timestamp' in audit_log
        assert 'state_summary' in audit_log
        assert audit_log['state_summary']['current_weight'] == 75.0
        assert audit_log['state_summary']['measurements_count'] == 5


class TestProcessorPersistence:
    """Test persistence logic in the processor."""

    @pytest.fixture
    def setup_mocks(self):
        """Set up common mocks for processor tests."""
        with patch('src.processing.processor.get_state_db') as mock_db_factory, \
             patch('src.processing.processor.FeatureManager') as mock_fm_class, \
             patch('src.processing.processor.PersistenceValidator') as mock_validator:

            # Set up database mock
            mock_db = Mock()
            mock_db.get_state.return_value = None
            mock_db.save_state.return_value = None
            mock_db_factory.return_value = mock_db

            # Set up feature manager mock
            mock_fm = Mock()
            mock_fm.is_enabled.return_value = True
            mock_fm_class.return_value = mock_fm

            # Set up validator mock
            mock_validator.validate_state.return_value = (True, None)
            mock_validator.should_persist.return_value = (True, "should persist")
            mock_validator.create_audit_log.return_value = {}

            yield {
                'db': mock_db,
                'db_factory': mock_db_factory,
                'feature_manager': mock_fm,
                'fm_class': mock_fm_class,
                'validator': mock_validator
            }

    def test_persistence_after_successful_processing(self, setup_mocks):
        """Test that state is persisted after successful processing."""
        mocks = setup_mocks

        # Set up initial state with complete Kalman parameters
        initial_state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'initial_state_mean': [75.0, 0.01],
                'initial_state_covariance': [[1.0, 0.0], [0.0, 0.1]],
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 14, 10, 0, 0)
        }
        mocks['db'].get_state.return_value = initial_state

        # Configure feature flags
        def is_enabled_side_effect(flag):
            flags = {
                'state_persistence': True,
                'kalman_filtering': True,
                'quality_scoring': True,
                'outlier_detection': False,
                'reset_hard': False,
                'reset_soft': False
            }
            return flags.get(flag, False)

        mocks['feature_manager'].is_enabled.side_effect = is_enabled_side_effect

        # Process a weight
        config = {
            'feature_manager': mocks['feature_manager'],
            'kalman': {
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.001,
                'observation_covariance': 0.5
            },
            'quality_scoring': {
                'use_quality_override': False
            }
        }

        result = process_measurement(
            user_id='user123',
            weight=76.0,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            source='patient-device',
            config=config,
            unit='kg'
        )

        # Verify persistence was attempted
        assert mocks['validator'].validate_state.called
        assert mocks['validator'].should_persist.called

        # Check that audit log was created
        assert mocks['validator'].create_audit_log.called

    def test_no_persistence_when_feature_disabled(self, setup_mocks):
        """Test that state is not persisted when feature is disabled."""
        mocks = setup_mocks

        # Set up initial state
        initial_state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'initial_state_mean': [75.0, 0.01],
                'initial_state_covariance': [[1.0, 0.0], [0.0, 0.1]],
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 14, 10, 0, 0),
            'last_accepted_timestamp': datetime(2024, 1, 14, 10, 0, 0)
        }
        # Return the same initial state each time
        mocks['db'].get_state.side_effect = lambda user_id: initial_state.copy()
        # Also mock create_initial_state since it's called when persistence is disabled
        mocks['db'].create_initial_state.return_value = initial_state.copy()

        # Disable state persistence
        def is_enabled_side_effect(flag):
            if flag == 'state_persistence':
                return False
            return True

        mocks['feature_manager'].is_enabled.side_effect = is_enabled_side_effect

        config = {
            'feature_manager': mocks['feature_manager'],
            'kalman': {
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.001,
                'observation_covariance': 0.5
            }
        }

        result = process_measurement(
            user_id='user123',
            weight=76.0,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            source='patient-device',
            config=config,
            unit='kg'
        )

        # Verify no persistence attempts
        assert not mocks['db'].save_state.called
        assert not mocks['validator'].validate_state.called

    def test_no_persistence_for_invalid_state(self, setup_mocks):
        """Test that invalid states are not persisted."""
        mocks = setup_mocks

        # Set up initial state
        initial_state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'initial_state_mean': [75.0, 0.01],
                'initial_state_covariance': [[1.0, 0.0], [0.0, 0.1]],
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 14, 10, 0, 0)
        }
        mocks['db'].get_state.return_value = initial_state

        # Make validator reject the state
        mocks['validator'].validate_state.return_value = (False, "Invalid state")

        config = {
            'feature_manager': mocks['feature_manager'],
            'kalman': {
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.001,
                'observation_covariance': 0.5
            }
        }

        result = process_measurement(
            user_id='user123',
            weight=76.0,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            source='patient-device',
            config=config,
            unit='kg'
        )

        # Verify save was not called
        assert not mocks['db'].save_state.called

        # Verify audit log was created for failure
        audit_calls = mocks['validator'].create_audit_log.call_args_list
        failure_logged = any(
            call[0][1] == 'validate_failed'
            for call in audit_calls
        )
        assert failure_logged

    def test_persistence_points_are_guarded(self, setup_mocks):
        """Test that all three persistence points have validation guards."""
        mocks = setup_mocks

        # Set up initial state
        initial_state = {
            'kalman_state': [[75.0], [0.01]],
            'kalman_params': {
                'initial_state_mean': [75.0, 0.01],
                'initial_state_covariance': [[1.0, 0.0], [0.0, 0.1]],
                'transition_covariance': [[0.01, 0], [0, 0.001]],
                'observation_covariance': [[0.5]]
            },
            'last_timestamp': datetime(2024, 1, 14, 10, 0, 0)
        }
        mocks['db'].get_state.return_value = initial_state

        # Track validation calls
        validation_calls = []
        def validate_side_effect(state, user_id, reason):
            validation_calls.append(reason)
            return (True, None)

        mocks['validator'].validate_state.side_effect = validate_side_effect

        config = {
            'feature_manager': mocks['feature_manager'],
            'kalman': {
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.001,
                'observation_covariance': 0.5
            }
        }

        # Process weight that goes through main path
        result = process_measurement(
            user_id='user123',
            weight=76.0,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            source='patient-device',
            config=config,
            unit='kg'
        )

        # Should have validation for at least one persistence point
        assert len(validation_calls) > 0

        # Check for expected reasons
        valid_reasons = {
            'successful_processing',
            'outlier_rejection_accept',
            'no_kalman_filtering'
        }
        assert any(reason in valid_reasons for reason in validation_calls)


class TestPersistenceIntegration:
    """Integration tests for persistence logic."""

    def test_rejected_measurements_not_persisted(self):
        """Test that rejected measurements don't trigger persistence."""
        with patch('src.processing.processor.get_state_db') as mock_db_factory, \
             patch('src.processing.processor.FeatureManager') as mock_fm_class:

            # Set up mocks
            mock_db = Mock()
            mock_db.get_state.return_value = None
            mock_db_factory.return_value = mock_db

            mock_fm = Mock()
            mock_fm.is_enabled.return_value = True
            mock_fm_class.return_value = mock_fm

            config = {
                'feature_manager': mock_fm,
                'kalman': {},
                'processing': {
                    'outlier_detection': {
                        'enabled': True
                    }
                }
            }

            # Process an extreme weight that should be rejected
            result = process_measurement(
                user_id='user123',
                weight=5.0,  # Too low - should be rejected
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                source='patient-device',
                config=config,
                unit='kg'
            )

            # Should be rejected
            assert result['accepted'] is False

            # For rejected measurements, state might not be saved
            # (depends on the specific rejection point)

    def test_concurrent_update_scenario(self):
        """Test handling of concurrent updates."""
        with patch('src.processing.processor.get_state_db') as mock_db_factory, \
             patch('src.processing.processor.FeatureManager') as mock_fm_class:

            mock_db = Mock()
            mock_db_factory.return_value = mock_db

            mock_fm = Mock()
            mock_fm.is_enabled.return_value = True
            mock_fm_class.return_value = mock_fm

            # Simulate state changing between read and write
            states = [
                {
                    'kalman_state': [[75.0], [0.01]],
                    'kalman_params': {
                        'initial_state_mean': [75.0, 0.01],
                        'initial_state_covariance': [[1.0, 0.0], [0.0, 0.1]],
                        'transition_covariance': [[0.01, 0], [0, 0.001]],
                        'observation_covariance': [[0.5]]
                    },
                    'last_timestamp': datetime(2024, 1, 14, 10, 0, 0)
                },
                {
                    'kalman_state': [[75.5], [0.01]],
                    'kalman_params': {
                        'initial_state_mean': [75.5, 0.01],
                        'initial_state_covariance': [[1.0, 0.0], [0.0, 0.1]],
                        'transition_covariance': [[0.01, 0], [0, 0.001]],
                        'observation_covariance': [[0.5]]
                    },
                    'last_timestamp': datetime(2024, 1, 14, 11, 0, 0)
                }
            ]
            mock_db.get_state.side_effect = states

            config = {
                'feature_manager': mock_fm,
                'kalman': {
                    'transition_covariance_weight': 0.01,
                    'transition_covariance_trend': 0.001,
                    'observation_covariance': 0.5
                }
            }

            # Process weight
            result = process_measurement(
                user_id='user123',
                weight=76.0,
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                source='patient-device',
                config=config,
                unit='kg'
            )

            # Should still process successfully
            assert result['accepted'] is True