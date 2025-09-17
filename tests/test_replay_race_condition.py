"""
Test suite for state restoration race condition fix.

Tests the atomic check-and-restore operations that prevent crashes
when snapshots are None or invalid.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import logging

from src.database.database import ProcessorStateDB
from src.replay.replay_manager import ReplayManager


class TestRaceConditionFix:
    """Test the race condition fix in state restoration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db = ProcessorStateDB()
        self.replay_manager = ReplayManager(self.db, {})
        self.user_id = "test_user"
        self.timestamp = datetime.now()

    def test_restore_with_none_snapshot(self):
        """Test that None snapshot doesn't crash but returns error."""
        # Directly test the validation
        result = self.db.restore_state_from_snapshot(self.user_id, None)

        assert not result['success']
        assert 'snapshot is None' in result['error']
        # Should not crash!

    def test_restore_with_invalid_snapshot_type(self):
        """Test that invalid snapshot type is handled gracefully."""
        # Try with string instead of dict
        result = self.db.restore_state_from_snapshot(self.user_id, "invalid")

        assert not result['success']
        assert 'invalid snapshot type' in result['error']

    def test_restore_with_missing_fields(self):
        """Test that missing required fields are detected."""
        incomplete_snapshot = {
            'kalman_state': [150.0, 0.0],
            # Missing 'kalman_covariance' and 'timestamp'
        }

        result = self.db.restore_state_from_snapshot(self.user_id, incomplete_snapshot)

        assert not result['success']
        assert 'missing required fields' in result['error']
        assert 'kalman_covariance' in result['error'] or 'timestamp' in result['error']

    def test_atomic_check_and_restore_no_snapshot(self):
        """Test atomic operation when no snapshot exists."""
        # No snapshots exist yet
        result = self.db.check_and_restore_snapshot(self.user_id, self.timestamp)

        assert not result['success']
        assert 'No snapshot found' in result['error']
        assert result['user_id'] == self.user_id

    def test_atomic_check_and_restore_success(self):
        """Test successful atomic check and restore."""
        # Create a valid state with snapshot
        state = {
            'last_state': np.array([150.0, 0.0]),
            'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'last_timestamp': self.timestamp - timedelta(hours=1),
            'kalman_params': {'Q': 0.1, 'R': 1.0},
            'state_history': []
        }
        self.db.save_state(self.user_id, state)

        # Save a snapshot
        self.db.save_state_snapshot(self.user_id, self.timestamp - timedelta(hours=1))

        # Now restore atomically
        result = self.db.check_and_restore_snapshot(
            self.user_id,
            self.timestamp  # Restore to before this time
        )

        assert result['success']
        assert 'snapshot' in result
        assert result['user_id'] == self.user_id

    def test_transaction_rollback_on_failure(self):
        """Test that transaction rolls back on failure."""
        # Create initial state
        initial_state = {
            'last_state': np.array([150.0, 0.0]),
            'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'last_timestamp': self.timestamp,
            'kalman_params': {'Q': 0.1, 'R': 1.0}
        }
        self.db.save_state(self.user_id, initial_state)

        # Mock restore to fail after modifying state
        with patch.object(self.db, 'save_state', side_effect=Exception("Save failed")):
            with pytest.raises(Exception):
                with self.db.transaction():
                    # Modify state
                    state = self.db.get_state(self.user_id)
                    state['last_state'] = np.array([200.0, 0.0])  # Change weight
                    self.db.states[self.user_id] = state  # Direct modification
                    # This should trigger rollback
                    self.db.save_state(self.user_id, state)

        # State should be rolled back to initial
        rolled_back_state = self.db.get_state(self.user_id)
        assert np.array_equal(rolled_back_state['last_state'], initial_state['last_state'])

    def test_replay_manager_retry_logic(self):
        """Test that replay manager retries on transient failures."""
        # Mock the database method to fail twice then succeed
        attempt_count = {'count': 0}

        def mock_check_and_restore(user_id, timestamp):
            attempt_count['count'] += 1
            if attempt_count['count'] < 3:
                # Fail first two attempts
                return {
                    'success': False,
                    'error': 'Transient database error',
                    'user_id': user_id
                }
            else:
                # Succeed on third attempt
                return {
                    'success': True,
                    'snapshot': {'kalman_state': [150.0, 0.0]},
                    'snapshot_timestamp': timestamp,
                    'user_id': user_id
                }

        with patch.object(self.db, 'check_and_restore_snapshot', side_effect=mock_check_and_restore):
            result = self.replay_manager._restore_state_to_buffer_start(
                self.user_id,
                self.timestamp
            )

        assert result['success']
        assert result['attempts'] == 3
        assert attempt_count['count'] == 3

    def test_replay_manager_no_retry_on_missing_snapshot(self):
        """Test that replay manager doesn't retry when snapshot doesn't exist."""
        attempt_count = {'count': 0}

        def mock_check_and_restore(user_id, timestamp):
            attempt_count['count'] += 1
            return {
                'success': False,
                'error': 'No snapshot found for user before timestamp',
                'user_id': user_id
            }

        with patch.object(self.db, 'check_and_restore_snapshot', side_effect=mock_check_and_restore):
            result = self.replay_manager._restore_state_to_buffer_start(
                self.user_id,
                self.timestamp
            )

        assert not result['success']
        assert 'No snapshot found' in result['error']
        assert attempt_count['count'] == 1  # Should not retry

    def test_concurrent_restore_operations(self):
        """Test that concurrent restores don't interfere."""
        import threading
        import time

        # Create states for multiple users
        for i in range(3):
            user_id = f"user_{i}"
            state = {
                'last_state': np.array([150.0 + i, 0.0]),
                'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
                'last_timestamp': self.timestamp,
                'kalman_params': {'Q': 0.1, 'R': 1.0},
                'state_history': []
            }
            self.db.save_state(user_id, state)
            self.db.save_state_snapshot(user_id, self.timestamp)

        results = []
        errors = []

        def restore_user(user_id):
            try:
                result = self.db.check_and_restore_snapshot(
                    user_id,
                    self.timestamp + timedelta(hours=1)
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent restores
        threads = []
        for i in range(3):
            thread = threading.Thread(target=restore_user, args=(f"user_{i}",))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 3
        assert all(r['success'] for r in results)

    def test_invalid_kalman_state_type(self):
        """Test handling of invalid kalman_state type in snapshot."""
        invalid_snapshot = {
            'kalman_state': "not_an_array",  # Should be array/list
            'kalman_covariance': [[1.0, 0.0], [0.0, 1.0]],
            'timestamp': self.timestamp
        }

        result = self.db.restore_state_from_snapshot(self.user_id, invalid_snapshot)

        assert not result['success']
        assert 'invalid kalman_state type' in result['error']

    @patch('src.database.database.logger')
    def test_comprehensive_error_logging(self, mock_logger):
        """Test that all error paths log appropriately."""
        # Test None snapshot logging
        self.db.restore_state_from_snapshot(self.user_id, None)
        mock_logger.error.assert_called()
        assert 'snapshot is None' in mock_logger.error.call_args[0][0]

        # Reset mock
        mock_logger.reset_mock()

        # Test missing fields logging
        incomplete_snapshot = {'kalman_state': [150.0, 0.0]}
        self.db.restore_state_from_snapshot(self.user_id, incomplete_snapshot)
        mock_logger.error.assert_called()
        assert 'missing required fields' in mock_logger.error.call_args[0][0]

        # Reset mock
        mock_logger.reset_mock()

        # Test atomic operation logging
        self.db.check_and_restore_snapshot(self.user_id, self.timestamp)
        mock_logger.info.assert_called()  # Should log the attempt
        mock_logger.warning.assert_called()  # Should warn about no snapshot


class TestIntegrationScenarios:
    """Integration tests for the complete replay flow with race condition fix."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.db = ProcessorStateDB()
        self.replay_manager = ReplayManager(self.db, {})

    def test_full_replay_with_missing_snapshot(self):
        """Test that full replay handles missing snapshot gracefully."""
        user_id = "integration_user"
        clean_measurements = [
            {'weight': 150.0, 'timestamp': datetime.now()}
        ]

        # No snapshot exists, should handle gracefully
        result = self.replay_manager.replay_clean_measurements(
            user_id,
            clean_measurements,
            datetime.now() - timedelta(hours=1),
            datetime.now()
        )

        assert not result['success']
        assert 'snapshot' in result['error'].lower() or 'restore' in result['error'].lower()

    def test_replay_with_valid_state(self):
        """Test successful replay with valid state and snapshot."""
        user_id = "valid_user"

        # Create initial state
        initial_state = {
            'last_state': np.array([149.0, 0.0]),
            'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'last_timestamp': datetime.now() - timedelta(hours=2),
            'kalman_params': {
                'Q': np.array([[0.1, 0], [0, 0.01]]),
                'R': 1.0,
                'H': np.array([[1.0, 0.0]])
            },
            'measurement_history': [],
            'state_history': []
        }
        self.db.save_state(user_id, initial_state)

        # Save snapshot
        self.db.save_state_snapshot(user_id, datetime.now() - timedelta(hours=2))

        # Create clean measurements
        clean_measurements = [
            {
                'weight': 150.0,
                'timestamp': datetime.now() - timedelta(hours=1),
                'source': 'test'
            },
            {
                'weight': 150.5,
                'timestamp': datetime.now() - timedelta(minutes=30),
                'source': 'test'
            }
        ]

        # Mock the processor to avoid complex setup
        with patch.object(self.replay_manager, 'processor') as mock_processor:
            mock_processor.process_measurement.return_value = ({
                'weight': 150.0,
                'filtered_weight': 150.0,
                'accepted': True
            }, {})

            result = self.replay_manager.replay_clean_measurements(
                user_id,
                clean_measurements,
                datetime.now() - timedelta(hours=2),
                datetime.now()
            )

        # Should succeed with proper state restoration
        assert result['success']
        assert result['measurements_replayed'] == 2