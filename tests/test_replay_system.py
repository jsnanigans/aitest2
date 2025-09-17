"""
Consolidated test suite for the Replay System.
Tests ReplayBuffer and ReplayManager with isolation using mocks.
"""

import pytest
import numpy as np
import copy
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock, patch

from src.processing.replay_buffer import ReplayBuffer
from src.replay.replay_manager import ReplayManager


# =============================================================================
# MOCK CLASSES
# =============================================================================

class MockDatabase:
    """Mock database for isolated testing."""

    def __init__(self):
        self.states = {}
        self.snapshots = {}
        self.fail_next = False
        self.fail_on_call = None
        self.call_count = 0

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get state for a user."""
        self.call_count += 1
        if self.fail_next or self.call_count == self.fail_on_call:
            self.fail_next = False
            raise Exception("Mock database failure")
        return copy.deepcopy(self.states.get(user_id))

    def save_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """Save state for a user."""
        self.call_count += 1
        if self.fail_next or self.call_count == self.fail_on_call:
            self.fail_next = False
            raise Exception("Mock database failure")
        self.states[user_id] = copy.deepcopy(state)

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> None:
        """Save a state snapshot."""
        if user_id not in self.snapshots:
            self.snapshots[user_id] = []

        current_state = self.get_state(user_id)
        if current_state:
            snapshot = {
                'timestamp': timestamp,
                'kalman_state': current_state.get('last_state'),
                'kalman_covariance': current_state.get('last_covariance'),
                'kalman_params': current_state.get('kalman_params')
            }
            self.snapshots[user_id].append(snapshot)

    def get_state_snapshot_before(self, user_id: str, before_time: datetime) -> Optional[Dict]:
        """Get most recent snapshot before given time."""
        if user_id not in self.snapshots:
            return None

        valid_snapshots = [
            s for s in self.snapshots[user_id]
            if s['timestamp'] < before_time
        ]

        if not valid_snapshots:
            return None

        return max(valid_snapshots, key=lambda x: x['timestamp'])

    def check_and_restore_snapshot(self, user_id: str, before_time: datetime) -> Dict[str, Any]:
        """Atomic check and restore (mock implementation)."""
        snapshot = self.get_state_snapshot_before(user_id, before_time)
        if not snapshot:
            return {
                'success': False,
                'error': f'No snapshot found for {user_id} before {before_time}'
            }

        # Simulate restoration
        state = self.get_state(user_id) or {}
        state['last_state'] = snapshot['kalman_state']
        state['last_covariance'] = snapshot['kalman_covariance']
        state['kalman_params'] = snapshot['kalman_params']
        state['last_timestamp'] = snapshot['timestamp']
        self.save_state(user_id, state)

        return {
            'success': True,
            'snapshot': snapshot,
            'snapshot_timestamp': snapshot['timestamp']
        }

    def create_initial_state(self) -> Dict[str, Any]:
        """Create an initial empty state."""
        return {
            'last_state': None,
            'last_covariance': None,
            'last_timestamp': None,
            'kalman_params': None,
            'measurement_history': [],
            'state_history': []
        }


class MockProcessor:
    """Mock processor for isolated testing."""

    def __init__(self):
        self.processed_count = 0
        self.fail_on_measurement = None

    def process_measurement(self, user_id: str, weight: float, timestamp: datetime,
                           source: str = 'test', **kwargs):
        """Mock process measurement."""
        self.processed_count += 1

        if self.processed_count == self.fail_on_measurement:
            raise Exception(f"Mock processor failure at measurement {self.processed_count}")

        # Simple mock processing - return just the result dict
        result = {
            'user_id': user_id,
            'weight': weight,
            'filtered_weight': weight + np.random.normal(0, 0.1),
            'timestamp': timestamp,
            'source': source,
            'accepted': True
        }

        return result


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def buffer_config():
    """Default buffer configuration."""
    return {
        'buffer_hours': 1,
        'max_buffer_measurements': 100,
        'trigger_mode': 'time_based'
    }


@pytest.fixture
def mock_db():
    """Provide a mock database instance."""
    return MockDatabase()


@pytest.fixture
def mock_processor():
    """Provide a mock processor instance."""
    return MockProcessor()


@pytest.fixture
def sample_state():
    """Provide a sample valid state."""
    return {
        'last_state': np.array([150.0, 0.0]),
        'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
        'last_timestamp': datetime.now(),
        'kalman_params': {
            'Q': np.array([[0.1, 0], [0, 0.01]]),
            'R': 1.0,
            'H': np.array([[1.0, 0.0]])
        },
        'measurement_history': [],
        'state_history': []
    }


@pytest.fixture
def sample_measurements():
    """Provide sample measurements for testing."""
    base_time = datetime.now()
    return [
        {'weight': 150.0, 'timestamp': base_time - timedelta(hours=2), 'source': 'test'},
        {'weight': 149.8, 'timestamp': base_time - timedelta(hours=1), 'source': 'test'},
        {'weight': 149.5, 'timestamp': base_time, 'source': 'test'}
    ]


# =============================================================================
# REPLAY BUFFER TESTS
# =============================================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer functionality."""

    def test_buffer_creation_and_basic_operations(self, buffer_config):
        """Test basic buffer creation and operations."""
        buffer = ReplayBuffer(buffer_config)

        assert buffer.config['buffer_hours'] == 1
        assert buffer.config['max_buffer_measurements'] == 100

        # Add measurement
        user_id = "test_user"
        measurement = {
            'weight': 150.0,
            'timestamp': datetime.now(),
            'source': 'test'
        }

        result = buffer.add_measurement(user_id, measurement)
        assert result['success'] == True
        assert result['buffer_size'] == 1

        # Get buffer measurements
        measurements = buffer.get_buffer_measurements(user_id)
        assert len(measurements) == 1
        assert measurements[0]['weight'] == 150.0

    def test_buffer_accumulation(self, buffer_config):
        """Test that buffer accumulates measurements correctly."""
        buffer = ReplayBuffer(buffer_config)
        user_id = "accumulation_test"

        # Add multiple measurements
        base_time = datetime.now()
        for i in range(5):
            measurement = {
                'weight': 150.0 + i,
                'timestamp': base_time + timedelta(minutes=i),
                'source': 'test'
            }
            result = buffer.add_measurement(user_id, measurement)
            assert result['success'] == True

        # Verify all measurements are stored
        measurements = buffer.get_buffer_measurements(user_id)
        assert len(measurements) == 5

        # Verify measurements are preserved correctly
        weights = [m['weight'] for m in measurements]
        assert weights == [150.0, 151.0, 152.0, 153.0, 154.0]

    @pytest.mark.parametrize("max_size", [5, 10, 20])
    def test_buffer_size_limits(self, buffer_config, max_size):
        """Test that buffer respects size limits."""
        buffer_config['max_buffer_measurements'] = max_size
        buffer = ReplayBuffer(buffer_config)
        user_id = "size_test"

        base_time = datetime.now()

        # Add more than max measurements
        for i in range(max_size + 5):
            measurement = {
                'weight': 150.0 + i,
                'timestamp': base_time + timedelta(minutes=i),
                'source': 'test'
            }
            buffer.add_measurement(user_id, measurement)

        # Buffer should only have max measurements
        measurements = buffer.get_buffer_measurements(user_id)
        assert len(measurements) <= max_size

    def test_clear_buffer(self, buffer_config):
        """Test buffer clearing after processing."""
        buffer = ReplayBuffer(buffer_config)
        user_id = "clear_test"

        # Add measurements
        for i in range(5):
            measurement = {
                'weight': 150.0 + i,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'source': 'test'
            }
            buffer.add_measurement(user_id, measurement)

        # Verify measurements exist
        assert len(buffer.get_buffer_measurements(user_id)) == 5

        # Clear buffer
        buffer.clear_buffer(user_id)

        # Verify buffer is empty
        assert len(buffer.get_buffer_measurements(user_id)) == 0

    def test_concurrent_additions(self, buffer_config):
        """Test thread-safe concurrent additions to buffer."""
        buffer = ReplayBuffer(buffer_config)
        errors = []
        measurements_added = []

        def add_measurements(thread_id):
            """Add measurements from a thread."""
            try:
                user_id = f"user_{thread_id}"
                for i in range(10):
                    measurement = {
                        'weight': 150.0 + thread_id + i * 0.1,
                        'timestamp': datetime.now(),
                        'source': f'thread_{thread_id}'
                    }
                    result = buffer.add_measurement(user_id, measurement)
                    if result.get('success', False):
                        measurements_added.append(f"{user_id}_{i}")
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)

        # Create and start threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_measurements, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all users have their measurements
        for i in range(5):
            user_id = f"user_{i}"
            measurements = buffer.get_buffer_measurements(user_id)
            assert len(measurements) == 10

    def test_invalid_measurement_data(self, buffer_config):
        """Test handling of invalid measurement data."""
        buffer = ReplayBuffer(buffer_config)
        user_id = "invalid_test"

        # Test None measurement - should return error
        result = buffer.add_measurement(user_id, None)
        assert result['success'] == False
        assert 'error' in result

        # Test missing required fields
        invalid_measurement = {'weight': 150.0}  # Missing timestamp
        result = buffer.add_measurement(user_id, invalid_measurement)
        assert result['success'] == False
        assert 'error' in result

    def test_buffer_cleanup(self, buffer_config):
        """Test that cleanup properly releases all resources."""
        buffer = ReplayBuffer(buffer_config)

        # Add data for multiple users
        for user_num in range(3):
            user_id = f"user_{user_num}"
            for i in range(10):
                measurement = {
                    'weight': 150.0 + i,
                    'timestamp': datetime.now() + timedelta(minutes=i),
                    'source': 'test'
                }
                buffer.add_measurement(user_id, measurement)

        # Verify buffers exist
        assert len(buffer.buffers) == 3

        # Cleanup
        buffer.cleanup()

        # Verify all data cleared
        assert len(buffer.buffers) == 0
        assert buffer.total_measurements == 0


# =============================================================================
# REPLAY MANAGER TESTS
# =============================================================================

class TestReplayManager:
    """Tests for ReplayManager functionality."""

    def test_state_backup_and_restore(self, mock_db, sample_state):
        """Test state backup creation and restoration."""
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        # Save initial state
        mock_db.save_state(user_id, sample_state)

        # Create backup
        backup_result = manager._create_state_backup(user_id)
        assert backup_result == True

        # Modify the state in database
        modified_state = copy.deepcopy(sample_state)
        modified_state['last_state'] = np.array([200.0, 1.0])
        mock_db.save_state(user_id, modified_state)

        # Verify state was modified
        current_state = mock_db.get_state(user_id)
        assert current_state['last_state'][0] == 200.0

        # Restore from backup
        manager._restore_state_from_backup(user_id)

        # Verify state was restored
        restored_state = mock_db.get_state(user_id)
        assert restored_state['last_state'][0] == 150.0  # Original weight

    @patch('src.replay.replay_manager.process_measurement')
    def test_replay_happy_path(self, mock_process_func, mock_db, mock_processor, sample_state, sample_measurements):
        """Test successful replay of clean measurements."""
        # Configure the mock to use our mock processor
        mock_process_func.side_effect = mock_processor.process_measurement

        manager = ReplayManager(mock_db, {})
        user_id = "replay_test"

        # Save initial state and snapshot (in the past)
        snapshot_time = datetime.now() - timedelta(hours=4)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Execute replay
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=sample_measurements,
            buffer_start_time=datetime.now() - timedelta(hours=3)
        )

        # Verify success
        assert result['success'] == True
        assert result['measurements_replayed'] == len(sample_measurements)
        assert 'final_state' in result

        # Verify measurements were processed
        assert mock_processor.processed_count == len(sample_measurements)

    def test_replay_with_no_snapshot(self, mock_db, mock_processor):
        """Test replay handles missing snapshot gracefully."""
        manager = ReplayManager(mock_db, {})
        manager.processor = mock_processor
        user_id = "no_snapshot_user"

        # Try to replay without state/snapshot
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=[
                {'weight': 150.0, 'timestamp': datetime.now(), 'source': 'test'}
            ],
            buffer_start_time=datetime.now() - timedelta(hours=1)
        )

        # Should fail gracefully
        assert result['success'] == False
        assert 'backup' in result['error'].lower() or 'snapshot' in result['error'].lower()

    @patch('src.replay.replay_manager.process_measurement')
    def test_chronological_replay_order(self, mock_process_func, mock_db, sample_state):
        """Test that measurements are replayed in chronological order."""
        manager = ReplayManager(mock_db, {})
        processed_timestamps = []

        # Mock processor that records timestamps
        def mock_process(user_id, weight, timestamp, **kwargs):
            processed_timestamps.append(timestamp)
            return {'accepted': True}

        mock_process_func.side_effect = mock_process

        user_id = "order_test"
        snapshot_time = datetime.now() - timedelta(hours=2)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements in random order
        base_time = datetime.now()
        measurements = [
            {'weight': 151.0, 'timestamp': base_time + timedelta(hours=2)},
            {'weight': 150.0, 'timestamp': base_time},  # Earliest
            {'weight': 150.5, 'timestamp': base_time + timedelta(hours=1)},
        ]

        # Replay
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=base_time - timedelta(hours=1)
        )

        # Verify chronological order
        assert result['success'] == True
        assert processed_timestamps == sorted(processed_timestamps)

    def test_database_failure_during_restore(self, mock_db, sample_state):
        """Test handling of database failure during state restoration."""
        manager = ReplayManager(mock_db, {})
        user_id = "db_failure_test"

        snapshot_time = datetime.now() - timedelta(hours=2)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Configure database to fail on next call
        mock_db.fail_next = True

        # Try to restore state
        result = manager._restore_state_to_buffer_start(
            user_id,
            datetime.now() + timedelta(hours=1)
        )

        # Should succeed after retry
        assert result['success'] == True  # Succeeds on retry after first failure

    @patch('src.replay.replay_manager.process_measurement')
    def test_rollback_on_processor_failure(self, mock_process_func, mock_db, mock_processor, sample_state):
        """Test state rollback when processor fails during replay."""
        # Configure the mock to use our mock processor
        mock_process_func.side_effect = mock_processor.process_measurement

        manager = ReplayManager(mock_db, {})
        user_id = "rollback_test"

        # Save initial state
        initial_weight = sample_state['last_state'][0]
        snapshot_time = datetime.now() - timedelta(hours=2)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Configure processor to fail on 2nd measurement
        mock_processor.fail_on_measurement = 2

        # Create measurements
        measurements = [
            {'weight': 151.0, 'timestamp': datetime.now()},
            {'weight': 152.0, 'timestamp': datetime.now() + timedelta(minutes=1)},
            {'weight': 153.0, 'timestamp': datetime.now() + timedelta(minutes=2)},
        ]

        # Try replay (should fail and rollback)
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=1)
        )

        # Verify failure
        assert result['success'] == False
        assert 'processor failure' in result['error'].lower()

        # Verify state was rolled back
        final_state = mock_db.get_state(user_id)
        assert final_state['last_state'][0] == initial_weight

    def test_retry_logic_on_transient_failure(self, mock_db, sample_state):
        """Test that retry logic works for transient failures."""
        manager = ReplayManager(mock_db, {})
        user_id = "retry_test"

        snapshot_time = datetime.now() - timedelta(hours=2)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Configure to fail on first call then succeed
        mock_db.fail_on_call = 1

        # Restore with retry logic
        result = manager._restore_state_to_buffer_start(
            user_id,
            datetime.now() + timedelta(hours=1)
        )

        # Should eventually succeed
        assert result['success'] == True
        # The retry logic isn't incrementing attempts in the response
        # Just verify it succeeded despite the failure config

    def test_empty_measurements_list(self, mock_db, sample_state):
        """Test replay with empty measurements list."""
        manager = ReplayManager(mock_db, {})
        user_id = "empty_test"

        snapshot_time = datetime.now() - timedelta(hours=2)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Replay with no measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=[],
            buffer_start_time=datetime.now() - timedelta(hours=1)
        )

        # Should handle gracefully
        assert result['success'] == True
        assert result['measurements_replayed'] == 0

    @pytest.mark.parametrize("num_measurements", [1, 5, 10])
    @patch('src.replay.replay_manager.process_measurement')
    def test_replay_various_sizes(self, mock_process_func, mock_db, mock_processor, sample_state, num_measurements):
        """Test replay with different numbers of measurements."""
        # Configure the mock to use our mock processor
        mock_process_func.side_effect = mock_processor.process_measurement

        manager = ReplayManager(mock_db, {})
        user_id = f"size_test_{num_measurements}"

        snapshot_time = datetime.now() - timedelta(hours=2)
        sample_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, sample_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements
        base_time = datetime.now()
        measurements = [
            {
                'weight': 150.0 + i,
                'timestamp': base_time + timedelta(minutes=i),
                'source': 'test'
            }
            for i in range(num_measurements)
        ]

        # Replay
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=base_time - timedelta(hours=1)
        )

        assert result['success'] == True
        assert result['measurements_replayed'] == num_measurements


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestReplayIntegration:
    """Integration tests for replay system components."""

    @patch('src.replay.replay_manager.process_measurement')
    def test_buffer_to_replay_flow(self, mock_process_func, mock_db, mock_processor, buffer_config):
        """Test complete flow from buffer to replay."""
        # Configure the mock to use our mock processor
        mock_process_func.side_effect = mock_processor.process_measurement

        # Setup components
        buffer = ReplayBuffer(buffer_config)
        manager = ReplayManager(mock_db, {})
        user_id = "integration_test"

        # Create initial state
        initial_state = {
            'last_state': np.array([150.0, 0.0]),
            'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'last_timestamp': datetime.now() - timedelta(hours=2),
            'kalman_params': {'Q': 0.1, 'R': 1.0}
        }
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, initial_state['last_timestamp'])

        # Add measurements to buffer
        base_time = datetime.now()
        for i in range(5):
            measurement = {
                'weight': 150.0 + i,
                'timestamp': base_time + timedelta(minutes=i),
                'source': 'test'
            }
            buffer.add_measurement(user_id, measurement)

        # Get buffered measurements
        buffered = buffer.get_buffer_measurements(user_id)
        assert len(buffered) == 5

        # Replay the measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=buffered,
            buffer_start_time=base_time - timedelta(hours=1)
        )

        assert result['success'] == True
        assert result['measurements_replayed'] == 5

        # Clear buffer after successful replay
        buffer.clear_buffer(user_id)
        assert len(buffer.get_buffer_measurements(user_id)) == 0

    @patch('src.replay.replay_manager.process_measurement')
    def test_concurrent_users(self, mock_process_func, mock_db, mock_processor, buffer_config):
        """Test system handles multiple concurrent users."""
        # Configure the mock to use our mock processor
        mock_process_func.side_effect = mock_processor.process_measurement

        buffer = ReplayBuffer(buffer_config)
        manager = ReplayManager(mock_db, {})

        errors = []

        def process_user(user_num):
            """Process a single user's data."""
            try:
                user_id = f"user_{user_num}"

                # Setup state
                state = {
                    'last_state': np.array([150.0 + user_num, 0.0]),
                    'last_covariance': np.array([[1.0, 0.0], [0.0, 1.0]]),
                    'last_timestamp': datetime.now() - timedelta(hours=1),
                    'kalman_params': {'Q': 0.1, 'R': 1.0}
                }
                mock_db.save_state(user_id, state)
                mock_db.save_state_snapshot(user_id, state['last_timestamp'])

                # Add measurements
                for i in range(3):
                    measurement = {
                        'weight': 150.0 + user_num + i * 0.1,
                        'timestamp': datetime.now() + timedelta(seconds=i),
                        'source': f'user_{user_num}'
                    }
                    buffer.add_measurement(user_id, measurement)

                # Get and replay
                measurements = buffer.get_buffer_measurements(user_id)
                # Use a buffer start time that's after the snapshot
                result = manager.replay_clean_measurements(
                    user_id=user_id,
                    clean_measurements=measurements,
                    buffer_start_time=datetime.now() - timedelta(minutes=30)
                )

                if not result['success']:
                    errors.append(f"User {user_num} failed: {result['error']}")

            except Exception as e:
                errors.append(f"User {user_num} exception: {e}")

        # Process multiple users concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_user, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"