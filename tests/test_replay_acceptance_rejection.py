"""
Test the core acceptance/rejection logic of the replay system.
Ensures measurements are properly filtered based on quality and outlier detection.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import copy

from src.processing.replay_buffer import ReplayBuffer
from src.replay.replay_manager import ReplayManager


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.states = {}
        self.snapshots = {}

    def get_state(self, user_id: str):
        """Get state for a user."""
        return copy.deepcopy(self.states.get(user_id))

    def save_state(self, user_id: str, state):
        """Save state for a user."""
        self.states[user_id] = copy.deepcopy(state)

    def save_state_snapshot(self, user_id: str, timestamp: datetime):
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

    def get_state_snapshot_before(self, user_id: str, before_time: datetime):
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

    def check_and_restore_snapshot(self, user_id: str, before_time: datetime):
        """Atomic check and restore."""
        snapshot = self.get_state_snapshot_before(user_id, before_time)
        if not snapshot:
            return {
                'success': False,
                'error': f'No snapshot found for {user_id} before {before_time}'
            }

        # Restore state from snapshot
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


class TestReplayAcceptanceRejection:
    """Test the core acceptance/rejection logic."""

    @pytest.fixture
    def mock_db(self):
        """Provide mock database."""
        return MockDatabase()

    @pytest.fixture
    def initial_state(self):
        """Provide initial Kalman state."""
        return {
            'last_state': np.array([150.0, 0.0]),  # 150kg, no trend
            'last_covariance': np.array([[1.0, 0.0], [0.0, 0.1]]),
            'last_timestamp': datetime.now() - timedelta(hours=5),
            'kalman_params': {
                'Q': np.array([[0.1, 0], [0, 0.01]]),
                'R': 1.0,
                'H': np.array([[1.0, 0.0]])
            },
            'last_raw_weight': 150.0,
            'last_accepted_weight': 150.0,
            'measurement_history': []
        }

    def create_mock_process_measurement(self, acceptance_rules):
        """Create a mock process_measurement function with specific acceptance rules."""
        def mock_process(user_id, weight, timestamp, source='test', **kwargs):
            """Mock processor that accepts/rejects based on rules."""

            # Default to accepting
            result = {
                'user_id': user_id,
                'weight': weight,
                'timestamp': timestamp,
                'source': source,
                'accepted': True,
                'filtered_weight': weight,
                'quality_score': 0.8,
                'stage': 'kalman_update'
            }

            # Apply acceptance rules
            for rule in acceptance_rules:
                if rule['type'] == 'weight_range':
                    if not (rule['min'] <= weight <= rule['max']):
                        result['accepted'] = False
                        result['rejected'] = True
                        result['reason'] = f"Weight {weight}kg outside acceptable range [{rule['min']}, {rule['max']}]"
                        result['stage'] = 'physiological_validation'
                        break

                elif rule['type'] == 'quality_threshold':
                    # Simulate quality score based on weight deviation
                    expected_weight = rule.get('expected_weight', 150.0)
                    deviation = abs(weight - expected_weight)
                    quality_score = max(0, 1.0 - (deviation / 50.0))  # Loses 0.02 per kg deviation

                    result['quality_score'] = quality_score
                    if quality_score < rule['threshold']:
                        result['accepted'] = False
                        result['rejected'] = True
                        result['reason'] = f"Quality score {quality_score:.2f} below threshold {rule['threshold']}"
                        result['stage'] = 'quality_scoring'
                        break

                elif rule['type'] == 'outlier_detection':
                    # Simulate outlier detection based on prediction
                    predicted_weight = rule.get('predicted_weight', 150.0)
                    max_deviation = rule.get('max_deviation', 5.0)  # kg

                    if abs(weight - predicted_weight) > max_deviation:
                        result['accepted'] = False
                        result['rejected'] = True
                        result['reason'] = f"Outlier: {weight}kg deviates {abs(weight - predicted_weight):.1f}kg from predicted {predicted_weight}kg"
                        result['stage'] = 'outlier_detection'
                        result['is_outlier'] = True
                        break

                elif rule['type'] == 'source_rejection':
                    # Reject specific sources
                    if source in rule['rejected_sources']:
                        result['accepted'] = False
                        result['rejected'] = True
                        result['reason'] = f"Source '{source}' is unreliable"
                        result['stage'] = 'source_validation'
                        break

            return result

        return mock_process

    @patch('src.replay.replay_manager.process_measurement')
    def test_physiological_limits_rejection(self, mock_process_func, mock_db, initial_state):
        """Test that measurements outside physiological limits are rejected."""

        # Setup acceptance rules - reject weights outside 30-400kg
        acceptance_rules = [
            {'type': 'weight_range', 'min': 30, 'max': 400}
        ]
        mock_process_func.side_effect = self.create_mock_process_measurement(acceptance_rules)

        # Setup manager and state
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        snapshot_time = datetime.now() - timedelta(hours=6)
        initial_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements with extreme values
        measurements = [
            {'weight': 150.0, 'timestamp': datetime.now() - timedelta(hours=3), 'source': 'test'},  # Normal
            {'weight': 25.0, 'timestamp': datetime.now() - timedelta(hours=2), 'source': 'test'},   # Too low
            {'weight': 450.0, 'timestamp': datetime.now() - timedelta(hours=1), 'source': 'test'},  # Too high
            {'weight': 151.0, 'timestamp': datetime.now(), 'source': 'test'},                       # Normal
        ]

        # Replay measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=4)
        )

        # Verify replay completed
        assert result['success'] == True

        # Check that extreme values were rejected
        assert mock_process_func.call_count == 4

        # Verify specific rejections
        calls = mock_process_func.call_args_list

        # First call (150kg) should be accepted
        result_150 = mock_process_func.side_effect(user_id, 150.0, measurements[0]['timestamp'])
        assert result_150['accepted'] == True

        # Second call (25kg) should be rejected
        result_25 = mock_process_func.side_effect(user_id, 25.0, measurements[1]['timestamp'])
        assert result_25['accepted'] == False
        assert 'outside acceptable range' in result_25['reason']

        # Third call (450kg) should be rejected
        result_450 = mock_process_func.side_effect(user_id, 450.0, measurements[2]['timestamp'])
        assert result_450['accepted'] == False
        assert 'outside acceptable range' in result_450['reason']

        # Fourth call (151kg) should be accepted
        result_151 = mock_process_func.side_effect(user_id, 151.0, measurements[3]['timestamp'])
        assert result_151['accepted'] == True

    @patch('src.replay.replay_manager.process_measurement')
    def test_quality_score_rejection(self, mock_process_func, mock_db, initial_state):
        """Test that low quality measurements are rejected."""

        # Setup acceptance rules - reject if quality score < 0.6
        acceptance_rules = [
            {'type': 'quality_threshold', 'threshold': 0.6, 'expected_weight': 150.0}
        ]
        mock_process_func.side_effect = self.create_mock_process_measurement(acceptance_rules)

        # Setup manager and state
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        snapshot_time = datetime.now() - timedelta(hours=6)
        initial_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements with varying quality
        measurements = [
            {'weight': 150.0, 'timestamp': datetime.now() - timedelta(hours=3), 'source': 'test'},  # High quality (deviation=0)
            {'weight': 155.0, 'timestamp': datetime.now() - timedelta(hours=2), 'source': 'test'},  # Good quality (deviation=5)
            {'weight': 180.0, 'timestamp': datetime.now() - timedelta(hours=1), 'source': 'test'},  # Low quality (deviation=30)
            {'weight': 151.0, 'timestamp': datetime.now(), 'source': 'test'},                       # High quality (deviation=1)
        ]

        # Replay measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=4)
        )

        assert result['success'] == True

        # Verify quality-based acceptance/rejection
        result_150 = mock_process_func.side_effect(user_id, 150.0, measurements[0]['timestamp'])
        assert result_150['accepted'] == True
        assert result_150['quality_score'] == 1.0

        result_155 = mock_process_func.side_effect(user_id, 155.0, measurements[1]['timestamp'])
        assert result_155['accepted'] == True
        assert result_155['quality_score'] == 0.9  # 5kg deviation -> 0.9 score

        result_180 = mock_process_func.side_effect(user_id, 180.0, measurements[2]['timestamp'])
        assert result_180['accepted'] == False
        assert result_180['quality_score'] == 0.4  # 30kg deviation -> 0.4 score
        assert 'below threshold' in result_180['reason']

        result_151 = mock_process_func.side_effect(user_id, 151.0, measurements[3]['timestamp'])
        assert result_151['accepted'] == True
        assert result_151['quality_score'] == 0.98

    @patch('src.replay.replay_manager.process_measurement')
    def test_outlier_detection_rejection(self, mock_process_func, mock_db, initial_state):
        """Test that statistical outliers are rejected."""

        # Setup acceptance rules - reject if > 5kg from predicted weight
        acceptance_rules = [
            {'type': 'outlier_detection', 'predicted_weight': 150.0, 'max_deviation': 5.0}
        ]
        mock_process_func.side_effect = self.create_mock_process_measurement(acceptance_rules)

        # Setup manager and state
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        snapshot_time = datetime.now() - timedelta(hours=6)
        initial_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements with outliers
        measurements = [
            {'weight': 150.0, 'timestamp': datetime.now() - timedelta(hours=4), 'source': 'test'},  # Normal
            {'weight': 152.0, 'timestamp': datetime.now() - timedelta(hours=3), 'source': 'test'},  # Normal variation
            {'weight': 160.0, 'timestamp': datetime.now() - timedelta(hours=2), 'source': 'test'},  # Outlier (10kg jump)
            {'weight': 149.0, 'timestamp': datetime.now() - timedelta(hours=1), 'source': 'test'},  # Normal
            {'weight': 140.0, 'timestamp': datetime.now(), 'source': 'test'},                       # Outlier (10kg drop)
        ]

        # Replay measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=5)
        )

        assert result['success'] == True

        # Verify outlier detection
        result_150 = mock_process_func.side_effect(user_id, 150.0, measurements[0]['timestamp'])
        assert result_150['accepted'] == True

        result_152 = mock_process_func.side_effect(user_id, 152.0, measurements[1]['timestamp'])
        assert result_152['accepted'] == True

        result_160 = mock_process_func.side_effect(user_id, 160.0, measurements[2]['timestamp'])
        assert result_160['accepted'] == False
        assert 'Outlier' in result_160['reason']
        assert result_160.get('is_outlier') == True

        result_149 = mock_process_func.side_effect(user_id, 149.0, measurements[3]['timestamp'])
        assert result_149['accepted'] == True

        result_140 = mock_process_func.side_effect(user_id, 140.0, measurements[4]['timestamp'])
        assert result_140['accepted'] == False
        assert 'Outlier' in result_140['reason']

    @patch('src.replay.replay_manager.process_measurement')
    def test_source_based_rejection(self, mock_process_func, mock_db, initial_state):
        """Test that unreliable sources are rejected."""

        # Setup acceptance rules - reject iglucose.com source
        acceptance_rules = [
            {'type': 'source_rejection', 'rejected_sources': ['iglucose.com', 'untrusted']}
        ]
        mock_process_func.side_effect = self.create_mock_process_measurement(acceptance_rules)

        # Setup manager and state
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        snapshot_time = datetime.now() - timedelta(hours=6)
        initial_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements from different sources
        measurements = [
            {'weight': 150.0, 'timestamp': datetime.now() - timedelta(hours=3), 'source': 'care-team-upload'},  # Trusted
            {'weight': 151.0, 'timestamp': datetime.now() - timedelta(hours=2), 'source': 'iglucose.com'},      # Untrusted
            {'weight': 150.5, 'timestamp': datetime.now() - timedelta(hours=1), 'source': 'patient-device'},    # Trusted
            {'weight': 152.0, 'timestamp': datetime.now(), 'source': 'untrusted'},                              # Untrusted
        ]

        # Replay measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=4)
        )

        assert result['success'] == True

        # Verify source-based rejection
        result_care = mock_process_func.side_effect(user_id, 150.0, measurements[0]['timestamp'], 'care-team-upload')
        assert result_care['accepted'] == True

        result_iglucose = mock_process_func.side_effect(user_id, 151.0, measurements[1]['timestamp'], 'iglucose.com')
        assert result_iglucose['accepted'] == False
        assert 'unreliable' in result_iglucose['reason']

        result_device = mock_process_func.side_effect(user_id, 150.5, measurements[2]['timestamp'], 'patient-device')
        assert result_device['accepted'] == True

        result_untrusted = mock_process_func.side_effect(user_id, 152.0, measurements[3]['timestamp'], 'untrusted')
        assert result_untrusted['accepted'] == False

    @patch('src.replay.replay_manager.process_measurement')
    def test_mixed_acceptance_rejection(self, mock_process_func, mock_db, initial_state):
        """Test a realistic mix of accepted and rejected measurements."""

        # Setup combined acceptance rules
        acceptance_rules = [
            {'type': 'weight_range', 'min': 30, 'max': 400},
            {'type': 'quality_threshold', 'threshold': 0.5, 'expected_weight': 150.0},
            {'type': 'outlier_detection', 'predicted_weight': 150.0, 'max_deviation': 8.0}
        ]
        mock_process_func.side_effect = self.create_mock_process_measurement(acceptance_rules)

        # Setup manager and state
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        snapshot_time = datetime.now() - timedelta(hours=12)
        initial_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create a realistic series of measurements
        measurements = [
            {'weight': 150.0, 'timestamp': datetime.now() - timedelta(hours=10), 'source': 'test'},  # Accept
            {'weight': 149.5, 'timestamp': datetime.now() - timedelta(hours=9), 'source': 'test'},   # Accept
            {'weight': 148.8, 'timestamp': datetime.now() - timedelta(hours=8), 'source': 'test'},   # Accept
            {'weight': 165.0, 'timestamp': datetime.now() - timedelta(hours=7), 'source': 'test'},   # Reject - outlier
            {'weight': 149.0, 'timestamp': datetime.now() - timedelta(hours=6), 'source': 'test'},   # Accept
            {'weight': 25.0, 'timestamp': datetime.now() - timedelta(hours=5), 'source': 'test'},    # Reject - physiological
            {'weight': 148.5, 'timestamp': datetime.now() - timedelta(hours=4), 'source': 'test'},   # Accept
            {'weight': 180.0, 'timestamp': datetime.now() - timedelta(hours=3), 'source': 'test'},   # Reject - quality
            {'weight': 148.0, 'timestamp': datetime.now() - timedelta(hours=2), 'source': 'test'},   # Accept
            {'weight': 147.5, 'timestamp': datetime.now() - timedelta(hours=1), 'source': 'test'},   # Accept
        ]

        # Replay measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=11)
        )

        assert result['success'] == True
        assert result['measurements_replayed'] == 10

        # Count accepted vs rejected
        accepted_count = 0
        rejected_count = 0

        for m in measurements:
            r = mock_process_func.side_effect(user_id, m['weight'], m['timestamp'], m['source'])
            if r['accepted']:
                accepted_count += 1
            else:
                rejected_count += 1

        # Should have 7 accepted, 3 rejected
        assert accepted_count == 7
        assert rejected_count == 3

    @patch('src.replay.replay_manager.process_measurement')
    def test_state_update_on_acceptance(self, mock_process_func, mock_db, initial_state):
        """Test that accepted measurements update the Kalman state."""

        # Track state updates
        state_updates = []

        def mock_process_with_state_update(user_id, weight, timestamp, **kwargs):
            """Mock that simulates state updates."""
            result = {
                'user_id': user_id,
                'weight': weight,
                'timestamp': timestamp,
                'accepted': True,
                'filtered_weight': weight + 0.1,  # Small Kalman correction
                'quality_score': 0.9
            }

            # Simulate state update
            if result['accepted']:
                new_state = {
                    'last_state': np.array([result['filtered_weight'], 0.0]),
                    'last_timestamp': timestamp,
                    'last_accepted_weight': result['filtered_weight']
                }
                state_updates.append(new_state)

                # Update the mock database state
                current_state = mock_db.get_state(user_id) or {}
                current_state.update(new_state)
                mock_db.save_state(user_id, current_state)

            return result

        mock_process_func.side_effect = mock_process_with_state_update

        # Setup manager and state
        manager = ReplayManager(mock_db, {})
        user_id = "test_user"

        snapshot_time = datetime.now() - timedelta(hours=6)
        initial_state['last_timestamp'] = snapshot_time
        mock_db.save_state(user_id, initial_state)
        mock_db.save_state_snapshot(user_id, snapshot_time)

        # Create measurements
        measurements = [
            {'weight': 150.0, 'timestamp': datetime.now() - timedelta(hours=3), 'source': 'test'},
            {'weight': 149.8, 'timestamp': datetime.now() - timedelta(hours=2), 'source': 'test'},
            {'weight': 149.5, 'timestamp': datetime.now() - timedelta(hours=1), 'source': 'test'},
        ]

        # Replay measurements
        result = manager.replay_clean_measurements(
            user_id=user_id,
            clean_measurements=measurements,
            buffer_start_time=datetime.now() - timedelta(hours=4)
        )

        assert result['success'] == True

        # Verify state was updated for each accepted measurement
        assert len(state_updates) == 3

        # Check final state
        final_state = mock_db.get_state(user_id)
        assert final_state is not None

        # Last accepted weight should be from the last measurement (with Kalman correction)
        expected_final_weight = 149.5 + 0.1  # Last measurement + Kalman correction
        assert np.isclose(final_state['last_state'][0], expected_final_weight, atol=0.01)