"""
Test the core acceptance/rejection logic of the replay system.

Ensures measurements are properly filtered based on quality and outlier detection.
Follows pytest best practices with fixtures, parametrization, and markers.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import copy

from src.processing.replay_buffer import ReplayBuffer
from src.replay.replay_manager import ReplayManager


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_database():
    """Provide a clean mock database for testing."""
    class MockDatabase:
        """Mock database for isolated testing."""

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

    return MockDatabase()


@pytest.fixture
def initial_kalman_state():
    """Provide initial Kalman filter state."""
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


@pytest.fixture
def test_user_id():
    """Standard test user ID."""
    return "test_user"


@pytest.fixture
def base_timestamp():
    """Base timestamp for consistent testing."""
    return datetime(2024, 1, 1, 12, 0)


@pytest.fixture
def measurement_factory():
    """Factory for creating test measurements."""
    def create_measurements(weights, hours_ago, source='test', base_time=None):
        """Create measurements with specified weights and times."""
        if base_time is None:
            base_time = datetime.now()

        return [
            {
                'weight': weight,
                'timestamp': base_time - timedelta(hours=hours),
                'source': source
            }
            for weight, hours in zip(weights, hours_ago)
        ]
    return create_measurements


@pytest.fixture
def replay_manager(mock_database):
    """Provide ReplayManager with mock database."""
    return ReplayManager(mock_database, {})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_process_measurement(acceptance_rules):
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


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.unit
class TestPhysiologicalValidation:
    """Test physiological limit validation."""

    @pytest.mark.parametrize("weight,should_accept", [
        (150.0, True),   # Normal weight
        (70.0, True),    # Normal low weight
        (200.0, True),   # Normal high weight
        (25.0, False),   # Too low
        (450.0, False),  # Too high
        (30.0, True),    # Boundary - minimum
        (400.0, True),   # Boundary - maximum
        (29.9, False),   # Just below minimum
        (400.1, False),  # Just above maximum
    ])
    @patch('src.replay.replay_manager.process_measurement')
    def test_weight_range_validation(
        self, mock_process_func, weight, should_accept,
        mock_database, initial_kalman_state, test_user_id, base_timestamp
    ):
        """Test that measurements are validated against physiological limits.

        Given: Weight measurements at various values
        When: Processing through physiological validation
        Then: Should accept/reject based on 30-400kg range
        """
        # Setup acceptance rules - reject weights outside 30-400kg
        acceptance_rules = [
            {'type': 'weight_range', 'min': 30, 'max': 400}
        ]
        mock_process = create_mock_process_measurement(acceptance_rules)

        # Test the validation
        result = mock_process(test_user_id, weight, base_timestamp)

        assert result['accepted'] == should_accept, \
            f"Weight {weight}kg should be {'accepted' if should_accept else 'rejected'}"

        if not should_accept:
            assert 'outside acceptable range' in result['reason'], \
                "Rejection reason should mention acceptable range"
            assert result['stage'] == 'physiological_validation', \
                "Rejection should occur at physiological validation stage"


@pytest.mark.unit
class TestQualityScoring:
    """Test quality score based acceptance/rejection."""

    @pytest.mark.parametrize("weight,expected_weight,threshold,should_accept", [
        (150.0, 150.0, 0.6, True),   # Perfect match - quality 1.0
        (155.0, 150.0, 0.6, True),   # 5kg deviation - quality 0.9
        (160.0, 150.0, 0.6, True),   # 10kg deviation - quality 0.8
        (170.0, 150.0, 0.6, True),   # 20kg deviation - quality 0.6
        (175.0, 150.0, 0.6, False),  # 25kg deviation - quality 0.5
        (180.0, 150.0, 0.6, False),  # 30kg deviation - quality 0.4
        (200.0, 150.0, 0.6, False),  # 50kg deviation - quality 0.0
        (145.0, 150.0, 0.9, True),   # 5kg deviation with strict threshold
        (145.0, 150.0, 0.95, False), # 5kg deviation with very strict threshold
    ])
    def test_quality_threshold_validation(
        self, weight, expected_weight, threshold, should_accept,
        test_user_id, base_timestamp
    ):
        """Test quality score based acceptance.

        Given: Measurements with various deviations from expected
        When: Applying quality threshold validation
        Then: Should accept/reject based on calculated quality score
        """
        acceptance_rules = [
            {
                'type': 'quality_threshold',
                'threshold': threshold,
                'expected_weight': expected_weight
            }
        ]
        mock_process = create_mock_process_measurement(acceptance_rules)

        result = mock_process(test_user_id, weight, base_timestamp)

        # Calculate expected quality score
        deviation = abs(weight - expected_weight)
        expected_quality = max(0, 1.0 - (deviation / 50.0))

        assert result['quality_score'] == pytest.approx(expected_quality, abs=0.01), \
            f"Quality score should be {expected_quality:.2f}"

        assert result['accepted'] == should_accept, \
            f"Weight {weight}kg with quality {expected_quality:.2f} should be {'accepted' if should_accept else 'rejected'}"

        if not should_accept:
            assert 'below threshold' in result['reason'], \
                "Rejection should mention threshold"


@pytest.mark.unit
class TestOutlierDetection:
    """Test statistical outlier detection."""

    @pytest.mark.parametrize("weight,predicted,max_dev,should_accept", [
        (150.0, 150.0, 5.0, True),   # Exact match
        (152.0, 150.0, 5.0, True),   # 2kg deviation - within limit
        (155.0, 150.0, 5.0, True),   # 5kg deviation - at limit
        (156.0, 150.0, 5.0, False),  # 6kg deviation - outlier
        (160.0, 150.0, 5.0, False),  # 10kg deviation - outlier
        (145.0, 150.0, 5.0, True),   # -5kg deviation - at limit
        (144.0, 150.0, 5.0, False),  # -6kg deviation - outlier
        (150.0, 150.0, 2.0, True),   # Strict threshold - exact
        (152.1, 150.0, 2.0, False),  # Strict threshold - outlier
    ])
    def test_outlier_detection_thresholds(
        self, weight, predicted, max_dev, should_accept,
        test_user_id, base_timestamp
    ):
        """Test outlier detection with various thresholds.

        Given: Measurements with deviations from predicted
        When: Applying outlier detection
        Then: Should flag outliers based on max deviation
        """
        acceptance_rules = [
            {
                'type': 'outlier_detection',
                'predicted_weight': predicted,
                'max_deviation': max_dev
            }
        ]
        mock_process = create_mock_process_measurement(acceptance_rules)

        result = mock_process(test_user_id, weight, base_timestamp)

        deviation = abs(weight - predicted)

        assert result['accepted'] == should_accept, \
            f"Weight {weight}kg with {deviation}kg deviation should be {'accepted' if should_accept else 'rejected'}"

        if not should_accept:
            assert result.get('is_outlier') == True, \
                "Should be flagged as outlier"
            assert 'Outlier' in result['reason'], \
                "Reason should mention outlier"
            assert f"{deviation:.1f}kg" in result['reason'], \
                f"Reason should mention deviation amount {deviation:.1f}kg"


@pytest.mark.unit
class TestSourceReliability:
    """Test source-based acceptance/rejection."""

    @pytest.mark.parametrize("source,rejected_sources,should_accept", [
        ('care-team-upload', ['iglucose.com'], True),
        ('patient-device', ['iglucose.com'], True),
        ('iglucose.com', ['iglucose.com'], False),
        ('untrusted', ['untrusted', 'iglucose.com'], False),
        ('test', [], True),
        ('any-source', ['any-source'], False),
    ])
    def test_source_rejection(
        self, source, rejected_sources, should_accept,
        test_user_id, base_timestamp
    ):
        """Test source-based rejection.

        Given: Measurements from various sources
        When: Applying source reliability rules
        Then: Should reject unreliable sources
        """
        acceptance_rules = [
            {
                'type': 'source_rejection',
                'rejected_sources': rejected_sources
            }
        ]
        mock_process = create_mock_process_measurement(acceptance_rules)

        result = mock_process(test_user_id, 150.0, base_timestamp, source)

        assert result['accepted'] == should_accept, \
            f"Source '{source}' should be {'accepted' if should_accept else 'rejected'}"

        if not should_accept:
            assert 'unreliable' in result['reason'], \
                "Rejection should mention unreliability"
            assert result['stage'] == 'source_validation', \
                "Should be rejected at source validation stage"


@pytest.mark.integration
class TestReplayAcceptanceIntegration:
    """Integration tests for full replay acceptance/rejection flow."""

    @patch('src.replay.replay_manager.process_measurement')
    def test_mixed_acceptance_rejection(
        self, mock_process_func, mock_database, initial_kalman_state,
        test_user_id, replay_manager, measurement_factory, base_timestamp
    ):
        """Test complex scenario with multiple acceptance/rejection criteria.

        Given: Various measurements with different characteristics
        When: Replaying through full validation pipeline
        Then: Should correctly accept/reject each measurement
        """
        # Complex acceptance rules combining multiple criteria
        acceptance_rules = [
            {'type': 'weight_range', 'min': 30, 'max': 400},
            {'type': 'quality_threshold', 'threshold': 0.5, 'expected_weight': 150.0},
            {'type': 'outlier_detection', 'predicted_weight': 150.0, 'max_deviation': 10.0},
            {'type': 'source_rejection', 'rejected_sources': ['untrusted']}
        ]

        mock_process_func.side_effect = create_mock_process_measurement(acceptance_rules)

        # Setup initial state
        snapshot_time = base_timestamp - timedelta(hours=7)  # Earlier than buffer start
        initial_kalman_state['last_timestamp'] = snapshot_time
        mock_database.save_state(test_user_id, initial_kalman_state)
        mock_database.save_state_snapshot(test_user_id, snapshot_time)

        # Create diverse measurements
        measurements = [
            {'weight': 150.0, 'timestamp': base_timestamp - timedelta(hours=5), 'source': 'care-team-upload'},  # ✓ All good
            {'weight': 25.0, 'timestamp': base_timestamp - timedelta(hours=4), 'source': 'patient-device'},     # ✗ Too low
            {'weight': 180.0, 'timestamp': base_timestamp - timedelta(hours=3), 'source': 'patient-device'},     # ✗ Low quality
            {'weight': 165.0, 'timestamp': base_timestamp - timedelta(hours=2), 'source': 'patient-device'},     # ✗ Outlier
            {'weight': 151.0, 'timestamp': base_timestamp - timedelta(hours=1), 'source': 'untrusted'},         # ✗ Bad source
            {'weight': 149.0, 'timestamp': base_timestamp, 'source': 'care-team-upload'},                       # ✓ All good
        ]

        # Replay measurements
        result = replay_manager.replay_clean_measurements(
            user_id=test_user_id,
            clean_measurements=measurements,
            buffer_start_time=base_timestamp - timedelta(hours=6)
        )

        assert result['success'] == True, "Replay should complete successfully"

        # Verify each measurement was processed correctly
        expected_results = [
            (150.0, True, None),                                      # Accepted
            (25.0, False, "outside acceptable range"),                # Physiological limit
            (180.0, False, "below threshold"),                        # Low quality
            (165.0, False, "Outlier"),                               # Outlier
            (151.0, False, "unreliable"),                            # Bad source
            (149.0, True, None),                                      # Accepted
        ]

        for i, (weight, should_accept, reason) in enumerate(expected_results):
            result = mock_process_func.side_effect(
                test_user_id,
                measurements[i]['weight'],
                measurements[i]['timestamp'],
                measurements[i]['source']
            )

            assert result['accepted'] == should_accept, \
                f"Measurement {i+1} ({weight}kg) should be {'accepted' if should_accept else 'rejected'}"

            if not should_accept and reason:
                assert reason in result['reason'], \
                    f"Measurement {i+1} rejection reason should contain '{reason}'"


@pytest.mark.slow
class TestPerformance:
    """Performance tests for acceptance/rejection logic."""

    @patch('src.replay.replay_manager.process_measurement')
    def test_large_batch_processing(
        self, mock_process_func, mock_database, initial_kalman_state,
        test_user_id, replay_manager, base_timestamp
    ):
        """Test performance with large batch of measurements.

        Given: 1000+ measurements
        When: Processing through acceptance/rejection
        Then: Should complete in reasonable time
        """
        import time

        # Simple acceptance rules for performance test
        acceptance_rules = [
            {'type': 'weight_range', 'min': 30, 'max': 400}
        ]
        mock_process_func.side_effect = create_mock_process_measurement(acceptance_rules)

        # Setup initial state
        snapshot_time = base_timestamp - timedelta(days=50)  # Earlier than buffer start
        initial_kalman_state['last_timestamp'] = snapshot_time
        mock_database.save_state(test_user_id, initial_kalman_state)
        mock_database.save_state_snapshot(test_user_id, snapshot_time)

        # Create large batch of measurements
        measurements = []
        for i in range(1000):
            weight = 150.0 + np.random.normal(0, 2)  # Normal variation
            timestamp = base_timestamp - timedelta(hours=1000-i)
            measurements.append({
                'weight': weight,
                'timestamp': timestamp,
                'source': 'test'
            })

        # Measure processing time
        start_time = time.time()

        result = replay_manager.replay_clean_measurements(
            user_id=test_user_id,
            clean_measurements=measurements,
            buffer_start_time=base_timestamp - timedelta(days=45)
        )

        elapsed_time = time.time() - start_time

        assert result['success'] == True, "Large batch should process successfully"
        assert elapsed_time < 10.0, f"Processing 1000 measurements should complete within 10 seconds, took {elapsed_time:.2f}s"