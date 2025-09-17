"""
Unit tests for ResetManager - focusing on critical functionality.
"""

import pytest
from datetime import datetime, timedelta
from src.processing.reset_manager import ResetManager, ResetType, MANUAL_DATA_SOURCES


@pytest.fixture
def base_config():
    """Standard config with all reset types enabled."""
    return {
        'kalman': {
            'reset': {
                'initial': {'enabled': True},
                'hard': {
                    'enabled': True,
                    'gap_threshold_days': 30
                },
                'soft': {
                    'enabled': True,
                    'min_weight_change_kg': 5,
                    'cooldown_days': 3
                }
            }
        }
    }


@pytest.fixture
def valid_state():
    """State with existing Kalman parameters."""
    return {
        'kalman_params': {'weight': [75.0, 0.0]},
        'last_timestamp': datetime(2024, 1, 1),
        'last_raw_weight': 75.0,
        'measurements_since_reset': 5,
        'last_accepted_timestamp': datetime(2024, 1, 1),
        'reset_events': []
    }


@pytest.fixture
def reset_state():
    """State immediately after a soft reset."""
    return {
        'reset_timestamp': datetime(2024, 1, 15),
        'reset_type': 'soft',
        'reset_parameters': {
            'adaptation_measurements': 15,
            'adaptation_days': 10,
            'adaptation_decay_rate': 4
        },
        'measurements_since_reset': 0,
        'kalman_params': None
    }


class TestResetDetection:
    """Test reset detection logic for all three types."""

    def test_initial_reset_no_state(self, base_config):
        """Empty state triggers INITIAL reset."""
        result = ResetManager.should_trigger_reset(
            state={},
            weight=75.0,
            timestamp=datetime(2024, 1, 1),
            source='patient-device',
            config=base_config
        )
        assert result == ResetType.INITIAL

    def test_initial_reset_no_kalman_params(self, base_config):
        """Missing kalman_params triggers INITIAL reset."""
        state = {'last_timestamp': datetime(2024, 1, 1)}
        result = ResetManager.should_trigger_reset(
            state=state,
            weight=75.0,
            timestamp=datetime(2024, 1, 2),
            source='patient-device',
            config=base_config
        )
        assert result == ResetType.INITIAL

    def test_hard_reset_gap_detection(self, valid_state, base_config):
        """30+ day gap triggers HARD reset."""
        new_timestamp = datetime(2024, 2, 5)  # 35 days later
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=75.0,
            timestamp=new_timestamp,
            source='patient-device',
            config=base_config
        )
        assert result == ResetType.HARD

    def test_hard_reset_threshold_boundary(self, valid_state, base_config):
        """Test exactly 30 days vs 29.99 days."""
        # Exactly 30 days - should trigger
        timestamp_30 = datetime(2024, 1, 31)
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=75.0,
            timestamp=timestamp_30,
            source='patient-device',
            config=base_config
        )
        assert result == ResetType.HARD

        # 29.9 days - should not trigger
        timestamp_29 = datetime(2024, 1, 30, 23, 0)
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=75.0,
            timestamp=timestamp_29,
            source='patient-device',
            config=base_config
        )
        assert result is None

    @pytest.mark.parametrize("source", ['questionnaire', 'patient-upload', 'care-team-upload'])
    def test_soft_reset_manual_sources(self, valid_state, base_config, source):
        """Each manual source triggers SOFT with 5kg+ change."""
        new_timestamp = datetime(2024, 1, 2)
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=81.0,  # 6kg change
            timestamp=new_timestamp,
            source=source,
            config=base_config
        )
        assert result == ResetType.SOFT

    def test_soft_reset_weight_change_threshold(self, valid_state, base_config):
        """Test exactly 5kg vs 4.99kg change."""
        new_timestamp = datetime(2024, 1, 2)

        # Exactly 5kg - should trigger
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=80.0,  # Exactly 5kg change
            timestamp=new_timestamp,
            source='questionnaire',
            config=base_config
        )
        assert result == ResetType.SOFT

        # 4.9kg - should not trigger
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=79.9,  # 4.9kg change
            timestamp=new_timestamp,
            source='questionnaire',
            config=base_config
        )
        assert result is None

    def test_soft_reset_cooldown(self, valid_state, base_config):
        """Verify cooldown prevents repeated soft resets."""
        # Add recent reset event
        valid_state['reset_events'] = [{
            'timestamp': datetime(2024, 1, 1, 12),
            'type': 'soft'
        }]

        # Try reset 2 days later (within 3-day cooldown)
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=81.0,  # 6kg change
            timestamp=datetime(2024, 1, 3, 12),
            source='questionnaire',
            config=base_config
        )
        assert result is None

        # Try reset 4 days later (after cooldown)
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=81.0,  # 6kg change
            timestamp=datetime(2024, 1, 5, 13),
            source='questionnaire',
            config=base_config
        )
        assert result == ResetType.SOFT

    def test_no_reset_normal_conditions(self, valid_state, base_config):
        """Verify no false positives under normal conditions."""
        # Small time gap, small weight change, non-manual source
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=75.5,  # Small change
            timestamp=datetime(2024, 1, 2),  # 1 day gap
            source='patient-device',  # Non-manual
            config=base_config
        )
        assert result is None


class TestResetPriority:
    """Test priority ordering when multiple reset conditions exist."""

    def test_initial_beats_hard(self, base_config):
        """No kalman_params + 30 day gap = INITIAL."""
        state = {
            'last_timestamp': datetime(2024, 1, 1),
            'last_raw_weight': 75.0
        }
        result = ResetManager.should_trigger_reset(
            state=state,
            weight=75.0,
            timestamp=datetime(2024, 2, 5),  # 35 days later
            source='patient-device',
            config=base_config
        )
        assert result == ResetType.INITIAL

    def test_initial_beats_soft(self, base_config):
        """No kalman_params + manual source = INITIAL."""
        state = {
            'last_timestamp': datetime(2024, 1, 1),
            'last_raw_weight': 75.0
        }
        result = ResetManager.should_trigger_reset(
            state=state,
            weight=81.0,  # 6kg change
            timestamp=datetime(2024, 1, 2),
            source='questionnaire',
            config=base_config
        )
        assert result == ResetType.INITIAL

    def test_hard_beats_soft(self, valid_state, base_config):
        """30 day gap + manual source = HARD."""
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=81.0,  # 6kg change
            timestamp=datetime(2024, 2, 5),  # 35 days later
            source='questionnaire',
            config=base_config
        )
        assert result == ResetType.HARD


class TestResetParameters:
    """Test parameter retrieval for different reset types."""

    def test_initial_parameters_defaults(self, base_config):
        """Verify default INITIAL parameters."""
        params = ResetManager.get_reset_parameters(ResetType.INITIAL, base_config)
        assert params['initial_variance_multiplier'] == 10
        assert params['weight_noise_multiplier'] == 50
        assert params['adaptation_measurements'] == 20
        assert params['adaptation_days'] == 21
        assert params['quality_acceptance_threshold'] == 0.25

    def test_hard_parameters_defaults(self, base_config):
        """Verify default HARD parameters."""
        params = ResetManager.get_reset_parameters(ResetType.HARD, base_config)
        assert params['initial_variance_multiplier'] == 5
        assert params['weight_noise_multiplier'] == 20
        assert params['adaptation_measurements'] == 10
        assert params['adaptation_days'] == 7
        assert params['quality_acceptance_threshold'] == 0.35

    def test_soft_parameters_defaults(self, base_config):
        """Verify default SOFT parameters."""
        params = ResetManager.get_reset_parameters(ResetType.SOFT, base_config)
        assert params['initial_variance_multiplier'] == 2
        assert params['weight_noise_multiplier'] == 5
        assert params['adaptation_measurements'] == 15
        assert params['adaptation_days'] == 10
        assert params['quality_acceptance_threshold'] == 0.45

    def test_parameters_from_config(self):
        """Config overrides defaults correctly."""
        config = {
            'kalman': {
                'reset': {
                    'soft': {
                        'weight_noise_multiplier': 10,
                        'adaptation_days': 5
                    }
                }
            }
        }
        params = ResetManager.get_reset_parameters(ResetType.SOFT, config)
        assert params['weight_noise_multiplier'] == 10  # Overridden
        assert params['adaptation_days'] == 5  # Overridden
        assert params['initial_variance_multiplier'] == 2  # Default


class TestAdaptationBehavior:
    """Test adaptive period detection and factor calculation."""

    def test_is_in_adaptive_period_measurements(self, reset_state):
        """True until measurement threshold."""
        reset_state['measurements_since_reset'] = 5
        is_adaptive, params = ResetManager.is_in_adaptive_period(
            reset_state,
            datetime(2024, 1, 16)  # 1 day after reset
        )
        assert is_adaptive is True
        assert params == reset_state['reset_parameters']

        # After both thresholds (measurements AND days)
        reset_state['measurements_since_reset'] = 20
        is_adaptive, params = ResetManager.is_in_adaptive_period(
            reset_state,
            datetime(2024, 1, 26)  # 11 days after reset (exceeds 10 day threshold)
        )
        assert is_adaptive is False

    def test_is_in_adaptive_period_days(self, reset_state):
        """True until day threshold."""
        reset_state['measurements_since_reset'] = 20  # Exceed measurements

        # Within days threshold
        is_adaptive, params = ResetManager.is_in_adaptive_period(
            reset_state,
            datetime(2024, 1, 24)  # 9 days after reset
        )
        assert is_adaptive is True

        # After days threshold
        is_adaptive, params = ResetManager.is_in_adaptive_period(
            reset_state,
            datetime(2024, 1, 26)  # 11 days after reset
        )
        assert is_adaptive is False

    def test_adaptive_factor_decay_curve(self, reset_state):
        """Factor increases 0->1 with decay rate."""
        # At reset (0 measurements)
        factor = ResetManager.get_adaptive_factor(reset_state, datetime(2024, 1, 15))
        assert factor == pytest.approx(0.0, abs=0.01)

        # After some measurements
        reset_state['measurements_since_reset'] = 4
        factor = ResetManager.get_adaptive_factor(reset_state, datetime(2024, 1, 16))
        assert 0.5 < factor < 0.8

        # Many measurements later
        reset_state['measurements_since_reset'] = 20
        factor = ResetManager.get_adaptive_factor(reset_state, datetime(2024, 1, 25))
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_adaptive_factor_no_reset(self, valid_state):
        """Returns 1.0 without reset state."""
        factor = ResetManager.get_adaptive_factor(valid_state, datetime(2024, 1, 2))
        assert factor == 1.0


class TestStateTransitions:
    """Test state changes during reset operations."""

    def test_perform_reset_initial(self, base_config):
        """Initial reset creates clean state."""
        state = {}
        new_state, reset_event = ResetManager.perform_reset(
            state=state,
            reset_type=ResetType.INITIAL,
            timestamp=datetime(2024, 1, 1),
            weight=75.0,
            source='patient-device',
            config=base_config
        )

        # Check new state
        assert new_state['kalman_params'] is None
        assert new_state['measurements_since_reset'] == 0
        assert new_state['reset_type'] == 'initial'
        assert 'reset_parameters' in new_state
        assert len(new_state['reset_events']) == 1

        # Check reset event
        assert reset_event['type'] == 'initial'
        assert reset_event['weight'] == 75.0
        assert reset_event['reason'] == 'initial_measurement'

    def test_perform_reset_preserves_history(self, valid_state, base_config):
        """Reset events accumulate in history."""
        # Add existing reset event
        valid_state['reset_events'] = [{
            'timestamp': datetime(2023, 12, 1),
            'type': 'initial'
        }]

        new_state, reset_event = ResetManager.perform_reset(
            state=valid_state,
            reset_type=ResetType.HARD,
            timestamp=datetime(2024, 2, 5),
            weight=72.0,
            source='patient-device',
            config=base_config
        )

        # Should have 2 events now
        assert len(new_state['reset_events']) == 2
        assert new_state['reset_events'][0]['type'] == 'initial'
        assert new_state['reset_events'][1]['type'] == 'hard'

    def test_reset_event_structure(self, valid_state, base_config):
        """Event contains all required fields."""
        new_state, reset_event = ResetManager.perform_reset(
            state=valid_state,
            reset_type=ResetType.SOFT,
            timestamp=datetime(2024, 1, 5),
            weight=81.0,
            source='questionnaire',
            config=base_config
        )

        assert 'timestamp' in reset_event
        assert 'type' in reset_event
        assert 'source' in reset_event
        assert 'weight' in reset_event
        assert 'last_weight' in reset_event
        assert 'gap_days' in reset_event
        assert 'reason' in reset_event
        assert 'parameters' in reset_event

        # Check reason generation
        assert 'manual_entry_change_6.0kg' in reset_event['reason']


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_string_timestamp_conversion(self, valid_state, base_config):
        """ISO string timestamps handled correctly."""
        valid_state['last_timestamp'] = '2024-01-01T00:00:00'

        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=75.0,
            timestamp=datetime(2024, 2, 5),
            source='patient-device',
            config=base_config
        )
        assert result == ResetType.HARD

    def test_missing_state_fields(self, base_config):
        """Graceful handling of incomplete state."""
        # State with some but not all fields
        partial_state = {
            'kalman_params': {'weight': [75.0, 0.0]},
            # Missing last_timestamp, last_raw_weight, etc.
        }

        result = ResetManager.should_trigger_reset(
            state=partial_state,
            weight=75.0,
            timestamp=datetime(2024, 1, 1),
            source='patient-device',
            config=base_config
        )
        assert result is None  # No reset triggered

    def test_config_missing_sections(self):
        """Missing config sections use defaults."""
        minimal_config = {}

        # Should still get default parameters
        params = ResetManager.get_reset_parameters(ResetType.INITIAL, minimal_config)
        assert params['initial_variance_multiplier'] == 10
        assert params['adaptation_measurements'] == 20

    def test_custom_trigger_sources(self, valid_state):
        """Config trigger_sources extend defaults."""
        config = {
            'kalman': {
                'reset': {
                    'soft': {
                        'enabled': True,
                        'min_weight_change_kg': 5,
                        'trigger_sources': ['custom-source']
                    }
                }
            }
        }

        # Custom source should trigger
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=81.0,
            timestamp=datetime(2024, 1, 2),
            source='custom-source',
            config=config
        )
        assert result == ResetType.SOFT

        # Default manual sources still work
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=81.0,
            timestamp=datetime(2024, 1, 2),
            source='questionnaire',
            config=config
        )
        assert result == ResetType.SOFT

    def test_negative_gap_days(self, valid_state, base_config):
        """Future timestamps don't crash."""
        # Timestamp before last_timestamp
        result = ResetManager.should_trigger_reset(
            state=valid_state,
            weight=75.0,
            timestamp=datetime(2023, 12, 15),  # Before last_timestamp
            source='patient-device',
            config=base_config
        )
        assert result is None  # No crash, no reset