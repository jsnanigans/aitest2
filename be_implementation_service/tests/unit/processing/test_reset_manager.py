"""Unit tests for ResetManager.

Tests reset detection logic which is CRITICAL for maintaining Kalman filter accuracy.
Wrong resets corrupt state for days/weeks.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from src.core.processing.kalman import ResetManager, ResetType


class TestResetDetection:
    """Critical tests for reset detection logic."""

    def test_initial_reset_triggers_when_no_kalman_params(self, base_config, base_timestamp):
        """Test INITIAL reset triggers when no Kalman params exist.

        INITIAL reset is the most aggressive adaptation - needed for first measurement.
        This has highest priority.

        Expected behavior:
        - Missing kalman_params triggers INITIAL reset
        - None state triggers INITIAL reset
        - Empty state triggers INITIAL reset
        """
        # Test 1: None state
        reset_type = ResetManager.should_trigger_reset(
            state=None,
            weight=70.0,
            timestamp=base_timestamp,
            source="manual",
            config=base_config,
        )
        assert reset_type == ResetType.INITIAL, "None state should trigger INITIAL reset"

        # Test 2: Empty state with no kalman_params
        state = {
            "measurement_history": [],
            "measurements_since_reset": 0,
        }
        reset_type = ResetManager.should_trigger_reset(
            state=state,
            weight=70.0,
            timestamp=base_timestamp,
            source="manual",
            config=base_config,
        )
        assert reset_type == ResetType.INITIAL, "Missing kalman_params should trigger INITIAL reset"

        # Test 3: State with explicitly None kalman_params
        state = {
            "kalman_params": None,
            "measurement_history": [],
        }
        reset_type = ResetManager.should_trigger_reset(
            state=state,
            weight=70.0,
            timestamp=base_timestamp,
            source="manual",
            config=base_config,
        )
        assert reset_type == ResetType.INITIAL, "None kalman_params should trigger INITIAL reset"

    def test_hard_reset_triggers_after_30_day_gap(self, base_config, base_timestamp):
        """Test HARD reset triggers after 30+ day gap.

        HARD reset threshold: exactly 30 days (configurable).
        This is CRITICAL: long gaps invalidate Kalman predictions.

        Expected behavior:
        - Gap >= 30 days triggers HARD reset
        - Gap < 30 days does not trigger reset
        - Boundary condition: exactly 30.0 days triggers reset
        """
        # Create state with existing Kalman params (so INITIAL doesn't trigger)
        state = {
            "kalman_params": {
                "observation_covariance": 3.49,
                "transition_covariance": [[0.1, 0], [0, 0.01]],
            },
            "last_state": np.array([[70.0, 0.0], [70.0, 0.0]]),
            "last_timestamp": base_timestamp - timedelta(days=31),
            "last_raw_weight": 70.0,
        }

        # Test 1: 31 day gap (should trigger)
        reset_type = ResetManager.should_trigger_reset(
            state=state,
            weight=70.5,
            timestamp=base_timestamp,
            source="manual",
            config=base_config,
        )
        assert reset_type == ResetType.HARD, "31 day gap should trigger HARD reset"

        # Test 2: Exactly 30 day gap (should trigger - boundary)
        state["last_timestamp"] = base_timestamp - timedelta(days=30)
        reset_type = ResetManager.should_trigger_reset(
            state=state,
            weight=70.5,
            timestamp=base_timestamp,
            source="manual",
            config=base_config,
        )
        assert reset_type == ResetType.HARD, "Exactly 30 day gap should trigger HARD reset"

        # Test 3: 29 day gap (should NOT trigger)
        state["last_timestamp"] = base_timestamp - timedelta(days=29)
        reset_type = ResetManager.should_trigger_reset(
            state=state,
            weight=70.5,
            timestamp=base_timestamp,
            source="manual",
            config=base_config,
        )
        assert reset_type is None, "29 day gap should NOT trigger HARD reset"

    def test_soft_reset_triggers_for_manual_entry_with_5kg_change(
        self, base_config, base_timestamp
    ):
        """Test SOFT reset triggers for manual source + >= 5kg change + no recent reset.

        SOFT reset threshold: 5.0kg change (configurable).
        This is CRITICAL: detects scale changes or genuine weight shifts.

        Expected behavior:
        - Manual source + >= 5kg change + > 3 day cooldown triggers SOFT reset
        - Change < 5kg does not trigger
        - Within 3-day cooldown does not trigger
        - Non-manual source does not trigger
        """
        base_state = {
            "kalman_params": {
                "observation_covariance": 3.49,
                "transition_covariance": [[0.1, 0], [0, 0.01]],
            },
            "last_state": np.array([[70.0, 0.0], [70.0, 0.0]]),
            "last_timestamp": base_timestamp - timedelta(days=5),
            "last_raw_weight": 70.0,
            "reset_events": [
                {
                    "timestamp": base_timestamp - timedelta(days=10),
                    "type": "initial",
                }
            ],
        }

        # Test 1: Manual source + 6kg change + outside cooldown (should trigger)
        reset_type = ResetManager.should_trigger_reset(
            state=base_state.copy(),
            weight=76.0,  # 6kg change
            timestamp=base_timestamp,
            source="internal-questionnaire",  # Manual source
            config=base_config,
        )
        assert reset_type == ResetType.SOFT, "Manual source + 6kg change should trigger SOFT reset"

        # Test 2: Manual source + exactly 5kg change (should trigger - boundary)
        reset_type = ResetManager.should_trigger_reset(
            state=base_state.copy(),
            weight=75.0,  # Exactly 5kg change
            timestamp=base_timestamp,
            source="care-team-upload",  # Manual source
            config=base_config,
        )
        assert reset_type == ResetType.SOFT, "Exactly 5kg change should trigger SOFT reset"

        # Test 3: Manual source + 4.9kg change (should NOT trigger)
        reset_type = ResetManager.should_trigger_reset(
            state=base_state.copy(),
            weight=74.9,  # 4.9kg change
            timestamp=base_timestamp,
            source="questionnaire",  # Manual source
            config=base_config,
        )
        assert reset_type is None, "4.9kg change should NOT trigger SOFT reset"

        # Test 4: Within 3-day cooldown (should NOT trigger even with large change)
        state_recent_reset = base_state.copy()
        state_recent_reset["reset_events"] = [
            {
                "timestamp": base_timestamp - timedelta(days=2),  # 2 days ago
                "type": "soft",
            }
        ]
        reset_type = ResetManager.should_trigger_reset(
            state=state_recent_reset,
            weight=76.0,  # 6kg change
            timestamp=base_timestamp,
            source="care-team-upload",
            config=base_config,
        )
        assert reset_type is None, "Should NOT trigger within 3-day cooldown"

    def test_reset_priority_order_initial_hard_soft(self, base_config, base_timestamp):
        """Test reset priority order: INITIAL > HARD > SOFT.

        When multiple reset conditions are met, highest priority wins.
        This is CRITICAL: ensures correct reset parameters are applied.

        Expected behavior:
        - No params + 31 day gap + 6kg change → INITIAL (not HARD or SOFT)
        - With params + 31 day gap + 6kg change → HARD (not SOFT)
        """
        # Test 1: No params + long gap + large change → INITIAL (highest priority)
        state_no_params = {
            "kalman_params": None,  # Triggers INITIAL
            "last_timestamp": base_timestamp - timedelta(days=31),  # Would trigger HARD
            "last_raw_weight": 70.0,  # 6kg change would trigger SOFT
        }
        reset_type = ResetManager.should_trigger_reset(
            state=state_no_params,
            weight=76.0,
            timestamp=base_timestamp,
            source="questionnaire",  # Manual source for SOFT
            config=base_config,
        )
        assert reset_type == ResetType.INITIAL, "INITIAL should have highest priority"

        # Test 2: With params + long gap + large change → HARD (not SOFT)
        state_with_params = {
            "kalman_params": {
                "observation_covariance": 3.49,
                "transition_covariance": [[0.1, 0], [0, 0.01]],
            },
            "last_state": np.array([[70.0, 0.0], [70.0, 0.0]]),
            "last_timestamp": base_timestamp - timedelta(days=31),  # Triggers HARD
            "last_raw_weight": 70.0,  # 6kg change would trigger SOFT
            "reset_events": [],
        }
        reset_type = ResetManager.should_trigger_reset(
            state=state_with_params,
            weight=76.0,
            timestamp=base_timestamp,
            source="questionnaire",  # Manual source for SOFT
            config=base_config,
        )
        assert reset_type == ResetType.HARD, "HARD should have priority over SOFT"

    def test_perform_reset_clears_state_and_applies_parameters(self, base_config, base_timestamp):
        """Test perform_reset clears state and applies correct parameters.

        This is CRITICAL: incomplete reset corrupts all future measurements.

        Expected behavior:
        - kalman_params set to None
        - last_state cleared
        - measurements_since_reset = 0
        - reset_timestamp set
        - reset_parameters applied correctly for each type
        """
        initial_state = {
            "kalman_params": {
                "observation_covariance": 3.49,
                "transition_covariance": [[0.1, 0], [0, 0.01]],
            },
            "last_state": np.array([[70.0, 0.0], [70.0, 0.0]]),
            "last_covariance": np.array([[[0.361, 0.0], [0.0, 0.001]], [[0.361, 0.0], [0.0, 0.001]]]),
            "last_timestamp": base_timestamp - timedelta(days=1),
            "last_raw_weight": 70.0,
            "measurements_since_reset": 5,
            "measurement_history": [{"weight": 70.0, "timestamp": base_timestamp.isoformat()}],
            "reset_events": [],
        }

        # Test INITIAL reset
        new_state, reset_event = ResetManager.perform_reset(
            state=initial_state.copy(),
            reset_type=ResetType.INITIAL,
            timestamp=base_timestamp,
            weight=70.5,
            source="manual",
            config=base_config,
        )

        # Verify state cleared
        assert new_state["kalman_params"] is None, "kalman_params should be None after reset"
        assert new_state["last_state"] is None, "last_state should be None after reset"
        assert new_state["last_covariance"] is None, "last_covariance should be None after reset"
        assert new_state["measurements_since_reset"] == 0, "Counter should reset to 0"
        assert new_state["measurement_history"] == [], "History should be cleared"

        # Verify reset metadata
        assert new_state["reset_type"] == "initial", "Reset type should be recorded"
        assert new_state["reset_timestamp"] == base_timestamp, "Reset timestamp should be set"
        assert "reset_parameters" in new_state, "Reset parameters should be stored"

        # Verify reset event created
        assert reset_event["type"] == "initial", "Reset event should have correct type"
        assert reset_event["timestamp"] == base_timestamp
        assert "parameters" in reset_event, "Reset event should include parameters"

        # Verify reset parameters for INITIAL type
        reset_params = new_state["reset_parameters"]
        assert "initial_variance_multiplier" in reset_params
        assert reset_params["initial_variance_multiplier"] == 10, "INITIAL should have multiplier=10"

        # Test HARD reset parameters
        new_state_hard, _ = ResetManager.perform_reset(
            state=initial_state.copy(),
            reset_type=ResetType.HARD,
            timestamp=base_timestamp,
            weight=70.5,
            source="manual",
            config=base_config,
        )
        assert new_state_hard["reset_parameters"]["initial_variance_multiplier"] == 5, \
            "HARD should have multiplier=5"

        # Test SOFT reset parameters
        new_state_soft, _ = ResetManager.perform_reset(
            state=initial_state.copy(),
            reset_type=ResetType.SOFT,
            timestamp=base_timestamp,
            weight=75.5,
            source="questionnaire",
            config=base_config,
        )
        assert new_state_soft["reset_parameters"]["initial_variance_multiplier"] == 2, \
            "SOFT should have multiplier=2"
