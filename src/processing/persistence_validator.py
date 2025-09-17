"""
Persistence Validator Module

Validates state before persistence to prevent invalid or corrupted data from being saved.
Provides audit trail and ensures state integrity.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PersistenceValidator:
    """Validates state before persistence operations."""

    # Required fields that must be present in every state
    REQUIRED_FIELDS = {
        'last_state',
        'kalman_params',
        'last_timestamp'
    }

    # Fields that should be numeric if present
    NUMERIC_FIELDS = {
        'measurements_since_reset',
        'last_raw_weight'
    }

    # Maximum reasonable weight in kg
    MAX_WEIGHT_KG = 500
    MIN_WEIGHT_KG = 10

    @classmethod
    def validate_state(cls, state: Dict[str, Any], user_id: str,
                      reason: str = "unknown") -> Tuple[bool, Optional[str]]:
        """
        Validate state before persistence.

        Args:
            state: The state dictionary to validate
            user_id: User identifier for logging
            reason: Reason for persistence (for audit trail)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not state:
            return False, "State is None or empty"

        # Check required fields
        missing_fields = cls.REQUIRED_FIELDS - set(state.keys())
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        # Validate Kalman state structure
        kalman_state = state.get('last_state')
        if not cls._validate_kalman_state(kalman_state):
            return False, "Invalid Kalman state structure"

        # Validate Kalman parameters
        kalman_params = state.get('kalman_params')
        if not cls._validate_kalman_params(kalman_params):
            return False, "Invalid Kalman parameters"

        # Validate timestamp
        last_timestamp = state.get('last_timestamp')
        if not cls._validate_timestamp(last_timestamp):
            return False, "Invalid last_timestamp"

        # Validate numeric fields
        for field in cls.NUMERIC_FIELDS:
            if field in state:
                value = state[field]
                if not isinstance(value, (int, float)):
                    return False, f"Field '{field}' must be numeric, got {type(value)}"

                # Special validation for weight
                if field == 'last_raw_weight':
                    if not cls.MIN_WEIGHT_KG <= value <= cls.MAX_WEIGHT_KG:
                        return False, f"Weight {value} outside valid range [{cls.MIN_WEIGHT_KG}, {cls.MAX_WEIGHT_KG}]"

        # Check for consistency between related fields
        if 'last_accepted_timestamp' in state and 'last_timestamp' in state:
            last_ts = state['last_timestamp']
            accepted_ts = state['last_accepted_timestamp']
            if isinstance(accepted_ts, datetime) and isinstance(last_ts, datetime):
                if accepted_ts > last_ts:
                    return False, "last_accepted_timestamp cannot be after last_timestamp"

        # Log successful validation with audit trail
        logger.debug(f"State validation passed for user {user_id}, reason: {reason}")

        return True, None

    @classmethod
    def _validate_kalman_state(cls, kalman_state: Any) -> bool:
        """Validate Kalman state structure."""
        # Handle numpy arrays or lists
        import numpy as np

        # Allow None for initial states
        if kalman_state is None:
            return True

        try:
            if isinstance(kalman_state, np.ndarray):
                # For numpy arrays, check shape
                if kalman_state.size == 0:
                    return True  # Empty array is OK for initial state

                if kalman_state.ndim == 1:
                    # 1D array [weight, trend]
                    if kalman_state.size < 1:
                        return False
                    weight = float(kalman_state[0])
                elif kalman_state.ndim == 2:
                    # 2D array [[weight], [trend]] or multiple states
                    if kalman_state.shape[0] == 0:
                        return True  # Empty is OK
                    # Get the last state
                    weight = float(kalman_state[-1][0] if kalman_state.shape[1] >= 1 else kalman_state[-1])
                else:
                    return False
            elif isinstance(kalman_state, list):
                if len(kalman_state) == 0:
                    return True  # Empty list is OK

                # Could be [weight, trend] or [[weight], [trend]]
                if len(kalman_state) >= 2 and isinstance(kalman_state[0], list):
                    # Legacy format: list of lists [[weight], [trend]]
                    weight = float(kalman_state[0][0])
                elif len(kalman_state) >= 1:
                    # Simple list [weight, trend]
                    weight = float(kalman_state[0])
                else:
                    return False
            else:
                return False

            # Weight should be reasonable
            if not cls.MIN_WEIGHT_KG <= weight <= cls.MAX_WEIGHT_KG:
                return False

        except (IndexError, TypeError, ValueError):
            return False

        return True

    @classmethod
    def _validate_kalman_params(cls, kalman_params: Any) -> bool:
        """Validate Kalman parameters structure."""
        if not isinstance(kalman_params, dict):
            return False

        # Check required parameter fields
        required_params = {'transition_covariance', 'observation_covariance'}
        if not all(param in kalman_params for param in required_params):
            return False

        # Validate transition covariance (2x2 matrix)
        trans_cov = kalman_params['transition_covariance']
        if not isinstance(trans_cov, list) or len(trans_cov) != 2:
            return False
        for row in trans_cov:
            if not isinstance(row, list) or len(row) != 2:
                return False
            for val in row:
                if not isinstance(val, (int, float)) or val < 0:
                    return False

        # Validate observation covariance (scalar or 1x1 matrix)
        obs_cov = kalman_params['observation_covariance']
        if isinstance(obs_cov, list):
            if len(obs_cov) != 1 or not isinstance(obs_cov[0], list) or len(obs_cov[0]) != 1:
                return False
            if not isinstance(obs_cov[0][0], (int, float)) or obs_cov[0][0] < 0:
                return False
        elif not isinstance(obs_cov, (int, float)) or obs_cov < 0:
            return False

        return True

    @classmethod
    def _validate_timestamp(cls, timestamp: Any) -> bool:
        """Validate timestamp field."""
        if timestamp is None:
            return False

        # Accept either datetime or ISO string
        if isinstance(timestamp, datetime):
            return True

        if isinstance(timestamp, str):
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return True
            except (ValueError, AttributeError):
                return False

        return False

    @classmethod
    def should_persist(cls, state: Dict[str, Any], previous_state: Optional[Dict[str, Any]],
                       user_id: str, reason: str = "processing") -> Tuple[bool, str]:
        """
        Determine if state should be persisted based on changes and validity.

        Args:
            state: Current state
            previous_state: Previous state (if any)
            user_id: User identifier
            reason: Reason for potential persistence

        Returns:
            Tuple of (should_persist, audit_message)
        """
        # First validate the current state
        is_valid, error = cls.validate_state(state, user_id, reason)
        if not is_valid:
            audit_msg = f"State validation failed for user {user_id}: {error}"
            logger.warning(audit_msg)
            return False, audit_msg

        # If no previous state, we should persist
        if previous_state is None:
            audit_msg = f"Initial state persistence for user {user_id}, reason: {reason}"
            logger.info(audit_msg)
            return True, audit_msg

        # Check if there are meaningful changes
        has_changes = cls._has_meaningful_changes(state, previous_state)

        if has_changes:
            audit_msg = f"State has meaningful changes for user {user_id}, reason: {reason}"
            logger.debug(audit_msg)
            return True, audit_msg
        else:
            audit_msg = f"No meaningful state changes for user {user_id}, skipping persistence"
            logger.debug(audit_msg)
            return False, audit_msg

    @classmethod
    def _has_meaningful_changes(cls, state: Dict[str, Any],
                                previous_state: Dict[str, Any]) -> bool:
        """
        Check if there are meaningful changes between states.

        Ignores minor floating point differences and focuses on significant changes.
        """
        # Fields that indicate meaningful changes
        significant_fields = {
            'last_state',
            'last_accepted_timestamp',
            'last_raw_weight',
            'measurements_since_reset',
            'reset_type',
            'last_source'
        }

        for field in significant_fields:
            if field not in state and field not in previous_state:
                continue

            # Field added or removed
            if (field in state) != (field in previous_state):
                return True

            current_val = state.get(field)
            prev_val = previous_state.get(field)

            # Special handling for Kalman state (check for significant weight change)
            if field == 'last_state' and current_val and prev_val:
                try:
                    # Handle numpy arrays and lists
                    import numpy as np
                    if isinstance(current_val, np.ndarray):
                        if len(current_val.shape) == 1:
                            current_weight = current_val[0]
                        else:
                            current_weight = current_val[-1][0]
                    else:
                        current_weight = current_val[0][0]
                    if isinstance(prev_val, np.ndarray):
                        if len(prev_val.shape) == 1:
                            prev_weight = prev_val[0]
                        else:
                            prev_weight = prev_val[-1][0]
                    else:
                        prev_weight = prev_val[0][0]
                    # Consider > 0.01 kg change as significant
                    if abs(current_weight - prev_weight) > 0.01:
                        return True
                except (IndexError, TypeError):
                    return True  # Structure changed

            # For other fields, check for any change
            elif current_val != prev_val:
                return True

        return False

    @classmethod
    def create_audit_log(cls, user_id: str, action: str, state: Dict[str, Any],
                        success: bool, reason: str, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an audit log entry for persistence operations.

        Args:
            user_id: User identifier
            action: Action being performed (e.g., "persist", "skip", "validate")
            state: State being processed
            success: Whether the operation succeeded
            reason: Reason for the operation
            error: Error message if operation failed

        Returns:
            Audit log entry dictionary
        """
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'action': action,
            'success': success,
            'reason': reason
        }

        if error:
            audit_entry['error'] = error

        # Add state summary (avoid logging full state for privacy)
        if state:
            audit_entry['state_summary'] = {
                'has_kalman_state': 'last_state' in state,
                'has_timestamp': 'last_timestamp' in state,
                'measurements_count': state.get('measurements_since_reset', 0)
            }

            # Add weight if present (useful for debugging)
            if 'last_state' in state:
                try:
                    import numpy as np
                    last_state = state['last_state']
                    if isinstance(last_state, np.ndarray):
                        if len(last_state.shape) == 1:
                            weight = last_state[0]
                        else:
                            weight = last_state[-1][0]
                    else:
                        weight = last_state[0][0]
                    audit_entry['state_summary']['current_weight'] = round(weight, 2)
                except (IndexError, TypeError):
                    pass

        logger.info(f"Persistence audit: {audit_entry}")
        return audit_entry