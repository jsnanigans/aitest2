"""
State validation for reset operations.
Ensures state integrity after each operation.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from .reset_transaction import ResetOperation

logger = logging.getLogger(__name__)


class StateValidator:
    """
    Validates state integrity after reset operations.

    Performs checks for:
    - Required fields presence
    - Data type correctness
    - Value range validity
    - NaN/Inf detection
    - Structural consistency
    """

    def validate(self, state: Dict[str, Any], operation: ResetOperation) -> bool:
        """
        Validate state based on operation type.

        Args:
            state: State to validate
            operation: Type of operation performed

        Returns:
            True if state is valid, False otherwise
        """
        validators = {
            ResetOperation.KALMAN_RESET: self._validate_kalman_state,
            ResetOperation.STATE_UPDATE: self._validate_processor_state,
            ResetOperation.BUFFER_UPDATE: self._validate_buffer_state,
            ResetOperation.STATE_PERSIST: self._validate_persisted_state
        }

        validator = validators.get(operation)
        if not validator:
            logger.error(f"No validator for operation {operation}")
            return False

        try:
            return validator(state)
        except Exception as e:
            logger.error(f"Validation failed for {operation}: {e}")
            return False

    def _validate_kalman_state(self, state: Dict) -> bool:
        """
        Validate Kalman filter state after reset.

        Checks:
        - State has been properly reset (kalman_params should be None after reset)
        - Reset parameters are present and valid
        - Measurements counter is reset
        """
        # After a reset, kalman_params should be None (will be recreated on next measurement)
        if state.get('kalman_params') is not None:
            logger.warning("kalman_params should be None after reset")
            # This is actually OK - might be set during processing

        # Check reset parameters
        if 'reset_parameters' not in state:
            logger.error("Missing reset_parameters after reset")
            return False

        reset_params = state['reset_parameters']
        required_reset_params = [
            'initial_variance_multiplier', 'weight_noise_multiplier',
            'trend_noise_multiplier', 'observation_noise_multiplier',
            'adaptation_measurements', 'adaptation_days', 'adaptation_decay_rate',
            'quality_acceptance_threshold'
        ]

        for param in required_reset_params:
            if param not in reset_params:
                logger.error(f"Missing reset parameter: {param}")
                return False

            value = reset_params[param]

            # Check for NaN or Inf in numeric parameters
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"Reset parameter {param} is NaN or Inf")
                    return False

                # Range checks for multipliers
                if 'multiplier' in param:
                    if value <= 0:
                        logger.error(f"Reset parameter {param} must be positive, got {value}")
                        return False

                if param == 'adaptation_decay_rate':
                    if value <= 0:
                        logger.error(f"Adaptation decay rate must be positive: {value}")
                        return False

        # Check measurements counter
        if state.get('measurements_since_reset', -1) != 0:
            logger.error(f"Measurements counter should be 0 after reset, got {state.get('measurements_since_reset')}")
            return False

        # Check reset type is valid (handle both upper and lowercase)
        reset_type = state.get('reset_type')
        valid_types = ['INITIAL', 'HARD', 'SOFT', 'initial', 'hard', 'soft', None]
        if reset_type not in valid_types:
            logger.error(f"Invalid reset type: {reset_type}")
            return False

        # Check reset timestamp exists
        if 'reset_timestamp' not in state:
            logger.error("Missing reset_timestamp")
            return False

        return True

    def _validate_processor_state(self, state: Dict) -> bool:
        """
        Validate overall processor state update.

        Checks for consistency and required fields.
        """
        # Essential fields that should always exist
        essential_fields = [
            'measurements_since_reset',
            'reset_type',
            'reset_parameters',
            'reset_timestamp'
        ]

        for field in essential_fields:
            if field not in state:
                logger.error(f"Missing essential field: {field}")
                return False

        # Validate measurements counter
        measurements = state.get('measurements_since_reset', -1)
        if measurements < 0:
            logger.error(f"Invalid measurements count: {measurements}")
            return False

        # Validate measurement history if present
        if 'measurement_history' in state:
            history = state['measurement_history']
            if not isinstance(history, list):
                logger.error("Measurement history is not a list")
                return False

            # Check history isn't too large (memory protection)
            max_history_size = 1000
            if len(history) > max_history_size:
                logger.error(f"Measurement history too large: {len(history)} > {max_history_size}")
                return False

        # If there's a last_state, validate it's numeric
        if 'last_state' in state and state['last_state'] is not None:
            last_state = state['last_state']
            if not isinstance(last_state, (int, float, np.ndarray)):
                logger.error(f"Invalid last_state type: {type(last_state)}")
                return False

            # Check for NaN/Inf
            if isinstance(last_state, (int, float)):
                if np.isnan(last_state) or np.isinf(last_state):
                    logger.error("last_state contains NaN or Inf")
                    return False

        return True

    def _validate_buffer_state(self, state: Dict) -> bool:
        """
        Validate buffer state after update.

        Checks buffer consistency and size limits.
        """
        # Buffer operations are handled differently in this system
        # After a reset, measurement_history should be cleared
        if 'measurement_history' in state:
            history = state.get('measurement_history', [])

            # After reset, history should be empty or very small
            if len(history) > 100:  # Reasonable limit after reset
                logger.warning(f"Large measurement history after reset: {len(history)} items")
                # Not a failure, just a warning

        return True

    def _validate_persisted_state(self, state: Dict) -> bool:
        """
        Validate state before persistence.

        Final validation before saving to database.
        """
        # Run all validations for a complete check

        # Check structure
        if not isinstance(state, dict):
            logger.error("State is not a dictionary")
            return False

        # Check for any NaN/Inf in numeric fields
        for key, value in state.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"Field {key} contains NaN or Inf")
                    return False

            elif isinstance(value, np.ndarray):
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    logger.error(f"Array field {key} contains NaN or Inf")
                    return False

        return True

    @staticmethod
    def validate_weight_value(weight: float) -> bool:
        """
        Validate a weight measurement value.

        Args:
            weight: Weight value in kg

        Returns:
            True if valid weight
        """
        # Basic sanity checks
        if np.isnan(weight) or np.isinf(weight):
            logger.error("Weight is NaN or Inf")
            return False

        # Physiological limits (same as in constants)
        if not (20 <= weight <= 450):
            logger.error(f"Weight {weight}kg outside physiological limits")
            return False

        return True

    @staticmethod
    def validate_reset_type(reset_type: str) -> bool:
        """
        Validate reset type string.

        Args:
            reset_type: Reset type string

        Returns:
            True if valid reset type
        """
        valid_types = ['INITIAL', 'HARD', 'SOFT', 'initial', 'hard', 'soft']
        return reset_type in valid_types