"""
Kalman State Validator

Validates and recovers Kalman filter states from database storage.
Handles common corruption scenarios and provides detailed error reporting.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np

from src.exceptions import StateValidationError, RecoveryFailedError

logger = logging.getLogger(__name__)


@dataclass
class KalmanStateShape:
    """Expected shapes for Kalman filter state components."""
    x: Tuple[int, int] = (1, 1)  # State mean
    P: Tuple[int, int] = (1, 1)  # State covariance
    F: Tuple[int, int] = (1, 1)  # State transition
    H: Tuple[int, int] = (1, 1)  # Observation model
    Q: Tuple[int, int] = (1, 1)  # Process noise
    R: Tuple[int, int] = (1, 1)  # Observation noise


class KalmanStateValidator:
    """
    Validates and recovers Kalman filter states.

    This class provides comprehensive validation and recovery mechanisms
    for Kalman filter states stored in the database. It can detect common
    corruption patterns and attempt automatic recovery where possible.
    """

    def __init__(self):
        """Initialize the validator with expected shapes and statistics tracking."""
        self.expected_shapes = KalmanStateShape()
        self.validation_stats = {
            'total': 0,
            'passed': 0,
            'recovered': 0,
            'failed': 0,
            'shape_mismatches': {},
            'value_errors': {},
            'recovery_types': {}
        }

    def validate_shape(self, array: np.ndarray, expected: Tuple[int, int],
                      component_name: str) -> bool:
        """
        Validate array shape matches expected dimensions.

        Args:
            array: NumPy array to validate
            expected: Expected shape tuple
            component_name: Name of the component for logging

        Returns:
            True if shape matches, False otherwise
        """
        if array.shape != expected:
            logger.warning(
                f"Shape mismatch for {component_name}: "
                f"expected {expected}, got {array.shape}"
            )
            # Track shape mismatches for monitoring
            self.validation_stats['shape_mismatches'][component_name] = \
                self.validation_stats['shape_mismatches'].get(component_name, 0) + 1
            return False
        return True

    def validate_values(self, array: np.ndarray, component_name: str) -> bool:
        """
        Check for NaN, Inf, and other invalid values.

        Args:
            array: NumPy array to validate
            component_name: Name of the component for logging

        Returns:
            True if values are valid, False otherwise
        """
        if np.any(np.isnan(array)):
            logger.error(f"NaN detected in {component_name}")
            self.validation_stats['value_errors'][f"{component_name}_nan"] = \
                self.validation_stats['value_errors'].get(f"{component_name}_nan", 0) + 1
            return False

        if np.any(np.isinf(array)):
            logger.error(f"Inf detected in {component_name}")
            self.validation_stats['value_errors'][f"{component_name}_inf"] = \
                self.validation_stats['value_errors'].get(f"{component_name}_inf", 0) + 1
            return False

        # Additional validation for specific components
        if component_name in ['P', 'Q', 'R']:  # Covariance matrices
            if np.any(array < 0):
                logger.error(f"Negative values in covariance matrix {component_name}")
                self.validation_stats['value_errors'][f"{component_name}_negative"] = \
                    self.validation_stats['value_errors'].get(f"{component_name}_negative", 0) + 1
                return False

        return True

    def attempt_recovery(self, data: Any, expected_shape: Tuple[int, int],
                        component_name: str) -> Optional[np.ndarray]:
        """
        Attempt to recover from common corruption scenarios.

        Args:
            data: Input data (could be array, list, scalar, etc.)
            expected_shape: Expected shape for the component
            component_name: Name of the component for logging

        Returns:
            Recovered array if successful, None otherwise
        """
        # Handle None values early
        if data is None:
            logger.error(f"Cannot recover {component_name}: value is None")
            return None

        try:
            # First ensure we have a numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            original_shape = data.shape

            # Case 1: Scalar value (0-dimensional)
            if data.ndim == 0:
                logger.info(f"Converting scalar {component_name} to array")
                self.validation_stats['recovery_types']['scalar_conversion'] = \
                    self.validation_stats['recovery_types'].get('scalar_conversion', 0) + 1
                return np.array([[data.item()]])

            # Case 2: Flattened array that should be 2D
            if data.ndim == 1 and expected_shape == (1, 1) and data.size == 1:
                logger.info(f"Recovering {component_name} from 1D to 2D")
                self.validation_stats['recovery_types']['flatten_recovery'] = \
                    self.validation_stats['recovery_types'].get('flatten_recovery', 0) + 1
                return data.reshape(1, 1)

            # Case 3: Wrong 2D shape but correct number of elements
            if data.size == np.prod(expected_shape):
                logger.info(f"Reshaping {component_name} from {original_shape} to {expected_shape}")
                self.validation_stats['recovery_types']['reshape'] = \
                    self.validation_stats['recovery_types'].get('reshape', 0) + 1
                return data.reshape(expected_shape)

            # Case 4: Nested lists/arrays (e.g., [[[value]]])
            if data.size == 1:
                logger.info(f"Extracting single value from nested {component_name}")
                self.validation_stats['recovery_types']['nested_extraction'] = \
                    self.validation_stats['recovery_types'].get('nested_extraction', 0) + 1
                return np.array([[data.flat[0]]])

            logger.error(f"Cannot recover {component_name} with shape {original_shape} and size {data.size}")
            return None

        except Exception as e:
            logger.error(f"Recovery failed for {component_name}: {e}")
            return None

    def validate_and_fix(self, state_dict: Dict) -> Optional[Dict]:
        """
        Main validation and recovery method.

        Args:
            state_dict: Dictionary containing Kalman state components

        Returns:
            Validated state dictionary if successful, None if validation fails

        Raises:
            StateValidationError: If critical validation fails
        """
        self.validation_stats['total'] += 1
        validated_state = {}
        all_valid = True
        recovery_performed = False

        # Check for required components
        required_components = ['x', 'P', 'F', 'H', 'Q', 'R']
        missing_components = []

        for component_name in required_components:
            if component_name not in state_dict:
                logger.error(f"Missing required component: {component_name}")
                missing_components.append(component_name)
                all_valid = False

        if missing_components:
            self.validation_stats['failed'] += 1
            raise StateValidationError(f"Missing required components: {missing_components}")

        # Validate and potentially recover each component
        for component_name, expected_shape in vars(self.expected_shapes).items():
            if component_name not in state_dict:
                # Optional components can be missing
                continue

            data = state_dict[component_name]

            # Try to convert to numpy array if needed
            try:
                if not isinstance(data, np.ndarray):
                    array = np.array(data)
                else:
                    array = data
            except Exception as e:
                logger.error(f"Cannot convert {component_name} to array: {e}")
                all_valid = False
                continue

            # First try validation
            if self.validate_shape(array, expected_shape, component_name) and \
               self.validate_values(array, component_name):
                validated_state[component_name] = array
                continue

            # Try recovery if validation failed
            recovered = self.attempt_recovery(data, expected_shape, component_name)

            if recovered is not None and self.validate_values(recovered, component_name):
                validated_state[component_name] = recovered
                recovery_performed = True
                logger.info(f"Successfully recovered {component_name}")
            else:
                all_valid = False
                logger.error(f"Failed to validate/recover {component_name}")

        # Update statistics
        if all_valid:
            if recovery_performed:
                self.validation_stats['recovered'] += 1
            else:
                self.validation_stats['passed'] += 1
            return validated_state
        else:
            self.validation_stats['failed'] += 1
            return None

    def get_validation_metrics(self) -> Dict:
        """
        Get validation metrics for monitoring.

        Returns:
            Dictionary containing validation statistics
        """
        total = self.validation_stats['total']
        if total == 0:
            success_rate = 1.0
        else:
            successful = self.validation_stats['passed'] + self.validation_stats['recovered']
            success_rate = successful / total

        return {
            'total_validations': total,
            'passed': self.validation_stats['passed'],
            'recovered': self.validation_stats['recovered'],
            'failed': self.validation_stats['failed'],
            'success_rate': success_rate,
            'shape_mismatches': dict(self.validation_stats['shape_mismatches']),
            'value_errors': dict(self.validation_stats['value_errors']),
            'recovery_types': dict(self.validation_stats['recovery_types'])
        }

    def reset_stats(self):
        """Reset validation statistics for fresh monitoring period."""
        self.validation_stats = {
            'total': 0,
            'passed': 0,
            'recovered': 0,
            'failed': 0,
            'shape_mismatches': {},
            'value_errors': {},
            'recovery_types': {}
        }