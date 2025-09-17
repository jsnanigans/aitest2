"""
Unit tests for KalmanStateValidator.

Tests validation, recovery, and error handling for Kalman filter states.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.processing.kalman_state_validator import KalmanStateValidator, KalmanStateShape
from src.exceptions import StateValidationError, RecoveryFailedError


class TestKalmanStateValidator:
    """Test suite for KalmanStateValidator."""

    @pytest.fixture
    def validator(self):
        """Create a fresh validator instance for each test."""
        return KalmanStateValidator()

    @pytest.fixture
    def valid_state(self):
        """Create a valid Kalman state for testing."""
        return {
            'x': np.array([[100.0]]),
            'P': np.array([[1.0]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

    def test_valid_state_passes(self, validator, valid_state):
        """Test that valid state passes validation without modification."""
        result = validator.validate_and_fix(valid_state)

        assert result is not None
        assert validator.validation_stats['passed'] == 1
        assert validator.validation_stats['recovered'] == 0
        assert validator.validation_stats['failed'] == 0

        # Check all components are present and correct
        for key in valid_state:
            assert key in result
            np.testing.assert_array_equal(result[key], valid_state[key])

    def test_flattened_array_recovery(self, validator):
        """Test recovery from flattened (1D) arrays."""
        state = {
            'x': np.array([100.0]),  # Should be [[100.0]]
            'P': np.array([1.0]),
            'F': np.array([1.0]),
            'H': np.array([1.0]),
            'Q': np.array([0.1]),
            'R': np.array([1.0])
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert validator.validation_stats['recovered'] == 1
        assert result['x'].shape == (1, 1)
        assert result['x'][0, 0] == 100.0

    def test_scalar_to_array_conversion(self, validator):
        """Test conversion of scalar values to arrays."""
        state = {
            'x': 100.0,  # Scalar, should be [[100.0]]
            'P': 1.0,
            'F': 1.0,
            'H': 1.0,
            'Q': 0.1,
            'R': 1.0
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert validator.validation_stats['recovered'] == 1
        assert result['x'].shape == (1, 1)
        assert result['x'][0, 0] == 100.0
        assert result['P'][0, 0] == 1.0

    def test_nan_detection(self, validator):
        """Test that NaN values are detected and fail validation."""
        state = {
            'x': np.array([[np.nan]]),
            'P': np.array([[1.0]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        result = validator.validate_and_fix(state)

        assert result is None
        assert validator.validation_stats['failed'] == 1
        assert 'x_nan' in validator.validation_stats['value_errors']

    def test_inf_detection(self, validator):
        """Test that Inf values are detected and fail validation."""
        state = {
            'x': np.array([[100.0]]),
            'P': np.array([[np.inf]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        result = validator.validate_and_fix(state)

        assert result is None
        assert validator.validation_stats['failed'] == 1
        assert 'P_inf' in validator.validation_stats['value_errors']

    def test_negative_covariance_detection(self, validator):
        """Test that negative values in covariance matrices are detected."""
        state = {
            'x': np.array([[100.0]]),
            'P': np.array([[-1.0]]),  # Negative covariance
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        result = validator.validate_and_fix(state)

        assert result is None
        assert validator.validation_stats['failed'] == 1
        assert 'P_negative' in validator.validation_stats['value_errors']

    def test_missing_component_raises_error(self, validator):
        """Test that missing required components raise StateValidationError."""
        state = {
            'x': np.array([[100.0]]),
            'P': np.array([[1.0]]),
            # Missing F, H, Q, R
        }

        with pytest.raises(StateValidationError) as exc_info:
            validator.validate_and_fix(state)

        assert "Missing required components" in str(exc_info.value)
        assert validator.validation_stats['failed'] == 1

    def test_nested_array_extraction(self, validator):
        """Test extraction of values from nested arrays."""
        state = {
            'x': np.array([[[100.0]]]),  # Extra nesting
            'P': np.array([[[1.0]]]),
            'F': np.array([[[1.0]]]),
            'H': np.array([[[1.0]]]),
            'Q': np.array([[[0.1]]]),
            'R': np.array([[[1.0]]]),
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert validator.validation_stats['recovered'] == 1
        assert result['x'].shape == (1, 1)
        assert result['x'][0, 0] == 100.0

    def test_wrong_shape_but_correct_elements(self, validator):
        """Test reshaping arrays with wrong shape but correct number of elements."""
        state = {
            'x': np.array([[100.0, 0.0]]).T,  # (2, 1) instead of (1, 1)
            'P': np.array([[1.0]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        # This should fail because x has 2 elements, not 1
        result = validator.validate_and_fix(state)

        assert result is None  # Cannot recover when element count is wrong
        assert validator.validation_stats['failed'] == 1

    def test_list_input_conversion(self, validator):
        """Test that lists are converted to numpy arrays."""
        state = {
            'x': [[100.0]],  # List instead of array
            'P': [[1.0]],
            'F': [[1.0]],
            'H': [[1.0]],
            'Q': [[0.1]],
            'R': [[1.0]]
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert isinstance(result['x'], np.ndarray)
        assert result['x'].shape == (1, 1)

    def test_mixed_valid_and_recovered(self, validator):
        """Test state with some valid and some recoverable components."""
        state = {
            'x': np.array([[100.0]]),  # Valid
            'P': np.array([1.0]),       # Needs recovery
            'F': 1.0,                   # Needs recovery
            'H': np.array([[1.0]]),     # Valid
            'Q': [0.1],                 # Needs recovery
            'R': np.array([[1.0]])      # Valid
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert validator.validation_stats['recovered'] == 1
        assert all(result[key].shape == (1, 1) for key in result)

    def test_validation_metrics(self, validator, valid_state):
        """Test validation metrics tracking."""
        # Run several validations
        validator.validate_and_fix(valid_state)
        validator.validate_and_fix({'x': np.array([100.0]), 'P': 1.0, 'F': 1.0, 'H': 1.0, 'Q': 0.1, 'R': 1.0})
        validator.validate_and_fix({'x': np.array([[np.nan]]), 'P': 1.0, 'F': 1.0, 'H': 1.0, 'Q': 0.1, 'R': 1.0})

        metrics = validator.get_validation_metrics()

        assert metrics['total_validations'] == 3
        assert metrics['passed'] == 1
        assert metrics['recovered'] == 1
        assert metrics['failed'] == 1
        assert metrics['success_rate'] == 2/3

    def test_reset_stats(self, validator, valid_state):
        """Test statistics reset functionality."""
        # Run some validations
        validator.validate_and_fix(valid_state)
        validator.validate_and_fix({'x': np.array([[np.nan]]), 'P': 1.0, 'F': 1.0, 'H': 1.0, 'Q': 0.1, 'R': 1.0})

        # Reset stats
        validator.reset_stats()

        assert validator.validation_stats['total'] == 0
        assert validator.validation_stats['passed'] == 0
        assert validator.validation_stats['recovered'] == 0
        assert validator.validation_stats['failed'] == 0

    def test_shape_mismatch_tracking(self, validator):
        """Test that shape mismatches are properly tracked."""
        state = {
            'x': np.array([[100.0, 50.0]]),  # Wrong shape (1, 2)
            'P': np.array([[1.0, 0.5], [0.5, 1.0]]),  # Wrong shape (2, 2)
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        result = validator.validate_and_fix(state)

        assert result is None
        assert 'x' in validator.validation_stats['shape_mismatches']
        assert 'P' in validator.validation_stats['shape_mismatches']

    def test_recovery_type_tracking(self, validator):
        """Test that recovery types are properly tracked."""
        # Test scalar conversion
        validator.validate_and_fix({'x': 100.0, 'P': 1.0, 'F': 1.0, 'H': 1.0, 'Q': 0.1, 'R': 1.0})
        assert 'scalar_conversion' in validator.validation_stats['recovery_types']

        # Reset and test flattened recovery
        validator.reset_stats()
        validator.validate_and_fix({'x': np.array([100.0]), 'P': np.array([1.0]),
                                   'F': np.array([1.0]), 'H': np.array([1.0]),
                                   'Q': np.array([0.1]), 'R': np.array([1.0])})
        assert 'flatten_recovery' in validator.validation_stats['recovery_types']

    def test_empty_state_dict(self, validator):
        """Test handling of empty state dictionary."""
        with pytest.raises(StateValidationError):
            validator.validate_and_fix({})

    def test_none_values_in_state(self, validator):
        """Test handling of None values in state."""
        state = {
            'x': None,
            'P': np.array([[1.0]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        result = validator.validate_and_fix(state)

        # None values cannot be recovered, so validation should fail
        assert result is None
        assert validator.validation_stats['failed'] == 1

    def test_complex_recovery_scenario(self, validator):
        """Test complex recovery with multiple issues."""
        state = {
            'x': [100],           # List, needs array conversion and reshape
            'P': np.array(1.0),   # Scalar array, needs reshape
            'F': [[1]],           # Nested list
            'H': 1,               # Scalar
            'Q': [[[0.1]]],       # Triple nested
            'R': np.array([1])    # 1D array
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert validator.validation_stats['recovered'] == 1
        for key in result:
            assert result[key].shape == (1, 1)

    def test_validation_with_logging(self, validator, caplog):
        """Test that appropriate log messages are generated."""
        import logging
        caplog.set_level(logging.INFO)

        state = {
            'x': 100.0,  # Will trigger scalar conversion
            'P': np.array([[1.0]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        result = validator.validate_and_fix(state)

        assert result is not None
        assert "Converting scalar x to array" in caplog.text

    def test_expected_shapes_configuration(self):
        """Test that expected shapes are properly configured."""
        validator = KalmanStateValidator()
        shapes = validator.expected_shapes

        assert shapes.x == (1, 1)
        assert shapes.P == (1, 1)
        assert shapes.F == (1, 1)
        assert shapes.H == (1, 1)
        assert shapes.Q == (1, 1)
        assert shapes.R == (1, 1)

    def test_partial_recovery_failure(self, validator):
        """Test when some components can be recovered but others cannot."""
        state = {
            'x': 100.0,                    # Can recover
            'P': np.array([[np.nan]]),     # Cannot recover - NaN
            'F': 1.0,                       # Can recover
            'H': np.array([[np.inf]]),     # Cannot recover - Inf
            'Q': 0.1,                       # Can recover
            'R': 1.0                        # Can recover
        }

        result = validator.validate_and_fix(state)

        assert result is None  # Should fail overall
        assert validator.validation_stats['failed'] == 1