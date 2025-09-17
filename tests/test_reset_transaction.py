"""
Tests for transactional reset operations.
"""

import pytest
import copy
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.processing.reset_transaction import (
    ResetTransaction,
    ResetOperation,
    TransactionCheckpoint,
    atomic_reset
)
from src.processing.state_validator import StateValidator
from src.processing.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError
)


class TestResetTransaction:
    """Test reset transaction management."""

    def test_successful_transaction(self):
        """Test that successful operations commit."""
        with ResetTransaction("user1") as txn:
            # Simulate successful operations
            state = {
                'kalman_params': None,
                'reset_parameters': {
                    'initial_variance_multiplier': 10,
                    'weight_noise_multiplier': 50,
                    'trend_noise_multiplier': 500,
                    'observation_noise_multiplier': 0.3,
                    'adaptation_measurements': 20,
                    'adaptation_days': 21,
                    'adaptation_decay_rate': 1.5,
                    'quality_acceptance_threshold': 0.25
                },
                'measurements_since_reset': 0,
                'reset_type': 'HARD',
                'reset_timestamp': datetime.now()
            }

            txn.save_checkpoint(ResetOperation.KALMAN_RESET, state)
            assert txn.validate_checkpoint(ResetOperation.KALMAN_RESET)
            txn.mark_completed(ResetOperation.KALMAN_RESET)

        assert len(txn.completed_operations) == 1
        assert not txn.is_failed

    def test_failed_validation_triggers_failure(self):
        """Test that validation failure marks transaction as failed."""
        with ResetTransaction("user1") as txn:
            # This will fail validation - missing required fields
            invalid_state = {
                'kalman_params': None,
                'reset_parameters': {
                    'initial_covariance': -1.0,  # Invalid negative value
                },
                'measurements_since_reset': -5  # Invalid negative count
            }

            txn.save_checkpoint(ResetOperation.KALMAN_RESET, invalid_state)
            is_valid = txn.validate_checkpoint(ResetOperation.KALMAN_RESET)

            assert not is_valid
            assert txn.is_failed
            assert txn.failure_reason is not None

    def test_exception_in_context_triggers_rollback(self):
        """Test that exceptions trigger rollback."""
        original_state = {'value': 100}

        try:
            with ResetTransaction("user1") as txn:
                txn.save_original_state(ResetOperation.STATE_UPDATE, original_state)
                txn.save_checkpoint(ResetOperation.STATE_UPDATE, {'value': 200})
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass  # Expected

        # Check we can retrieve original state
        assert txn.get_original_state(ResetOperation.STATE_UPDATE) == original_state

    def test_checkpoint_retrieval(self):
        """Test checkpoint storage and retrieval."""
        txn = ResetTransaction("user1")

        state1 = {'step': 1}
        state2 = {'step': 2}

        txn.save_checkpoint(ResetOperation.STATE_UPDATE, state1)
        txn.save_checkpoint(ResetOperation.KALMAN_RESET, state2)

        # Should get the last checkpoint for each operation
        checkpoint1 = txn._get_last_checkpoint(ResetOperation.STATE_UPDATE)
        checkpoint2 = txn._get_last_checkpoint(ResetOperation.KALMAN_RESET)

        assert checkpoint1.state_snapshot == state1
        assert checkpoint2.state_snapshot == state2

    def test_multiple_operations_in_transaction(self):
        """Test transaction with multiple operations."""
        with ResetTransaction("user1") as txn:
            # Operation 1: State update
            state = {
                'kalman_params': None,
                'reset_parameters': {
                    'initial_variance_multiplier': 10,
                    'weight_noise_multiplier': 50,
                    'trend_noise_multiplier': 500,
                    'observation_noise_multiplier': 0.3,
                    'adaptation_measurements': 20,
                    'adaptation_days': 21,
                    'adaptation_decay_rate': 1.5,
                    'quality_acceptance_threshold': 0.25
                },
                'measurements_since_reset': 0,
                'reset_type': 'HARD',
                'reset_timestamp': datetime.now()
            }

            txn.save_checkpoint(ResetOperation.STATE_UPDATE, state)
            assert txn.validate_checkpoint(ResetOperation.STATE_UPDATE)
            txn.mark_completed(ResetOperation.STATE_UPDATE)

            # Operation 2: Kalman reset
            txn.save_checkpoint(ResetOperation.KALMAN_RESET, state)
            assert txn.validate_checkpoint(ResetOperation.KALMAN_RESET)
            txn.mark_completed(ResetOperation.KALMAN_RESET)

        assert len(txn.completed_operations) == 2
        assert not txn.is_failed

    def test_atomic_reset_context_manager(self):
        """Test the atomic_reset convenience function."""
        with atomic_reset("user1") as txn:
            state = {
                'kalman_params': None,
                'reset_parameters': {
                    'initial_variance_multiplier': 10,
                    'weight_noise_multiplier': 50,
                    'trend_noise_multiplier': 500,
                    'observation_noise_multiplier': 0.3,
                    'adaptation_measurements': 20,
                    'adaptation_days': 21,
                    'adaptation_decay_rate': 1.5,
                    'quality_acceptance_threshold': 0.25
                },
                'measurements_since_reset': 0,
                'reset_type': 'SOFT',
                'reset_timestamp': datetime.now()
            }

            txn.save_checkpoint(ResetOperation.STATE_UPDATE, state)
            assert txn.validate_checkpoint(ResetOperation.STATE_UPDATE)

        # Should complete successfully
        assert not txn.is_failed


class TestStateValidator:
    """Test state validation logic."""

    def test_validate_kalman_state_valid(self):
        """Test validation of valid Kalman state."""
        validator = StateValidator()

        valid_state = {
            'kalman_params': None,  # Should be None after reset
            'reset_parameters': {
                'initial_variance_multiplier': 1.5,
                'weight_noise_multiplier': 20,
                'trend_noise_multiplier': 200,
                'observation_noise_multiplier': 0.5,
                'adaptation_measurements': 20,
                'adaptation_days': 7,
                'adaptation_decay_rate': 2.5,
                'quality_acceptance_threshold': 0.4
            },
            'measurements_since_reset': 0,
            'reset_type': 'HARD',
            'reset_timestamp': datetime.now()
        }

        assert validator.validate(valid_state, ResetOperation.KALMAN_RESET)

    def test_validate_kalman_state_invalid_parameters(self):
        """Test validation catches invalid reset parameters."""
        validator = StateValidator()

        invalid_state = {
            'reset_parameters': {
                'initial_variance_multiplier': -10,  # Invalid negative
                'weight_noise_multiplier': 0,  # Invalid zero
                'trend_noise_multiplier': float('inf'),  # Invalid infinity
                'observation_noise_multiplier': -0.3,  # Invalid negative
                'adaptation_measurements': 20,
                'adaptation_days': 7,
                'adaptation_decay_rate': 0,  # Invalid zero
                'quality_acceptance_threshold': 0.4
            },
            'measurements_since_reset': 0,
            'reset_type': 'HARD',
            'reset_timestamp': datetime.now()
        }

        assert not validator.validate(invalid_state, ResetOperation.KALMAN_RESET)

    def test_validate_weight_value(self):
        """Test weight value validation."""
        validator = StateValidator()

        # Valid weights
        assert validator.validate_weight_value(70.0)
        assert validator.validate_weight_value(150.0)

        # Invalid weights
        assert not validator.validate_weight_value(float('nan'))
        assert not validator.validate_weight_value(float('inf'))
        assert not validator.validate_weight_value(10.0)  # Too low
        assert not validator.validate_weight_value(500.0)  # Too high

    def test_validate_reset_type(self):
        """Test reset type validation."""
        validator = StateValidator()

        assert validator.validate_reset_type('INITIAL')
        assert validator.validate_reset_type('HARD')
        assert validator.validate_reset_type('SOFT')

        assert not validator.validate_reset_type('INVALID')
        assert not validator.validate_reset_type('')


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, name="test")

        def failing_func():
            raise ValueError("Test failure")

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Circuit should be open now
        assert breaker.is_open
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "test")

    def test_circuit_recovery(self):
        """Test circuit recovery in half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            timeout=0.1,  # 100ms timeout
            success_threshold=2,
            name="test"
        )

        def failing_func():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.is_open

        # Wait for timeout
        time.sleep(0.2)

        # Should enter half-open and succeed twice to close
        result1 = breaker.call(lambda: "success1")
        assert result1 == "success1"
        assert breaker.state == CircuitState.HALF_OPEN

        result2 = breaker.call(lambda: "success2")
        assert result2 == "success2"
        assert breaker.is_closed

    def test_circuit_breaker_reopen_on_half_open_failure(self):
        """Test circuit reopens if it fails in half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            timeout=0.1,
            name="test"
        )

        def failing_func():
            raise ValueError("Test failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Wait for timeout
        time.sleep(0.2)

        # Try to recover but fail
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Should be open again
        assert breaker.is_open

    def test_circuit_breaker_status(self):
        """Test circuit breaker status reporting."""
        breaker = CircuitBreaker(failure_threshold=2, name="test")

        status = breaker.get_status()
        assert status['name'] == 'test'
        assert status['state'] == 'closed'
        assert status['failure_count'] == 0

        # Fail once
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test")))
        except ValueError:
            pass

        status = breaker.get_status()
        assert status['failure_count'] == 1
        assert status['state'] == 'closed'

        # Fail again to open
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test")))
        except ValueError:
            pass

        status = breaker.get_status()
        assert status['state'] == 'open'
        assert 'last_error' in status

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=1, name="test")

        # Open the circuit
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test")))
        except ValueError:
            pass

        assert breaker.is_open

        # Manual reset
        breaker.reset()

        assert breaker.is_closed
        assert breaker.failure_count == 0

        # Should work again
        result = breaker.call(lambda: "success")
        assert result == "success"