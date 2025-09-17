# Fix Plan: Cascading Reset Failures
**Priority:** 🔴 CRITICAL
**File:** `src/processing/processor.py:320-335`
**Created:** September 17, 2025

## Problem Statement

The reset operation sequence has no error boundaries, causing cascading failures:
1. **No validation between operations** - Corrupted state propagates
2. **No rollback mechanism** - Partial failures leave inconsistent state
3. **No error recovery** - System continues with corrupted data
4. **Dependencies ignored** - Later operations use potentially bad data

```python
# CURRENT PROBLEMATIC CODE
kalman_state = self.reset_manager.apply_reset(
    kalman_state, reset_type, measurement
)
# No error check here! If this fails partially, we continue anyway

adaptation_state = self.reset_manager.create_adaptation_state(
    reset_type, measurement.timestamp
)
# Uses potentially corrupted kalman_state

buffer_state = self._update_buffer_after_reset(
    buffer_state, kalman_state  # Corrupted state propagates
)
# Now buffer is corrupted too
```

## Solution Design

### Core Concept: Transactional Reset Operations

All reset operations must be atomic - either all succeed or all rollback.

### Phase 1: Transaction Infrastructure (Week 1)

#### 1.1 Reset Transaction Manager
```python
# src/processing/reset_transaction.py
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import copy
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ResetOperation(Enum):
    """Types of operations in a reset transaction"""
    KALMAN_RESET = "kalman_reset"
    ADAPTATION_CREATE = "adaptation_create"
    BUFFER_UPDATE = "buffer_update"
    STATE_PERSIST = "state_persist"

@dataclass
class TransactionCheckpoint:
    """Snapshot of state at a point in transaction"""
    operation: ResetOperation
    timestamp: float
    state_snapshot: Dict[str, Any]
    validation_passed: bool = False

class ResetTransaction:
    """
    Manages atomic reset operations with automatic rollback
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.checkpoints: List[TransactionCheckpoint] = []
        self.original_states = {}
        self.completed_operations = []
        self.failed = False
        self.failure_reason = None

    def __enter__(self):
        """Start transaction - capture initial state"""
        logger.info(f"Starting reset transaction for user {self.user_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction - commit or rollback"""
        if exc_type is not None:
            logger.error(f"Reset transaction failed: {exc_val}")
            self.rollback(str(exc_val))
            return False  # Re-raise exception

        if self.failed:
            self.rollback(self.failure_reason)
            return False

        # All operations succeeded
        self.commit()
        return True

    def save_checkpoint(self, operation: ResetOperation, state: Dict[str, Any]):
        """Save state snapshot after an operation"""
        checkpoint = TransactionCheckpoint(
            operation=operation,
            timestamp=time.time(),
            state_snapshot=copy.deepcopy(state),
            validation_passed=False
        )
        self.checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint saved for {operation.value}")

    def validate_checkpoint(self, operation: ResetOperation) -> bool:
        """Validate the state after an operation"""
        checkpoint = self._get_last_checkpoint(operation)
        if not checkpoint:
            logger.error(f"No checkpoint found for {operation.value}")
            return False

        try:
            validator = StateValidator()
            is_valid = validator.validate(checkpoint.state_snapshot, operation)

            checkpoint.validation_passed = is_valid
            if not is_valid:
                self.failed = True
                self.failure_reason = f"Validation failed for {operation.value}"
                logger.error(self.failure_reason)

            return is_valid

        except Exception as e:
            logger.error(f"Validation error for {operation.value}: {e}")
            self.failed = True
            self.failure_reason = str(e)
            return False

    def mark_completed(self, operation: ResetOperation):
        """Mark an operation as successfully completed"""
        self.completed_operations.append(operation)
        logger.info(f"Operation completed: {operation.value}")

    def rollback(self, reason: str):
        """Rollback all completed operations"""
        logger.warning(f"Rolling back reset transaction: {reason}")

        # Restore original states in reverse order
        for operation in reversed(self.completed_operations):
            try:
                self._rollback_operation(operation)
                logger.info(f"Rolled back {operation.value}")
            except Exception as e:
                logger.error(f"Failed to rollback {operation.value}: {e}")
                # Continue rollback attempt for other operations

        # Clear transaction state
        self.checkpoints.clear()
        self.completed_operations.clear()

    def commit(self):
        """Commit all operations - make permanent"""
        logger.info(f"Committing reset transaction for user {self.user_id}")
        # No action needed - states already updated
        # Could add audit logging here

    def _get_last_checkpoint(self, operation: ResetOperation) -> Optional[TransactionCheckpoint]:
        """Get the most recent checkpoint for an operation"""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.operation == operation:
                return checkpoint
        return None

    def _rollback_operation(self, operation: ResetOperation):
        """Rollback a specific operation"""
        if operation not in self.original_states:
            return

        original_state = self.original_states[operation]
        # Implementation depends on operation type
        # This would call appropriate restore methods
```

#### 1.2 State Validator
```python
# src/processing/state_validator.py
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class StateValidator:
    """Validates state integrity after operations"""

    def validate(self, state: Dict[str, Any], operation: ResetOperation) -> bool:
        """
        Validate state based on operation type
        """
        validators = {
            ResetOperation.KALMAN_RESET: self._validate_kalman_state,
            ResetOperation.ADAPTATION_CREATE: self._validate_adaptation_state,
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

    def _validate_kalman_state(self, kalman_state: Dict) -> bool:
        """Validate Kalman filter state after reset"""
        required_keys = ['x', 'P', 'F', 'H', 'Q', 'R']

        # Check all required keys exist
        for key in required_keys:
            if key not in kalman_state:
                logger.error(f"Missing Kalman state component: {key}")
                return False

        # Validate shapes
        expected_shapes = {
            'x': (1, 1),
            'P': (1, 1),
            'F': (1, 1),
            'H': (1, 1),
            'Q': (1, 1),
            'R': (1, 1)
        }

        for key, expected_shape in expected_shapes.items():
            array = kalman_state[key]
            if not isinstance(array, np.ndarray):
                logger.error(f"{key} is not a numpy array")
                return False

            if array.shape != expected_shape:
                logger.error(f"{key} has wrong shape: {array.shape} != {expected_shape}")
                return False

            # Check for NaN or Inf
            if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                logger.error(f"{key} contains NaN or Inf values")
                return False

        # Validate covariance is positive semi-definite
        P = kalman_state['P']
        if P[0, 0] < 0:
            logger.error("Covariance P is negative")
            return False

        # Validate process noise is positive
        Q = kalman_state['Q']
        if Q[0, 0] <= 0:
            logger.error("Process noise Q is non-positive")
            return False

        return True

    def _validate_adaptation_state(self, adaptation_state: Dict) -> bool:
        """Validate adaptation state after creation"""
        required_keys = ['is_adapting', 'adaptation_start', 'adaptation_factor',
                        'measurements_since_reset', 'reset_type']

        for key in required_keys:
            if key not in adaptation_state:
                logger.error(f"Missing adaptation state key: {key}")
                return False

        # Validate adaptation factor is reasonable
        factor = adaptation_state['adaptation_factor']
        if not (1.0 <= factor <= 100.0):
            logger.error(f"Invalid adaptation factor: {factor}")
            return False

        # Validate measurements count
        count = adaptation_state['measurements_since_reset']
        if count < 0:
            logger.error(f"Invalid measurements count: {count}")
            return False

        return True

    def _validate_buffer_state(self, buffer_state: Dict) -> bool:
        """Validate buffer state after update"""
        if 'buffer' not in buffer_state:
            logger.error("Missing buffer in buffer_state")
            return False

        buffer = buffer_state['buffer']

        # Check buffer is a list
        if not isinstance(buffer, list):
            logger.error("Buffer is not a list")
            return False

        # Validate buffer size constraints
        max_buffer_size = 1000  # Configure as needed
        if len(buffer) > max_buffer_size:
            logger.error(f"Buffer exceeds max size: {len(buffer)}")
            return False

        return True

    def _validate_persisted_state(self, state: Dict) -> bool:
        """Validate state before persistence"""
        # This would validate the complete state object
        return True
```

#### 1.3 Updated Processor with Transactions
```python
# src/processing/processor.py (updated reset section)

def _handle_reset(self, kalman_state: Dict, reset_type: str,
                 measurement: WeightMeasurement, buffer_state: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Handle reset operations with transaction safety

    Returns:
        Tuple of (kalman_state, adaptation_state, buffer_state)
    """
    user_id = measurement.user_id

    # Create transaction context
    with ResetTransaction(user_id) as txn:
        try:
            # Store original states for rollback
            txn.original_states[ResetOperation.KALMAN_RESET] = copy.deepcopy(kalman_state)
            txn.original_states[ResetOperation.BUFFER_UPDATE] = copy.deepcopy(buffer_state)

            # Operation 1: Apply Kalman reset
            logger.info(f"Applying {reset_type} reset for user {user_id}")
            new_kalman_state = self.reset_manager.apply_reset(
                kalman_state, reset_type, measurement
            )

            txn.save_checkpoint(ResetOperation.KALMAN_RESET, new_kalman_state)
            if not txn.validate_checkpoint(ResetOperation.KALMAN_RESET):
                raise ValueError("Kalman reset validation failed")
            txn.mark_completed(ResetOperation.KALMAN_RESET)

            # Operation 2: Create adaptation state
            logger.info(f"Creating adaptation state for {reset_type}")
            adaptation_state = self.reset_manager.create_adaptation_state(
                reset_type, measurement.timestamp
            )

            txn.save_checkpoint(ResetOperation.ADAPTATION_CREATE, adaptation_state)
            if not txn.validate_checkpoint(ResetOperation.ADAPTATION_CREATE):
                raise ValueError("Adaptation state validation failed")
            txn.mark_completed(ResetOperation.ADAPTATION_CREATE)

            # Operation 3: Update buffer
            logger.info("Updating buffer after reset")
            new_buffer_state = self._update_buffer_after_reset(
                buffer_state, new_kalman_state, adaptation_state
            )

            txn.save_checkpoint(ResetOperation.BUFFER_UPDATE, new_buffer_state)
            if not txn.validate_checkpoint(ResetOperation.BUFFER_UPDATE):
                raise ValueError("Buffer update validation failed")
            txn.mark_completed(ResetOperation.BUFFER_UPDATE)

            # All operations succeeded
            return new_kalman_state, adaptation_state, new_buffer_state

        except Exception as e:
            logger.error(f"Reset failed for user {user_id}: {e}")
            # Transaction will automatically rollback
            # Return original states
            return kalman_state, self._get_default_adaptation_state(), buffer_state
```

### Phase 2: Circuit Breaker Pattern (Week 2)

#### 2.1 Circuit Breaker Implementation
```python
# src/processing/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """
    Prevents cascading failures by breaking the circuit after repeated failures
    """

    def __init__(self, failure_threshold: int = 3, timeout: int = 60,
                 success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # Seconds before attempting recovery
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_attempt_time = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitOpenError(
                    f"Circuit open due to {self.failure_count} failures"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True

        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.timeout

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info("Circuit breaker recovered, entering CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0  # Reset on success in CLOSED state

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker recovery failed, reopening circuit")
            self.state = CircuitState.OPEN
            self.success_count = 0

        elif self.failure_count >= self.failure_threshold:
            logger.error(f"Circuit breaker opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN

class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    pass
```

#### 2.2 Apply Circuit Breaker to Reset Operations
```python
# src/processing/processor.py (with circuit breaker)

class Processor:
    def __init__(self):
        # ... existing init ...
        self.reset_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60,
            success_threshold=2
        )

    def _handle_reset_with_breaker(self, kalman_state: Dict, reset_type: str,
                                  measurement: WeightMeasurement, buffer_state: Dict):
        """
        Handle reset with circuit breaker protection
        """
        try:
            return self.reset_circuit_breaker.call(
                self._handle_reset,
                kalman_state, reset_type, measurement, buffer_state
            )
        except CircuitOpenError as e:
            logger.error(f"Reset circuit open: {e}")
            # Use fallback strategy
            return self._fallback_reset_strategy(
                kalman_state, measurement, buffer_state
            )

    def _fallback_reset_strategy(self, kalman_state: Dict,
                                measurement: WeightMeasurement,
                                buffer_state: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Fallback when reset operations are failing repeatedly
        """
        logger.warning(f"Using fallback reset strategy for user {measurement.user_id}")

        # Option 1: Use last known good state from database
        last_good = self.db.get_last_known_good_state(measurement.user_id)
        if last_good:
            return last_good['kalman_state'], last_good['adaptation_state'], buffer_state

        # Option 2: Create minimal safe state
        safe_kalman_state = self._create_safe_kalman_state(measurement.value)
        safe_adaptation_state = self._create_safe_adaptation_state()

        return safe_kalman_state, safe_adaptation_state, buffer_state
```

### Phase 3: Recovery Service (Week 3)

#### 3.1 Reset Recovery Service
```python
# src/processing/reset_recovery.py
class ResetRecoveryService:
    """
    Handles recovery from reset failures
    """

    def __init__(self, database, reset_manager):
        self.db = database
        self.reset_manager = reset_manager
        self.recovery_attempts = {}

    def recover_from_failed_reset(self, user_id: str, measurement: WeightMeasurement,
                                 failed_state: Dict) -> Optional[Dict]:
        """
        Attempt to recover from a failed reset operation
        """
        logger.info(f"Attempting reset recovery for user {user_id}")

        # Track recovery attempts
        if user_id not in self.recovery_attempts:
            self.recovery_attempts[user_id] = []

        self.recovery_attempts[user_id].append({
            'timestamp': datetime.now(),
            'measurement': measurement
        })

        # Strategy 1: Use historical data
        historical_state = self._recover_from_history(user_id, measurement)
        if historical_state:
            return historical_state

        # Strategy 2: Interpolate from nearby measurements
        interpolated_state = self._interpolate_state(user_id, measurement)
        if interpolated_state:
            return interpolated_state

        # Strategy 3: Create conservative new state
        return self._create_conservative_state(measurement)

    def _recover_from_history(self, user_id: str,
                             measurement: WeightMeasurement) -> Optional[Dict]:
        """
        Recover using historical measurements
        """
        # Get recent measurements
        history = self.db.get_recent_measurements(user_id, days=7)

        if len(history) < 3:
            return None

        # Calculate statistics from history
        weights = [m.value for m in history]
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)

        # Create state based on historical data
        kalman_state = {
            'x': np.array([[mean_weight]]),
            'P': np.array([[std_weight ** 2]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }

        adaptation_state = {
            'is_adapting': True,
            'adaptation_start': measurement.timestamp,
            'adaptation_factor': 10.0,
            'measurements_since_reset': 0,
            'reset_type': 'RECOVERY'
        }

        return {
            'kalman_state': kalman_state,
            'adaptation_state': adaptation_state
        }

    def _interpolate_state(self, user_id: str,
                          measurement: WeightMeasurement) -> Optional[Dict]:
        """
        Interpolate state from surrounding measurements
        """
        # Get measurements before and after
        before = self.db.get_measurement_before(user_id, measurement.timestamp)
        after = self.db.get_measurement_after(user_id, measurement.timestamp)

        if not before or not after:
            return None

        # Linear interpolation
        time_ratio = (measurement.timestamp - before.timestamp) / \
                    (after.timestamp - before.timestamp)

        interpolated_weight = before.value + \
                            (after.value - before.value) * time_ratio

        return self._create_state_from_weight(interpolated_weight)

    def _create_conservative_state(self, measurement: WeightMeasurement) -> Dict:
        """
        Create a conservative state with high uncertainty
        """
        return {
            'kalman_state': {
                'x': np.array([[measurement.value]]),
                'P': np.array([[100.0]]),  # High uncertainty
                'F': np.array([[1.0]]),
                'H': np.array([[1.0]]),
                'Q': np.array([[1.0]]),  # High process noise
                'R': np.array([[10.0]])  # High observation noise
            },
            'adaptation_state': {
                'is_adapting': True,
                'adaptation_start': measurement.timestamp,
                'adaptation_factor': 50.0,  # Very high adaptation
                'measurements_since_reset': 0,
                'reset_type': 'CONSERVATIVE'
            }
        }
```

## Testing Strategy

### Unit Tests
```python
# tests/test_reset_transaction.py
import pytest
from src.processing.reset_transaction import ResetTransaction, ResetOperation

class TestResetTransaction:

    def test_successful_transaction(self):
        """Test that successful operations commit"""
        with ResetTransaction("user1") as txn:
            # Simulate successful operations
            txn.save_checkpoint(ResetOperation.KALMAN_RESET, {"x": [[100]]})
            assert txn.validate_checkpoint(ResetOperation.KALMAN_RESET)
            txn.mark_completed(ResetOperation.KALMAN_RESET)

        assert len(txn.completed_operations) == 1

    def test_failed_validation_rollback(self):
        """Test rollback on validation failure"""
        with pytest.raises(ValueError):
            with ResetTransaction("user1") as txn:
                # This will fail validation
                txn.save_checkpoint(ResetOperation.KALMAN_RESET, {"invalid": "state"})
                txn.validate_checkpoint(ResetOperation.KALMAN_RESET)
                # Should trigger rollback

    def test_exception_triggers_rollback(self):
        """Test that exceptions trigger rollback"""
        with pytest.raises(RuntimeError):
            with ResetTransaction("user1") as txn:
                txn.save_checkpoint(ResetOperation.KALMAN_RESET, {"x": [[100]]})
                raise RuntimeError("Simulated failure")
                # Should trigger rollback

# tests/test_circuit_breaker.py
class TestCircuitBreaker:

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        breaker = CircuitBreaker(failure_threshold=3)

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError()))

        # Circuit should be open now
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "test")

    def test_circuit_recovery(self):
        """Test circuit recovery in half-open state"""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0, success_threshold=2)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError()))

        # Wait and succeed twice to close
        time.sleep(0.1)
        breaker.call(lambda: "success1")
        breaker.call(lambda: "success2")

        assert breaker.state == CircuitState.CLOSED
```

## Rollout Plan

### Week 1: Core Implementation
- [ ] Implement ResetTransaction class
- [ ] Implement StateValidator
- [ ] Add circuit breaker
- [ ] Unit tests

### Week 2: Integration
- [ ] Integrate with existing processor
- [ ] Add recovery service
- [ ] Integration tests
- [ ] Performance testing

### Week 3: Deployment
- [ ] Deploy to staging
- [ ] Monitor for 48 hours
- [ ] Fix any issues
- [ ] Production deployment

### Week 4: Monitoring
- [ ] Monitor metrics
- [ ] Tune thresholds
- [ ] Documentation
- [ ] Team training

## Success Metrics

- **Transaction Success Rate:** >99.5%
- **Recovery Success Rate:** >95%
- **Circuit Breaker Triggers:** <1 per day
- **Performance Impact:** <5ms per reset
- **Rollback Success Rate:** 100%

## Rollback Procedure

If issues arise:
1. Disable transaction wrapper via feature flag
2. Revert to original reset code
3. Analyze failure logs
4. Fix issues and redeploy

## Monitoring and Alerts

### Metrics to Track
- Reset success/failure rates
- Transaction rollback frequency
- Circuit breaker state changes
- Recovery attempt rates
- Performance impact

### Alert Thresholds
- Reset failure rate >1%
- Circuit breaker opens >3 times/hour
- Transaction rollback rate >0.5%
- Performance degradation >10ms

---

*Plan created by System Architecture Council*
*Implementation priority: CRITICAL - Begin after data corruption fix*