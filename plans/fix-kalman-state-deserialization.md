# Fix Plan: Kalman State Deserialization Data Corruption
**Priority:** 🔴 CRITICAL
**File:** `src/database/database.py:248-255`
**Created:** September 17, 2025

## Problem Statement

The `_deserialize_kalman_state` method has critical vulnerabilities:
1. **No shape validation** - Arrays could be any dimension
2. **Silent failures** - Returns `None` on error, causing downstream crashes
3. **No data integrity checks** - Accepts NaN, Inf, corrupted values
4. **Generic exception handling** - Hides specific problems

```python
# CURRENT PROBLEMATIC CODE
def _deserialize_kalman_state(self, state_json: str) -> Optional[Dict]:
    try:
        state_dict = json.loads(state_json)
        kalman_state = {
            'x': np.array(state_dict['x']),  # Could be any shape!
            'P': np.array(state_dict['P']),
            'F': np.array(state_dict['F']),
            # No validation whatsoever
        }
        return kalman_state
    except Exception as e:
        logger.error(f"Failed to deserialize: {e}")
        return None  # SILENT FAILURE!
```

## Solution Design

### Phase 1: Core Validation (CRITICAL - Week 1)

#### 1.1 Create KalmanStateValidator Class
```python
# src/processing/kalman_state_validator.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class KalmanStateShape:
    """Expected shapes for Kalman filter state components"""
    x: Tuple[int, int] = (1, 1)  # State mean
    P: Tuple[int, int] = (1, 1)  # State covariance
    F: Tuple[int, int] = (1, 1)  # State transition
    H: Tuple[int, int] = (1, 1)  # Observation model
    Q: Tuple[int, int] = (1, 1)  # Process noise
    R: Tuple[int, int] = (1, 1)  # Observation noise

class KalmanStateValidator:
    """Validates and recovers Kalman filter states"""

    def __init__(self):
        self.expected_shapes = KalmanStateShape()
        self.validation_stats = {
            'total': 0,
            'passed': 0,
            'recovered': 0,
            'failed': 0
        }

    def validate_shape(self, array: np.ndarray, expected: Tuple[int, int],
                      component_name: str) -> bool:
        """Validate array shape matches expected dimensions"""
        if array.shape != expected:
            logger.warning(
                f"Shape mismatch for {component_name}: "
                f"expected {expected}, got {array.shape}"
            )
            return False
        return True

    def validate_values(self, array: np.ndarray, component_name: str) -> bool:
        """Check for NaN, Inf, and other invalid values"""
        if np.any(np.isnan(array)):
            logger.error(f"NaN detected in {component_name}")
            return False
        if np.any(np.isinf(array)):
            logger.error(f"Inf detected in {component_name}")
            return False
        return True

    def attempt_recovery(self, array: np.ndarray, expected_shape: Tuple[int, int],
                        component_name: str) -> Optional[np.ndarray]:
        """Attempt to recover from common corruption scenarios"""
        original_shape = array.shape

        # Case 1: Flattened array that should be 2D
        if array.ndim == 1 and expected_shape == (1, 1):
            logger.info(f"Recovering {component_name} from 1D to 2D")
            return array.reshape(1, 1)

        # Case 2: Wrong 2D shape but correct number of elements
        if array.size == np.prod(expected_shape):
            logger.info(f"Reshaping {component_name} from {original_shape} to {expected_shape}")
            return array.reshape(expected_shape)

        # Case 3: Scalar value
        if array.ndim == 0:
            logger.info(f"Converting scalar {component_name} to array")
            return np.array([[array.item()]])

        logger.error(f"Cannot recover {component_name} with shape {original_shape}")
        return None

    def validate_and_fix(self, state_dict: Dict) -> Dict:
        """Main validation and recovery method"""
        self.validation_stats['total'] += 1
        validated_state = {}
        all_valid = True

        for component_name, expected_shape in vars(self.expected_shapes).items():
            if component_name not in state_dict:
                logger.error(f"Missing component: {component_name}")
                all_valid = False
                continue

            array = np.array(state_dict[component_name])

            # First try validation
            if self.validate_shape(array, expected_shape, component_name) and \
               self.validate_values(array, component_name):
                validated_state[component_name] = array
                continue

            # Try recovery
            recovered = self.attempt_recovery(array, expected_shape, component_name)
            if recovered is not None and self.validate_values(recovered, component_name):
                validated_state[component_name] = recovered
                self.validation_stats['recovered'] += 1
                logger.info(f"Successfully recovered {component_name}")
            else:
                all_valid = False
                logger.error(f"Failed to validate/recover {component_name}")

        if all_valid:
            self.validation_stats['passed'] += 1
        else:
            self.validation_stats['failed'] += 1

        return validated_state if all_valid else None
```

#### 1.2 Update Database Deserialization
```python
# src/database/database.py
class Database:
    def __init__(self):
        self.validator = KalmanStateValidator()
        self.enable_validation = True  # Feature flag

    def _deserialize_kalman_state(self, state_json: str, user_id: str) -> Dict:
        """
        Deserialize Kalman state with validation and recovery

        Raises:
            ValueError: If state cannot be deserialized or validated
            DataCorruptionError: If state is corrupted beyond recovery
        """
        # Parse JSON
        try:
            state_dict = json.loads(state_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed for user {user_id}: {e}")
            raise ValueError(f"Invalid JSON for Kalman state: {e}")

        # Skip validation if disabled (for rollback)
        if not self.enable_validation:
            return self._legacy_deserialize(state_dict)

        # Validate and attempt recovery
        validated_state = self.validator.validate_and_fix(state_dict)

        if validated_state is None:
            # Log detailed error information
            logger.error(
                f"Kalman state validation failed for user {user_id}. "
                f"Stats: {self.validator.validation_stats}"
            )

            # Attempt fallback to last known good state
            last_good = self._get_last_known_good_state(user_id)
            if last_good:
                logger.warning(f"Using last known good state for user {user_id}")
                return last_good

            # No recovery possible
            raise DataCorruptionError(
                f"Kalman state corrupted and unrecoverable for user {user_id}"
            )

        return validated_state
```

### Phase 2: Comprehensive Error Handling (Week 2)

#### 2.1 Custom Exception Classes
```python
# src/exceptions.py
class DataCorruptionError(Exception):
    """Raised when data corruption is detected"""
    pass

class StateValidationError(Exception):
    """Raised when state validation fails"""
    pass

class RecoveryFailedError(Exception):
    """Raised when automatic recovery attempts fail"""
    pass
```

#### 2.2 Recovery Service
```python
# src/database/recovery_service.py
class StateRecoveryService:
    """Handles recovery from corrupted states"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.recovery_attempts = {}

    def create_checkpoint(self, user_id: str, state: Dict) -> str:
        """Create a checkpoint before risky operations"""
        checkpoint_id = f"{user_id}_{int(time.time())}"
        self.db.save_checkpoint(checkpoint_id, state)
        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> Dict:
        """Restore from a checkpoint"""
        return self.db.get_checkpoint(checkpoint_id)

    def find_last_valid_state(self, user_id: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Find the most recent valid state within time window"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        states = self.db.get_states_since(user_id, cutoff_time)
        validator = KalmanStateValidator()

        for state in reversed(states):  # Most recent first
            if validator.validate_and_fix(state):
                return state

        return None

    def repair_state_history(self, user_id: str) -> Dict:
        """Scan and repair historical states"""
        repair_log = {
            'scanned': 0,
            'repaired': 0,
            'failed': 0,
            'details': []
        }

        states = self.db.get_all_states(user_id)
        validator = KalmanStateValidator()

        for state_record in states:
            repair_log['scanned'] += 1

            validated = validator.validate_and_fix(state_record['state'])
            if validated:
                if validated != state_record['state']:
                    # State was repaired
                    self.db.update_state(state_record['id'], validated)
                    repair_log['repaired'] += 1
                    repair_log['details'].append({
                        'id': state_record['id'],
                        'timestamp': state_record['timestamp'],
                        'action': 'repaired'
                    })
            else:
                repair_log['failed'] += 1
                repair_log['details'].append({
                    'id': state_record['id'],
                    'timestamp': state_record['timestamp'],
                    'action': 'failed_validation'
                })

        return repair_log
```

### Phase 3: Monitoring and Alerting (Week 3)

#### 3.1 Validation Metrics
```python
# src/monitoring/validation_metrics.py
class ValidationMetrics:
    """Track validation success/failure rates"""

    def __init__(self):
        self.metrics = {
            'validation_success': 0,
            'validation_recovered': 0,
            'validation_failed': 0,
            'shape_mismatches': {},
            'value_errors': {},
            'recovery_types': {}
        }

    def record_validation(self, result: str, details: Dict = None):
        """Record validation outcome"""
        self.metrics[f'validation_{result}'] += 1

        if details:
            if 'shape_mismatch' in details:
                component = details['component']
                self.metrics['shape_mismatches'][component] = \
                    self.metrics['shape_mismatches'].get(component, 0) + 1

            if 'recovery_type' in details:
                recovery = details['recovery_type']
                self.metrics['recovery_types'][recovery] = \
                    self.metrics['recovery_types'].get(recovery, 0) + 1

    def get_success_rate(self) -> float:
        """Calculate validation success rate"""
        total = sum([
            self.metrics['validation_success'],
            self.metrics['validation_recovered'],
            self.metrics['validation_failed']
        ])
        if total == 0:
            return 1.0

        successful = self.metrics['validation_success'] + self.metrics['validation_recovered']
        return successful / total

    def should_alert(self) -> bool:
        """Check if metrics warrant an alert"""
        success_rate = self.get_success_rate()
        return success_rate < 0.95  # Alert if <95% success
```

## Testing Strategy

### Unit Tests
```python
# tests/test_kalman_state_validator.py
import pytest
import numpy as np
from src.processing.kalman_state_validator import KalmanStateValidator

class TestKalmanStateValidator:

    def test_valid_state_passes(self):
        """Test that valid state passes validation"""
        validator = KalmanStateValidator()
        state = {
            'x': np.array([[100.0]]),
            'P': np.array([[1.0]]),
            'F': np.array([[1.0]]),
            'H': np.array([[1.0]]),
            'Q': np.array([[0.1]]),
            'R': np.array([[1.0]])
        }
        result = validator.validate_and_fix(state)
        assert result is not None

    def test_flattened_array_recovery(self):
        """Test recovery from flattened arrays"""
        validator = KalmanStateValidator()
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
        assert result['x'].shape == (1, 1)

    def test_nan_detection(self):
        """Test that NaN values are detected"""
        validator = KalmanStateValidator()
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

    def test_missing_component(self):
        """Test handling of missing components"""
        validator = KalmanStateValidator()
        state = {
            'x': np.array([[100.0]]),
            'P': np.array([[1.0]]),
            # Missing F, H, Q, R
        }
        result = validator.validate_and_fix(state)
        assert result is None

    def test_scalar_to_array_conversion(self):
        """Test scalar value conversion"""
        validator = KalmanStateValidator()
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
        assert result['x'].shape == (1, 1)
        assert result['x'][0, 0] == 100.0
```

### Integration Tests
```python
# tests/test_database_validation_integration.py
class TestDatabaseValidation:

    def test_corrupt_state_recovery(self, database):
        """Test recovery from corrupted database state"""
        # Insert corrupted state
        corrupt_json = '{"x": [100], "P": "invalid", "F": [1]}'
        database.execute(
            "INSERT INTO states (user_id, state) VALUES (?, ?)",
            ("test_user", corrupt_json)
        )

        # Attempt to load with validation
        with pytest.raises(DataCorruptionError):
            database.get_state("test_user")

    def test_fallback_to_last_good(self, database):
        """Test fallback to last known good state"""
        # Insert good state
        good_state = {
            'x': [[100.0]], 'P': [[1.0]],
            'F': [[1.0]], 'H': [[1.0]],
            'Q': [[0.1]], 'R': [[1.0]]
        }
        database.save_state("test_user", good_state)

        # Insert corrupted state
        database.execute(
            "INSERT INTO states (user_id, state) VALUES (?, ?)",
            ("test_user", '{"x": "corrupt"}')
        )

        # Should fallback to good state
        result = database.get_state_with_recovery("test_user")
        assert result is not None
        assert result['x'][0][0] == 100.0
```

## Rollout Plan

### Week 1: Implementation
- [ ] Create KalmanStateValidator class
- [ ] Add unit tests
- [ ] Update database deserialization
- [ ] Deploy to development environment

### Week 2: Testing & Monitoring
- [ ] Enable feature flag for 5% of users
- [ ] Monitor validation metrics
- [ ] Fix any edge cases discovered
- [ ] Expand to 25% of users

### Week 3: Full Deployment
- [ ] Enable for 50% of users
- [ ] Run repair script on historical data
- [ ] Monitor for 24 hours
- [ ] Full rollout if metrics are good

## Rollback Procedure

If issues arise:
1. Set `enable_validation = False` in feature flag
2. Revert to `_legacy_deserialize` method
3. Investigate failures in logs
4. Fix issues and re-attempt

## Success Metrics

- **Validation Success Rate:** >99%
- **Recovery Success Rate:** >95% of corrupted states
- **Performance Impact:** <5ms per deserialization
- **Error Rate:** <0.1% increase in overall errors

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance degradation | High | Benchmark before deployment, cache validators |
| False positives | Medium | Conservative validation initially, monitor closely |
| Breaking existing data | High | Feature flag, gradual rollout, backup data |
| Recovery failures | Medium | Multiple recovery strategies, fallback options |

## Documentation Updates

- Update API documentation with new exceptions
- Add troubleshooting guide for validation errors
- Document recovery procedures for operations team
- Update development guidelines with validation requirements

## Long-term Improvements

1. **State Versioning:** Add version field to track schema changes
2. **Automatic Migration:** Build migration system for state upgrades
3. **Compression:** Consider protobuf/msgpack for better serialization
4. **Validation Cache:** Cache validation results for repeated states

---

*Plan created by System Architecture Council*
*Implementation priority: CRITICAL - Begin immediately*