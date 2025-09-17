# Plan: Transactional Reset Safety

## Decision
**Approach**: Implement atomic reset operations with validation checkpoints and automatic rollback
**Why**: Current reset operations lack error boundaries, risking data corruption and cascade failures
**Risk Level**: High

## Implementation Steps

### Phase 1: Create Transaction Infrastructure
1. **Create `src/processing/reset_transaction.py`** - New transaction manager
   - Implement `ResetTransaction` context manager for atomic operations
   - Add state snapshot/rollback mechanism
   - Include validation checkpoints after each operation

2. **Create `src/processing/state_validator.py`** - State integrity validator
   - Validate Kalman parameters (P matrix positive definite, Q/R > 0)
   - Check buffer consistency (timestamps ordered, no duplicates)
   - Verify adaptation state bounds (decay rates, measurement counts)

### Phase 2: Refactor Reset Operations
3. **Modify `src/processing/reset_manager.py:175-230`** - Add transaction wrapper
   - Wrap perform_reset in transaction context
   - Add validation after each state mutation
   - Return detailed error information on failure

4. **Modify `src/processing/processor.py:174-186`** - Integrate transaction handling
   - Use ResetTransaction context manager around reset operations
   - Add fallback strategy for reset failures
   - Log all reset attempts with outcomes

### Phase 3: Error Recovery
5. **Create `src/processing/reset_recovery.py`** - Recovery strategies
   - Implement circuit breaker for repeated reset failures
   - Add fallback to last known good state
   - Create manual recovery procedures

6. **Update `src/database.py`** - Add transaction support
   - Implement save_state with transaction semantics
   - Add state versioning for rollback capability
   - Create audit log table for reset events

### Phase 4: Testing
7. **Create `tests/test_reset_transaction.py`** - Transaction tests
   - Test rollback on validation failure
   - Test partial failure scenarios
   - Test concurrent reset attempts

8. **Update `tests/test_reset_manager.py`** - Add failure cases
   - Test corrupted state handling
   - Test recovery procedures
   - Test circuit breaker activation

## Files to Change

- `src/processing/reset_transaction.py` (NEW) - Transaction manager implementation
- `src/processing/state_validator.py` (NEW) - State validation logic
- `src/processing/reset_recovery.py` (NEW) - Recovery procedures
- `src/processing/reset_manager.py:175-230` - Wrap perform_reset in transaction
- `src/processing/processor.py:174-186` - Use transaction context
- `src/database.py:150-200` - Add transaction support to save_state
- `tests/test_reset_transaction.py` (NEW) - Transaction test suite
- `tests/test_reset_manager.py:50+` - Add failure test cases

## Detailed Design

### ResetTransaction Context Manager
```python
class ResetTransaction:
    def __init__(self, state, validator):
        self.original_state = deepcopy(state)
        self.validator = validator
        self.checkpoints = []

    def __enter__(self):
        return self

    def checkpoint(self, state, operation):
        """Validate and save checkpoint"""
        if not self.validator.validate(state):
            raise ResetValidationError(f"{operation} produced invalid state")
        self.checkpoints.append((operation, deepcopy(state)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on any exception
            return self.original_state
```

### State Validation Rules
1. **Kalman State**:
   - P matrix: symmetric, positive definite, reasonable bounds
   - Q, R: positive scalars, within configured limits
   - State estimate: within physiological bounds

2. **Buffer State**:
   - Timestamps: monotonically increasing
   - Measurements: within safety bounds
   - Size: within configured limits

3. **Adaptation State**:
   - Decay factors: 0 < factor <= 1
   - Measurement count: non-negative integer
   - Reset timestamp: valid datetime

### Recovery Procedures
1. **Immediate Recovery**: Rollback to pre-reset state
2. **Delayed Recovery**: Use last known good state from database
3. **Manual Recovery**: Admin-triggered state reconstruction
4. **Circuit Break**: Disable resets after N failures, alert ops

## Acceptance Criteria

- [ ] All reset operations are atomic (succeed completely or rollback)
- [ ] Invalid states are detected before persistence
- [ ] Failed resets don't corrupt existing state
- [ ] Recovery procedures restore service within 1 minute
- [ ] Audit log captures all reset attempts with outcomes
- [ ] Performance impact < 5ms per measurement
- [ ] Zero data loss during rollback scenarios
- [ ] Circuit breaker prevents cascade failures

## Risks & Mitigations

**Main Risk**: Transaction overhead impacts real-time processing
**Mitigation**: Use lightweight copy-on-write for state snapshots, validate async where possible

**Secondary Risk**: Complex recovery logic introduces new bugs
**Mitigation**: Extensive testing with chaos engineering, gradual rollout with feature flags

**Data Risk**: State versioning increases storage requirements
**Mitigation**: Implement retention policy (keep last 10 versions), compress older states

## Out of Scope

- Distributed transaction support (single-node only)
- Cross-user transaction coordination
- Automatic state repair (only detection and rollback)
- Migration of existing corrupted states