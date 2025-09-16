# Plan: Fix State Restoration Race Condition in Replay Manager

## Decision
**Approach**: Make snapshot operations atomic in database layer with existence check
**Why**: Race condition allows restore_state_from_snapshot to operate on None, causing crashes
**Risk Level**: High (data corruption potential if not fixed properly)

## Implementation Steps

### Step 1: Add Atomic Snapshot Check-and-Restore Method
**File**: `src/database/database.py:310`
- Add new method `check_and_restore_snapshot()` that combines existence check and restore
- Use database transaction to ensure atomicity
- Return detailed result with error codes

### Step 2: Refactor restore_state_from_snapshot for Safety
**File**: `src/database/database.py:282`
- Add explicit None/empty check at method start
- Add type validation for snapshot structure
- Log warning for invalid snapshot attempts
- Return error details in result dictionary

### Step 3: Update Replay Manager to Use Safe Methods
**File**: `src/replay/replay_manager.py:209`
- Replace separate get/restore calls with atomic method
- Add retry logic with exponential backoff for transient failures
- Implement proper error aggregation and reporting

### Step 4: Add Database Transaction Support
**File**: `src/database/database.py:50`
- Add context manager for transactions
- Ensure all snapshot operations use transactions
- Add rollback capability for failed operations

### Step 5: Enhance Error Handling and Logging
**File**: `src/replay/replay_manager.py:220-254`
- Add structured logging with correlation IDs
- Capture partial restore states for debugging
- Add metrics for restore success/failure rates

### Step 6: Add Integration Tests
**File**: `tests/test_replay_manager.py` (new)
- Test race condition scenarios
- Test partial data returns
- Test rollback on failure
- Test concurrent access patterns

## Files to Change

- `src/database/database.py:282-310` - Add atomic check-and-restore, improve validation
- `src/database/database.py:50` - Add transaction context manager
- `src/replay/replay_manager.py:209-254` - Use atomic operations, add retries
- `src/replay/replay_manager.py:89-95` - Update to use new safe methods
- `tests/test_replay_manager.py` - New comprehensive test suite

## Detailed Changes

### Database Layer (database.py)
```python
# Line 282: Enhanced restore method
def restore_state_from_snapshot(self, user_id: str, snapshot: Optional[Dict]) -> Dict[str, Any]:
    """Return detailed result instead of bool"""
    if snapshot is None:
        return {'success': False, 'error': 'Snapshot is None'}
    if not isinstance(snapshot, dict):
        return {'success': False, 'error': f'Invalid snapshot type: {type(snapshot)}'}
    # Validate required fields
    required = ['kalman_state', 'kalman_covariance', 'timestamp']
    missing = [f for f in required if f not in snapshot]
    if missing:
        return {'success': False, 'error': f'Missing fields: {missing}'}
    # Proceed with restoration...

# New method after line 310
def check_and_restore_snapshot(self, user_id: str, before_timestamp: datetime) -> Dict[str, Any]:
    """Atomic check and restore operation"""
    with self.transaction():
        snapshot = self.get_state_snapshot_before(user_id, before_timestamp)
        if not snapshot:
            return {'success': False, 'error': 'No snapshot found', 'timestamp': before_timestamp}
        result = self.restore_state_from_snapshot(user_id, snapshot)
        result['snapshot'] = snapshot
        return result
```

### Replay Manager (replay_manager.py)
```python
# Line 209: Replace _restore_state_to_buffer_start
def _restore_state_to_buffer_start(self, user_id: str, buffer_start_time: datetime) -> Dict[str, Any]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = self.db.check_and_restore_snapshot(user_id, buffer_start_time)
            if result['success']:
                logger.info(f"Restored state for {user_id} to {result['snapshot']['timestamp']}")
                return result
            elif attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            else:
                return result
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                return {'success': False, 'error': f'All restore attempts failed: {e}'}
```

## Acceptance Criteria

- [ ] No crashes when snapshot is None or invalid
- [ ] Atomic operations prevent race conditions
- [ ] Rollback works correctly on partial failures
- [ ] All error paths return structured error information
- [ ] Retry logic handles transient failures
- [ ] Tests pass for concurrent access scenarios
- [ ] Performance impact < 5ms per operation

## Risks & Mitigations

**Main Risk**: Database locking could impact performance
**Mitigation**: Use row-level locks, add monitoring, set transaction timeout to 1s

**Secondary Risk**: Breaking existing code that expects bool return
**Mitigation**: Keep backward compatible wrapper methods, deprecate gradually

## Testing Strategy

1. Unit tests for each new method
2. Integration tests for full replay flow
3. Stress test with concurrent operations
4. Chaos testing with database failures
5. Performance benchmarking before/after

## Rollout Plan

1. Deploy database changes with feature flag
2. Monitor error rates for 24 hours
3. Enable for 10% of replay operations
4. Full rollout after validation

## Out of Scope

- Optimizing snapshot storage format
- Adding snapshot compression
- Implementing snapshot versioning
- Changing underlying database engine