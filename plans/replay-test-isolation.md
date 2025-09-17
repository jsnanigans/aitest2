# Plan: Replay System Isolated Test Suite

## Decision
**Approach**: Create comprehensive isolated test suite using heavy mocking and test factories
**Why**: Enable testing of all edge cases without full system dependencies, ensuring thread safety and failure recovery
**Risk Level**: Low

## Implementation Steps

1. **Test Structure Setup** - Create `tests/replay/` directory structure
   - `test_replay_buffer_isolated.py` - Buffer-specific tests
   - `test_replay_manager_isolated.py` - Manager-specific tests
   - `test_replay_concurrency.py` - Thread safety tests
   - `test_replay_integration.py` - Component integration tests
   - `conftest.py` - Shared fixtures and factories

2. **Mock Factory Creation** - Build `tests/replay/mocks.py`
   - MockDatabase class with in-memory state storage
   - MockProcessor for measurement processing simulation
   - MockOutlierDetector with configurable detection behavior
   - MockKalmanFilter for state management simulation

3. **Data Factory Setup** - Create `tests/replay/factories.py`
   - MeasurementFactory with realistic weight patterns
   - BufferStateFactory for various buffer scenarios
   - UserStateFactory for different user conditions
   - TimeSeriesFactory for temporal test data

4. **Buffer Isolation Tests** - Implement buffer-specific tests
   - Test CRUD operations with various data sizes
   - Verify thread-safe concurrent access
   - Test memory limits and eviction policies
   - Validate time-based and count-based triggers

5. **Manager Isolation Tests** - Implement manager-specific tests
   - Test state backup/restore mechanisms
   - Verify rollback on various failure modes
   - Test trajectory continuity checks
   - Validate atomic operation guarantees

6. **Concurrency Tests** - Thread safety verification
   - Concurrent buffer additions from multiple threads
   - Race condition testing for state transitions
   - Deadlock prevention verification
   - Memory consistency under load

7. **Failure Mode Tests** - Edge case coverage
   - Database connection failures during replay
   - Timeout handling during long operations
   - Corrupted state recovery
   - Out-of-memory conditions

8. **Performance Tests** - System characteristics
   - Buffer throughput benchmarks
   - State restoration latency
   - Memory usage profiling
   - Lock contention analysis

## Files to Change

- `tests/replay/__init__.py` - Package initialization
- `tests/replay/conftest.py` - Pytest fixtures and configuration
- `tests/replay/mocks.py` - Mock implementations
- `tests/replay/factories.py` - Test data factories
- `tests/replay/test_replay_buffer_isolated.py` - Buffer tests
- `tests/replay/test_replay_manager_isolated.py` - Manager tests
- `tests/replay/test_replay_concurrency.py` - Thread safety tests
- `tests/replay/test_replay_failures.py` - Failure scenario tests
- `tests/replay/test_replay_performance.py` - Performance benchmarks

## Acceptance Criteria

- [ ] All tests run without real database/processor dependencies
- [ ] 100% code coverage for ReplayBuffer class
- [ ] 100% code coverage for ReplayManager class
- [ ] Thread safety verified with 100+ concurrent operations
- [ ] All failure modes have explicit test coverage
- [ ] Performance benchmarks establish baselines
- [ ] Mock factories generate realistic test data
- [ ] Tests complete in under 10 seconds total

## Test Categories

### Buffer Unit Tests
```python
# Key test cases:
- test_buffer_creation_and_cleanup
- test_add_measurement_success
- test_buffer_size_limits
- test_time_window_rotation
- test_trigger_conditions
- test_get_ready_buffers
- test_cleanup_old_buffers
- test_concurrent_additions
- test_memory_eviction
```

### Manager Unit Tests
```python
# Key test cases:
- test_state_backup_creation
- test_state_restoration
- test_rollback_on_failure
- test_trajectory_continuity
- test_chronological_replay
- test_timeout_handling
- test_atomic_operations
- test_retry_logic
```

### Concurrency Tests
```python
# Key test cases:
- test_multiple_users_concurrent
- test_buffer_thread_safety
- test_state_race_conditions
- test_deadlock_prevention
- test_memory_consistency
```

### Failure Tests
```python
# Key test cases:
- test_db_failure_during_backup
- test_db_failure_during_restore
- test_processor_failure_during_replay
- test_timeout_during_replay
- test_corrupted_state_handling
- test_oom_conditions
```

## Mock Strategy

### MockDatabase
```python
class MockDatabase:
    def __init__(self):
        self.states = {}
        self.snapshots = defaultdict(list)
        self.fail_on_next_call = False

    def get_state(self, user_id):
        if self.fail_on_next_call:
            raise Exception("Simulated DB failure")
        return self.states.get(user_id)
```

### MeasurementFactory
```python
class MeasurementFactory:
    @staticmethod
    def create_weight_series(
        start_weight=70.0,
        days=7,
        measurements_per_day=3,
        noise_std=0.5
    ):
        # Generate realistic weight patterns
```

## Risks & Mitigations

**Main Risk**: Over-mocking may hide real integration issues
**Mitigation**: Maintain separate integration test suite with real components

**Secondary Risk**: Thread safety tests may be non-deterministic
**Mitigation**: Use deterministic scheduling and multiple test iterations

## Out of Scope
- Full system integration tests (separate test suite)
- UI/API testing
- Load testing with production data volumes
- Cross-platform compatibility testing