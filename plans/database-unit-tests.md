# Plan: ProcessorStateDB Unit Tests

## Decision
**Approach**: Comprehensive pytest-based unit tests with fixtures for isolation
**Why**: Database persistence is critical - data corruption leads to system failure
**Risk Level**: High (persistence layer affects all users)

## Implementation Steps
1. **Basic CRUD Tests** - Create `tests/test_database.py` with CRUD operations
2. **Serialization Tests** - Add numpy/datetime serialization test suite
3. **Transaction Tests** - Implement transaction atomicity and rollback tests
4. **Concurrency Tests** - Add multi-user concurrent access scenarios
5. **Storage Tests** - Test disk persistence and recovery paths
6. **Migration Tests** - Verify schema evolution and backward compatibility

## Files to Change
- `tests/test_database.py` - New comprehensive test file (~800 lines)
- `tests/fixtures/test_states.json` - Sample state fixtures for testing
- `tests/conftest.py:15` - Add database fixtures if needed

## Pytest Best Practices Requirements

### Mandatory Implementation Standards
1. **Fixtures** - Use `@pytest.fixture` for shared test data and setup
   - Create reusable fixtures in conftest.py for common configurations
   - Use fixture composition for complex test scenarios
   - Implement proper fixture scoping (function, class, module, session)

2. **Parametrization** - Use `@pytest.mark.parametrize` for data-driven tests
   - Test multiple input combinations with single test methods
   - Include edge cases and boundary values in parameters
   - Use descriptive parameter IDs for clear test output

3. **Markers** - Use `@pytest.mark` for test categorization
   - `@pytest.mark.slow` for long-running tests
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests
   - Define markers in conftest.py with descriptions

4. **Assertions** - Use pytest assertion features
   - `pytest.approx()` for floating-point comparisons
   - Descriptive assertion messages with f-strings
   - Multiple assertions with clear failure context

5. **Test Organization**
   - Group related tests in classes
   - Follow naming convention: `test_<what>_<condition>_<expected>`
   - Use docstrings with Given/When/Then format
   - Keep tests isolated and independent

6. **Mock Usage** - Proper mocking with unittest.mock
   - Mock external dependencies
   - Use `MagicMock` for complex objects
   - Verify mock calls with assertions

7. **Error Testing** - Test error conditions
   - Use `pytest.raises` for exception testing
   - Test error messages and error types
   - Validate error recovery paths

## Acceptance Criteria
- [ ] 100% coverage of ProcessorStateDB public methods
- [ ] All numpy array shapes preserved through serialization
- [ ] Transaction rollback works on any exception
- [ ] Concurrent user updates don't corrupt data
- [ ] Disk persistence survives process crashes
- [ ] State snapshots restore correctly
- [ ] CSV export produces valid output

## Risks & Mitigations
**Main Risk**: File corruption during concurrent writes
**Mitigation**: Use file locks and atomic write operations (write-rename pattern)

**Secondary Risk**: Memory exhaustion with large measurement histories
**Mitigation**: Test buffer limits (30 items) and history pruning (100 snapshots)

## Out of Scope
- Performance benchmarking (separate test suite)
- Integration with Kalman filter (covered in processor tests)
- Real SQLite migration (future enhancement)

---

## Detailed Test Cases

### 1. Basic CRUD Operations
```python
test_create_initial_state()           # Empty state creation
test_save_and_retrieve_state()        # Basic save/get
test_update_existing_state()          # Overwrite state
test_delete_state()                    # Remove user
test_clear_state()                     # Reset to initial
test_get_nonexistent_user()           # None return
test_get_all_users()                   # List all user IDs
test_get_stats()                       # Database statistics
```

### 2. Numpy Serialization
```python
test_serialize_1d_array()              # State vector [weight, trend]
test_serialize_2d_array()              # Covariance matrix
test_serialize_nested_arrays()         # Arrays in dicts
test_preserve_array_shape()            # Shape integrity
test_preserve_array_dtype()            # Float64 preservation
test_serialize_empty_array()           # Edge case
test_serialize_large_array()           # 100x100 matrix
test_round_trip_precision()            # No precision loss
```

### 3. Datetime Handling
```python
test_serialize_datetime()              # ISO format conversion
test_serialize_none_datetime()         # Null timestamps
test_timezone_handling()               # UTC vs local
test_microsecond_precision()           # Sub-second accuracy
```

### 4. Transaction Management
```python
test_transaction_commit()               # Successful transaction
test_transaction_rollback()             # Exception rollback
test_nested_transactions()              # Inner transaction handling
test_transaction_isolation()            # No partial states
test_rollback_preserves_original()     # Complete restoration
test_transaction_with_multiple_users()  # Batch operations
```

### 5. Measurement History Buffer
```python
test_measurement_history_append()       # Add measurements
test_measurement_history_limit()        # 30 item max
test_measurement_history_fifo()         # First in, first out
test_empty_measurement_history()        # Initial state
test_history_serialization()            # Complex nested data
```

### 6. State Snapshots
```python
test_save_snapshot()                    # Create snapshot
test_snapshot_limit()                   # 100 snapshot max
test_get_snapshot_before_date()        # Time-based retrieval
test_restore_from_snapshot()           # State restoration
test_restore_invalid_snapshot()        # Error handling
test_atomic_check_and_restore()        # Transactional restore
test_snapshot_with_no_kalman()         # Skip invalid states
test_snapshot_history_count()          # Count snapshots
```

### 7. Disk Persistence
```python
test_save_to_disk()                    # File creation
test_load_from_disk()                  # File reading
test_persistence_across_instances()    # New instance loads data
test_corrupt_file_handling()           # Graceful degradation
test_partial_write_recovery()          # Incomplete writes
test_disk_full_handling()              # Storage exhaustion
test_concurrent_file_access()          # Multiple processes
```

### 8. Concurrent Access
```python
test_concurrent_user_updates()         # Parallel saves
test_concurrent_transactions()         # Transaction conflicts
test_race_condition_prevention()       # No data races
test_deadlock_prevention()             # No circular waits
```

### 9. Data Validation
```python
test_invalid_state_rejection()         # Type checking
test_missing_field_handling()          # Partial states
test_extra_field_preservation()        # Forward compatibility
test_malformed_json_recovery()         # Corrupt data files
```

### 10. CSV Export
```python
test_export_empty_database()           # No users
test_export_with_users()               # Multiple users
test_export_field_formatting()         # Decimal precision
test_export_missing_fields()           # Handle None values
test_export_creates_directory()        # Path creation
test_export_return_count()             # User count accuracy
```

### 11. Edge Cases
```python
test_extremely_long_user_id()          # 1000+ chars
test_unicode_user_ids()                # Non-ASCII IDs
test_special_chars_in_paths()          # Filesystem safety
test_zero_size_arrays()                # Empty matrices
test_infinity_nan_values()             # Float edge cases
```

### 12. Memory Management
```python
test_memory_cleanup()                  # No memory leaks
test_large_database_handling()         # 10,000+ users
test_state_copy_independence()         # Deep copies returned
```

## Test Fixtures

```python
@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary directory for database files."""
    return tmp_path / "test_db"

@pytest.fixture
def db_instance(temp_db_path):
    """Fresh database instance for each test."""
    return ProcessorStateDB(storage_path=str(temp_db_path))

@pytest.fixture
def sample_state():
    """Valid state with all fields populated."""
    return {
        'last_state': np.array([70.5, 0.1]),
        'last_covariance': np.array([[1.0, 0.0], [0.0, 0.01]]),
        'last_timestamp': datetime.now(),
        'kalman_params': {
            'transition_covariance': [[0.01, 0], [0, 0.001]],
            'observation_covariance': [[1.0]],
            'initial_state_covariance': [[100, 0], [0, 1]]
        },
        'last_source': 'patient-device',
        'last_raw_weight': 71.2,
        'measurement_history': [{'weight': 70.0, 'timestamp': datetime.now()}]
    }
```

## Success Metrics
- All tests pass in < 5 seconds
- No file handles leaked
- No temporary files left behind
- Coverage report shows 100% line coverage
- Mutation testing survives < 10% of mutations