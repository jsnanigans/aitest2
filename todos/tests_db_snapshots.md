# Unit Tests for Snapshot Improvements

## Overview
Tests for periodic snapshot creation functionality implemented to support replay mechanism.

**Related Files:**
- `weight_values/src/core/database/dynamodb_store.py` - DynamoDB snapshot methods
- `weight_values/src/core/database/database.py` - In-memory snapshot methods
- `weight_values/src/core/processing/processor.py` - Periodic snapshot logic

**Related Plans:**
- `plans/snapshot-mechanism-analysis.md` - Snapshot strategy analysis

---

## Database Layer Tests

### DynamoDB Store Tests (`test_dynamodb_snapshots.py`)

#### Basic Snapshot Queries
- [ ] `test_get_latest_snapshot_returns_most_recent()`
  - Create 3 snapshots at different times
  - Verify get_latest_snapshot() returns the newest one
  - Verify fields: snapshotTime, state, userId

- [ ] `test_get_latest_snapshot_returns_none_when_no_snapshots()`
  - Call get_latest_snapshot() for new user
  - Verify returns None

- [ ] `test_get_snapshot_before_timestamp_returns_correct_snapshot()`
  - Create snapshots at T1, T2, T3
  - Query for snapshot before T2.5
  - Verify returns T2 snapshot

- [ ] `test_get_snapshot_before_timestamp_returns_none_when_no_suitable_snapshot()`
  - Create snapshot at T2
  - Query for snapshot before T1 (earlier than all snapshots)
  - Verify returns None

#### TTL and Retention
- [ ] `test_snapshot_ttl_set_to_10_days()`
  - Create snapshot
  - Verify TTL field is timestamp + 10 days
  - Verify TTL is an integer timestamp

- [ ] `test_old_snapshots_expire_after_ttl()`
  - Integration test (may need local DynamoDB)
  - Create snapshot with short TTL
  - Wait for expiration
  - Verify snapshot no longer queryable

### In-Memory Database Tests (`test_memory_database_snapshots.py`)

#### Snapshot Storage and Retrieval
- [ ] `test_save_state_snapshot_creates_snapshot()`
  - Save state for user
  - Create snapshot
  - Verify snapshot exists in _snapshots dict
  - Verify snapshot has timestamp and state fields

- [ ] `test_get_latest_snapshot_returns_most_recent()`
  - Create 3 snapshots for user
  - Verify get_latest_snapshot() returns newest
  - Verify snapshot data is deep copy (not reference)

- [ ] `test_get_latest_snapshot_returns_none_for_no_snapshots()`
  - Call get_latest_snapshot() for new user
  - Verify returns None

- [ ] `test_get_snapshot_before_timestamp()`
  - Create snapshots at T1, T2, T3
  - Query for snapshot before T2.5
  - Verify returns T2 snapshot state

- [ ] `test_get_snapshot_returns_none_when_timestamp_before_all_snapshots()`
  - Create snapshot at T2
  - Query for snapshot before T1
  - Verify returns None

#### Automatic Cleanup
- [ ] `test_snapshot_list_cleanup_keeps_last_10()`
  - Create 15 snapshots for user over time
  - Verify _snapshots[user_id] has exactly 10 items
  - Verify oldest 5 were removed
  - Verify newest 10 are retained

- [ ] `test_snapshots_remain_sorted_by_timestamp()`
  - Create snapshots out of order
  - Verify list is sorted by timestamp (ascending)

#### State Restoration
- [ ] `test_check_and_restore_snapshot_uses_time_based_lookup()`
  - Create snapshots at T1, T2, T3
  - Call check_and_restore_snapshot(user_id, T2.5)
  - Verify state restored to T2 snapshot
  - Verify success=True in response

- [ ] `test_check_and_restore_snapshot_fails_when_no_suitable_snapshot()`
  - Create snapshot at T2
  - Call check_and_restore_snapshot(user_id, T1)
  - Verify success=False
  - Verify error message indicates no snapshot found

---

## Processor Logic Tests

### Periodic Snapshot Creation (`test_processor_periodic_snapshots.py`)

#### Snapshot Trigger Logic
- [ ] `test_periodic_snapshot_created_after_24_hours()`
  - Process measurement at T0 (creates initial snapshot)
  - Process measurement at T0 + 25 hours
  - Verify new snapshot was created
  - Verify db.save_state_snapshot() called twice

- [ ] `test_periodic_snapshot_not_created_before_24_hours()`
  - Process measurement at T0 (creates initial snapshot)
  - Process measurement at T0 + 12 hours
  - Verify no new snapshot created
  - Verify db.save_state_snapshot() called only once

- [ ] `test_initial_snapshot_created_for_new_user()`
  - Process first measurement for new user
  - Verify snapshot created immediately
  - Verify snapshot has correct timestamp

#### Configuration Handling
- [ ] `test_periodic_snapshot_respects_disabled_flag()`
  - Set config["snapshot"]["periodic_enabled"] = False
  - Process measurements over 48 hours
  - Verify no periodic snapshots created
  - Verify only post-reset snapshots created (if resets occur)

- [ ] `test_periodic_snapshot_respects_custom_interval()`
  - Set config["snapshot"]["interval_hours"] = 12
  - Process measurement at T0 (initial)
  - Process measurement at T0 + 13 hours
  - Verify new snapshot created (12-hour interval)

- [ ] `test_periodic_snapshot_handles_missing_config()`
  - Don't provide snapshot config section
  - Verify defaults to 24-hour interval, enabled=True
  - Verify processing continues normally

#### Error Handling
- [ ] `test_periodic_snapshot_failure_does_not_block_processing()`
  - Mock db.save_state_snapshot() to raise exception
  - Process measurement
  - Verify measurement still processed successfully
  - Verify result returned (snapshot failure logged, not raised)

- [ ] `test_periodic_snapshot_handles_snapshot_without_timestamp()`
  - Create snapshot with missing last_timestamp field
  - Process new measurement
  - Verify new snapshot created (fallback behavior)

#### Integration with Post-Reset Snapshots
- [ ] `test_post_reset_snapshot_and_periodic_snapshot_both_created()`
  - Process measurement that triggers reset
  - Verify post-reset snapshot created
  - Verify periodic snapshot check still runs
  - If 24 hours elapsed, both snapshots should exist

- [ ] `test_periodic_snapshot_continues_after_reset_snapshot()`
  - Create reset snapshot at T0
  - Process measurement at T0 + 25 hours
  - Verify periodic snapshot created (reset doesn't interfere)

---

## Integration Tests

### End-to-End Snapshot Coverage (`test_snapshot_coverage_integration.py`)

- [ ] `test_snapshots_created_over_10_days()`
  - Simulate measurements over 10 days
  - Process 1-2 measurements per day
  - Verify 10 snapshots exist at end
  - Verify snapshots span the full time range

- [ ] `test_replay_finds_periodic_snapshot_before_window()`
  - Process measurements over 5 days (creates periodic snapshots)
  - Trigger replay with 72-hour window
  - Verify pre-window snapshot found
  - Verify replay succeeds

- [ ] `test_snapshot_coverage_improves_replay_success_rate()`
  - Simulate 100 users over 30 days
  - Mix of users with/without resets
  - Trigger replay for all users
  - Verify >95% success rate (vs ~30% without periodic snapshots)

---

## Performance Tests

### Overhead and Efficiency (`test_snapshot_performance.py`)

- [ ] `test_periodic_snapshot_overhead_under_20ms()`
  - Process 1000 measurements for single user
  - Measure time for periodic snapshot checks
  - Verify average overhead < 20ms per measurement

- [ ] `test_snapshot_storage_growth_predictable()`
  - Simulate 10,000 users over 30 days
  - Verify snapshot count = users × 10 (10-day retention)
  - Verify no unbounded growth

---

## Test Utilities and Fixtures

### Shared Fixtures (`conftest.py`)

```python
@pytest.fixture
def mock_db_with_snapshots():
    """In-memory database with sample snapshots."""
    db = ProcessorStateDB()
    # Create user with state
    # Create 3 snapshots at T0, T0+24h, T0+48h
    return db

@pytest.fixture
def snapshot_config():
    """Default snapshot configuration."""
    return {
        "snapshot": {
            "periodic_enabled": True,
            "interval_hours": 24,
            "retention_days": 10,
        }
    }

@pytest.fixture
def disabled_snapshot_config():
    """Configuration with periodic snapshots disabled."""
    return {
        "snapshot": {
            "periodic_enabled": False,
            "interval_hours": 24,
            "retention_days": 10,
        }
    }
```

---

## Coverage Goals

- **Database layer:** >95% coverage for snapshot methods
- **Processor logic:** >90% coverage for _maybe_create_periodic_snapshot()
- **Edge cases:** All error paths tested
- **Integration:** Key replay + snapshot scenarios verified

---

## Test Execution

```bash
# Run all snapshot tests
uv run pytest tests/ -k snapshot -xvs

# Run specific test file
uv run pytest tests/test_processor_periodic_snapshots.py -xvs

# Run with coverage
uv run pytest tests/ -k snapshot --cov=weight_values/src/core/database --cov=weight_values/src/core/processing --cov-report=html
```

---

## Priority Order

### High Priority (Core Functionality)
1. ✅ Database: get_latest_snapshot() tests
2. ✅ Database: get_snapshot(timestamp) tests
3. ✅ Processor: 24-hour interval tests
4. ✅ Processor: initial snapshot tests
5. ✅ Integration: replay finds snapshot

### Medium Priority (Edge Cases)
6. Processor: disabled flag test
7. Processor: error handling tests
8. Database: TTL verification
9. In-memory: cleanup/retention tests

### Low Priority (Nice to Have)
10. Performance tests
11. Long-term integration tests (30-day simulation)

---

## Notes

- **DynamoDB tests** may require local DynamoDB container or mocking
- **Integration tests** should use in-memory database for speed
- **Performance tests** are optional but recommended for production confidence
- **Snapshot format compatibility** should be verified if schema changes

---

## Related Issues

- Snapshot improvements implemented in PR #XXX
- Related to replay mechanism enhancement (see `plans/replay-service-layer-simplification.md`)
- Addresses issue: "Replay fails for users without recent resets"