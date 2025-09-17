# Plan: Fix Inverted Persistence Logic Bug

## Decision
**Approach**: Audit and fix state persistence logic to ensure state is saved only when appropriate
**Why**: Current logic may have inverted conditions causing unnecessary DB writes and potential data corruption
**Risk Level**: High - affects data integrity and system reliability

## Implementation Steps

### Phase 1: Investigation & Audit (2 hours)

1. **Audit Current Persistence Points** - Search `src/processing/processor.py` for all `db.save_state()` calls
   - Document when each save occurs
   - Identify the conditions that trigger saves
   - Map the data flow from measurement to persistence

2. **Trace Historical Code** - Check git history for `processor.py:143-146`
   - Find when the bug was introduced
   - Determine if it was already fixed
   - Check for related commits that might have addressed it

3. **Analyze Persistence Semantics** - Review `src/database/database.py`
   - Understand what `save_state()` actually persists
   - Check for any update vs insert logic
   - Identify potential data corruption scenarios

### Phase 2: Fix Implementation (1 hour)

4. **Clarify Naming Convention** - Update method names in `processor.py`
   - Rename ambiguous `should_update` to `should_persist_state`
   - Add clear docstrings explaining when persistence should occur
   - Document the difference between state updates and persistence

5. **Fix Persistence Logic** - Modify `processor.py` persistence conditions
   - Ensure state is saved AFTER successful processing
   - Add validation before persistence
   - Implement proper error handling around saves

6. **Add Persistence Guards** - Implement safety checks
   - Verify state validity before persistence
   - Add timestamp checks to prevent stale overwrites
   - Implement optimistic locking if needed

### Phase 3: Testing & Validation (2 hours)

7. **Create Unit Tests** - Add to `tests/test_processor.py`
   - Test persistence occurs after accepted measurements
   - Test no persistence for rejected measurements
   - Test persistence with feature flag disabled
   - Test concurrent update scenarios

8. **Integration Tests** - Create `tests/test_persistence_integration.py`
   - Test full pipeline with real database
   - Verify state recovery after failures
   - Test rollback scenarios

9. **Data Integrity Check** - Script to audit existing data
   - Create `scripts/audit_persistence.py`
   - Check for inconsistent states in database
   - Identify potentially corrupted records

## Files to Change

- `src/processing/processor.py:234-236` - Fix persistence after outlier rejection
- `src/processing/processor.py:432-435` - Fix persistence in early return path
- `src/processing/processor.py:546-548` - Verify main persistence logic
- `src/database/database.py` - Add validation to `save_state()`
- `tests/test_processor.py` - Add persistence-specific test cases
- `scripts/audit_persistence.py` - New file for data integrity check

## Acceptance Criteria

- [ ] State is persisted only after successful measurement processing
- [ ] No state persistence occurs for rejected measurements
- [ ] Feature flag correctly controls persistence behavior
- [ ] All existing tests pass without modification
- [ ] New tests verify correct persistence behavior
- [ ] Audit script identifies any corrupted data
- [ ] Performance impact < 5% (no unnecessary DB writes)

## Risks & Mitigations

**Main Risk**: Existing data may be corrupted from inverted logic
**Mitigation**: Run audit script before deployment, create backup, implement recovery procedure

**Secondary Risk**: Breaking existing state recovery mechanisms
**Mitigation**: Extensive integration testing with real data scenarios

**Performance Risk**: Fixing logic might increase DB writes
**Mitigation**: Implement write batching or caching if needed

## Rollback Procedure

1. Keep database backup before deployment
2. Feature flag to disable new persistence logic
3. Revert code changes if corruption detected
4. Run recovery script to fix corrupted states
5. Monitor error rates and data quality metrics

## Out of Scope

- Complete database schema changes
- Migration to different storage backend
- Implementing event sourcing or CQRS
- Optimizing database performance
- Adding distributed locking mechanisms

## Open Questions for Review

1. Should we implement versioning for state objects?
2. Is optimistic locking needed for concurrent updates?
3. Should we add metrics/monitoring for persistence operations?
4. Do we need a separate audit log for state changes?