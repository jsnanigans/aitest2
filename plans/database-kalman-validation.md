# Plan: Database Kalman State Validation

## Decision
**Approach**: Add shape validation and explicit error handling to _deserialize method in database.py
**Why**: Prevent silent data corruption and ensure Kalman filter receives valid state arrays
**Risk Level**: High (data corruption in production)

## Implementation Steps

### Phase 1: Add Validation Helper (Priority: CRITICAL)
1. **Create validation method** - Add `_validate_kalman_arrays()` in `database.py:165`
   - Validate `last_state` is shape (n,2) or (2,)
   - Validate `last_covariance` is shape (n,2,2) or (2,2)
   - Validate kalman_params arrays match expected dimensions
   - Return validation result with specific error messages

2. **Update _deserialize method** - Modify `database.py:154-165`
   - Call validation after array reconstruction
   - Raise ValueError with detailed message on validation failure
   - Log all validation failures with context

3. **Add recovery mechanism** - Create `_attempt_recovery()` method
   - Try to reshape arrays if dimensions are wrong but data salvageable
   - Log recovery attempts
   - Return None if unrecoverable (explicit failure, not silent)

### Phase 2: Update Error Handling
4. **Fix get_state method** - Modify `database.py:77-87`
   - Wrap _deserialize in try/catch
   - On validation error, attempt recovery
   - On recovery failure, return clean initial state (not None)
   - Log all recovery attempts with user context

5. **Add state versioning** - Insert at `database.py:136`
   - Add 'state_version' field to serialized state
   - Support migration from v1 (current) to v2 (validated)
   - Enable future format changes

### Phase 3: Add Safety Checks
6. **Update save_state method** - Modify `database.py:89-102`
   - Validate state before saving
   - Reject invalid states with clear error
   - Add pre-save validation hook

7. **Create integrity check utility** - New method `validate_all_states()`
   - Scan all stored states for corruption
   - Report validation failures
   - Attempt batch recovery where possible

## Files to Change
- `src/database/database.py:154-165` - Core validation in _deserialize
- `src/database/database.py:77-87` - Error handling in get_state
- `src/database/database.py:89-102` - Pre-save validation
- `src/database/database.py:136` - Add state versioning
- `tests/test_database_validation.py` - New comprehensive test file

## Validation Logic Details

```python
# Expected shapes for Kalman state arrays
KALMAN_SHAPES = {
    'last_state': [(2,), (None, 2)],  # (2,) or (n, 2)
    'last_covariance': [(2, 2), (None, 2, 2)],  # (2x2) or (n, 2x2)
    'kalman_params': {
        'initial_state_mean': (2,),
        'initial_state_covariance': (2, 2),
        'transition_covariance': (2, 2),
        'observation_covariance': (1, 1)
    }
}
```

## Test Cases Required
1. **Corruption scenarios**:
   - Wrong shape arrays (1D when expecting 2D)
   - Missing dimensions
   - Non-numeric data in arrays
   - Infinity/NaN values
   - Mismatched array sizes

2. **Recovery scenarios**:
   - Reshapeable arrays (flattened but correct size)
   - Partial state (some fields valid)
   - Version migration

3. **Edge cases**:
   - Empty arrays
   - Single measurement states
   - Very large state histories

## Acceptance Criteria
- [ ] No silent failures - all errors logged with context
- [ ] Shape validation catches 100% of dimension mismatches
- [ ] Recovery succeeds for 80%+ of reshapeable arrays
- [ ] Performance impact < 5ms per state load
- [ ] All existing states pass validation or are recovered
- [ ] Comprehensive test coverage (>95%)

## Risks & Mitigations
**Main Risk**: Breaking existing production states during deployment
**Mitigation**:
- Run validation in report-only mode first
- Backup all states before migration
- Implement gradual rollout with feature flag
- Add rollback procedure

**Secondary Risk**: Performance degradation from validation
**Mitigation**:
- Cache validation results
- Validate only on first load
- Use numpy's built-in shape checking

## Rollback Plan
1. Feature flag `ENABLE_STATE_VALIDATION` (default: False)
2. Gradual rollout: 1% → 10% → 50% → 100%
3. Monitor error rates and performance metrics
4. Quick disable via config without code deployment

## Out of Scope
- Changing Kalman filter implementation
- Modifying state storage format (JSON remains)
- Database migration to different backend
- Automatic state repair without logging