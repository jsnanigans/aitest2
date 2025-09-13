# Plan: Improve Reset Logic - Only Reset on True Data Gaps

## Summary
Modify the reset mechanism to only trigger when there are NO measurements (accepted or rejected) during the gap period, rather than resetting based solely on time since last accepted measurement. This prevents unnecessary resets when data exists but was rejected.

## Context
- **Source**: User request with visualization showing inappropriate reset despite rejected data points
- **Assumptions**: 
  - System should maintain continuity when data exists, even if rejected
  - True gaps (no data at all) still warrant reset for fresh start
  - Rejected measurements indicate ongoing monitoring, not absence
- **Constraints**:
  - Must maintain stateless processor architecture
  - Cannot break existing state management patterns
  - Must preserve backward compatibility with existing data

## Requirements

### Functional
- Reset should only trigger when NO measurements exist in gap period
- System must check for both accepted and rejected measurements
- Gap calculation should consider actual data absence, not just accepted data
- Maintain separate thresholds for questionnaire (10 days) and standard (30 days) sources

### Non-functional
- Performance: Checking historical data should not significantly impact processing time
- Reliability: Must handle edge cases (partial data, mixed sources)
- Maintainability: Clear separation between gap detection and reset logic

## Alternatives

### Option A: Store Rejected Measurements in Database
**Approach**: Extend ProcessorStateDB to store rejected measurements alongside state
- **Pros**: 
  - Complete history available for gap checking
  - Can analyze rejection patterns over time
  - Enables future analytics on data quality
- **Cons**: 
  - Requires database schema changes
  - Increases storage requirements
  - More complex state management

### Option B: Track Last Attempt Timestamp in State
**Approach**: Add `last_attempt_timestamp` to state, updated for any measurement attempt
- **Pros**: 
  - Minimal state changes
  - Simple implementation
  - Maintains stateless architecture
- **Cons**: 
  - Loses granularity of attempts within gap
  - Cannot distinguish single vs multiple rejections
  - May miss edge cases with clustered rejections

### Option C: Query Historical Data During Reset Check
**Approach**: Add method to check for any measurements in date range before reset
- **Pros**: 
  - Most accurate - checks actual data
  - No state schema changes needed
  - Can leverage existing data storage
- **Cons**: 
  - Requires access to historical data store
  - Potential performance impact
  - Dependency on external data source

## Recommendation
**Option B: Track Last Attempt Timestamp** with enhancement to also track rejection count.

**Rationale**: 
- Maintains stateless architecture principle
- Minimal changes to existing codebase
- Sufficient for solving the core problem
- Can be enhanced later if needed

## High-Level Design

### Architecture Changes
```
State Structure (Enhanced):
{
  'last_state': [...],
  'last_covariance': [...],
  'last_timestamp': datetime,        # Last accepted measurement
  'last_attempt_timestamp': datetime, # Last measurement attempt (any)
  'last_source': str,
  'rejection_count_since_accept': int,
  'kalman_params': {...}
}
```

### Flow Modifications
1. **On Any Measurement**: Update `last_attempt_timestamp`
2. **On Rejection**: Increment `rejection_count_since_accept`
3. **On Accept**: Reset `rejection_count_since_accept`, update both timestamps
4. **Reset Check**: Use `last_attempt_timestamp` instead of `last_timestamp`

### Affected Components
- `src/processor.py`: WeightProcessor.process_weight() - lines 130-175
- `src/database.py`: ProcessorStateDB state structure
- `src/kalman.py`: KalmanFilterManager state updates
- Tests: All reset-related test files

## Implementation Plan (No Code)

### Step 1: Update State Structure
- Add `last_attempt_timestamp` field to state
- Add `rejection_count_since_accept` field
- Update state initialization in KalmanFilterManager
- Ensure backward compatibility with existing states

### Step 2: Modify Measurement Processing
- Update WeightProcessor.process_weight() to track attempts
- Set `last_attempt_timestamp` for every measurement
- Increment rejection counter on rejections
- Reset counter on successful acceptance

### Step 3: Update Reset Logic
- Change gap calculation to use `last_attempt_timestamp`
- Keep `last_timestamp` for other validation logic
- Maintain questionnaire vs standard threshold logic
- Add logging for reset decisions

### Step 4: Handle Edge Cases
- First measurement (no prior state)
- State migration from old format
- Concurrent measurements (same timestamp)
- Clock skew/time travel scenarios

### Step 5: Update Visualization
- Show attempted measurements differently
- Indicate why reset didn't trigger
- Add visual markers for true gaps vs rejected periods

## Validation & Rollout

### Test Strategy
1. **Unit Tests**:
   - Test gap calculation with attempts
   - Verify rejection counting
   - Test state migration
   - Edge case coverage

2. **Integration Tests**:
   - End-to-end processing with mixed accept/reject
   - Multi-user scenarios
   - Long gap scenarios with/without attempts

3. **Regression Tests**:
   - Ensure existing test suite passes
   - Verify backward compatibility
   - Performance benchmarks

### Manual Verification Checklist
- [ ] Process user 0040872d data - verify no reset with rejected data
- [ ] Create true 30+ day gap - verify reset still triggers
- [ ] Test questionnaire → device transition with rejections
- [ ] Verify visualization shows correct reset points
- [ ] Check state persistence and recovery

### Rollout Plan
1. **Phase 1**: Deploy with feature flag (disabled by default)
2. **Phase 2**: Enable for test users, monitor behavior
3. **Phase 3**: Gradual rollout to all users
4. **Rollback**: Revert to timestamp-only logic if issues

## Risks & Mitigations

### Risk 1: State Migration Issues
- **Impact**: Existing users lose state or get errors
- **Mitigation**: Auto-migrate old state format, add fallback logic
- **Monitoring**: Log state version mismatches

### Risk 2: Performance Degradation
- **Impact**: Slower processing for high-volume users
- **Mitigation**: Optimize state updates, consider caching
- **Monitoring**: Track processing time metrics

### Risk 3: Incorrect Gap Detection
- **Impact**: Inappropriate resets or missed resets
- **Mitigation**: Extensive testing, gradual rollout
- **Monitoring**: Alert on unusual reset patterns

## Acceptance Criteria
- ✅ Reset only triggers when NO data exists in gap period
- ✅ Rejected measurements prevent reset
- ✅ True data gaps still trigger appropriate reset
- ✅ Backward compatible with existing states
- ✅ No performance regression
- ✅ Clear logging of reset decisions

## Out of Scope
- Historical data migration
- Rejection reason analysis
- Advanced gap detection algorithms
- Multi-source correlation
- Predictive reset timing

## Open Questions
1. Should we distinguish between different rejection reasons?
2. How many rejections in a row should trigger a different behavior?
3. Should visualization show rejection density in gaps?
4. Do we need to store rejection history for analytics?

## Review Cycle

### Self-Review Notes
- Considered storing full rejection history but opted for simpler approach
- Tracking last attempt is sufficient for core requirement
- Can enhance with more sophisticated tracking if needed
- Maintains architectural principles of stateless processing

### Revisions
- Added rejection counter to provide more context
- Clarified that both timestamps are needed for different purposes
- Added consideration for state migration path