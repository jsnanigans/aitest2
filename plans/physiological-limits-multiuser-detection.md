# Plan: Physiological Limits & Multi-User Detection

## Summary
Enhance the weight stream processor to properly handle multi-user scales by implementing graduated physiological limits and optional user profiling, while maintaining the stateless architecture and Kalman filter integrity.

## Context
- Source: Investigation of user `0040872d-333a-4ace-8c5a-b2fcd056e65a`
- Problem: Current 50% change threshold allows impossible weight jumps (40kg→60kg)
- Evidence: 118 impossible changes >5kg/24h, clear multi-user patterns
- Framework guidance: docs/framework-overview-01.md recommends 2-3% daily limits
- Constraints: Must maintain stateless processor, preserve Kalman filter best practices

## Requirements

### Functional
1. Replace percentage-based validation with time-based absolute limits
2. Detect and handle multi-user scale scenarios
3. Maintain backward compatibility with existing state structure
4. Preserve Kalman filter mathematical integrity

### Non-functional
- No performance degradation (O(1) per measurement)
- Maintain stateless architecture
- Keep state size minimal (~1KB per user)
- Clear separation between validation and filtering

## Alternatives

### Option A: Minimal Change - Physiological Limits Only
- **Approach**: Replace line 208 validation with graduated limits
- **Pros**: 
  - Simple, focused change
  - No state structure changes
  - Easy to test and validate
- **Cons**: 
  - Doesn't address multi-user root cause
  - May over-reject in shared scale scenarios
- **Risk**: Low

### Option B: Full Multi-User Detection with Separate States
- **Approach**: Detect users via clustering, maintain multiple Kalman states
- **Pros**: 
  - Properly handles shared scales
  - Better accuracy per user
- **Cons**: 
  - Complex state management
  - Breaks current state structure
  - Requires user identification logic
- **Risk**: High - significant architecture change

### Option C: Physiological Limits + Session Grouping (Recommended)
- **Approach**: Add time-based limits + detect rapid measurement sessions
- **Pros**: 
  - Addresses both issues
  - Minimal state changes
  - Backward compatible
  - Aligns with framework document recommendations
- **Cons**: 
  - Slightly more complex than Option A
- **Risk**: Low-Medium

## Recommendation
**Option C** - Implement physiological limits with session-based grouping. This provides immediate improvement without breaking the architecture, and sets foundation for future multi-user enhancements.

## High-Level Design

### Architecture Overview
```
WeightProcessor (stateless)
    ├── _validate_weight() [UPDATED]
    │   ├── Basic bounds check (30-400kg)
    │   └── Physiological plausibility check [NEW]
    │       └── Time-based graduated limits
    ├── _detect_session() [NEW]
    │   └── Group measurements <5min apart
    └── process_weight()
        ├── Session detection
        ├── Validation with context
        └── Kalman update (unchanged)

ProcessorDatabase (state)
    └── State Structure [MINIMAL CHANGE]
        ├── existing fields...
        └── last_session_time [NEW - optional]
```

### Data Flow
1. Measurement arrives → Check session boundary
2. Apply physiological validation with time context
3. If valid → Update Kalman (unchanged)
4. Store minimal session info for next measurement

## Implementation Plan (No Code)

### Phase 1: Core Physiological Limits
1. **Update `_validate_weight()` in processor.py**
   - Add time_delta parameter
   - Implement graduated limits logic
   - Return validation result with reason

2. **Modify `process_weight()` flow**
   - Calculate time since last measurement
   - Pass time_delta to validation
   - Log rejection reasons for debugging

3. **Update config.toml**
   - Add `[physiological]` section
   - Define configurable limits
   - Default to safe medical values

### Phase 2: Session Detection
1. **Add `_detect_measurement_session()`**
   - Check if within 5 minutes of last
   - Mark as same/new session
   - Track session statistics

2. **Enhance validation context**
   - Detect high-variance sessions
   - Flag potential multi-user scenarios
   - Add to rejection reasoning

### Phase 3: State Migration
1. **Extend state structure (backward compatible)**
   - Add optional `session_info` field
   - Preserve all existing fields
   - Handle missing fields gracefully

2. **Update state persistence**
   - Version state structure
   - Auto-migrate old states
   - No breaking changes

## Validation & Rollout

### Test Strategy
1. **Unit Tests** (tests/test_physiological_limits.py)
   - All limit boundaries
   - Edge cases (first measurement, long gaps)
   - Session detection logic

2. **Integration Tests**
   - Process known multi-user data
   - Verify Kalman stability
   - Check state compatibility

3. **Regression Tests**
   - Run on existing test users
   - Compare acceptance rates
   - Verify no single-user degradation

### Manual QA Checklist
- [ ] Process user `0040872d-333a-4ace-8c5a-b2fcd056e65a`
- [ ] Verify 40kg→75kg jumps rejected
- [ ] Confirm 2kg bathroom changes accepted
- [ ] Check Kalman convergence unchanged
- [ ] Test with single-user data (no regression)

### Rollout Plan
1. **Phase 1**: Deploy physiological limits (low risk)
   - Monitor rejection rates
   - Collect validation reasons
   - Tune limits if needed

2. **Phase 2**: Enable session detection (optional flag)
   - Test on known multi-user accounts
   - Gather session statistics
   - Refine detection thresholds

3. **Phase 3**: Full deployment
   - Enable for all users
   - Monitor performance metrics
   - Document new behavior

## Risks & Mitigations

### Risk 1: Over-rejection of Valid Data
- **Mitigation**: Conservative initial limits, monitoring, config tuning
- **Monitoring**: Track rejection rates by reason

### Risk 2: Kalman Filter Instability
- **Mitigation**: Validation happens before Kalman, filter unchanged
- **Monitoring**: Track innovation/confidence metrics

### Risk 3: State Compatibility Issues
- **Mitigation**: Backward compatible changes only, version field
- **Recovery**: State migration logic, fallback handling

## Acceptance Criteria
1. ✅ Impossible changes (>5kg/hour) consistently rejected
2. ✅ Normal changes (2kg meals, 0.5kg/day) accepted
3. ✅ Kalman filter behavior unchanged for valid data
4. ✅ State remains backward compatible
5. ✅ No performance degradation
6. ✅ Clear rejection reasoning in logs

## Out of Scope
- Full multi-user profile management
- User identification/authentication
- Historical data reprocessing
- UI changes for multi-user display
- Complex clustering algorithms

## Open Questions
1. Should we make session detection optional via config?
2. What's the optimal session timeout (5 min vs 10 min)?
3. Should rejection reasons be stored in state for analysis?
4. Do we need different limits for children vs adults?

## Implementation Code Structure

### Key Files to Modify
```
src/processor.py
├── _validate_weight()           # Main change here
├── _calculate_time_delta()      # New helper
├── _detect_session()            # New helper
└── process_weight()             # Minor flow update

config.toml
└── [physiological]              # New section

tests/test_physiological_limits.py  # New test file
```

### State Structure (Minimal Change)
```python
{
    'initialized': bool,
    'init_buffer': [],
    'kalman_params': {...},        # Unchanged
    'last_state': np.array([...]), # Unchanged
    'last_covariance': np.array([...]), # Unchanged
    'last_timestamp': datetime,    # Unchanged
    'adapted_params': {...},        # Unchanged
    'last_session_time': datetime, # NEW (optional)
    'session_stats': {...}          # NEW (optional)
}
```

## Review Checklist
- [x] Maintains stateless architecture
- [x] Preserves Kalman filter integrity
- [x] Backward compatible state
- [x] Clear validation logic
- [x] Testable components
- [x] Configurable thresholds
- [x] Performance considerations
- [x] Error handling strategy