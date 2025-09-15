# Plan: Improved Kalman Gap Adjustment

## Summary
Implement a cleaner gap handling strategy that performs a full reset after gap_threshold_days, uses fixed "safe" parameters during warmup, and gradually transitions back to normal operation without complex parameter tuning.

## Context
- Source: User request for better gap handling
- Current issue: Gap handling uses complex buffering and adaptation that doesn't work well
- Goal: Simpler, more predictable behavior after data gaps

## Requirements

### Functional
- Full Kalman reset after max_gap_days (default: 30 days)
- Fixed, conservative parameters during warmup phase
- Minimum 7 measurements over 7+ different days for warmup completion
- Smooth transition from warmup to normal operation
- No parameter discontinuities that cause jumps

### Non-functional
- Maintain medical data safety standards
- Predictable, testable behavior
- Minimal computational overhead
- Clear state transitions

## Alternatives

### Option A: IQR-based Parameter Tuning (Original Proposal)
**Approach**: Collect 7+ measurements, perform IQR analysis, tune Kalman parameters
**Pros**: 
- Adaptive to post-gap data characteristics
- Could handle varying data quality

**Cons**: 
- Insufficient data for robust statistics (need 25+ points)
- Complex state machine with multiple failure modes
- Parameter updates could cause discontinuities
- Hard to test and validate

### Option B: Fixed Warmup with Gradual Transition
**Approach**: Use conservative fixed parameters during warmup, gradually blend to normal
**Pros**: 
- Simple, predictable behavior
- No parameter discontinuities
- Easy to test and validate
- Safe for medical data

**Cons**: 
- Less adaptive to data characteristics
- May be overly conservative initially

### Option C: Dual-Track Kalman
**Approach**: Run two Kalman filters - one conservative, one normal - blend results
**Pros**: 
- Smooth transitions
- No parameter changes

**Cons**: 
- Double computation
- Complex blending logic
- Memory overhead

## Recommendation
**Option B: Fixed Warmup with Gradual Transition**

Rationale:
- Simplicity and safety are paramount for medical data
- Statistical robustness requires more data than available
- Predictable behavior aids debugging and validation
- Gradual transitions prevent discontinuities

## High-Level Design

### Architecture
```
Gap Detection → Full Reset → Warmup Phase → Transition Phase → Normal Operation
```

### State Machine
```python
States:
- NORMAL: Regular Kalman filtering
- GAP_DETECTED: Gap > threshold, prepare for reset  
- WARMUP: Collecting initial measurements (min 7 over 7 days)
- TRANSITION: Gradually shifting from conservative to normal parameters
- NORMAL: Back to regular operation
```

### Key Parameters
- `gap_threshold_days`: 30 (triggers full reset)
- `warmup_min_measurements`: 7
- `warmup_min_days`: 7  
- `warmup_observation_covariance`: 10.0 (3x normal, more tolerant)
- `warmup_transition_covariance`: 0.05 (3x normal, allows more change)
- `transition_duration_measurements`: 5 (gradual blend over 5 measurements)

## Implementation Plan (No Code)

### Step 1: Refactor Gap Detection
- Location: `src/processor.py` lines 150-180
- Simplify gap detection to binary decision: gap > threshold → reset
- Remove complex gap_buffer logic
- Add clear state tracking: `gap_state: 'normal' | 'warmup' | 'transition'`

### Step 2: Implement Full Reset Logic
- Location: `src/kalman.py` - new method `reset_after_gap()`
- Clear all Kalman state
- Initialize with first post-gap measurement
- Set conservative parameters for warmup
- Track warmup start time and measurement count

### Step 3: Warmup Phase Management
- Location: `src/processor.py` - new warmup handling section
- Track measurements: count and unique days
- Use fixed conservative parameters (no adaptation)
- Accept all physiologically valid measurements
- No extreme deviation checks during warmup

### Step 4: Transition Phase Implementation  
- Location: `src/kalman.py` - new method `blend_parameters()`
- After warmup complete, enter transition phase
- Linearly blend from warmup to normal parameters over 5 measurements
- Formula: `param = warmup_param * (1 - α) + normal_param * α` where α goes 0→1

### Step 5: State Persistence Updates
- Location: `src/database.py`
- Add fields: `gap_state`, `warmup_measurements`, `warmup_days`, `transition_progress`
- Ensure state survives restarts
- Add migration for existing states

### Step 6: Update Quality Scoring
- Location: `src/quality_scorer.py`
- Disable or reduce quality scoring during warmup
- Use relaxed thresholds during transition
- Full scoring only in normal state

## Validation & Rollout

### Test Strategy
1. Unit tests for each state transition
2. Integration tests with various gap scenarios:
   - Single long gap
   - Multiple short gaps  
   - Gap during warmup
   - Gap during transition
3. Edge cases:
   - Exactly 7 measurements over 6 days (should not complete)
   - Measurements all on same day (should not complete)
   - Very noisy data during warmup

### Manual QA Checklist
- [ ] Verify reset triggers at correct gap threshold
- [ ] Confirm warmup requires both 7 measurements AND 7 days
- [ ] Check smooth parameter transitions (no jumps in filtered weight)
- [ ] Test with real historical data containing gaps
- [ ] Verify state persistence across restarts

### Rollout Plan
1. Feature flag: `enable_improved_gap_handling`
2. Deploy to test environment
3. Run parallel comparison with old logic
4. Gradual rollout: 10% → 50% → 100%
5. Monitor for anomalies in filtered weights

## Risks & Mitigations

### Risk 1: Over-conservative During Warmup
**Impact**: May accept bad data during warmup
**Mitigation**: Keep basic physiological validation active, just with relaxed thresholds

### Risk 2: Transition Discontinuities  
**Impact**: Jumps in filtered weight when parameters change
**Mitigation**: Gradual blending over 5 measurements, monitor max change

### Risk 3: State Corruption
**Impact**: Incorrect gap handling if state is corrupted
**Mitigation**: Add state validation, auto-recovery to safe defaults

## Acceptance Criteria
- [ ] Full reset occurs after 30-day gaps
- [ ] Warmup requires exactly 7 measurements over 7+ different days
- [ ] No jumps > 2kg in filtered weight during transitions
- [ ] All existing tests pass with new logic
- [ ] Performance impact < 5% on processing time

## Out of Scope
- Complex parameter tuning algorithms
- IQR or other statistical analysis during warmup
- Machine learning approaches
- Historical data reprocessing

## Open Questions
1. Should warmup parameters be configurable or hard-coded for safety?
2. How to handle questionnaire sources during warmup (currently they trigger resets)?
3. Should we track gap frequency to identify problematic users?
4. What metrics to monitor during rollout?

## Review Cycle
- Self-review: Simplified from original IQR approach for safety and testability
- Removed statistical analysis due to insufficient data points
- Added gradual transition to prevent discontinuities
- Next: Review with medical team for warmup parameter values