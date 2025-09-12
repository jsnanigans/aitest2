# Plan: Remove Deviation Check After Gaps - Use BMI Validation Only

## Summary
Remove the relative deviation check (extreme_threshold) for measurements after gaps. Instead, only reject values that fall outside physiologically possible BMI ranges. This prevents the system from being stuck rejecting valid measurements that differ significantly from stale baselines.

## Context
- Source: User investigation of rejected measurements for users 0040872d and 08f3aee0e
- Assumptions:
  - After gaps, the previous baseline may be stale/incorrect
  - Relative comparisons to old baselines are unreliable
  - BMI ranges provide absolute physiological boundaries
- Constraints:
  - Must maintain backward compatibility with existing data
  - Must preserve Kalman filter functionality for continuous measurements

## Requirements

### Functional
- Remove extreme_threshold deviation check after gaps (>30 days)
- Accept any weight within valid BMI range (typically 10-60) after gaps
- Reset Kalman filter with new weight if within BMI range
- Maintain deviation checks for continuous measurements (no gap)

### Non-functional
- No performance degradation
- Clear logging of reset decisions
- Maintain data quality for continuous tracking

## Alternatives

### Option A: Complete Removal of Deviation Check
- Approach: Remove all deviation checks, rely only on BMI validation
- Pros: 
  - Simplest implementation
  - No relative comparisons ever
  - Consistent behavior
- Cons:
  - May accept erratic measurements during continuous tracking
  - Loss of outlier detection for data entry errors
- Risks: Reduced data quality for continuous measurements

### Option B: Gap-Conditional Deviation Check
- Approach: Disable deviation check only when gap > threshold
- Pros:
  - Maintains quality checks for continuous data
  - Accepts valid measurements after gaps
  - Balanced approach
- Cons:
  - More complex logic
  - Need to define gap threshold
- Risks: Edge cases around gap threshold

### Option C: Graduated Deviation Tolerance
- Approach: Increase deviation tolerance based on gap duration
- Pros:
  - Smooth transition between behaviors
  - Maintains some outlier protection
- Cons:
  - Complex formula needed
  - Harder to explain to users
- Risks: May still reject valid measurements after long gaps

## Recommendation
**Option B: Gap-Conditional Deviation Check**

Rationale:
- Solves the immediate problem of stuck baselines after gaps
- Preserves data quality for continuous tracking
- Clear, explainable logic
- Minimal code changes required

## High-Level Design

### Architecture Changes
- Modify `WeightProcessor._process_weight_internal()` in `src/processor.py`
- Add BMI validation module or enhance existing validation
- Update reset decision logic flow

### Data Flow
1. Measurement arrives with timestamp
2. Calculate gap from last measurement
3. If gap > threshold (30 days):
   - Validate weight against BMI range only
   - If valid: Reset Kalman and accept
   - If invalid: Reject (impossible BMI)
4. If gap ≤ threshold:
   - Apply existing deviation checks
   - Process normally

### Affected Components
- `src/processor.py`: Main processing logic (lines 130-185)
- `src/validation.py`: BMI validation functions
- `src/models.py`: BMI calculation utilities

## Implementation Plan (No Code)

### Step 1: Add BMI Validation Infrastructure
- Create method to calculate BMI from weight and user height
- Define physiological BMI boundaries (min: 10, max: 60)
- Add method to validate weight against BMI range
- Handle missing height data gracefully

### Step 2: Modify Gap Reset Logic
- In `processor.py` lines 135-147:
  - Before accepting post-gap weight, validate BMI
  - Only reset if BMI is within valid range
  - Log reset decision with BMI value

### Step 3: Remove Deviation Check After Gaps
- In `processor.py` lines 163-184:
  - Add condition to skip deviation check if recent reset
  - Or check if gap > threshold before deviation check
  - Ensure continuous measurements still get deviation check

### Step 4: Update Rejection Messages
- Change rejection reason from "Extreme deviation" to "Invalid BMI" when applicable
- Include actual BMI value in rejection metadata
- Maintain clear audit trail

### Step 5: Add Configuration
- Add config parameters:
  - `min_valid_bmi`: 10
  - `max_valid_bmi`: 60
  - `skip_deviation_after_gap_days`: 30
- Allow per-user height configuration

## Validation & Rollout

### Test Strategy
1. Unit tests:
   - Test BMI validation with various weights/heights
   - Test gap detection logic
   - Test deviation skip after gaps
   - Test normal processing for continuous data

2. Integration tests:
   - Test with user 0040872d data
   - Test with user 08f3aee0e data
   - Verify extreme values accepted after gaps if BMI valid
   - Verify continuous tracking still rejects outliers

3. Edge cases:
   - Missing height data
   - BMI exactly at boundaries
   - Gap exactly at threshold
   - Multiple resets in sequence

### Manual QA Checklist
- [ ] Process historical data for affected users
- [ ] Verify 100+kg measurements accepted if BMI valid
- [ ] Verify <30kg measurements rejected if BMI invalid
- [ ] Check continuous measurements still filtered
- [ ] Review logs for clear reset reasons

### Rollout Plan
1. Deploy to test environment
2. Process historical data for known problem users
3. Validate results match expectations
4. Deploy to production with feature flag
5. Enable for affected users first
6. Monitor for 1 week
7. Enable globally

## Risks & Mitigations

### Risk 1: Missing Height Data
- Impact: Cannot calculate BMI for validation
- Mitigation: Fall back to absolute weight bounds (30-200kg)
- Monitoring: Log cases of missing height

### Risk 2: Invalid Height Data  
- Impact: Incorrect BMI calculations
- Mitigation: Validate height is reasonable (1.0-2.5m)
- Monitoring: Alert on suspicious BMI values

### Risk 3: Legitimate Extreme Weights
- Impact: May reject valid measurements for very tall/short users
- Mitigation: Use generous BMI bounds, allow manual override
- Monitoring: Track rejection rates by user

## Acceptance Criteria

1. After 30+ day gap, system accepts any weight with BMI 10-60
2. After 30+ day gap, system rejects weights with BMI <10 or >60
3. For continuous measurements (<30 day gap), deviation check still applies
4. Clear logging indicates when deviation check is skipped
5. Historical data for users 0040872d and 08f3aee0e processes correctly

## Out of Scope

- Changing deviation thresholds for continuous measurements
- Modifying Kalman filter parameters
- Adding user-specific BMI ranges
- Implementing graduated deviation tolerance
- Changing the 30-day gap threshold

## Open Questions

1. Should we make the BMI range configurable per deployment?
2. How should we handle users with medically extreme but valid BMIs?
3. Should we add an override mechanism for support teams?
4. Do we need to migrate historical rejected measurements?
5. Should the gap threshold be configurable or fixed at 30 days?

## Review Cycle

### Self-Review Notes
- Verified the plan addresses the root cause identified in investigation
- Confirmed BMI validation is more appropriate than relative deviation after gaps
- Ensured backward compatibility is maintained
- Added clear acceptance criteria for testing

### Revisions
- Initial version focused on complete removal of deviation check
- Revised to conditional removal based on gap duration
- Added explicit BMI boundary configuration
- Clarified that continuous measurements keep deviation check