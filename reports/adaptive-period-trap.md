# Investigation: Adaptive Period Measurement Trap

## Bottom Line

**Root Cause**: `measurements_since_reset` only increments on accepted measurements, trapping system in infinite rejection loop.
**Fix Location**: `src/processing/processor.py:396,465,495` (rejection points)
**Confidence**: High

## What's Happening

User's weight measurements (~82kg) are permanently rejected after system accepts erroneous manual entry (116.57kg). The Kalman filter locks onto the wrong value and cannot recover because the adaptive period never ends.

## Why It Happens

**Primary Cause**: Circular dependency - need accepted measurements to exit adaptive period, but can't accept measurements while in adaptive period with bad state
**Trigger**: `src/processing/processor.py:723` - Counter increments only on acceptance
**Decision Point**: `src/processing/processor.py:332` - Checks if measurements_since_reset < 15

Failure sequence:
1. Manual entry 116.57kg accepted (34.7kg jump from previous 81.88kg)
2. Soft reset triggered, Kalman state = 116.57kg, measurements_since_reset = 0
3. Real measurements (~82kg) show -34.4kg innovation from prediction
4. Kalman fit score = 0.06 (needs 0.45 during soft reset adaptive period)
5. Rejection doesn't increment counter, stays at 0
6. System stuck: needs 15 accepted measurements to exit adaptive period

## Evidence

- **Key File**: `output/results_test_no_date.json:line 230-400` - User 9575e299 data
- **Search Used**: `jq '.users["9575e299-bef6-4c76-9f27-aecc5fdf13a4"]'` - Found pattern
- **Counter Logic**: `processor.py:723` - Increments only after acceptance
- **Check Logic**: `processor.py:332` - Uses counter to determine adaptive period

## Next Steps

1. Increment `measurements_since_reset` for ALL measurements during adaptive period
2. Add safety valve: max 20 consecutive rejections forces adaptive exit
3. Validate reset-triggering measurements (reject implausible BMI changes)

## Risks

- **Critical**: Patient receives medical advice based on 116kg instead of actual 82kg
- **Current state**: System permanently locked, manual database intervention required
- **Data integrity**: Erroneous manual entries can corrupt state permanently
