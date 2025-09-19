# Investigation: Low Quality Score (0.42) Acceptance

## Bottom Line

**Root Cause**: Adaptive period threshold reduction after soft reset
**Fix Location**: `src/processing/processor.py:349`
**Confidence**: High

## What's Happening

A weight measurement with quality score 0.424 was accepted on 2025-01-08 despite being below the normal threshold of 0.6. This occurred because the system was in an adaptive period following a soft reset, which lowered the acceptance threshold to 0.4.

## Why It Happens

**Primary Cause**: Post-reset adaptive period activates relaxed quality thresholds
**Trigger**: `src/processing/processor.py:332` - Checks if `measurements_since_reset < adaptation_measurements` (1 < 10)
**Decision Point**: `src/processing/processor.py:349` - Sets `threshold = 0.4` during adaptation

## Evidence

- **Key File**: `src/processing/processor.py:329-350` - Adaptive period detection and threshold adjustment
- **Search Used**: `rg "measurements_since_reset" src/` - Found counter tracking logic
- **State Reset**: `src/processing/kalman.py:648` - Sets `measurements_since_reset = 0` on reset
- **Increment Logic**: `src/processing/processor.py:723` - Increments only after accepted measurements

## Sequence of Events

1. Measurement 4: Soft reset triggered (weight jump to 116.6kg)
2. State reset with `measurements_since_reset = 0`, then incremented to 1
3. Measurements 5-9: Rejected (Q < 0.4), counter stays at 1
4. Measurement 10: Counter=1 < 10, adaptive threshold=0.4, Q=0.424 > 0.4 → Accepted

## Next Steps

1. Review if adaptive threshold of 0.4 is appropriate for post-reset scenarios
2. Consider if rejected measurements should count toward adaptation exit
3. Add explicit logging when adaptive thresholds are applied

## Risks

- Too-lenient adaptive threshold may allow poor quality data through
- Current logic only counts accepted measurements, potentially extending adaptive period
