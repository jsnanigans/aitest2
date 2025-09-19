# Investigation: Kalman State Persisted Before Quality Scoring

## Bottom Line

**Root Cause**: Kalman state is persisted before quality validation during initialization
**Fix Location**: `src/processing/processor.py:260`
**Confidence**: High

## What's Happening

During Kalman filter initialization (first measurement or after reset), the system persists state immediately after creating it, before quality scoring runs. If quality scoring rejects the measurement, the Kalman state has already been saved with potentially invalid data.

## Why It Happens

**Primary Cause**: Incorrect code flow assumption
**Trigger**: `src/processing/processor.py:189-287` - Initialization path returns early
**Decision Point**: `src/processing/processor.py:260` - State persisted with reason "outlier_rejection_accept"

## Evidence

- **Key File**: `src/processing/processor.py:240` - Comment says "after outlier rejection" but it's before quality scoring
- **Search Used**: `rg "outlier_rejection_accept"` - Found premature persistence
- **Code Flow**: Lines 189-287 handle initialization, return at 287, quality scoring starts at 289

## Next Steps

1. Move state persistence after quality scoring validation (after line 383)
2. Add rollback mechanism if quality score rejects after state update
3. Fix misleading comment about "outlier rejection" on line 240

## Risks

- Invalid weights corrupt Kalman state permanently
- All future measurements use incorrect baseline
- Reset mechanisms may not trigger due to bad state
