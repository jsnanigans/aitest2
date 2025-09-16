# Investigation: Retrospective Outlier Inconsistency

## Bottom Line
**Root Cause**: Quality score override protection is not temporal-aware, allowing contradictory decisions for temporally close measurements
**Fix Location**: `src/outlier_detection.py:68-80` - protected_indices logic
**Confidence**: High

## What's Happening
The retrospective processing accepts an outlier below 80kg while rejecting another above 120kg despite being temporally close. This creates non-straight trend lines instead of smooth trajectories.

## Why It Happens
**Primary Cause**: Quality score override uses point-in-time evaluation without temporal consistency checks
**Trigger**: `src/outlier_detection.py:73-79` - Quality score > 0.7 grants absolute protection
**Decision Point**: `src/outlier_detection.py:108` - Protected measurements skip all outlier checks

## Evidence
- **Key File**: `src/outlier_detection.py:73-75` - Shows unconditional quality score protection
- **Search Used**: `rg "quality_score.*protected" src/` - Found protection logic
- **Config**: `config.toml:165` - quality_score_threshold = 0.7 (default)
- **Replay Logic**: `src/replay_manager.py:301-310` - Reduces quality threshold to 0.25 during replay

## Root Cause Analysis

The issue stems from three interacting factors:

1. **Quality Score Protection is Absolute**: Any measurement with quality > 0.7 is protected from outlier detection regardless of context (`outlier_detection.py:73-75`)

2. **No Temporal Consistency**: The system doesn't check if protected measurements are consistent with each other across time windows

3. **Replay Leniency**: During retrospective replay, quality thresholds drop to 0.25 (`replay_manager.py:302`), allowing more measurements through

This creates the scenario where:
- An 80kg outlier gets quality score > 0.7 → Protected → Accepted
- A 120kg outlier gets quality score < 0.7 → Not protected → Rejected  
- These contradictory decisions happen close in time, creating trajectory discontinuity

## Next Steps
1. Add temporal consistency check for protected measurements in `outlier_detection.py`
2. Implement sliding window validation for quality-protected outliers
3. Consider reducing quality_score_threshold from 0.7 to 0.6 for better balance

## Risks
- Trajectory discontinuities affect Kalman filter stability
- Medical decisions based on inconsistent weight trends
- User trust erosion from unexplained trajectory changes
