# Plan to Fix Cliffing Issue in Weight Processing System

## Problem Statement
The Kalman filter follows steep trends ("cliffs") when multiple measurements are taken close in time (within minutes), leading to unrealistic weight trajectory changes. This is particularly evident for user `678639d5-e18b-4014-8859-fa3f1b436a99` where rapid measurements cause the filter to deviate significantly from a reasonable weight trajectory.

## Root Causes Identified

1. **Zero Weight for Temporal Consistency**: The `temporal_consistency` component in quality scoring has weight = 0, completely ignoring temporal factors
2. **Insufficient Temporal Penalties**: Rapid measurements (minutes apart) receive only mild penalties (0.5-0.7 quality scores)
3. **Kalman Time Delta Too Permissive**: Minimum time delta of 0.1 days allows rapid adaptation to closely-spaced measurements
4. **Burst Pattern Penalties Too Mild**: Multiple measurements in short periods receive only 0.6-0.8 penalty multipliers

## Proposed Solutions

### Solution 1: Enable and Strengthen Temporal Consistency (Immediate Fix)
**Files to modify**: `config.toml`

```toml
[quality_scoring.component_weights]
kalman_fit = 0.40              # Reduced from 0.65
temporal_consistency = 0.30    # Increased from 0!!
anomaly_detection = 0.20       # Reduced from 0.25
source_reliability = 0.05      # Added back
trend_alignment = 0.05         # Reduced from 0.1
```

**Rationale**: Temporal consistency is critical for preventing rapid changes. Zero weight means the system completely ignores time factors in quality assessment.

### Solution 2: Implement Aggressive Rapid Measurement Rejection
**Files to modify**: `src/processing/unified_quality_scorer.py`

Add new configuration and logic:
```python
# Enhanced rapid measurement thresholds
MIN_MEASUREMENT_INTERVAL_MINUTES = 15  # Minimum time between accepted measurements
RAPID_CHANGE_PENALTY_FACTOR = 0.001   # Aggressive penalty for rapid changes
```

Modify temporal consistency calculation to:
1. Reject measurements < 15 minutes apart (except from trusted sources)
2. Apply exponential penalty based on time proximity: `penalty = exp(-5 * (15 - time_diff_minutes) / 15)` for measurements < 15 minutes
3. Consider measurement density: If > 3 measurements in 1 hour, apply additional 0.5x multiplier

### Solution 3: Enhance Kalman Filter Time-Aware Processing
**Files to modify**: `src/processing/kalman.py`

```python
# Modify time_delta_days calculation
if delta < 0.01:  # Less than ~15 minutes
    # Apply heavy damping for rapid measurements
    time_delta_days = 0.01
    # Increase observation noise for rapid measurements
    obs_cov *= 10.0  # Make Kalman trust rapid measurements much less
elif delta < 0.04:  # Less than 1 hour
    time_delta_days = max(0.1, delta)
    obs_cov *= 3.0   # Moderate trust reduction
else:
    time_delta_days = max(0.1, min(30.0, delta))
```

### Solution 4: Add Measurement Density Awareness
**Files to modify**: `src/processing/processor.py`, `src/processing/unified_quality_scorer.py`

Track measurement density in a sliding window:
- Count measurements in last 1 hour, 6 hours, 24 hours
- Apply progressive penalties:
  - > 5 measurements/hour: quality *= 0.3
  - > 10 measurements/6 hours: quality *= 0.5
  - > 20 measurements/24 hours: quality *= 0.7

### Solution 5: Implement Smart Buffering for Rapid Measurements
**Files to modify**: `src/replay/replay_buffer.py`

When multiple measurements arrive rapidly:
1. Buffer them for 15-30 minutes
2. Take the median value
3. Process as a single measurement with timestamp = median timestamp
4. Mark as "aggregated" for transparency

## Implementation Priority

1. **Immediate (Config Only)**: Enable temporal_consistency weight (Solution 1)
2. **High Priority**: Implement aggressive rapid measurement penalties (Solution 2)
3. **Medium Priority**: Enhance Kalman time handling (Solution 3)
4. **Low Priority**: Add measurement density tracking (Solution 4)
5. **Optional**: Smart buffering system (Solution 5)

## Testing Strategy

1. **Before Changes**: Save current results for user `678639d5-e18b-4014-8859-fa3f1b436a99`
2. **After Each Change**: Re-run processing and compare visualization
3. **Success Metrics**:
   - No steep cliffs in filtered weight trajectory
   - Smooth transitions even with rapid measurements
   - Quality scores < 0.3 for measurements < 5 minutes apart
   - Acceptance rate for rapid bursts < 20%

## Rollback Plan

If changes cause issues:
1. Revert config.toml to original weights
2. Use git to revert code changes
3. Re-run with original settings

## Alternative Approaches Considered

1. **Hard time-based gating**: Reject all measurements < X minutes apart
   - Rejected: Too rigid, may lose valid data

2. **Statistical outlier detection only**: Rely on MAD/IQR
   - Rejected: Doesn't address temporal aspect

3. **Separate "rapid measurement mode"**: Different processing for rapid data
   - Rejected: Too complex, adds branching logic

## Recommended Next Steps

1. Start with config change (Solution 1) - test impact
2. If insufficient, implement Solution 2 (rapid measurement penalties)
3. Monitor results and iterate based on actual data patterns
4. Consider Solutions 3-5 if problems persist

## Expected Outcome

After implementing these solutions, the system should:
- Maintain smooth weight trajectories even with rapid measurement bursts
- Reject or heavily penalize measurements taken seconds/minutes apart
- Trust measurements spaced hours/days apart more than rapid bursts
- Eliminate the "cliffing" pattern while preserving legitimate weight changes