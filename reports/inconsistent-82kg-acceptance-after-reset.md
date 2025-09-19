# Investigation: Inconsistent ~82kg Measurement Acceptance After 116kg Reset

## Bottom Line

**Root Cause**: Quality scores gradually improve as Kalman filter converges from 116kg toward 82kg, creating inconsistent acceptance near the 0.40-0.47 quality threshold boundary.
**Fix Location**: `src/processing/processor.py:574` - adaptive quality threshold logic
**Confidence**: High

## What's Happening

After a soft reset to 116kg (manual entry), subsequent ~82kg measurements show inconsistent acceptance patterns. Some measurements around 82kg are accepted while similar values are rejected, creating a seemingly random pattern in the visualization.

## Why It Happens

**Primary Cause**: Kalman fit component progressively improves as the filter adapts
**Trigger**: `src/processing/reset_manager.py:137` - Soft reset sets quality_acceptance_threshold to 0.35
**Decision Point**: `src/processing/processor.py:574` - Uses threshold of 0.4 during adaptation

The Kalman filter starts at 116kg and slowly converges toward 82kg. Each measurement improves the Kalman fit score:
- Initial 82kg measurements: Kalman fit ~0.06 (30% deviation from 116kg)
- After 5 days: Kalman fit ~0.22
- After 8 days: Kalman fit ~0.28 (crosses threshold when temporal consistency improves)
- After 20 days: Kalman fit ~0.40 (consistently accepted)

Temporal consistency adds variability:
- 0.4 for measurements < 2 days apart
- 0.6 for measurements 2+ days apart
- 0.8-1.0 for measurements 3+ days apart

This creates a quality score formula: `(kalman_fit * 0.5) + (temporal * 0.3) + (trend * 0.2)`

## Evidence

- **Key Measurements**: Jan 6 rejected (Q=0.324), Jan 8 accepted (Q=0.424)
- **Threshold Range**: Rejected max = 0.4017, Accepted min = 0.4734
- **Pattern**: All 19 measurements with Q < 0.40 rejected, 2 with Q > 0.42 accepted
- **Search Used**: `jq analysis of results_test_no_date.json`

## Next Steps

1. Consider using progressive thresholds: start at 0.25 after reset, increase to 0.4 over 10 measurements
2. Weight temporal consistency lower during adaptation period (e.g., 0.1 instead of 0.3)
3. Add explicit "adaptation grace period" that accepts measurements within 15% of any previously accepted value

## Risks

- Users see valid measurements rejected after manual corrections
- Trust erosion when system appears random/unpredictable
- Data gaps during critical weight change periods
