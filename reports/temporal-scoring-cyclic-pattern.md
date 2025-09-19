# Investigation: Cyclical Accept/Reject Pattern for User bc1c9d20

## Bottom Line

**Root Cause**: Temporal consistency scoring creates artificial cycles with gradual score improvement followed by reset
**Fix Location**: `src/processing/unified_quality_scorer.py:calculate_temporal_consistency`
**Confidence**: High

## What's Happening

User bc1c9d20 shows a regular pattern: 9-10 rejections building up quality scores from ~0.08 to ~0.5, then acceptance, followed by immediate quality drop and the cycle repeats.

## Why It Happens

**Primary Cause**: Temporal consistency scoring progressively increases from 0.2→0.3→0.4→0.6 based on time gaps
**Trigger**: `unified_quality_scorer.py:329-344` - Fixed temporal consistency thresholds
**Decision Point**: When quality score exceeds ~0.48, measurement gets accepted, resetting the cycle

### The Cycle Mechanism:

1. **Initial Rejection** (Quality ~0.08): 
   - First measurement after acceptance gets temporal_consistency=0.3
   - Low Kalman fit (~0.03) due to deviation
   - Overall quality too low → REJECTED

2. **Score Building** (10-11 measurements):
   - Each daily measurement increases temporal_consistency (0.3→0.4→0.6)
   - Kalman fit gradually improves (0.03→0.5)
   - Quality score climbs: 0.08→0.16→0.25→0.35→0.48

3. **Acceptance Threshold** (Quality ~0.48-0.51):
   - After ~10 rejections, quality reaches acceptance level
   - Temporal consistency=0.6, Kalman fit=0.3+
   - Measurement ACCEPTED

4. **Reset and Repeat**:
   - Next measurement: temporal drops back to 0.3
   - Kalman fit resets due to new baseline
   - Cycle restarts

## Evidence

- **Pattern Data**: `output/results_test_no_date.json`
  - Rejection runs: 10, 9, 10, 5 measurements
  - Quality progression: 0.08→0.50 over each cycle
  - Temporal consistency: 0.2→0.6 progression

- **Key Files**:
  - `unified_quality_scorer.py:329-344` - Temporal threshold logic
  - `config.toml:87-92` - Temporal thresholds (6h=0.5kg, 24h=2kg)

- **Search Used**: 
  - `jq analysis of user data` - Found consistent 10-measurement cycles
  - `rg "calculate_temporal_consistency"` - Located scoring logic

## Next Steps

1. Implement adaptive temporal scoring that doesn't reset after acceptance
2. Consider exponential decay instead of step functions for temporal scores
3. Add hysteresis to prevent rapid accept/reject oscillations

## Risks

- Users experience frustrating rejection patterns despite consistent measurements
- Acceptance becomes predictable rather than quality-based
- Temporal scoring creates artificial periodicity unrelated to actual weight patterns
