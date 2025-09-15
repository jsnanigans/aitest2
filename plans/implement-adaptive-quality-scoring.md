# Plan: Adaptive Quality Scoring During Learning Period

## Summary
Make quality scoring more lenient during the adaptive period (initial measurements and post-reset) to prevent rejection of valid data while the Kalman filter is still learning the weight pattern.

## Problem
Even with adaptive Kalman parameters, measurements were still being rejected due to quality scoring:
- Quality score 0.58 < threshold 0.6
- Plausibility component failing (0.34) because Kalman prediction is still adjusting
- Valid measurements rejected during learning period

## Solution Design

### Adaptive Period Detection
Considered in adaptive period if:
- First 10 measurements after initialization/reset
- OR within 7 days of initialization/reset

### Adaptive Quality Parameters
During adaptive period:
- **Threshold**: 0.4 (vs 0.6 normal) - more lenient
- **Component weights**:
  - Safety: 0.45 (vs 0.35) - keep safety high
  - Plausibility: 0.10 (vs 0.25) - much lower during learning
  - Consistency: 0.15 (vs 0.25) - lower during learning
  - Reliability: 0.30 (vs 0.15) - higher to trust source

## Implementation
1. Check if in adaptive period (measurements_since_reset < 10 OR days < 7)
2. Create adaptive quality config with adjusted parameters
3. Pass adaptive config to quality scorer
4. Result: More forgiving scoring during learning period

## Expected Impact
- Measurement at 108.4kg with score 0.58 → Now accepted
- Reduces false rejections during initial/post-reset period
- Maintains safety checks while allowing adaptation
- Better user experience with valid data accepted

## Testing
- Initial 120kg → 108kg measurements: All accepted ✓
- Quality scores during adaptive period properly adjusted ✓
- Safety still enforced (extreme values still rejected) ✓
