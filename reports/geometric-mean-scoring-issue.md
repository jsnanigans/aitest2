# Investigation: Unified Quality Scorer Rejecting Valid Measurements

## Bottom Line

**Root Cause**: UnifiedQualityScorer uses weighted geometric mean despite config setting `use_harmonic_mean = true`
**Fix Location**: `src/processing/unified_quality_scorer.py:182`
**Confidence**: High

## What's Happening

The unified quality scorer is rejecting 380 out of 406 measurements (93.6% rejection rate) for user eb61194e-491f-40e4-9064-d3d096e0fe64. Measurements that visually fit the trend perfectly are being rejected with scores around 0.51, below the 0.59 threshold.

## Why It Happens

**Primary Cause**: Geometric mean severely penalizes low component scores
**Trigger**: `src/processing/unified_quality_scorer.py:182` - Always calls `_calculate_weighted_geometric_mean`
**Decision Point**: `src/processing/unified_quality_scorer.py:487-515` - Geometric mean implementation

The UnifiedQualityScorer ignores the `use_harmonic_mean = true` configuration and always uses geometric mean. With weights:
- kalman_fit: 0.45 (45%)
- temporal_consistency: 0.28 (28%)
- anomaly_detection: 0.25 (25%)
- source_reliability: 0.01 (1%)
- trend_alignment: 0.01 (1%)

Even though source_reliability has only 1% weight, when it scores 0.20, the geometric mean formula `score^0.01` still significantly impacts the overall score. Geometric mean is particularly punitive - any component scoring low drags down the entire score regardless of weight.

## Evidence

- **Key File**: `src/processing/unified_quality_scorer.py:182` - Hard-coded geometric mean call
- **Config Ignored**: `config.toml:272` - `use_harmonic_mean = true` has no effect
- **Search Used**: `rg "use_harmonic_mean"` - Found in other scorers but not UnifiedQualityScorer
- **Data Evidence**: 93.6% rejection rate with source_reliability consistently at 0.20

## Next Steps

1. Add config check to UnifiedQualityScorer to respect `use_harmonic_mean` setting
2. Implement harmonic mean calculation method in UnifiedQualityScorer
3. Consider arithmetic mean option - geometric mean is too punitive for quality scoring

## Risks

- Valid measurements being systematically rejected
- User weight tracking data becoming sparse and unreliable
- Kalman filter receiving insufficient data to maintain accurate state
