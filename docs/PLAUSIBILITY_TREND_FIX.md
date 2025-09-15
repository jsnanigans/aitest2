# Plausibility Score Enhancement - Trend Detection

## Problem Identified

Users experiencing gradual weight loss (e.g., from 80.87kg to 78kg over several days) were having legitimate measurements rejected due to low plausibility scores.

### Root Cause

The original plausibility calculation:
1. Used a simple mean of recent weights
2. Applied a minimum standard deviation of 0.5kg
3. Did not account for trends in the data

For users with consistent weight loss/gain patterns, the small variance in their history combined with the minimum std of 0.5kg meant that any weight more than ~1.5kg from the mean would be flagged as implausible.

## Solution Implemented

Enhanced the `_calculate_plausibility` method in `quality_scorer.py` to:

1. **Detect Linear Trends**: Calculate slope and R² for recent weight history
2. **Project Expected Values**: When a strong trend exists (R² > 0.5), project the trend forward to estimate the expected next weight
3. **Adjust Tolerance**: For strong trends, increase the minimum standard deviation to allow for continued progression
4. **Blend Predictions**: Use a weighted average of historical mean and projected value based on trend strength

## Key Changes

### Trend Detection
```python
def _calculate_trend(weights) -> (slope, r_squared)
```
- Performs linear regression on weight history
- Returns slope (kg per measurement) and R² (fit quality)

### Dynamic Mean Adjustment
- R² > 0.8: Use projected next value as mean
- 0.5 < R² ≤ 0.8: Blend projection with historical mean
- R² ≤ 0.5: Use historical mean (no trend)

### Dynamic Standard Deviation
- For trends with |slope| > 0.1 and R² > 0.5: min_std = max(1.0, |slope| * 3)
- Otherwise: min_std = 0.5 (original behavior)

## Test Results

### Before Fix
- 80.87kg → 78.0kg: Plausibility = 0.147 (REJECTED)
- 80.87kg → 77.0kg: Plausibility = 0.054 (REJECTED)

### After Fix
- 80.87kg → 78.0kg: Plausibility = 0.700 (ACCEPTED)
- 80.87kg → 77.0kg: Plausibility = 0.500 (MARGINAL)

## Scenarios Validated

1. **Gradual Weight Loss** (0.3kg/day): ✓ Correctly accepts continued loss
2. **Steady Weight Gain** (0.2kg/day): ✓ Correctly accepts continued gain
3. **Stable Weight**: ✓ Maintains strict tolerance for non-trending data
4. **Rapid Weight Loss** (0.5kg/day): ✓ Accepts while maintaining safety bounds

## Impact

- Reduces false rejections for users with legitimate weight changes
- Maintains safety by keeping other quality components (safety, consistency)
- Improves user experience for those actively losing/gaining weight
- Preserves rejection of truly implausible measurements

## Files Modified

- `src/quality_scorer.py`: Added trend detection to `_calculate_plausibility` method and `_calculate_trend` helper