# Quality Scoring System Improvements

## Problem Identified

The consistency quality check was too strict, causing legitimate weight measurements to be rejected. The main issue was that the consistency score used a simple daily rate calculation that was inappropriate for short time periods.

### Specific Issues Found

1. **Short Time Period Problem**: Measurements taken within hours of each other were being penalized harshly
   - Example: 0.5 kg change in 1 hour = 12 kg/day rate → consistency score of 0.089 (rejected)
   - This is unrealistic as 0.5 kg variation within a day is completely normal

2. **User Case Study**: User `03de147f-5e59-49b5-864b-da235f1dab54`
   - Had legitimate measurements around 92-93 kg
   - One outlier at 42.22 kg (correctly rejected)
   - But subsequent normal measurements were being rejected due to overly strict consistency checks

## Solution Implemented

### Improved Consistency Scoring

The `calculate_consistency_score` method in `quality_scorer.py` has been updated with time-aware thresholds:

#### For measurements < 6 hours apart:
- Allow up to 3 kg variation (accounts for meals, hydration, bathroom visits)
- Typical variation: 1.5 kg
- More lenient scoring curve

#### For measurements 6-24 hours apart:
- Interpolate between short-term and daily limits
- Progressive scaling based on actual time elapsed
- Balanced approach for intra-day measurements

#### For measurements > 24 hours apart:
- Use original daily rate calculation
- Maintains strict control for longer-term changes
- Prevents unrealistic weight changes over days/weeks

### Code Changes

Updated `src/quality_scorer.py` lines 224-295 with the improved `calculate_consistency_score` method.

## Results

### Before Improvement
- 0.5 kg change in 1 hour: score = 0.089 ❌
- 1.0 kg change in 2 hours: score = 0.089 ❌
- 2.0 kg change in 6 hours: score = 0.308 ❌

### After Improvement
- 0.5 kg change in 1 hour: score = 1.000 ✅
- 1.0 kg change in 2 hours: score = 1.000 ✅
- 2.0 kg change in 6 hours: score = 0.933 ✅

## Additional Recommendations

1. **Consider adjusting component weights** if needed:
   - Current: safety=0.35, plausibility=0.25, consistency=0.25, reliability=0.15
   - Could reduce consistency weight if still too strict

2. **Monitor outlier handling**:
   - The 42.22 kg outlier is correctly rejected (plausibility score = 0.041)
   - System properly identifies true anomalies vs normal variations

3. **Consider threshold adjustments**:
   - Current acceptance threshold: 0.6
   - Could lower to 0.5 for more lenient acceptance if needed

## Testing

Run the following to verify improvements:
```bash
uv run python test_improved_scorer.py
```

This will test various scenarios and confirm that normal weight variations are now accepted while true outliers are still rejected.
