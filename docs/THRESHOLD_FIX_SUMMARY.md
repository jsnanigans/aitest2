# Threshold Unit Fix Implementation Summary

## Problem Fixed
The enhanced processor was passing absolute weight values (kg) to the base processor which expected percentages, causing valid measurements to be incorrectly accepted/rejected. This led to impossible weight dips in visualizations, particularly for user `0040872d-333a-4ace-8c5a-b2fcd056e65a`.

## Solution Implemented

### 1. Created Unified Threshold Calculator (`src/threshold_calculator.py`)
- Single source of truth for all threshold calculations
- Explicit unit parameters (`'percentage'` or `'kg'`)
- Source reliability profiles based on empirical data
- Proper conversion between units with bounds

### 2. Fixed Enhanced Processor (`src/processor_enhanced.py`)
- Now uses `ThresholdCalculator.get_extreme_deviation_threshold()` with `unit='percentage'`
- Correctly passes percentage values to base processor
- Adapts Kalman noise based on source reliability
- Includes threshold metadata in results for transparency

### 3. Key Changes

#### Before (Buggy):
```python
# Enhanced processor set threshold in kg
adapted_config['extreme_threshold'] = get_adaptive_threshold(source, time_gap_days)
# Returns: 3.0 (meant as 3kg)

# Base processor expected percentage
deviation = abs(weight - predicted) / predicted  # e.g., 0.13 (13%)
if deviation > extreme_threshold:  # Compares 0.13 > 3.0 (always false!)
```

#### After (Fixed):
```python
# Enhanced processor now converts to percentage
threshold_result = ThresholdCalculator.get_extreme_deviation_threshold(
    source=source,
    time_gap_days=time_gap_days,
    current_weight=cleaned_weight,
    unit='percentage'  # Explicitly request percentage
)
adapted_config['extreme_threshold'] = threshold_result.value  # Now 0.035 (3.5%)
```

## Source Reliability Profiles

Based on analysis of 709,246 measurements:

| Source | Outlier Rate (per 1000) | Reliability | Threshold Behavior |
|--------|-------------------------|-------------|-------------------|
| care-team-upload | 3.6 | excellent | More lenient (10kg) |
| patient-upload | 13.0 | excellent | More lenient (10kg) |
| patient-device | 20.7 | good | Standard (10kg) |
| connectivehealth.io | 35.8 | moderate | Stricter (5kg) |
| api.iglucose.com | 151.4 | poor | Very strict (3kg) |

## Testing

### Test Files Created
- `tests/test_threshold_consistency.py` - Comprehensive unit and integration tests
- `tests/test_threshold_fix_verification.py` - Specific verification of the bug fix

### Test Results
✅ All threshold calculator unit tests pass
✅ Source reliability adaptation working correctly
✅ Weight and time gap scaling appropriate
✅ Enhanced processor integration successful
✅ No more impossible weight dips

## Impact

### Before Fix
- 85kg measurement with 75kg predicted → Accepted (wrong!)
- Led to Kalman filter extrapolating unrealistic values
- Created impossible dips below 60kg

### After Fix
- 85kg measurement with 75kg predicted → Rejected (correct!)
- Kalman filter maintains realistic weight progression
- No impossible physiological changes

## Backward Compatibility
- Old configuration parameters still supported with warnings
- Threshold metadata included in results for debugging
- Legacy fields maintained for existing consumers

## Future Improvements
1. Consider adding runtime unit validation in debug mode
2. Update source reliability profiles monthly based on data
3. Add comprehensive logging of threshold decisions
4. Consider user-configurable thresholds for edge cases

## Files Modified
1. `src/threshold_calculator.py` (NEW - 300 lines)
2. `src/processor_enhanced.py` (Modified - ~30 lines changed)
3. `tests/test_threshold_consistency.py` (NEW - 358 lines)
4. `tests/test_threshold_fix_verification.py` (NEW - 350 lines)

## Verification Command
```bash
uv run python tests/test_threshold_consistency.py
uv run python tests/test_threshold_fix_verification.py
```

Both tests should pass with all assertions successful.