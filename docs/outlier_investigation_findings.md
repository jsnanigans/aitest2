# Investigation: Outlier Detection and Rejection Logic

## Date: 2025-09-14

## Summary
The system fails to reject obvious outliers (e.g., 100-110kg readings for users with 92kg baseline) because the rejection thresholds are too permissive and the Kalman filter adapts to outliers rather than rejecting them.

## The Complete Story

### 1. Trigger Point
**Location**: `src/processor.py:182-201`
**What happens**: Physiological validation is performed on incoming measurements
```python
validation_result = PhysiologicalValidator.validate_comprehensive(
    cleaned_weight,
    previous_weight=previous_weight,
    time_diff_hours=time_diff_hours,
    source=source
)
```

### 2. Processing Chain

#### Absolute Limits Check
**Location**: `src/validation.py:51-54`
**What happens**: Weight is checked against absolute physiological limits
**Why it matters**: Only rejects if weight > 400kg (extremely permissive)
```python
ABSOLUTE_MAX_WEIGHT = 400.0  # kg
if weight > PhysiologicalValidator.ABSOLUTE_MAX_WEIGHT:
    return False, f"Weight {weight:.1f}kg above absolute maximum"
```

#### Rate of Change Check
**Location**: `src/validation.py:67-92`
**What happens**: Daily rate of change is calculated and checked
**Why it matters**: Allows up to 6.44kg/day change
```python
MAX_DAILY_CHANGE_KG = 6.44
daily_rate = (weight_diff / time_diff_hours) * 24
if daily_rate > max_daily_change:
    return False, f"Change exceeds max rate"
```

#### Deviation from Kalman Prediction
**Location**: `src/processor.py:211-230`
**What happens**: Deviation from Kalman predicted weight is checked
**Why it matters**: Uses 20% threshold - too permissive
```python
extreme_threshold = 0.20  # From config
deviation = abs(cleaned_weight - predicted_weight) / predicted_weight
if deviation > extreme_threshold:
    return {'accepted': False, 'reason': f"Extreme deviation: {deviation:.1%}"}
```

### 3. Decision Points

For a user with 92kg baseline receiving 100kg measurement:
1. **Absolute limit**: 100kg < 400kg → PASS ✓
2. **Rate of change**: 8kg/day > 6.44kg → FAIL ✗ (but only if same day)
3. **Deviation**: (100-92)/92 = 8.7% < 20% → PASS ✓
4. **Result**: ACCEPTED (should be rejected!)

For 110kg measurement:
1. **Absolute limit**: 110kg < 400kg → PASS ✓
2. **Rate of change**: 18kg/day > 6.44kg → FAIL ✗ (but only if same day)
3. **Deviation**: (110-92)/92 = 19.6% < 20% → PASS ✓
4. **Result**: ACCEPTED (should be rejected!)

### 4. Final Outcome
**Location**: `src/processor.py:236-242`
**Result**: Outliers are accepted and Kalman filter adapts to them
**Root Cause**: Thresholds are too permissive for detecting outliers in normal weight ranges

## Key Insights

1. **Primary Cause**: The 20% deviation threshold is too high for users in normal weight ranges
2. **Contributing Factors**: 
   - Absolute limit of 400kg is meant for extreme cases, not outlier detection
   - Rate of change validation can be bypassed with time gaps
   - No statistical outlier detection (z-score based)
3. **Design Intent**: System was optimized to minimize false rejections but now under-rejects

## Evidence Trail

### Files Examined
- `src/processor.py`: Main processing logic and deviation check
- `src/validation.py`: Physiological limits and rate validation
- `src/constants.py`: Threshold definitions

### Test Results
```
Testing deviation from 92kg baseline:
- 100kg: 8.7% deviation (ACCEPTED - should reject)
- 105kg: 14.1% deviation (ACCEPTED - should reject)  
- 110kg: 19.6% deviation (ACCEPTED - borderline)
- 115kg: 25.0% deviation (REJECTED - correct)
```

## Confidence Assessment
**Overall Confidence**: High
**Reasoning**: Direct code inspection and simulation confirm the behavior
**Gaps**: Need to analyze actual user data to determine optimal thresholds

## Recommendations

### 1. Implement Adaptive Deviation Thresholds
```python
def get_deviation_threshold(baseline_weight):
    if baseline_weight < 100:
        return 0.10  # 10% for normal weight
    elif baseline_weight < 150:
        return 0.12  # 12% for overweight
    else:
        return 0.15  # 15% for obese
```

### 2. Add Statistical Outlier Detection
```python
def is_statistical_outlier(value, recent_values, z_threshold=3.0):
    mean = np.mean(recent_values)
    std = np.std(recent_values)
    z_score = abs(value - mean) / std
    return z_score > z_threshold
```

### 3. Implement Suspicious Range Rejection
Currently only warns for weights > 300kg, should reject or flag for review:
- Make suspicious range (300kg) actually reject, not just warn
- Add intermediate suspicious range (e.g., 150-300kg) with stricter validation

### 4. Add Context-Aware Validation
Consider multiple factors together:
- If weight > 100kg AND user baseline < 85kg → flag for review
- If BMI jump > 5 points in < 30 days → flag for review
- If multiple outlier indicators present → increase rejection likelihood

### 5. Prevent Kalman Adaptation to Outliers
- Don't update Kalman state if confidence is below threshold
- Use robust Kalman filter variant that's less sensitive to outliers
- Implement outlier detection before Kalman update, not after
