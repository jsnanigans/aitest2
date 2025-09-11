# Implementation Plan: BMI Validation and Deferred Reset

## Problem Statement

User 0040872d experienced catastrophic baseline corruption when the system accepted a drop from 87kg to 52kg (and even 32kg - BMI 10.4) after a gap. This corrupted the Kalman filter baseline, causing 164 subsequent valid measurements to be rejected.

## Root Cause

The reset mechanism (triggered after gaps) **bypasses ALL validation**, accepting any value regardless of physiological impossibility.

## Solution Overview

### 1. BMI Validation on Reset Path
- Never accept physiologically impossible values
- Validate BMI ranges (15-50 as critical thresholds)
- Check percentage changes (>50% = suspicious)
- Apply source-specific rules (stricter for 'iglucose')

### 2. Deferred Reset Processing
- Don't reset on first post-gap measurement
- Collect all measurements from the day
- At end-of-day, select the one closest to pre-gap baseline
- This reduces impact of erroneous readings

## Implementation Details

### Phase 1: BMI Validation (Immediate)

**File: `src/processor.py` (lines 124-137)**

Current problematic code:
```python
if delta > reset_gap_days:
    should_reset = True
    # Immediately accepts ANY value without validation!
    new_state = WeightProcessor._initialize_kalman_immediate(
        weight, timestamp, kalman_config
    )
```

Enhanced approach:
```python
if delta > reset_gap_days:
    # Validate BEFORE accepting
    last_weight = self._get_last_weight(state)
    
    # BMI validation if height available
    if height_m:
        bmi = weight / (height_m ** 2)
        if bmi < 15 or bmi > 50:
            return rejection_result(f"BMI {bmi:.1f} outside valid range")
    
    # Percentage change validation
    if last_weight:
        pct_change = abs(weight - last_weight) / last_weight
        if pct_change > 0.5:  # 50% threshold
            return rejection_result(f"Change of {pct_change:.1%} exceeds threshold")
    
    # Only reset if validation passes
    if validation_passed:
        should_reset = True
```

### Phase 2: Deferred Reset (Week 1)

**New functionality in `processor_enhanced.py`:**

1. **Mark measurements for deferred processing:**
```python
if delta > reset_gap_days:
    # Don't reset immediately
    state['pending_reset'] = {
        'triggered_at': timestamp,
        'gap_days': delta,
        'pre_gap_weight': last_weight,
        'measurements': []
    }
    
    # Collect this measurement
    state['pending_reset']['measurements'].append({
        'weight': weight,
        'source': source,
        'distance_from_baseline': abs(weight - last_weight)
    })
    
    return deferred_result()
```

2. **End-of-day processing:**
```python
def process_pending_resets(user_id):
    # Get all pending measurements
    measurements = state['pending_reset']['measurements']
    
    # Select closest to pre-gap baseline
    best = min(measurements, key=lambda m: m['distance_from_baseline'])
    
    # Reset with best value
    reset_kalman_with(best['weight'])
```

### Phase 3: Source Reliability (Week 2)

Track reliability scores:
- `patient-device`: 0.95 (highly reliable)
- `internal-questionnaire`: 0.70 (self-reported)
- `iglucose`: 0.40 (erratic, unreliable)
- `care-team-upload`: 0.80 (professional input)

Apply stricter validation for unreliable sources.

## Testing Strategy

### Test Case 1: BMI Validation
```python
# Should reject BMI < 15
baseline: 87kg (BMI 28.4)
after_gap: 32kg (BMI 10.4) → REJECT

# Should reject >50% change
baseline: 87kg
after_gap: 40kg (54% drop) → REJECT
```

### Test Case 2: Deferred Reset
```python
# Multiple measurements after gap
baseline: 80kg
day_after_gap:
  - 8am: 65kg (iglucose) - far
  - 12pm: 78kg (device) - close ✓
  - 6pm: 72kg (device) - medium
  
end_of_day: Select 78kg (closest to 80kg)
```

### Test Case 3: Source-Specific
```python
# Same change, different sources
baseline: 85kg
after_gap: 60kg (29% drop)
  - from device: ACCEPT (reliable)
  - from iglucose: REJECT (unreliable + large change)
```

## Success Metrics

1. **No impossible BMIs accepted** (none < 13 or > 55)
2. **Reduced false rejections** after gaps
3. **Better baseline continuity** via closest-value selection
4. **Source-aware processing** reduces erratic data impact

## Risk Mitigation

- **Backward compatibility**: Keep original processor, use enhanced version for new deployments
- **Gradual rollout**: Test with known problematic users first
- **Monitoring**: Track reset decisions and validation rejections
- **Fallback**: Can disable deferred processing via config flag

## Configuration

```python
processing_config = {
    "enable_bmi_validation": True,
    "bmi_min": 15.0,
    "bmi_max": 50.0,
    "max_reset_change_pct": 0.5,  # 50%
    "enable_deferred_reset": True,
    "source_reliability": {
        "patient-device": 0.95,
        "iglucose": 0.40,
        "internal-questionnaire": 0.70
    }
}
```

## Timeline

- **Day 1**: Implement BMI validation in reset path
- **Day 2-3**: Test with user 0040872d data
- **Week 1**: Implement deferred reset processing
- **Week 2**: Add source reliability scoring
- **Week 3**: Production deployment with monitoring

## Expected Outcome

For user 0040872d:
- **Before**: Accepted 52kg drop, corrupted baseline, 164 rejections
- **After**: Rejects impossible values, selects plausible reset value, maintains valid baseline

This prevents catastrophic baseline corruption while maintaining system flexibility for legitimate weight changes.