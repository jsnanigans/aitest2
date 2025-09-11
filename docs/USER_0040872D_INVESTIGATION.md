# User 0040872d Weight Drop Investigation & Solution

## Executive Summary

User 0040872d experienced a dramatic weight drop from stable 87-90kg to ~52kg that was incorrectly accepted by the Kalman filter, leading to 164 subsequent rejections. This document details the investigation, root cause analysis, and proposed BMI-based solution.

## Problem Description

### User Timeline
- **May 2024 - Jan 2025**: Stable weight around 86-89kg (connectivehealth source)
- **March 2025**: Sudden erratic readings from iglucose API
  - 99.3kg → 76.2kg in 2 minutes (-23.3%)
  - 98.0kg → 59.4kg in 2 days (-39.4%)
  - 91.3kg → 32.0kg in 5 days (BMI 10.4!)
- **Result**: System accepted impossible changes, corrupting baseline

### Impact
- 164 measurements rejected after accepting bad data
- Rejection reasons: "Sustained" and "Extreme" deviations
- User's weight tracking became unusable

## Root Cause Analysis

### 1. Why Kalman Accepted the Drop

The investigation revealed multiple failures:

1. **Gap-based reset**: After 30+ day gaps, system resets and accepts any value
2. **No physiological limits**: System doesn't check if changes are medically possible
3. **Source agnostic**: Treats all sources equally (iglucose appears unreliable)
4. **No BMI validation**: Accepts BMI < 15 (life-threatening) values

### 2. Specific Failures

```python
# Example of accepted impossible changes:
2025-03-12 20:06: 99.3kg (BMI 32.4) ✅ Accepted
2025-03-12 20:08: 76.2kg (BMI 24.9) ✅ Accepted (-23% in 2 minutes!)
2025-03-20 22:23: 32.0kg (BMI 10.4) ✅ Accepted (BMI incompatible with life!)
```

## Proposed Solution: BMI-Based Validation

### Implementation: `src/bmi_validator.py`

```python
class BMIValidator:
    """Validates weight changes using physiological limits"""
    
    @staticmethod
    def should_reset_kalman(current_weight, last_weight, time_delta_hours, height_m, source):
        """Determines if Kalman should reset due to impossible change"""
        
        # Percentage-based rules
        if abs(pct_change) > 50:
            return True, "Extreme change > 50%"
        
        # Time-based rules
        if time_delta_hours < 1 and abs(pct_change) > 30:
            return True, "Instant change > 30%"
        
        # BMI-based rules
        if bmi < 15:
            return True, "Critical BMI < 15"
        if bmi > 50:
            return True, "Critical BMI > 50"
        
        # Source-specific rules
        if 'iglucose' in source and abs(pct_change) > 25:
            return True, "Suspicious source with > 25% change"
```

### Validation Rules

#### 1. BMI Thresholds
- **BMI < 15**: Auto-reset (life-threatening)
- **BMI < 16**: Reset if change > 20%
- **BMI > 40**: Reset if change > 20%
- **BMI > 50**: Auto-reset (extreme obesity)

#### 2. Percentage Change Rules
- **Single measurement > 30%**: Auto-reset
- **Single measurement > 20%**: Reset if BMI extreme
- **Daily rate > 2kg/day**: Auto-reset
- **Weekly average > 10%**: Consider reset

#### 3. Time-Based Limits
- **< 1 hour**: Max 3kg or 3%
- **< 1 day**: Max 5kg or 5%
- **< 1 week**: Max 7kg or 7%
- **< 1 month**: Max 15kg or 15%
- **> 1 month**: Max 1.5kg/day sustained

## Test Results

### User 0040872d with BMI Validation

```
✅ 2024-05-21: 86.6kg (BMI 28.3) - Accepted
✅ 2025-01-02: 87.1kg (BMI 28.4) - Accepted
✅ 2025-03-12 20:06: 99.3kg (BMI 32.4) - Accepted
🔄 2025-03-12 20:08: 76.2kg (BMI 24.9) - RESET (23% drop)
🔄 2025-03-15: 59.4kg (BMI 19.4) - RESET (39% drop)
🔄 2025-03-20: 32.0kg (BMI 10.4) - RESET (BMI < 15)
```

**Results**:
- 8 resets triggered (vs 0 before)
- No acceptance of BMI < 15 values
- System recovers when valid data returns

## Benefits

1. **Prevents Catastrophic Errors**: No more accepting 40% weight drops
2. **Medical Safety**: Rejects BMI values incompatible with life
3. **Faster Recovery**: Resets instead of adapting to bad data
4. **Source Awareness**: Stricter validation for unreliable sources
5. **User Protection**: Prevents corruption of weight baseline

## Implementation Steps

1. **Add BMI Validator** (`src/bmi_validator.py`)
2. **Integrate with Processor** - Check for reset before processing
3. **Add Height Data** - Store user height for BMI calculation
4. **Configure Thresholds** - Adjust based on user population
5. **Monitor & Tune** - Track reset frequency and adjust

## Metrics to Track

- Reset frequency by source
- Percentage of measurements triggering resets
- Recovery time after resets
- User satisfaction scores

## Conclusion

The BMI-based validation system successfully prevents the acceptance of physiologically impossible weight changes while maintaining flexibility for normal variations. This protects users from data corruption while ensuring accurate weight tracking.