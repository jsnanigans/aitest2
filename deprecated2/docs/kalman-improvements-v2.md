# Kalman Filter Improvements v2: Simple & Effective

## Executive Summary

Through analysis of user 0535471D070B445DA4B26B09D6DE0255's data (a reporting glitch with measurements ranging from 30-115kg within minutes), we discovered the Kalman filter was incorrectly dropping from ~102kg to ~78kg. 

**The solution was surprisingly simple**: Reduce process noise by 50-100x and check implied velocity. This prevented the 23.6kg drop entirely.

## Problem: March 2nd Reporting Glitch

User 0535471D070B445DA4B26B09D6DE0255 experienced a device glitch on March 2, 2025, with 20 measurements in ~3 hours:
- Measurements ranged wildly: 30kg, 32kg, 39kg, 43kg, 46kg, 50kg, 55kg, 62kg, 68kg, 75kg, 79kg, 82kg, 98kg, 100kg, 115kg
- All within minutes of each other
- True weight was ~110kg based on surrounding days

**Original Filter Behavior:**
- Dropped from 102.3kg → 78.7kg (23.6kg drop!) ❌
- Partially recovered after glitch ended
- Clearly wrong - humans can't lose 23kg in 3 hours

## Root Cause: Process Noise Too High

The original filter had:
```python
process_noise_weight = 0.5   # Too high!
process_noise_trend = 0.01   # Too high!
measurement_noise = 1.0       # Too low!
```

This told the filter: "Weight can jump around randomly by 0.5kg² variance" - which is wrong! Human weight has inertia.

## Solution 1: Reduce Process Noise (Most Effective)

```python
# New defaults - 50-100x reduction!
process_noise_weight = 0.01   # Weight has inertia
process_noise_trend = 0.001   # Velocity changes gradually  
measurement_noise = 2.0        # More realistic scale noise
```

**Why it works:** This tells the filter that weight follows physics, not random walks. The filter now trusts its model more than wild measurements.

## Solution 2: Velocity-Aware Measurement Noise

```python
# Calculate what velocity this measurement implies
implied_velocity = (weight - predicted_weight) / time_delta_days

# If physically impossible, dramatically increase measurement noise
if abs(implied_velocity) > max_reasonable_trend:  # 0.5 kg/day
    impossibility_factor = abs(implied_velocity) / max_reasonable_trend
    measurement_noise *= (impossibility_factor ** 2)
```

**Example:** 
- 32kg measurement when expecting 102kg = -70kg change
- Time delta: 0.1 days (2.4 hours)
- Implied velocity: -700 kg/day (!!)
- Impossibility factor: 1400
- Measurement noise scaled by: 1,960,000x
- Result: Measurement effectively ignored ✓

## Solution 3: Time-Adaptive Process Noise

```python
if time_delta_days < 0.1:  # Minutes/hours
    # Very little can change - use tiny process noise
    q_weight = base_process_noise * time_delta_days
elif time_delta_days < 7.0:  # Days
    # Normal scaling
    q_weight = base_process_noise * time_delta_days
else:  # Weeks
    # More uncertainty over long gaps
    q_weight = base_process_noise * sqrt(time_delta_days) * 2
```

**Benefit:** Handles both rapid glitches AND long vacation gaps correctly.

## Results: Night and Day Difference

### Test with Original Parameters
```
Pre-glitch:  102.3kg
32kg measurement → Filter: 100.5kg (dropped 1.8kg)
68kg measurement → Filter: 94.8kg (dropped 7.5kg)  
79kg measurement → Filter: 88.4kg (dropped 14kg)
Post-glitch: 78.7kg (total drop: 23.6kg) ❌
```

### Test with Improved Parameters
```
Pre-glitch:  109.8kg
32kg measurement → Filter: 109.8kg (R scaled by 977,540x)
68kg measurement → Filter: 109.8kg (R scaled by 61,033,875x)
79kg measurement → Filter: 109.8kg (R scaled by 17,885,927x)
Post-glitch: 109.8kg (total drop: 0.0kg) ✅
```

## Key Insights

### 1. Process Noise Is Critical
The single most important change was reducing process noise. This parameter fundamentally controls how much the filter trusts its physics model vs measurements.

### 2. Physics Over Statistics
Statistical outlier detection (3-sigma rule) isn't enough. Physical constraints (max possible velocity) are more reliable for impossible measurements.

### 3. Weight Has Inertia
Human weight doesn't randomly jump. It follows physics:
- Maximum realistic change: ~0.5 kg/day
- Water weight fluctuation: ±2kg over days
- Extreme diet/illness: ~1 kg/week
- Impossible: 20kg in hours

## Implementation in TrendKalmanFilter

The improvements are now in `src/filters/kalman_filter.py`:

```python
class TrendKalmanFilter:
    def __init__(self,
        process_noise_weight=0.01,  # Was 0.5
        process_noise_trend=0.001,  # Was 0.01  
        measurement_noise=2.0,       # Was 1.0
        max_reasonable_trend=0.5
    ):
        # ...
        
    def process_measurement(self, weight, timestamp):
        # Time-adaptive process noise
        if time_delta_days < 0.1:
            q_weight = base * time_delta_days
        # ...
        
        # Velocity checking
        implied_velocity = innovation / time_delta_days
        if abs(implied_velocity) > max_reasonable_trend:
            factor = abs(implied_velocity) / max_reasonable_trend
            measurement_noise *= (factor ** 2)
```

## Backwards Compatibility

All changes maintain API compatibility. Existing code automatically benefits from:
- Better default parameters
- Velocity-aware outlier rejection  
- Time-adaptive behavior

Only new field in results: `implied_velocity` for diagnostics.

## Testing & Validation

Test file: `tests/test_duplicate_timestamp_edge_case.py`
- Tests the exact March 2nd glitch scenario
- Verifies filter maintains weight around 110kg
- Confirms outliers are properly detected
- Validates long-gap handling still works

## Conclusion

Sometimes the best solution is the simplest. By recognizing that **weight has physical inertia** and adjusting process noise accordingly, we transformed the filter from fragile to robust. No complex heuristics needed - just better physics modeling through proper parameter tuning.

The filter now correctly:
- Ignores device glitches (impossibly fast changes)
- Tracks real weight changes (gradual trends)
- Handles data gaps (adaptive uncertainty)
- Maintains stability (physics-based constraints)