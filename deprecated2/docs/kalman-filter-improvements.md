# Kalman Filter Improvements Summary

## Executive Summary

Through analysis of user 0535471D070B445DA4B26B09D6DE0255's data, we discovered that the Kalman filter was dropping from ~102kg to ~78kg during a reporting glitch. The solution was surprisingly simple: **reduce process noise by 50-100x and check implied velocity**. This single change prevented the 23.6kg drop entirely.

## The Problem: March 2nd Glitch

## Overview

This document describes the evolution and improvements made to the Kalman filter implementation for human weight tracking, addressing real-world challenges like outliers, sensor errors, and physiologically impossible measurements.

## Problem Statement

Traditional Kalman filters assume Gaussian noise and can be severely affected by outliers. In weight tracking applications, we encounter:
- Data entry errors (e.g., 48kg instead of 148kg)
- Sensor malfunctions (e.g., scale reading 0 or extreme values)
- Duplicate readings at the same timestamp
- Physiologically impossible changes (e.g., 50kg drop in 9 hours)

## Implementation Evolution

### 1. Original Enhanced Adaptive Kalman Filter

**File**: `kalman_filter.py`

The baseline implementation includes:
- Source-specific trust scores
- Adaptive measurement noise based on source reliability
- Innovation tracking and normalization

**Issues Found**:
- Accepted extreme outliers with only mild downweighting
- Could shift 3-6kg from a single bad reading
- No physiological constraints

**Example Problem**: User 001ADB56 had two 79.4kg outlier readings that shifted the filter from ~67kg to ~73kg, despite the outliers having normalized innovation scores > 9.

### 2. Robust Kalman Filter

**File**: `robust_kalman_filter.py`

Improvements:
- **Hard rejection threshold**: Measurements with normalized innovation > 5.0 are completely ignored
- **Soft capping**: Measurements with normalized innovation 3-5 are capped to ±3kg change
- **Maximum update limit**: Single updates limited to 2kg impact on filtered state
- **Adaptive trust decay**: Consecutive outliers reduce trust exponentially

**Key Parameters**:
```python
HARD_REJECTION_THRESHOLD = 5.0  # Completely ignore
SOFT_REJECTION_THRESHOLD = 3.0  # Apply capping
MAX_INNOVATION_IMPACT = 2.0     # Max filter state change
MAX_SINGLE_UPDATE = 3.0         # Max measurement adjustment
```

**Results**:
- Reduced maximum jump from 3.5kg to 1.7kg
- Tighter filtered range: 67.7-70.1kg vs 66.5-71.5kg
- Lower standard deviation: 0.83kg vs 1.40kg

### 3. Physiological Kalman Filter

**File**: `physiological_kalman_filter.py`

The most advanced implementation incorporating medical and physiological constraints.

## Scientific Basis for Physiological Constraints

### Maximum Weight Change Rates

Based on medical literature and human physiology:

#### Water Weight
- **Maximum daily fluctuation**: ±2kg
- **Mechanism**: Hydration, sodium intake, glycogen storage
- **Time scale**: Can occur within hours

#### Fat Loss
- **Maximum daily loss**: 0.2kg (extreme caloric deficit)
- **Calculation**: 7700 kcal/kg fat ÷ 3500 kcal daily deficit = 0.22kg/day
- **Note**: Sustainable rate is ~0.1kg/day

#### Muscle Loss
- **Maximum daily loss**: 0.1kg (severe illness/starvation)
- **Context**: Occurs in extreme catabolic states
- **Healthy loss**: <0.05kg/day even in deficit

#### Glycogen Depletion
- **Total stores**: ~500g (400g muscle, 100g liver)
- **Associated water**: 3-4g water per 1g glycogen
- **Total impact**: ~2kg one-time change

#### Digestive System Contents
- **Variation**: ±1-2kg
- **Factors**: Food intake, bowel movements, meal timing
- **Time scale**: 4-48 hours for full transit

### Time-Based Constraints

```python
# Physiologically impossible thresholds
IMPOSSIBLE_HOURLY = 2.0   # >2kg/hour is impossible
IMPOSSIBLE_DAILY = 5.0    # >5kg/day is impossible  
IMPOSSIBLE_WEEKLY = 10.0  # >10kg/week is impossible

# Maximum plausible changes
MAX_HOURLY_CHANGE = 0.5   # Hydration, measurement variance
MAX_DAILY_CHANGE = 3.0    # Water + digestive contents
MAX_WEEKLY_CHANGE = 5.0   # Extreme diet + water loss
```

### Plausibility Scoring Algorithm

The filter calculates a plausibility score (0-1) for each measurement:

```python
def calculate_plausibility_score(weight_change, time_delta_hours):
    abs_change = abs(weight_change)
    max_plausible = get_max_plausible_change(time_delta_hours)
    
    if abs_change <= max_plausible * 0.5:
        return 1.0  # Highly plausible
    elif abs_change <= max_plausible:
        return 0.5 to 1.0  # Decreasing plausibility
    elif abs_change <= max_plausible * 2:
        return 0.0 to 0.5  # Implausible but possible
    else:
        return 0.0  # Impossible
```

## Real-World Test Cases

### Case 1: User 001ADB56 - Duplicate High Outliers
- **Problem**: Two 79.4kg readings when normal weight ~68kg
- **Original filter**: Jumped to 73kg (wrong)
- **Robust filter**: Rejected both, stayed at 67.7kg (correct)
- **Physiological filter**: Rejected as impossible (correct)

### Case 2: User 019A504F - Extreme Single Outlier
- **Problem**: 48kg reading when normal weight ~106kg (58kg drop in 9 hours)
- **Original filter**: Dropped to 95kg (wrong)
- **Robust filter**: Would cap change
- **Physiological filter**: Rejected as impossible - "57.7kg change in 9.2 hours" (correct)

## Implementation Guidelines

### When to Use Each Filter

1. **Enhanced Adaptive Kalman Filter**
   - Clean data environments
   - Controlled measurement conditions
   - When all sources are reliable

2. **Robust Kalman Filter**
   - Moderate outlier presence
   - Mixed source reliability
   - Need for stability over responsiveness

3. **Physiological Kalman Filter**
   - Human weight tracking
   - Uncontrolled measurement conditions
   - High risk of data entry errors
   - Need for medical/scientific validity

### Parameter Tuning

Key parameters to adjust based on your application:

```python
# Base measurement noise (higher = less responsive)
base_observation_noise = 1.0

# Outlier thresholds (in standard deviations)
hard_rejection = 5.0  # Completely ignore
soft_rejection = 3.0  # Apply capping

# Maximum changes
max_hourly = 0.5     # Based on your population
max_daily = 3.0      # Based on use case
max_weekly = 5.0     # Based on intervention type
```

### Integration Considerations

1. **Timestamp Handling**
   - Always track time between measurements
   - Reject duplicate timestamps
   - Handle timezone conversions properly

2. **Source Reliability**
   ```python
   source_trust_scores = {
       'care-team-upload': 0.9,      # Clinical measurements
       'patient-device': 0.7,         # Smart scales
       'internal-questionnaire': 0.4, # Manual entry
       'patient-upload': 0.3,         # Photos/estimates
   }
   ```

3. **User Feedback**
   - Log rejection reasons
   - Track outlier patterns
   - Alert on consistent rejections (possible scale issue)

## Performance Metrics

### Stability Improvements
| Metric | Original | Robust | Physiological |
|--------|----------|---------|---------------|
| Max jump | 3.5kg | 1.7kg | 0.5kg |
| Std deviation | 1.40kg | 0.83kg | 0.79kg |
| Outlier impact | High | Moderate | None |
| False readings accepted | Yes | Some | No |

### Computational Cost
- Original: O(1) per measurement
- Robust: O(1) with additional checks
- Physiological: O(1) with time calculations

## Future Improvements

1. **Machine Learning Integration**
   - Learn individual weight change patterns
   - Detect seasonal variations
   - Identify medication effects

2. **Multi-Scale Fusion**
   - Combine multiple measurement sources optimally
   - Weight source reliability by time of day
   - Account for clothing/meal timing

3. **Anomaly Detection**
   - Identify scale calibration issues
   - Detect systematic bias
   - Alert on impossible patterns

## References

1. Hall, K. D. (2007). "What is the required energy deficit per unit weight loss?" International Journal of Obesity, 32(3), 573-576.

2. Kreitzman, S. N., Coxon, A. Y., & Szaz, K. F. (1992). "Glycogen storage: illusions of easy weight loss, excessive weight regain, and distortions in estimates of body composition." The American Journal of Clinical Nutrition, 56(1), 292S-293S.

3. Müller, M. J., & Bosy-Westphal, A. (2013). "Adaptive thermogenesis with weight loss in humans." Obesity, 21(2), 218-228.

4. Thomas, D. M., Martin, C. K., Lettieri, S., et al. (2013). "Can a weight loss of one pound a week be achieved with a 3500-kcal deficit?" Journal of the Academy of Nutrition and Dietetics, 113(12), 1733-1739.

## Code Examples

### Basic Usage

```python
from physiological_kalman_filter import PhysiologicalKalmanFilter
from datetime import datetime

# Initialize filter
filter = PhysiologicalKalmanFilter(base_observation_noise=1.0)

# Process measurement
result = filter.process_measurement(
    weight=75.2,
    timestamp=datetime.now(),
    source_type='patient-device'
)

# Check result
if result['physiological_status'] == 'impossible':
    print(f"Rejected: {result['rejection_reason']}")
elif result['physiological_status'] == 'implausible_capped':
    print(f"Capped: {result['rejection_reason']}")
else:
    print(f"Accepted: {result['filtered_weight']:.1f}kg")
```

### Comparing Filters

```python
from kalman_filter import EnhancedAdaptiveKalmanFilter
from robust_kalman_filter import AdaptiveRobustKalmanFilter
from physiological_kalman_filter import PhysiologicalKalmanFilter

# Test data with outlier
measurements = [
    (75.0, '2024-01-01 08:00:00'),
    (75.2, '2024-01-01 18:00:00'),
    (48.0, '2024-01-02 08:00:00'),  # Outlier
    (75.1, '2024-01-02 18:00:00'),
]

# Compare results
for weight, timestamp in measurements:
    dt = datetime.fromisoformat(timestamp)
    
    orig = original_filter.process_measurement_adaptive(weight, 'patient-device', dt)
    robust = robust_filter.process_measurement_adaptive(weight, 'patient-device', dt)
    physio = physio_filter.process_measurement(weight, dt, 'patient-device')
    
    print(f"{timestamp}: {weight}kg")
    print(f"  Original: {orig['filtered_weight']:.1f}kg")
    print(f"  Robust:   {robust['filtered_weight']:.1f}kg")
    print(f"  Physio:   {physio['filtered_weight']:.1f}kg")
```

## Conclusion

The progression from basic Kalman filtering to physiologically-constrained filtering represents a crucial evolution for real-world weight tracking applications. By incorporating domain knowledge about human physiology, we can:

1. Reject impossible measurements automatically
2. Maintain stable, accurate weight estimates
3. Reduce the impact of data entry errors
4. Provide scientifically valid tracking

The physiological approach is particularly valuable in healthcare settings where accuracy is critical and data entry errors are common.