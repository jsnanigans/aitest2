# Current Kalman Filter Implementation

## Overview

This document describes the current state of our Kalman filter implementation in `/src/filters/simple_kalman.py`, how it works, and how to extend it for future requirements.

## Current Implementation

We have implemented a **simple, robust 1D Kalman filter** that effectively handles outliers while maintaining simplicity and performance.

### Core Components

#### 1. SimpleKalmanFilter (Base Implementation)

**Purpose**: Track weight measurements with automatic outlier rejection.

**State Vector**: 
- `x = [weight]` - Single dimension tracking current weight (kg)

**Model**:
- **State Transition**: Random walk model `x(t+1) = x(t)` (weight remains constant between measurements)
- **Process Noise (Q)**: 0.5 kg² - represents natural daily weight variation
- **Measurement Noise (R)**: 1.0 kg² - represents measurement uncertainty

**Key Features**:
```python
class SimpleKalmanFilter:
    def __init__(self, initial_weight=75.0, process_noise=0.5, measurement_noise=1.0):
        # PyKalman filter with configurable parameters
        self.kf = KalmanFilter(
            initial_state_mean=[initial_weight],
            initial_state_covariance=[[10.0]],  # High initial uncertainty
            transition_matrices=[[1.0]],         # Random walk
            observation_matrices=[[1.0]],        # Direct observation
            transition_covariance=[[process_noise]],
            observation_covariance=[[measurement_noise]]
        )
```

**State Representation**:
```python
# Current state maintained as:
self.current_mean = np.array([weight])           # Estimated weight
self.current_covariance = np.array([[variance]]) # Estimation uncertainty
self.measurement_count = 0                       # Number of processed measurements
```

**Outlier Rejection Algorithm**:
```python
# Automatic outlier handling based on normalized innovation
if normalized_innovation > 3.0:  # >3 sigma from prediction
    scale_factor = normalized_innovation / 3.0
    if normalized_innovation > 6.0:  # Extreme outliers
        scale_factor = scale_factor ** 2  # Exponential scaling
    
    # Increase measurement noise R to reduce outlier impact
    adjusted_R = base_R * scale_factor
```

#### 2. AdaptiveKalmanFilter (Extended Implementation)

**Purpose**: Dynamically adjust measurement noise based on data source quality.

**Additional Features**:
- Source-specific reliability mapping
- Time-of-day adjustments
- Gap penalty for long periods between measurements

**Source Reliability Mapping**:
```python
self.source_reliability = {
    'care-team-upload': 0.3,       # Most reliable (clinical)
    'patient-device': 0.5,          # Smart scales
    'internal-questionnaire': 1.0,  # Manual entry
    'patient-upload': 2.0,          # Photo uploads (least reliable)
    'unknown': 1.5
}
```

**Adaptive R Calculation**:
```python
def process_measurement_adaptive(weight, source_type, hours_since_last, hour_of_day):
    # Base R from source type
    base_R = self.source_reliability.get(source_type, 1.5)
    
    # Time factors
    time_factor = 1.0 if (5 <= hour_of_day <= 9) else 1.5  # Morning more reliable
    gap_factor = 1.0 + (hours_since_last / 168.0)         # Uncertainty increases with time
    
    # Final adaptive R
    adaptive_R = base_R * time_factor * gap_factor
```

## How the Filter Works

### Processing Flow

1. **Initialization**: First measurement initializes the state
2. **Prediction**: Estimate next weight based on current state
3. **Innovation**: Calculate difference between measurement and prediction
4. **Outlier Check**: Compute normalized innovation (σ-normalized)
5. **Adaptive Update**: Adjust R if outlier detected
6. **State Update**: Update state estimate with Kalman gain
7. **Return Results**: Provide filtered weight and uncertainty

### Key Algorithms

**Innovation and Outlier Detection**:
```python
# Innovation: difference between measurement and prediction
innovation = measured_weight - predicted_weight

# Normalized innovation for outlier detection
normalized_innovation = abs(innovation) / uncertainty

# Outlier thresholds:
# > 3σ: Likely outlier (increase R)
# > 6σ: Extreme outlier (exponentially increase R)
```

**Missing Data Handling**:
```python
def handle_missing_data(time_gap_days):
    # Predict forward without measurements
    for _ in range(time_gap_days):
        # Prediction-only update (no measurement)
        self.kf.filter_update(observation=None)
    
    # Uncertainty naturally increases with time
```

## Configuration and Parameters

### Tunable Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `initial_weight` | 75.0 kg | Starting weight estimate | Sets initial state |
| `process_noise` (Q) | 0.5 kg² | Daily weight variation | Higher = more responsive to changes |
| `measurement_noise` (R) | 1.0 kg² | Measurement uncertainty | Higher = less trust in measurements |
| `outlier_threshold` | 3.0σ | Outlier detection threshold | Higher = fewer outliers detected |

### Source Trust Configuration

Edit the `source_reliability` dictionary in `AdaptiveKalmanFilter`:

```python
# Current settings (lower = more trusted)
'care-team-upload': 0.3      # Clinical measurements
'patient-device': 0.5         # Smart scales  
'internal-questionnaire': 1.0 # Manual entry
'patient-upload': 2.0         # Photo uploads
```

## How to Extend the Filter

### Extension Option 1: Add Trend Tracking (2D State)

To track weight change trends:

```python
class TrendKalmanFilter(SimpleKalmanFilter):
    def __init__(self):
        # State: [weight, trend_kg_per_day]
        self.kf = KalmanFilter(
            initial_state_mean=[75.0, 0.0],
            transition_matrices=[
                [1.0, 1.0],  # weight += trend * dt
                [0.0, 1.0]   # trend remains constant
            ],
            observation_matrices=[[1.0, 0.0]],  # Only observe weight
            # ... other parameters
        )
    
    def get_trend(self):
        return self.current_mean[1]  # kg/day
```

### Extension Option 2: Add Physiological Model (3D State)

To model hydration/food effects:

```python
class PhysiologicalKalmanFilter(SimpleKalmanFilter):
    def __init__(self):
        # State: [true_weight, trend, hydration_offset]
        # Hydration is mean-reverting (returns to 0)
        self.kf = KalmanFilter(
            initial_state_mean=[75.0, 0.0, 0.0],
            transition_matrices=[
                [1.0, 1.0, 0.0],  # weight += trend
                [0.0, 1.0, 0.0],  # trend constant
                [0.0, 0.0, 0.8]   # hydration decays
            ],
            observation_matrices=[[1.0, 0.0, 1.0]],  # observe weight + hydration
        )
```

### Extension Option 3: Add Custom Constraints

Add domain-specific rules:

```python
class ConstrainedKalmanFilter(SimpleKalmanFilter):
    MAX_DAILY_CHANGE = 3.0  # kg/day physiological limit
    
    def process_measurement(self, weight):
        # Check physiological constraints
        if self.current_mean is not None:
            change = abs(weight - self.current_mean[0])
            if change > self.MAX_DAILY_CHANGE:
                # Reject or heavily downweight
                return self.handle_impossible_reading(weight)
        
        return super().process_measurement(weight)
```

### Extension Option 4: Add Parameter Learning

Use EM algorithm for automatic tuning:

```python
def optimize_for_user(historical_weights):
    # Learn optimal Q and R from data
    kf = KalmanFilter(n_dim_state=1, n_dim_obs=1)
    kf = kf.em(historical_weights, n_iter=10)
    
    learned_Q = kf.transition_covariance[0, 0]
    learned_R = kf.observation_covariance[0, 0]
    
    # Create optimized filter
    return SimpleKalmanFilter(
        process_noise=learned_Q,
        measurement_noise=learned_R
    )
```

## Integration Points

### Main System Integration (`main.py`)

```python
class UnifiedStreamProcessor:
    def start_user(self, user_id):
        if self.enable_kalman:
            if self.config.get('use_adaptive_kalman'):
                self.kalman_filter = AdaptiveKalmanFilter()
            else:
                self.kalman_filter = SimpleKalmanFilter()
    
    def process_reading(self, reading):
        # Process with Kalman filter
        kalman_result = self.kalman_filter.process_measurement(weight)
        
        # Store results
        self.kalman_results.append({
            'filtered_weight': kalman_result['filtered_weight'],
            'uncertainty': kalman_result['uncertainty'],
            'innovation': kalman_result['innovation'],
            # ...
        })
```

### Configuration (`config.toml`)

```toml
# Feature flags
enable_kalman = true
use_adaptive_kalman = false  # Use adaptive filter

# Source trust scores (for adaptive filter)
[source_type_trust_scores]
"care-team-upload" = 0.5
"patient-device" = 0.1
"internal-questionnaire" = 0.3
"patient-upload" = 0.2
```

### Output Structure

```json
{
  "kalman_summary": {
    "filter_initialized": true,
    "total_measurements": 67,
    "mean_innovation": -0.043,
    "mean_uncertainty": 0.712,
    "final_filtered_weight": 94.17,
    "kalman_outliers": 3,
    "kalman_outlier_rate": 0.045
  },
  "kalman_time_series": [
    {
      "date": "2025-01-17",
      "measured_weight": 95.6,
      "filtered_weight": 95.6,
      "uncertainty": 0.956,
      "innovation": 0.0,
      "normalized_innovation": 0.0
    },
    // ...
  ]
}
```

## Performance Characteristics

### Computational Performance
- **Processing speed**: ~0.1ms per measurement
- **Memory usage**: O(1) - constant regardless of data size
- **Scalability**: Linear with number of measurements

### Statistical Performance
- **Outlier rejection**: Successfully rejects >3σ outliers
- **Convergence**: Typically within 5-10 measurements
- **Uncertainty quantification**: Proper confidence bounds

### Benchmark Results

From `benchmark_kalman.py` testing:

| Test Case | Description | Result |
|-----------|-------------|--------|
| User with 61kg outliers | ~86kg baseline with erroneous 61kg readings | ✅ PASS - Max deviation 0.79kg |
| User with 32-34kg outliers | ~118kg baseline with extreme outliers | ✅ PASS - Max deviation 0.45kg |

## Best Practices for Extension

### 1. Maintain Backward Compatibility
```python
class ExtendedFilter(SimpleKalmanFilter):
    def __init__(self, *args, new_param=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_param = new_param
```

### 2. Add Comprehensive Testing
```python
def test_extension():
    filter = ExtendedFilter()
    
    # Test with normal data
    assert filter.process_measurement(75.0) is not None
    
    # Test with outliers
    result = filter.process_measurement(150.0)  # Extreme outlier
    assert result['normalized_innovation'] > 3.0
```

### 3. Document State Changes
```python
class ExtendedFilter(SimpleKalmanFilter):
    """
    Extended Kalman filter with [describe new features].
    
    State Vector:
    - x[0]: weight (kg)
    - x[1]: [new state] ([units])
    
    New Parameters:
    - new_param: [description]
    """
```

### 4. Preserve PyKalman Integration
Always use PyKalman's methods for numerical stability:
```python
# Good - use PyKalman's filter_update
self.kf.filter_update(state, covariance, observation)

# Bad - manual matrix operations
kalman_gain = P @ H.T @ inv(S)  # Avoid this
```

## Future Roadmap

### High Priority
1. **Add trend tracking** - Enable weight change monitoring
2. **Parameter learning** - Auto-tune Q and R per user
3. **Batch processing** - Efficient multi-user initialization

### Medium Priority
4. **Time-of-day patterns** - Learn and compensate for daily cycles
5. **Seasonal adjustments** - Handle holiday/vacation patterns
6. **Multi-device fusion** - Combine multiple scale readings

### Low Priority
7. **Advanced physiological model** - Full 3D state space
8. **Predictive capabilities** - Forecast future weights
9. **Anomaly explanations** - Identify outlier causes

## Troubleshooting

### Common Issues and Solutions

**Issue**: Filter not converging
- **Solution**: Check initial uncertainty is high enough (default 10.0)

**Issue**: Too sensitive to outliers
- **Solution**: Increase outlier threshold or base R value

**Issue**: Too slow to adapt to real changes
- **Solution**: Increase process noise Q or decrease R

**Issue**: Uncertainty growing unbounded
- **Solution**: Ensure measurements are being processed correctly

## Code References

- **Simple filter**: `src/filters/simple_kalman.py:10-89`
- **Adaptive filter**: `src/filters/simple_kalman.py:149-204`
- **Parameter learning**: `src/filters/simple_kalman.py:91-147`
- **Integration**: `main.py:94-100`
- **Configuration**: `config.toml:8-10`
- **Benchmarking**: `benchmark_kalman.py`

## Summary

The current Kalman filter implementation provides:

1. **Robust outlier handling** through adaptive measurement noise
2. **Simple, maintainable code** using PyKalman
3. **Proven effectiveness** on real problematic data
4. **Easy extensibility** for future requirements
5. **Production-ready integration** with the main system

The filter successfully balances mathematical rigor with practical robustness, making it suitable for production use while remaining simple enough to understand and extend.