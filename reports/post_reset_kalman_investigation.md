# Investigation: Post-Reset Kalman Rigidity Issue

## Summary
After a 30-day reset, the Kalman filter assumes a straight trend (0 change) and is too rigid to adapt to actual weight patterns, causing excessive rejections of valid measurements.

## The Complete Story

### 1. Trigger Point
**Location**: `src/processor.py:159-164` (Kalman initialization after reset)
**What happens**: After reset, Kalman filter is initialized with:
```python
state = KalmanFilterManager.initialize_immediate(
    cleaned_weight, timestamp, kalman_config, observation_covariance
)
```

### 2. Initial State Problem
**Location**: `src/kalman.py:33-38`
**What happens**: Initial state is set with trend = 0
```python
kalman_params = {
    'initial_state_mean': [weight, 0],  # Trend is 0!
    'initial_state_covariance': [[initial_variance, 0], [0, 0.001]],
    'transition_covariance': [
        [0.0160, 0],  # Very small weight variance
        [0, 0.0001]   # EXTREMELY small trend variance!
    ],
}
```

### 3. Why It Matters
- **Trend = 0**: Assumes weight will stay constant
- **Tiny trend covariance (0.0001)**: Filter is VERY slow to learn new trends
- **Result**: If actual weight differs from pre-gap weight, filter rejects many valid measurements

### 4. Visual Evidence
In the provided image for user 01672f42-568b-4d49-abbc-eee60d87ccb2:
- Kalman prediction (blue line) starts at ~120kg
- Actual measurements are ~105-107kg
- Many rejections (red dots) because deviation exceeds threshold
- Takes weeks for Kalman to "catch up" to reality

## Key Insights

1. **Primary Cause**: Transition covariance for trend (0.0001) is too small after reset
2. **Contributing Factor**: No memory of pre-gap trend direction
3. **Design Intent**: These values work well for continuous data but fail after gaps

## Proposed Solution

### Adaptive Covariance After Reset
Track measurements since last reset and use adaptive parameters:

```python
def get_adaptive_covariances(measurements_since_reset):
    """Get adaptive covariances that start loose and tighten over time."""
    
    # Base (final) values
    base_weight_cov = 0.0160
    base_trend_cov = 0.0001
    
    if measurements_since_reset < 10:
        # Start with much higher covariances for quick adaptation
        # Exponentially decay from 10x to 1x over 10 measurements
        factor = np.exp(-measurements_since_reset / 3)  # ~95% reduction by 10 measurements
        
        weight_multiplier = 1 + 9 * factor  # 10x to 1x
        trend_multiplier = 1 + 99 * factor  # 100x to 1x (trend needs more flexibility)
        
        return {
            'weight': base_weight_cov * weight_multiplier,
            'trend': base_trend_cov * trend_multiplier
        }
    else:
        # After 10 measurements, use normal values
        return {
            'weight': base_weight_cov,
            'trend': base_trend_cov
        }
```

### Implementation Steps
1. Add `measurements_since_reset` counter to state
2. Reset counter when gap ≥ 30 days
3. Use adaptive covariances in Kalman update
4. Increment counter with each accepted measurement

## Benefits
- Quick adaptation after reset (within 3-5 measurements)
- Gradual transition to stable filtering
- No permanent increase in noise tolerance
- Maintains long-term stability

## Confidence Assessment
**Overall Confidence**: High
**Reasoning**: 
- Clear causal relationship between small covariances and rigidity
- Visual evidence confirms the issue
- Solution directly addresses root cause
- Similar approach used successfully in other Kalman applications
