# Kalman Prediction Timing Issue

## Problem Summary

The `calculate_kalman_fit` method in `src/processing/unified_quality_scorer.py` is comparing measurements to an incorrect "prediction" that's calculated from the already-updated Kalman state, rather than a true prediction made before incorporating the measurement.

## Current (Incorrect) Implementation

In `src/processing/processor.py:228-236`, the code:
1. Gets the **current** state values from the Kalman filter (after last update)
2. Projects these values forward: `kalman_prediction = current_weight + current_trend * time_delta_days`
3. Passes this to quality scoring

This is problematic because `current_weight` and `current_trend` are the posterior values from AFTER the last measurement was incorporated, not predictions for the current timestamp.

## What Should Happen

The correct Kalman filter predict-update cycle:
1. **Predict Step**: Take the posterior state from the previous update and propagate it forward to the current timestamp using the transition matrix
2. **Innovation**: Compare this prediction to the actual measurement (this is what quality scoring needs)
3. **Update Step**: Incorporate the measurement to get the new posterior state

## Technical Details

### Kalman Prediction Formula
For a state `[weight, trend]` at time `t-1`, the prediction for time `t` should be:
```
predicted_state_t = F * state_(t-1)
predicted_covariance_t = F * P_(t-1) * F' + Q
```
Where:
- `F` is the transition matrix `[[1, Δt], [0, 1]]`
- `P` is the state covariance
- `Q` is the process noise

### Innovation (What Quality Scoring Uses)
```
innovation = measurement - H * predicted_state
innovation_covariance = H * predicted_covariance * H' + R
```
Where:
- `H` is the observation matrix `[1, 0]`
- `R` is the observation noise

## Impact

The current implementation:
1. **Underestimates prediction error** during normal operation because it's using an already-corrected state
2. **May incorrectly reject valid measurements** that deviate from the posterior state but would be acceptable given proper prediction uncertainty
3. **Reduces the effectiveness** of the Kalman filter's ability to handle time gaps properly

## Solution Required

Need to modify the Kalman filter implementation to:
1. Add a `predict` method that returns the predicted state and covariance WITHOUT updating
2. Call this prediction before quality scoring
3. Pass the true prediction (not projection of posterior) to `calculate_kalman_fit`

## Code Locations

- **Prediction calculation**: `src/processing/processor.py:224-256`
- **Quality scorer**: `src/processing/unified_quality_scorer.py:215-302`
- **Kalman update**: `src/processing/kalman.py:74-156`

## Example Fix Approach

```python
# In KalmanFilterManager, add:
@staticmethod
def predict_next_state(state, timestamp):
    """Get prediction for next timestamp WITHOUT updating state."""
    if not state or not state.get("last_state"):
        return None, None

    # Get time delta
    time_delta_days = calculate_time_delta_days(timestamp, state["last_timestamp"])

    # Build transition matrix
    F = np.array([[1, time_delta_days], [0, 1]])

    # Get last posterior
    last_state = state["last_state"][-1] if len(last_state.shape) > 1 else last_state
    last_cov = state["last_covariance"][-1] if len(last_covariance.shape) > 2 else last_covariance

    # Predict
    predicted_state = F @ last_state
    predicted_cov = F @ last_cov @ F.T + Q

    # Extract weight prediction and innovation covariance
    predicted_weight = predicted_state[0]
    innovation_cov = predicted_cov[0,0] + R  # H @ predicted_cov @ H.T + R

    return predicted_weight, innovation_cov
```

Then use this prediction in processor.py instead of the manual calculation.