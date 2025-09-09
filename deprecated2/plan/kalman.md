# Progressive Kalman Filter Implementation Plan

## Overview
This document outlines a phased approach to implementing a Kalman filter for physiological weight data processing using the `pykalman` package. The implementation starts with the simplest possible model and progressively adds complexity based on validation results.

## Why PyKalman?
- **Numerical Stability**: Battle-tested algorithms with proper matrix decomposition
- **Built-in Features**: Missing data handling, parameter learning (EM algorithm), smoothing
- **Less Code**: Focus on modeling physiology, not matrix math
- **Parameter Learning**: Automatic optimization of Q and R from data

## Dependencies
```bash
pip install pykalman==0.9.7
# Also requires: numpy, scipy (installed automatically)
```

## Design Principles
1. **Independent Pipeline**: Runs parallel to existing confidence scoring system
2. **No Dependencies**: Does not rely on current baseline or scoring logic
3. **Progressive Enhancement**: Each phase builds on previous, but is independently valuable
4. **Data-Driven**: Validate each phase before proceeding to next
5. **Library-First**: Leverage PyKalman's capabilities before custom code

## Phase 1: Minimal 1D Implementation (Baseline)

### Goal
Establish simplest working Kalman filter as proof of concept using PyKalman.

### State Model
- **State Vector**: `x = [weight]` (single dimension)
- **State Transition**: `x(t+1) = x(t)` (random walk model)
- **Process Noise**: `Q = 0.5 kg²` (allows natural daily variation)
- **Measurement Noise**: `R = 1.0 kg²` (fixed for all sources)

### Implementation with PyKalman
```python
from pykalman import KalmanFilter
import numpy as np

class WeightKalmanFilter:
    def __init__(self):
        # Configure 1D Kalman filter for weight tracking
        self.kf = KalmanFilter(
            initial_state_mean=[75.0],        # Reasonable starting weight
            initial_state_covariance=[[10.0]], # High initial uncertainty
            transition_matrices=[[1.0]],       # Random walk: x(t+1) = x(t)
            observation_matrices=[[1.0]],      # Direct observation
            transition_covariance=[[0.5]],     # Q: process noise
            observation_covariance=[[1.0]]     # R: measurement noise
        )
        self.current_mean = None
        self.current_covariance = None

    def process_measurement(self, weight):
        """Process a single weight measurement."""
        if self.current_mean is None:
            # First measurement - initialize
            self.current_mean = np.array([weight])
            self.current_covariance = np.array([[10.0]])

        # Update filter with new measurement
        (self.current_mean,
         self.current_covariance) = self.kf.filter_update(
            self.current_mean,
            self.current_covariance,
            observation=np.array([weight])
        )

        return float(self.current_mean[0])

    def handle_missing_data(self, time_gap_days):
        """Handle missing measurements by prediction only."""
        if self.current_mean is None:
            return None

        # Predict forward without update
        for _ in range(int(time_gap_days)):
            (self.current_mean,
             self.current_covariance) = self.kf.filter_update(
                self.current_mean,
                self.current_covariance,
                observation=None  # PyKalman handles missing data
            )

        return float(self.current_mean[0])
```

### Parameter Learning with EM
```python
def learn_parameters_from_data(weight_measurements):
    """Use EM algorithm to learn optimal Q and R from data."""
    kf = KalmanFilter(
        initial_state_mean=[np.mean(weight_measurements)],
        n_dim_state=1,
        n_dim_obs=1
    )

    # Learn parameters using EM algorithm
    kf = kf.em(weight_measurements, n_iter=10)

    print(f"Learned Q: {kf.transition_covariance}")
    print(f"Learned R: {kf.observation_covariance}")

    return kf
```

### Integration Points
- Add to `baseline_processor.py` as parallel processing
- Log both Kalman and current system outputs
- Compare performance metrics

### Success Criteria
- Filter converges to reasonable estimates
- Handles missing data gracefully
- Performance comparable to moving average

---

## Phase 2: Dynamic Measurement Noise

### Goal
Make measurement noise (R) adaptive based on data source quality.

### Enhancements
- Keep 1D state vector
- Calculate R dynamically based on:
  - Source type (smart scale vs manual entry)
  - Time since last measurement
  - Time of day (morning vs evening)

### Implementation with PyKalman
```python
class AdaptiveWeightKalmanFilter(WeightKalmanFilter):
    def __init__(self):
        super().__init__()
        self.last_measurement_time = None

    def calculate_R(self, source_type, hours_since_last, hour_of_day):
        """Calculate measurement noise based on data quality indicators."""
        # Base noise by source
        base_R = {
            'care-team-upload': 0.3,
            'patient-device': 0.5,
            'internal-questionnaire': 1.0,
            'patient-upload': 2.0,
            'unknown': 1.5
        }.get(source_type, 1.0)

        # Time of day factor (morning readings more reliable)
        time_factor = 1.0 if (5 <= hour_of_day <= 9) else 1.5

        # Gap factor (uncertainty increases with time)
        gap_factor = 1.0 + (hours_since_last / 168.0) if hours_since_last else 1.0

        return base_R * time_factor * gap_factor

    def process_measurement_adaptive(self, weight, source_type, timestamp):
        """Process measurement with adaptive R."""
        # Calculate time-based factors
        hours_since_last = 0
        if self.last_measurement_time:
            hours_since_last = (timestamp - self.last_measurement_time).total_seconds() / 3600

        hour_of_day = timestamp.hour

        # Calculate adaptive R
        R_adaptive = self.calculate_R(source_type, hours_since_last, hour_of_day)

        # Update observation covariance dynamically
        self.kf.observation_covariance = np.array([[R_adaptive]])

        # Process measurement
        result = self.process_measurement(weight)
        self.last_measurement_time = timestamp

        return result, R_adaptive
```

### Success Criteria
- Better handling of mixed-quality data sources
- Improved prediction accuracy
- Appropriate uncertainty estimates

---

## Phase 3: Add Trend Component (2D State)

### Goal
Separate underlying weight trend from daily fluctuations.

### State Model
- **State Vector**: `x = [weight, trend]`
  - weight: current weight (kg)
  - trend: rate of change (kg/day)
- **State Transition Matrix**:
  ```
  F = [[1, Δt],  # weight += trend * time
       [0, 1]]   # trend remains constant
  ```
- **Process Noise**:
  ```
  Q = [[0.25, 0   ],  # Weight noise
       [0,    0.001]]  # Trend changes slowly
  ```

### Implementation with PyKalman
```python
class TrendKalmanFilter:
    def __init__(self):
        # 2D state: [weight, trend]
        self.kf = KalmanFilter(
            initial_state_mean=[75.0, 0.0],   # Weight and zero initial trend
            initial_state_covariance=[[10.0, 0.0],
                                      [0.0, 0.01]],  # Uncertainty in both
            transition_matrices=[[1.0, 1.0],   # weight += trend * dt
                                [0.0, 1.0]],   # trend stays constant
            observation_matrices=[[1.0, 0.0]], # Observe only weight
            transition_covariance=[[0.25, 0.0],
                                   [0.0, 0.001]], # Q matrix
            observation_covariance=[[1.0]]     # R (scalar for 1D obs)
        )
        self.current_mean = None
        self.current_covariance = None

    def process_measurement_with_dt(self, weight, dt=1.0):
        """Process measurement with variable time step."""
        # Update transition matrix for time step
        self.kf.transition_matrices = np.array([[1.0, dt],
                                                [0.0, 1.0]])

        if self.current_mean is None:
            self.current_mean = np.array([weight, 0.0])
            self.current_covariance = np.array([[10.0, 0.0],
                                                [0.0, 0.01]])

        # Update with measurement
        (self.current_mean,
         self.current_covariance) = self.kf.filter_update(
            self.current_mean,
            self.current_covariance,
            observation=np.array([weight])
        )

        return {
            'weight': float(self.current_mean[0]),
            'trend': float(self.current_mean[1]),  # kg/day
            'weight_uncertainty': float(np.sqrt(self.current_covariance[0, 0])),
            'trend_uncertainty': float(np.sqrt(self.current_covariance[1, 1]))
        }

    def predict_future(self, days_ahead):
        """Predict weight days_ahead in the future."""
        if self.current_mean is None:
            return None

        predicted_weight = self.current_mean[0] + self.current_mean[1] * days_ahead

        # Uncertainty grows with prediction horizon
        F_pred = np.array([[1.0, days_ahead],
                          [0.0, 1.0]])
        predicted_cov = F_pred @ self.current_covariance @ F_pred.T
        predicted_cov += self.kf.transition_covariance * days_ahead

        return {
            'predicted_weight': float(predicted_weight),
            'prediction_uncertainty': float(np.sqrt(predicted_cov[0, 0]))
        }
```

### Learning Trend from Historical Data
```python
def learn_trend_model(weight_measurements, timestamps):
    """Learn optimal parameters for trend model using EM."""
    # Calculate time deltas
    time_deltas = np.diff(timestamps)

    # Create observation-aligned transition matrices
    transition_matrices = []
    for dt in time_deltas:
        transition_matrices.append([[1.0, dt], [0.0, 1.0]])

    kf = KalmanFilter(
        n_dim_state=2,
        n_dim_obs=1,
        initial_state_mean=[weight_measurements[0], 0.0]
    )

    # Learn parameters
    kf = kf.em(
        weight_measurements,
        n_iter=15,
        # PyKalman will learn Q, R, and initial conditions
    )

    return kf
```

### Initialization Strategy
- Weight: First measurement
- Trend: 0 (no initial assumption)
- Let filter learn trend from data

### Success Criteria
- Accurate trend detection
- Better long-term predictions
- Meaningful uncertainty bounds for missing data

---

## Phase 4: Adaptive Process Noise

### Goal
Adjust process noise (Q) based on observed behavior patterns.

### Approach
- Track innovation sequence (prediction errors)
- Calculate innovation variance over sliding window
- Adjust Q to match observed variance
- Use PyKalman's EM for periodic recalibration

### Implementation with PyKalman
```python
class AdaptiveProcessKalmanFilter(TrendKalmanFilter):
    def __init__(self):
        super().__init__()
        self.innovation_history = []
        self.measurement_history = []
        self.recalibration_interval = 50  # Recalibrate every 50 measurements

    def adapt_Q(self, window=20):
        """Adapt process noise based on innovation variance."""
        if len(self.innovation_history) < window:
            return

        recent_innovations = self.innovation_history[-window:]
        innovation_var = np.var(recent_innovations)

        # Calculate expected innovation variance
        S = self.current_covariance[0, 0] + self.kf.observation_covariance[0, 0]

        ratio = innovation_var / S
        if ratio > 2.0:
            # Model is surprised - increase Q
            current_Q = self.kf.transition_covariance
            self.kf.transition_covariance = current_Q * 1.1
        elif ratio < 0.5:
            # Model is too conservative - decrease Q
            current_Q = self.kf.transition_covariance
            self.kf.transition_covariance = current_Q * 0.9

        # Bound Q to prevent runaway
        self.kf.transition_covariance = np.clip(
            self.kf.transition_covariance, 0.01, 5.0
        )

    def process_measurement_adaptive(self, weight):
        """Process measurement with adaptive Q."""
        # Calculate innovation before update
        if self.current_mean is not None:
            predicted_weight = self.current_mean[0] + self.current_mean[1]  # 1-day prediction
            innovation = weight - predicted_weight
            self.innovation_history.append(innovation)

        # Store measurement for recalibration
        self.measurement_history.append(weight)

        # Adapt Q based on recent innovations
        self.adapt_Q()

        # Periodic recalibration using EM
        if len(self.measurement_history) % self.recalibration_interval == 0:
            self.recalibrate_with_em()

        # Process measurement
        return self.process_measurement_with_dt(weight)

    def recalibrate_with_em(self):
        """Use EM to recalibrate parameters from recent data."""
        if len(self.measurement_history) < 20:
            return

        # Use last 100 measurements for recalibration
        recent_measurements = self.measurement_history[-100:]

        # Create new KF and learn parameters
        kf_new = KalmanFilter(
            n_dim_state=2,
            n_dim_obs=1,
            initial_state_mean=self.current_mean.tolist()
        )

        try:
            kf_new = kf_new.em(recent_measurements, n_iter=5)

            # Update our filter with learned parameters
            self.kf.transition_covariance = kf_new.transition_covariance
            self.kf.observation_covariance = kf_new.observation_covariance

            print(f"Recalibrated Q: {self.kf.transition_covariance}")
            print(f"Recalibrated R: {self.kf.observation_covariance}")
        except:
            pass  # Keep current parameters if EM fails
```

### Success Criteria
- Better adaptation to individual user patterns
- Reduced prediction errors
- Stable Q values after initial learning period

---

## Phase 5: Physiological Hydration Model (3D State)

### Goal
Explicitly model short-term physiological variations.

### State Model
- **State Vector**: `x = [true_weight, trend, hydration_offset]`
  - true_weight: Actual body mass
  - trend: Long-term change (kg/day)
  - hydration_offset: Daily fluctuation (mean-reverting)

### Dynamics with PyKalman
```python
class PhysiologicalKalmanFilter:
    def __init__(self):
        # 3D state: [true_weight, trend, hydration]
        self.kf = KalmanFilter(
            initial_state_mean=[75.0, 0.0, 0.0],
            initial_state_covariance=[[10.0, 0.0, 0.0],
                                      [0.0, 0.01, 0.0],
                                      [0.0, 0.0, 1.0]],
            transition_matrices=[[1.0, 1.0, 0.0],   # weight += trend
                                [0.0, 1.0, 0.0],    # trend constant
                                [0.0, 0.0, 0.8]],   # hydration decays
            observation_matrices=[[1.0, 0.0, 1.0]], # observe weight + hydration
            transition_covariance=[[0.1, 0.0, 0.0],  # True weight noise
                                   [0.0, 0.001, 0.0], # Trend noise
                                   [0.0, 0.0, 0.5]],  # Hydration noise
            observation_covariance=[[0.5]]
        )
        self.current_mean = None
        self.current_covariance = None

    def process_measurement(self, weight, dt=1.0):
        """Process measurement accounting for hydration effects."""
        # Update transition matrix for time step
        decay_factor = 0.8 ** dt  # Hydration decay over time
        self.kf.transition_matrices = np.array([
            [1.0, dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, decay_factor]
        ])

        if self.current_mean is None:
            # Initialize with measurement
            self.current_mean = np.array([weight, 0.0, 0.0])
            self.current_covariance = self.kf.initial_state_covariance

        # Update filter
        (self.current_mean,
         self.current_covariance) = self.kf.filter_update(
            self.current_mean,
            self.current_covariance,
            observation=np.array([weight])
        )

        return {
            'true_weight': float(self.current_mean[0]),
            'trend': float(self.current_mean[1]),
            'hydration_offset': float(self.current_mean[2]),
            'measured_weight': weight,
            'weight_uncertainty': float(np.sqrt(self.current_covariance[0, 0]))
        }

    def detect_hydration_event(self):
        """Detect significant hydration changes."""
        if self.current_mean is None:
            return None

        hydration = abs(self.current_mean[2])
        hydration_std = np.sqrt(self.current_covariance[2, 2])

        if hydration > 2 * hydration_std:
            return {
                'significant_hydration': True,
                'hydration_kg': float(hydration),
                'confidence': float(1 - np.exp(-hydration / hydration_std))
            }
        return {'significant_hydration': False}
```

### Handling Different Physiological Patterns
```python
def configure_for_user_pattern(user_type):
    """Configure filter based on user's physiological patterns."""
    configs = {
        'stable': {
            'weight_noise': 0.1,
            'trend_noise': 0.0001,
            'hydration_noise': 0.3,
            'hydration_decay': 0.9
        },
        'weight_loss': {
            'weight_noise': 0.2,
            'trend_noise': 0.001,
            'hydration_noise': 0.4,
            'hydration_decay': 0.85
        },
        'athlete': {
            'weight_noise': 0.3,
            'trend_noise': 0.0001,
            'hydration_noise': 0.8,  # Higher hydration variation
            'hydration_decay': 0.7    # Faster recovery
        }
    }

    config = configs.get(user_type, configs['stable'])

    return KalmanFilter(
        transition_covariance=[
            [config['weight_noise'], 0, 0],
            [0, config['trend_noise'], 0],
            [0, 0, config['hydration_noise']]
        ],
        # ... other parameters
    )
```

### Success Criteria
- Better separation of true weight change from daily fluctuations
- Improved handling of outliers (pizza night, dehydration)
- More stable trend estimates

---

## Phase 6: Time-of-Day Patterns

### Goal
Model systematic variations based on measurement timing.

### Approach
- Learn user's typical weighing schedule
- Model expected offset by time of day
- Adjust measurements to common baseline (e.g., morning weight)

### Implementation
```python
class TimeAwareKalmanFilter:
    def __init__(self):
        self.daily_pattern = np.zeros(24)  # Hourly offsets
        self.pattern_counts = np.zeros(24)  # Sample counts

    def update_pattern(self, hour, innovation):
        # Exponential moving average of pattern
        alpha = 0.1
        self.daily_pattern[hour] = (1-alpha) * self.daily_pattern[hour] + alpha * innovation
        self.pattern_counts[hour] += 1

    def get_expected_offset(self, hour):
        if self.pattern_counts[hour] < 5:
            return 0.0  # Not enough data
        return self.daily_pattern[hour]
```

### Success Criteria
- Reduced systematic bias from timing variations
- Better predictions for unusual measurement times
- Improved innovation whiteness (residuals uncorrelated)

---

## Phase 7: Smart Initialization

### Goal
Use all available information for optimal filter initialization, leveraging PyKalman's EM algorithm.

### Strategy with PyKalman
```python
class SmartInitKalmanFilter(PhysiologicalKalmanFilter):
    def smart_initialize(self, initial_readings):
        """Initialize filter intelligently from available data."""
        if len(initial_readings) == 0:
            raise ValueError("Need at least one reading")

        weights = [r['weight'] for r in initial_readings]

        if len(initial_readings) == 1:
            # Single reading: use it with high uncertainty
            self.current_mean = np.array([weights[0], 0.0, 0.0])
            self.current_covariance = np.diag([10.0, 0.1, 1.0])

        elif len(initial_readings) < 7:
            # Few readings: use mean and variance
            mean_weight = np.mean(weights)
            var_weight = np.var(weights) if len(weights) > 1 else 10.0

            self.current_mean = np.array([mean_weight, 0.0, 0.0])
            self.current_covariance = np.diag([var_weight, 0.01, 0.5])

        else:
            # Many readings: use EM to learn initial parameters
            self._learn_initial_state_with_em(initial_readings)

    def _learn_initial_state_with_em(self, initial_readings):
        """Use EM algorithm to learn optimal initial state."""
        weights = np.array([r['weight'] for r in initial_readings])
        timestamps = np.array([r['timestamp'] for r in initial_readings])

        # Calculate time deltas for variable dt
        time_deltas = np.diff(timestamps)

        # Initialize KF for EM learning
        kf_init = KalmanFilter(
            n_dim_state=3,  # [weight, trend, hydration]
            n_dim_obs=1,
            initial_state_mean=[weights[0], 0.0, 0.0]
        )

        # Learn all parameters including initial state
        kf_learned = kf_init.em(
            X=weights,
            n_iter=20,
            em_vars=['initial_state_mean',
                    'initial_state_covariance',
                    'transition_covariance',
                    'observation_covariance']
        )

        # Extract learned initial state
        self.current_mean = kf_learned.initial_state_mean
        self.current_covariance = kf_learned.initial_state_covariance

        # Update filter parameters
        self.kf.transition_covariance = kf_learned.transition_covariance
        self.kf.observation_covariance = kf_learned.observation_covariance

        # Smooth historical data for better initialization
        (smoothed_states, smoothed_covariances) = kf_learned.smooth(weights)

        # Use last smoothed state as current state
        self.current_mean = smoothed_states[-1]
        self.current_covariance = smoothed_covariances[-1]

        return {
            'initial_weight': float(self.current_mean[0]),
            'initial_trend': float(self.current_mean[1]),
            'learned_Q': self.kf.transition_covariance.tolist(),
            'learned_R': float(self.kf.observation_covariance[0, 0])
        }
```

### Batch Initialization for Multiple Users
```python
def initialize_filters_for_users(user_data_dict):
    """Initialize filters for multiple users efficiently."""
    user_filters = {}

    for user_id, readings in user_data_dict.items():
        filter = SmartInitKalmanFilter()

        # Initialize based on available data
        if len(readings) >= 7:
            # Enough data for EM learning
            filter.smart_initialize(readings)
        else:
            # Fallback to simple initialization
            filter.smart_initialize(readings)

        user_filters[user_id] = filter

    return user_filters
```

### Success Criteria
- Faster convergence to accurate estimates
- Better initial uncertainty quantification
- Robust to various data availability scenarios

---

## Phase 8: Outlier Detection & Handling

### Goal
Use filter for anomaly detection and data quality monitoring.

### Approach with PyKalman
- Calculate Mahalanobis distance for each measurement
- Flag measurements > 3σ from prediction
- Options for handling outliers:
  1. Reject (don't update filter)
  2. Downweight (increase R dynamically)
  3. Track separately for analysis

### Implementation
```python
class OutlierAwareKalmanFilter(SmartInitKalmanFilter):
    def __init__(self, outlier_mode='downweight'):
        super().__init__()
        self.outlier_mode = outlier_mode  # 'reject', 'downweight', or 'track'
        self.outlier_threshold = 3.0  # Standard deviations
        self.outliers = []
        self.outlier_count = 0

    def detect_outlier(self, measurement):
        """Detect if measurement is an outlier using Mahalanobis distance."""
        if self.current_mean is None:
            return False, 0.0

        # Predict next state (without updating)
        (predicted_mean,
         predicted_cov) = self.kf.filter_update(
            self.current_mean,
            self.current_covariance,
            observation=None  # Prediction only
        )

        # Calculate innovation
        innovation = measurement - predicted_mean[0]

        # Innovation covariance (S = HPH' + R)
        H = self.kf.observation_matrices[0]
        S = H @ predicted_cov @ H.T + self.kf.observation_covariance[0, 0]

        # Mahalanobis distance
        mahalanobis = abs(innovation) / np.sqrt(S)

        is_outlier = mahalanobis > self.outlier_threshold

        return is_outlier, mahalanobis

    def process_measurement_with_outlier_handling(self, weight):
        """Process measurement with outlier detection and handling."""
        is_outlier, mahalanobis = self.detect_outlier(weight)

        if is_outlier:
            self.outlier_count += 1
            self.outliers.append({
                'weight': weight,
                'mahalanobis': mahalanobis,
                'expected': float(self.current_mean[0]) if self.current_mean is not None else None
            })

            if self.outlier_mode == 'reject':
                # Don't update filter with outlier
                return {
                    'filtered_weight': float(self.current_mean[0]),
                    'outlier_detected': True,
                    'outlier_action': 'rejected'
                }

            elif self.outlier_mode == 'downweight':
                # Increase measurement noise for this observation
                original_R = self.kf.observation_covariance.copy()

                # Increase R proportionally to Mahalanobis distance
                scale_factor = (mahalanobis / self.outlier_threshold) ** 2
                self.kf.observation_covariance = original_R * scale_factor

                # Process with increased uncertainty
                result = self.process_measurement(weight)

                # Restore original R
                self.kf.observation_covariance = original_R

                return {
                    **result,
                    'outlier_detected': True,
                    'outlier_action': 'downweighted',
                    'R_scale': scale_factor
                }

        # Normal processing for non-outliers
        result = self.process_measurement(weight)
        return {
            **result,
            'outlier_detected': False
        }

    def get_outlier_statistics(self):
        """Get statistics about detected outliers."""
        if not self.outliers:
            return {'outlier_rate': 0.0, 'outliers': []}

        total_measurements = self.outlier_count + len(self.outliers)

        return {
            'outlier_rate': len(self.outliers) / total_measurements,
            'outlier_count': len(self.outliers),
            'average_mahalanobis': np.mean([o['mahalanobis'] for o in self.outliers]),
            'max_mahalanobis': max(o['mahalanobis'] for o in self.outliers),
            'outliers': self.outliers[-10:]  # Last 10 outliers
        }
```

### Adaptive Outlier Threshold
```python
def adapt_outlier_threshold(self, user_stability_score):
    """Adapt outlier threshold based on user's weight stability."""
    if user_stability_score > 0.8:
        # Stable user - tighter threshold
        self.outlier_threshold = 2.5
    elif user_stability_score < 0.3:
        # Variable user - looser threshold
        self.outlier_threshold = 4.0
    else:
        # Default threshold
        self.outlier_threshold = 3.0
```

### Success Criteria
- Accurate outlier identification
- Improved robustness to bad data
- Useful data quality metrics

---

## Phase 9: Multi-Scale Dynamics

### Goal
Model weight dynamics at multiple timescales.

### State Model
- **State Vector**: `x = [weight, weekly_trend, daily_cycle, hydration]`
- Different process noises for different timescales
- Better handling of various prediction horizons

### Benefits
- Weekly trend for long-term goals
- Daily cycle for short-term predictions
- Hydration for physiological realism

---

## Phase 10: Full System Integration

### Goal
Integrate Kalman filter with existing system components.

### Integration Points
1. **Confidence Score Enhancement**: Use Kalman uncertainty to improve confidence scores
2. **Baseline Initialization**: Use baseline data for smart initialization
3. **Visualization**: Add Kalman predictions and uncertainty bands to charts
4. **A/B Testing**: Compare performance against current system

### Success Criteria
- Demonstrable improvement over current system
- Smooth integration with existing pipeline
- Production-ready performance and reliability

---

## Validation Strategy

### Synthetic Data Testing with PyKalman
```python
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_weight_data(days=90):
    """Generate realistic synthetic weight data for testing."""
    true_weight = 75.0
    true_trend = -0.05  # Losing 50g/day
    hydration_state = 0.0

    weights = []
    timestamps = []
    base_time = datetime.now()

    for day in range(days):
        # True weight evolution
        true_weight += true_trend + np.random.normal(0, 0.1)

        # Hydration state (mean-reverting)
        hydration_state = 0.8 * hydration_state + np.random.normal(0, 0.3)

        # Time of day effect
        hour = np.random.choice([6, 7, 8, 18, 19, 20], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
        time_of_day_offset = 0.5 if hour > 12 else 0.0

        # Source-dependent measurement noise
        if np.random.random() > 0.7:
            source = 'patient-upload'
            noise = np.random.normal(0, 1.0)
        else:
            source = 'care-team-upload'
            noise = np.random.normal(0, 0.3)

        measured = true_weight + hydration_state + time_of_day_offset + noise

        weights.append({
            'day': day,
            'timestamp': base_time + timedelta(days=day, hours=hour),
            'true_weight': true_weight,
            'measured_weight': measured,
            'source': source,
            'hydration': hydration_state
        })

    return weights
```

### Backtesting Framework with PyKalman
```python
def backtest_pykalman_filter(filter, historical_data, test_split=0.2):
    """Backtest a PyKalman-based filter on historical data."""
    n_test = int(len(historical_data) * test_split)
    train_data = historical_data[:-n_test]
    test_data = historical_data[-n_test:]

    # Train filter on training data
    train_weights = [d['measured_weight'] for d in train_data]

    # Use EM to learn parameters if filter supports it
    if hasattr(filter, 'smart_initialize'):
        filter.smart_initialize(train_data)
    else:
        # Basic initialization
        filter.current_mean = np.array([train_weights[0], 0.0])
        filter.current_covariance = np.eye(2) * 10.0

    # Process training data
    for weight in train_weights:
        filter.process_measurement(weight)

    # Test on held-out data
    predictions = []
    errors = []
    mahalanobis_distances = []

    for data_point in test_data:
        # Make prediction
        if hasattr(filter, 'predict_future'):
            pred = filter.predict_future(1)
            predicted_weight = pred['predicted_weight']
        else:
            # Simple prediction
            predicted_weight = filter.current_mean[0]

        predictions.append(predicted_weight)
        actual_weight = data_point['measured_weight']
        errors.append(actual_weight - predicted_weight)

        # Check if it would be flagged as outlier
        if hasattr(filter, 'detect_outlier'):
            is_outlier, mahal = filter.detect_outlier(actual_weight)
            mahalanobis_distances.append(mahal)

        # Update filter with actual measurement
        filter.process_measurement(actual_weight)

    # Calculate metrics
    errors_array = np.array(errors)
    rmse = np.sqrt(np.mean(errors_array**2))
    mae = np.mean(np.abs(errors_array))

    # Calculate prediction interval coverage
    within_2sigma = np.sum(np.abs(errors_array) < 2 * np.std(errors_array)) / len(errors_array)

    return {
        'rmse': rmse,
        'mae': mae,
        'mean_error': np.mean(errors_array),
        'std_error': np.std(errors_array),
        '2sigma_coverage': within_2sigma,
        'predictions': predictions,
        'errors': errors,
        'mahalanobis': mahalanobis_distances
    }
```

### Comparative Testing Framework
```python
def compare_filter_implementations():
    """Compare different filter implementations."""
    # Generate test data
    test_data = generate_synthetic_weight_data(days=180)

    filters = {
        '1D_Simple': WeightKalmanFilter(),
        '2D_Trend': TrendKalmanFilter(),
        '3D_Physiological': PhysiologicalKalmanFilter(),
        'Outlier_Aware': OutlierAwareKalmanFilter()
    }

    results = {}

    for name, filter in filters.items():
        print(f"Testing {name}...")
        result = backtest_pykalman_filter(filter, test_data)
        results[name] = result

        print(f"  RMSE: {result['rmse']:.3f}")
        print(f"  MAE: {result['mae']:.3f}")
        print(f"  2σ Coverage: {result['2sigma_coverage']:.1%}")

    return results
```

### Cross-Validation with Parameter Grid Search
```python
def optimize_filter_parameters(historical_data):
    """Find optimal Q and R parameters using cross-validation."""
    from sklearn.model_selection import TimeSeriesSplit

    # Parameter grid
    Q_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    R_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    tscv = TimeSeriesSplit(n_splits=5)
    weights = [d['measured_weight'] for d in historical_data]

    best_params = None
    best_score = float('inf')

    for Q in Q_values:
        for R in R_values:
            scores = []

            for train_idx, test_idx in tscv.split(weights):
                train_weights = [weights[i] for i in train_idx]
                test_weights = [weights[i] for i in test_idx]

                # Create and train filter
                kf = KalmanFilter(
                    transition_covariance=[[Q]],
                    observation_covariance=[[R]],
                    initial_state_mean=[train_weights[0]],
                    transition_matrices=[[1.0]],
                    observation_matrices=[[1.0]]
                )

                # Evaluate
                filtered_state_means, _ = kf.filter(train_weights)

                # Predict and score on test set
                test_errors = []
                for i, test_val in enumerate(test_weights):
                    if i > 0:
                        pred = filtered_state_means[-1][0]  # Last filtered value
                        test_errors.append((test_val - pred) ** 2)

                scores.append(np.mean(test_errors))

            avg_score = np.mean(scores)

            if avg_score < best_score:
                best_score = avg_score
                best_params = {'Q': Q, 'R': R}

    return best_params, best_score
```

### Performance Metrics
Track for each phase:
1. **Accuracy**: RMSE, MAE of predictions
2. **Uncertainty**: Calibration of confidence intervals
3. **Robustness**: Performance with missing data
4. **Stability**: Convergence time, numerical stability
5. **Outliers**: Detection rate, false positive rate

---

## Implementation Timeline (Accelerated with PyKalman)

### Week 1: Foundation (Phase 1-3)
- Day 1-2: Setup PyKalman, implement 1D filter
- Day 3-4: Add dynamic R calculation
- Day 5: Implement 2D trend model
- Use PyKalman's built-in features to accelerate development

### Week 2: Advanced Features (Phase 4-7)
- Day 1-2: Adaptive Q with EM recalibration
- Day 3-4: 3D physiological model
- Day 5: Smart initialization with EM learning
- Leverage PyKalman's EM algorithm for automatic parameter tuning

### Week 3: Robustness & Integration (Phase 8-10)
- Day 1-2: Outlier detection and handling
- Day 3-4: System integration with baseline processor
- Day 5: A/B testing framework and metrics
- Focus on integration since core filtering is handled by PyKalman

### Benefits of Accelerated Timeline
- **75% faster implementation** due to PyKalman
- **More time for testing** and validation
- **Better numerical stability** from the start
- **Automatic parameter learning** reduces tuning time

---

## Risk Mitigation

### Technical Risks
1. **Numerical Instability**: Use square-root form or UD decomposition if needed
2. **Parameter Tuning**: Start with conservative values, tune based on data
3. **Computational Cost**: Profile and optimize matrix operations

### Data Risks
1. **Poor Quality Data**: Robust outlier detection from Phase 8
2. **Missing Data**: Graceful degradation, increasing uncertainty
3. **Non-Stationarity**: Adaptive parameters from Phase 4

### Integration Risks
1. **Breaking Changes**: Parallel pipeline ensures no disruption
2. **Performance Impact**: Async processing if needed
3. **User Trust**: Gradual rollout with monitoring

---

## Success Metrics

### Phase 1 Success (Minimum Viable)
- 20% improvement over moving average baseline
- Handles 50% missing data without degradation
- Sub-millisecond processing time per reading

### Full Implementation Success
- 40% reduction in false positive outlier detection
- 90% of predictions within confidence intervals
- 50% improvement in trend detection accuracy
- Production deployment with <0.01% error rate
