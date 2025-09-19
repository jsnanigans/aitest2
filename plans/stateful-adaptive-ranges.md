# Plan: Stateful Adaptive Ranges (Recursive-Compatible)

## Problem Statement

Current system processes measurements recursively, maintaining only:
- Current measurement
- Previous state from database (Kalman state, buffers, etc.)
- Cannot store full historical data in memory

Need adaptive ranges that work within these constraints while still learning user patterns.

## Solution: Incremental Statistics in State

Instead of storing historical measurements, we maintain rolling statistics in the state that update incrementally with each new measurement.

## Implementation Design

### 1. Stateful User Profile (Stored in DB State)

```python
# src/processing/stateful_profile.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime

@dataclass
class StatefulUserProfile:
    """User profile that updates incrementally without storing history."""

    # Core statistics (updated incrementally)
    mean_weight: float = 0.0
    variance_weight: float = 0.0
    measurement_count: int = 0

    # Exponential moving averages (no history needed)
    ema_daily_change: float = 0.0
    ema_weekly_range: float = 0.0
    ema_noise_estimate: float = 0.5  # Start conservative

    # Percentile estimators (using P-Square algorithm)
    p10_estimator: Optional[Dict] = None  # P-Square state for 10th percentile
    p90_estimator: Optional[Dict] = None  # P-Square state for 90th percentile
    p95_estimator: Optional[Dict] = None  # P-Square state for 95th percentile

    # Rolling window statistics (fixed-size circular buffers)
    recent_changes_buffer: list = field(default_factory=lambda: [0.0] * 7)  # Last 7 changes
    recent_weights_buffer: list = field(default_factory=lambda: [0.0] * 7)  # Last 7 weights
    buffer_index: int = 0
    buffer_filled: bool = False

    # Pattern detection (incremental)
    direction_change_count: int = 0
    oscillation_score: float = 0.0
    trend_estimate: float = 0.0  # EMA-based trend

    # Adaptive thresholds (calculated from statistics)
    current_outlier_threshold: float = 3.0  # kg, starts conservative
    current_noise_multiplier: float = 1.0
    current_smoothing_strength: float = 0.5

    # State tracking
    last_weight: Optional[float] = None
    last_timestamp: Optional[datetime] = None
    last_direction: Optional[int] = None  # -1, 0, 1 for down, stable, up

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for database storage."""
        return {
            'mean_weight': self.mean_weight,
            'variance_weight': self.variance_weight,
            'measurement_count': self.measurement_count,
            'ema_daily_change': self.ema_daily_change,
            'ema_weekly_range': self.ema_weekly_range,
            'ema_noise_estimate': self.ema_noise_estimate,
            'p10_estimator': self.p10_estimator,
            'p90_estimator': self.p90_estimator,
            'p95_estimator': self.p95_estimator,
            'recent_changes_buffer': self.recent_changes_buffer,
            'recent_weights_buffer': self.recent_weights_buffer,
            'buffer_index': self.buffer_index,
            'buffer_filled': self.buffer_filled,
            'direction_change_count': self.direction_change_count,
            'oscillation_score': self.oscillation_score,
            'trend_estimate': self.trend_estimate,
            'current_outlier_threshold': self.current_outlier_threshold,
            'current_noise_multiplier': self.current_noise_multiplier,
            'current_smoothing_strength': self.current_smoothing_strength,
            'last_weight': self.last_weight,
            'last_timestamp': self.last_timestamp.isoformat() if self.last_timestamp else None,
            'last_direction': self.last_direction
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatefulUserProfile':
        """Deserialize from dictionary."""
        profile = cls()
        for key, value in data.items():
            if key == 'last_timestamp' and value:
                profile.last_timestamp = datetime.fromisoformat(value)
            elif hasattr(profile, key):
                setattr(profile, key, value)
        return profile
```

### 2. Incremental Profile Updater

```python
class IncrementalProfileUpdater:
    """Updates user profile incrementally without storing history."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alpha = config.get('ema_alpha', 0.1)  # EMA learning rate
        self.noise_alpha = config.get('noise_alpha', 0.05)  # Slower adaptation for noise

    def update_profile(self, profile: StatefulUserProfile,
                      new_weight: float,
                      timestamp: datetime,
                      kalman_prediction: Optional[float] = None) -> StatefulUserProfile:
        """
        Update profile with single new measurement.

        This is called for each measurement during recursive processing.
        """

        # First measurement initialization
        if profile.last_weight is None:
            profile.last_weight = new_weight
            profile.last_timestamp = timestamp
            profile.mean_weight = new_weight
            profile.measurement_count = 1
            profile.recent_weights_buffer[0] = new_weight
            self._initialize_percentile_estimators(profile, new_weight)
            return profile

        # Calculate time difference
        if profile.last_timestamp:
            time_diff_days = (timestamp - profile.last_timestamp).total_seconds() / 86400
        else:
            time_diff_days = 1.0

        # Update incremental statistics
        self._update_mean_variance(profile, new_weight)

        # Update change statistics if ~daily measurement
        if 0.5 <= time_diff_days <= 2.0:
            daily_change = abs(new_weight - profile.last_weight)
            self._update_change_statistics(profile, daily_change)

            # Update direction tracking for oscillation detection
            self._update_oscillation_tracking(profile, new_weight)

        # Update circular buffers
        self._update_buffers(profile, new_weight)

        # Update percentile estimators (P-Square algorithm)
        self._update_percentile_estimators(profile, new_weight)

        # Update noise estimate using innovation if Kalman prediction available
        if kalman_prediction is not None:
            innovation = abs(new_weight - kalman_prediction)
            profile.ema_noise_estimate = (1 - self.noise_alpha) * profile.ema_noise_estimate + self.noise_alpha * innovation

        # Update adaptive thresholds based on current statistics
        self._update_adaptive_thresholds(profile)

        # Update state for next iteration
        profile.last_weight = new_weight
        profile.last_timestamp = timestamp
        profile.measurement_count += 1

        return profile

    def _update_mean_variance(self, profile: StatefulUserProfile, new_weight: float):
        """Update mean and variance using Welford's online algorithm."""
        n = profile.measurement_count + 1
        delta = new_weight - profile.mean_weight
        profile.mean_weight += delta / n

        if n > 1:
            delta2 = new_weight - profile.mean_weight
            profile.variance_weight += (delta * delta2 - profile.variance_weight) / n

    def _update_change_statistics(self, profile: StatefulUserProfile, daily_change: float):
        """Update exponential moving averages of changes."""
        if profile.ema_daily_change == 0:
            profile.ema_daily_change = daily_change
        else:
            profile.ema_daily_change = (1 - self.alpha) * profile.ema_daily_change + self.alpha * daily_change

        # Update rolling buffer of changes
        profile.recent_changes_buffer[profile.buffer_index % 7] = daily_change

    def _update_buffers(self, profile: StatefulUserProfile, new_weight: float):
        """Update circular buffers."""
        profile.buffer_index = (profile.buffer_index + 1) % 7
        profile.recent_weights_buffer[profile.buffer_index] = new_weight

        if profile.buffer_index == 6 and not profile.buffer_filled:
            profile.buffer_filled = True

        # Update weekly range estimate if buffer is filled
        if profile.buffer_filled:
            weekly_range = max(profile.recent_weights_buffer) - min(profile.recent_weights_buffer)
            if profile.ema_weekly_range == 0:
                profile.ema_weekly_range = weekly_range
            else:
                profile.ema_weekly_range = (1 - self.alpha) * profile.ema_weekly_range + self.alpha * weekly_range

    def _update_oscillation_tracking(self, profile: StatefulUserProfile, new_weight: float):
        """Track oscillation patterns incrementally."""
        if profile.last_weight is not None:
            current_direction = 1 if new_weight > profile.last_weight else -1 if new_weight < profile.last_weight else 0

            if profile.last_direction is not None and current_direction != 0:
                if current_direction != profile.last_direction:
                    profile.direction_change_count += 1

                    # Update oscillation score (EMA of direction change rate)
                    change_rate = 1.0  # Direction changed
                else:
                    change_rate = 0.0  # Direction continued

                profile.oscillation_score = (1 - self.alpha) * profile.oscillation_score + self.alpha * change_rate

            profile.last_direction = current_direction

            # Update trend estimate
            weight_change = new_weight - profile.last_weight
            profile.trend_estimate = (1 - self.alpha) * profile.trend_estimate + self.alpha * weight_change

    def _update_adaptive_thresholds(self, profile: StatefulUserProfile):
        """
        Update thresholds based on current statistics.
        No history needed - uses current EMAs and estimates.
        """

        # Base threshold on EMA of daily changes
        if profile.ema_daily_change > 0:
            # Start with 3x the typical daily change
            base_threshold = profile.ema_daily_change * 3

            # Adjust based on oscillation score (0-1, where 1 is high oscillation)
            if profile.oscillation_score > 0.5:
                # High oscillation user - be more lenient
                profile.current_outlier_threshold = base_threshold * (1 + profile.oscillation_score)
                profile.current_noise_multiplier = 1.5
                profile.current_smoothing_strength = 0.3
            elif profile.oscillation_score > 0.3:
                # Moderate oscillation
                profile.current_outlier_threshold = base_threshold * 1.5
                profile.current_noise_multiplier = 1.2
                profile.current_smoothing_strength = 0.5
            else:
                # Stable user - be stricter
                profile.current_outlier_threshold = base_threshold
                profile.current_noise_multiplier = 0.8
                profile.current_smoothing_strength = 0.7

            # Cap thresholds for safety
            profile.current_outlier_threshold = max(1.0, min(10.0, profile.current_outlier_threshold))

            # Adjust for high variance users (using coefficient of variation)
            if profile.variance_weight > 0 and profile.mean_weight > 0:
                cv = np.sqrt(profile.variance_weight) / profile.mean_weight
                if cv > 0.05:  # High variance relative to mean
                    profile.current_outlier_threshold *= (1 + cv * 10)
                    profile.current_noise_multiplier *= (1 + cv * 5)

        else:
            # No data yet, use conservative defaults
            profile.current_outlier_threshold = 3.0
            profile.current_noise_multiplier = 1.0
            profile.current_smoothing_strength = 0.5

    def _initialize_percentile_estimators(self, profile: StatefulUserProfile, initial_value: float):
        """Initialize P-Square percentile estimators."""
        # P-Square algorithm maintains 5 markers for each percentile
        # This is a simplified initialization
        profile.p10_estimator = {'markers': [initial_value] * 5, 'positions': [1, 2, 3, 4, 5]}
        profile.p90_estimator = {'markers': [initial_value] * 5, 'positions': [1, 2, 3, 4, 5]}
        profile.p95_estimator = {'markers': [initial_value] * 5, 'positions': [1, 2, 3, 4, 5]}

    def _update_percentile_estimators(self, profile: StatefulUserProfile, new_value: float):
        """Update P-Square percentile estimators (simplified version)."""
        # This is a simplified version - full P-Square would be more complex
        # For now, we'll use the EMA approach for percentile-like estimates

        if profile.p10_estimator and profile.ema_daily_change > 0:
            # Approximate percentiles using multiples of EMA
            profile.p10_estimator['value'] = profile.ema_daily_change * 0.5
            profile.p90_estimator['value'] = profile.ema_daily_change * 2.0
            profile.p95_estimator['value'] = profile.ema_daily_change * 3.0
```

### 3. Stateful Validation with Adaptive Ranges

```python
class StatefulRangeValidator:
    """Validates measurements using stateful adaptive ranges."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.updater = IncrementalProfileUpdater(config)

    def validate_measurement(self,
                            measurement: Dict,
                            profile: StatefulUserProfile,
                            kalman_state: Optional[Dict] = None) -> Dict:
        """
        Validate measurement against user's adaptive range.

        Args:
            measurement: Current measurement
            profile: User's stateful profile
            kalman_state: Optional Kalman filter state

        Returns:
            Validation results and updated profile
        """
        weight = measurement['weight']
        timestamp = measurement['timestamp']

        # Get Kalman prediction if available
        kalman_prediction = kalman_state.get('state', profile.mean_weight) if kalman_state else profile.mean_weight

        # Update profile with new measurement
        updated_profile = self.updater.update_profile(profile, weight, timestamp, kalman_prediction)

        # Calculate deviation
        deviation = abs(weight - kalman_prediction)

        # Use adaptive threshold from profile
        threshold = updated_profile.current_outlier_threshold

        # Determine if outlier
        is_outlier = deviation > threshold

        # Calculate quality adjustment based on deviation
        if deviation < updated_profile.ema_daily_change:
            quality_multiplier = 1.0  # Normal for this user
        elif deviation < updated_profile.ema_daily_change * 2:
            quality_multiplier = 0.9  # Slightly unusual
        elif deviation < threshold:
            quality_multiplier = 0.7  # Unusual but acceptable
        else:
            quality_multiplier = 0.5  # Very unusual

        # Adjust for high oscillation users
        if updated_profile.oscillation_score > 0.5:
            quality_multiplier = min(1.0, quality_multiplier * 1.2)  # More tolerant

        return {
            'is_outlier': is_outlier,
            'deviation': deviation,
            'threshold': threshold,
            'quality_multiplier': quality_multiplier,
            'user_noise_estimate': updated_profile.ema_noise_estimate,
            'user_daily_variation': updated_profile.ema_daily_change,
            'oscillation_score': updated_profile.oscillation_score,
            'updated_profile': updated_profile,
            'adaptive_params': self._get_adaptive_params(updated_profile)
        }

    def _get_adaptive_params(self, profile: StatefulUserProfile) -> Dict:
        """Get current adaptive parameters from profile."""
        return {
            'kalman': {
                'process_noise': 0.1 * profile.current_noise_multiplier,
                'observation_noise': 1.0 * profile.current_noise_multiplier,
                'innovation_threshold': profile.current_outlier_threshold
            },
            'smoothing': {
                'strength': profile.current_smoothing_strength,
                'apply': profile.oscillation_score > 0.3
            },
            'outlier': {
                'threshold': profile.current_outlier_threshold,
                'strict_mode': profile.ema_daily_change < 0.5  # Strict for stable users
            }
        }
```

### 4. Integration with Recursive Processing

```python
# Integration with existing recursive processor

def process_measurement_with_adaptive_ranges(measurement: Dict,
                                            user_state: Dict,
                                            kalman_filter: AdaptiveKalmanFilter) -> Dict:
    """
    Process single measurement with adaptive ranges.
    This is called recursively for each measurement.
    """

    # Extract or initialize profile from state
    if 'user_profile' in user_state:
        profile = StatefulUserProfile.from_dict(user_state['user_profile'])
    else:
        profile = StatefulUserProfile()

    # Create validator
    validator = StatefulRangeValidator()

    # Validate and update profile in one pass
    validation_result = validator.validate_measurement(
        measurement,
        profile,
        user_state.get('kalman_state')
    )

    # Update measurement based on validation
    if validation_result['is_outlier']:
        measurement['outlier'] = True
        measurement['outlier_reason'] = f"Exceeds adaptive threshold ({validation_result['threshold']:.1f}kg)"

    # Adjust quality score
    measurement['quality_score'] *= validation_result['quality_multiplier']

    # Apply adaptive Kalman parameters
    adaptive_params = validation_result['adaptive_params']
    kalman_filter.update_parameters({
        'process_noise': adaptive_params['kalman']['process_noise'],
        'observation_noise': adaptive_params['kalman']['observation_noise']
    })

    # Update Kalman if not outlier
    if not measurement.get('outlier'):
        kalman_filter.update(measurement['weight'])

    # Save updated profile back to state
    user_state['user_profile'] = validation_result['updated_profile'].to_dict()

    # Add profile metrics to measurement for visibility
    measurement['profile_metrics'] = {
        'user_daily_variation': validation_result['user_daily_variation'],
        'oscillation_score': validation_result['oscillation_score'],
        'adaptive_threshold': validation_result['threshold']
    }

    return measurement

# Modified database state structure
class UserState:
    """Enhanced user state with profile."""

    def __init__(self):
        self.kalman_state = {}  # Existing Kalman state
        self.buffers = {}       # Existing buffers
        self.user_profile = {}  # NEW: Stateful profile
        self.last_processed = None

    def to_dict(self):
        return {
            'kalman_state': self.kalman_state,
            'buffers': self.buffers,
            'user_profile': self.user_profile,  # Persisted with state
            'last_processed': self.last_processed
        }
```

## Key Advantages of Stateful Approach

1. **No History Required**: Uses only current state + new measurement
2. **Incremental Updates**: All statistics update in O(1) time
3. **Memory Efficient**: Fixed-size buffers (7 days) for recent context
4. **Database Compatible**: Profile serializes with existing state
5. **Recursive-Friendly**: Each measurement updates independently

## Implementation Steps

### Phase 1: Core Stateful Profile (Week 1)
1. Implement `StatefulUserProfile` dataclass
2. Add incremental mean/variance calculation
3. Create EMA-based change tracking
4. Build circular buffer system

### Phase 2: Adaptive Thresholds (Week 2)
1. Implement threshold calculation from EMAs
2. Add oscillation scoring
3. Create noise estimation
4. Build parameter adaptation

### Phase 3: Integration (Week 3)
1. Modify state structure to include profile
2. Update database schema
3. Integrate with processor
4. Add profile metrics to output

### Phase 4: Testing & Tuning (Week 4)
1. Test with real user data
2. Tune EMA learning rates
3. Validate threshold calculations
4. Add monitoring metrics

## Configuration

```toml
[adaptive_ranges]
enabled = true
ema_alpha = 0.1  # Learning rate for changes
noise_alpha = 0.05  # Learning rate for noise estimate
buffer_size = 7  # Days of recent history

[adaptive_ranges.thresholds]
base_multiplier = 3.0  # Base threshold = multiplier * daily_variation
min_threshold = 1.0  # Minimum threshold in kg
max_threshold = 10.0  # Maximum threshold in kg

[adaptive_ranges.oscillation]
high_oscillation_score = 0.5
moderate_oscillation_score = 0.3
```

## Success Metrics

1. **Memory usage**: Constant per user (~1KB state)
2. **Processing time**: O(1) per measurement
3. **Adaptation speed**: Converges within 10-15 measurements
4. **False positive reduction**: >30% reduction

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| EMA lag | Medium | Dual learning rates (fast for changes, slow for noise) |
| Cold start | Low | Conservative defaults, faster initial learning |
| State corruption | High | Validation on deserialization, safe defaults |
| Oscillation misdetection | Medium | Multiple indicators (direction changes + variance) |

This stateful approach maintains all the benefits of adaptive ranges while working perfectly with your recursive architecture!