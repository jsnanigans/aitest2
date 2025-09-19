# Plan: Dynamic User-Adaptive Ranges for Noise Handling

## Problem Statement

Current oscillation handling uses fixed thresholds and parameters for all users, but analysis shows:
- User 07d08dd8: 8 direction changes with 21.9kg range
- User 44241501: Stable around 83kg with minimal variation
- Some users have consistent 2-3kg daily fluctuations (medical)
- Others have measurement noise of <0.5kg

Fixed thresholds either over-filter legitimate variation or under-filter noise depending on the user's normal pattern.

## Objectives

1. Build user-specific baseline profiles of normal variation
2. Dynamically adapt filtering thresholds to each user's pattern
3. Distinguish between "normal for this user" and "anomalous for this user"
4. Learn and update profiles over time
5. Handle new users with conservative defaults that adapt quickly

## Implementation Design

### 1. User Variation Profile

```python
# src/processing/user_variation_profile.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

@dataclass
class UserVariationProfile:
    """Tracks user-specific variation patterns."""
    user_id: str

    # Historical statistics
    baseline_weight: float = 0.0
    typical_daily_variation: float = 0.0      # Normal day-to-day change
    typical_weekly_range: float = 0.0         # Normal weekly min-max range
    noise_floor: float = 0.0                  # Minimum variation (measurement noise)

    # Percentile-based ranges (adaptive)
    p10_daily_change: float = 0.0             # 10th percentile daily change
    p90_daily_change: float = 0.0             # 90th percentile daily change
    p95_daily_change: float = 0.0             # 95th percentile (outlier threshold)

    # Pattern characteristics
    oscillation_frequency: Optional[float] = None
    oscillation_amplitude: Optional[float] = None
    has_medical_pattern: bool = False

    # Confidence and sample size
    measurement_count: int = 0
    confidence_score: float = 0.0             # 0-1, based on data quantity/quality
    last_updated: Optional[datetime] = None

    # Adaptive thresholds (dynamically calculated)
    outlier_threshold_multiplier: float = 3.0  # Start conservative
    smoothing_window_size: int = 3
    kalman_noise_adjustment: float = 1.0

    # Rolling statistics buffers
    recent_changes: List[float] = field(default_factory=list)
    recent_ranges: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize profile to dictionary."""
        return {
            'user_id': self.user_id,
            'baseline_weight': self.baseline_weight,
            'typical_daily_variation': self.typical_daily_variation,
            'typical_weekly_range': self.typical_weekly_range,
            'noise_floor': self.noise_floor,
            'p10_daily_change': self.p10_daily_change,
            'p90_daily_change': self.p90_daily_change,
            'p95_daily_change': self.p95_daily_change,
            'oscillation_frequency': self.oscillation_frequency,
            'oscillation_amplitude': self.oscillation_amplitude,
            'has_medical_pattern': self.has_medical_pattern,
            'measurement_count': self.measurement_count,
            'confidence_score': self.confidence_score,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'outlier_threshold_multiplier': self.outlier_threshold_multiplier,
            'smoothing_window_size': self.smoothing_window_size,
            'kalman_noise_adjustment': self.kalman_noise_adjustment
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserVariationProfile':
        """Deserialize profile from dictionary."""
        profile = cls(user_id=data['user_id'])
        for key, value in data.items():
            if key == 'last_updated' and value:
                setattr(profile, key, datetime.fromisoformat(value))
            elif hasattr(profile, key):
                setattr(profile, key, value)
        return profile
```

### 2. Profile Builder and Updater

```python
class UserProfileBuilder:
    """Builds and updates user variation profiles from historical data."""

    def __init__(self, min_measurements: int = 20):
        self.min_measurements = min_measurements

    def build_profile(self, user_id: str, measurements: List[Dict]) -> UserVariationProfile:
        """Build initial profile from historical measurements."""
        profile = UserVariationProfile(user_id=user_id)

        if len(measurements) < 3:
            return self._get_default_profile(user_id)

        # Sort measurements by timestamp
        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])
        weights = np.array([m['weight'] for m in sorted_measurements])
        timestamps = [m['timestamp'] for m in sorted_measurements]

        # Calculate baseline statistics
        profile.baseline_weight = np.median(weights)
        profile.measurement_count = len(measurements)

        # Calculate daily changes
        daily_changes = []
        for i in range(1, len(sorted_measurements)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400
            if 0.5 <= time_diff <= 2:  # Consider only ~daily measurements
                weight_change = abs(weights[i] - weights[i-1])
                daily_changes.append(weight_change)

        if daily_changes:
            # Calculate percentile-based thresholds
            profile.p10_daily_change = np.percentile(daily_changes, 10)
            profile.p90_daily_change = np.percentile(daily_changes, 90)
            profile.p95_daily_change = np.percentile(daily_changes, 95)
            profile.typical_daily_variation = np.median(daily_changes)

            # Estimate noise floor (minimum consistent variation)
            profile.noise_floor = profile.p10_daily_change

        # Calculate weekly ranges
        weekly_ranges = self._calculate_weekly_ranges(sorted_measurements)
        if weekly_ranges:
            profile.typical_weekly_range = np.median(weekly_ranges)

        # Detect patterns
        self._detect_patterns(profile, sorted_measurements)

        # Set adaptive thresholds based on variation
        self._update_adaptive_thresholds(profile)

        # Calculate confidence
        profile.confidence_score = self._calculate_confidence(profile)
        profile.last_updated = datetime.now()

        return profile

    def update_profile(self, profile: UserVariationProfile,
                      new_measurements: List[Dict]) -> UserVariationProfile:
        """Update existing profile with new measurements."""
        if not new_measurements:
            return profile

        sorted_new = sorted(new_measurements, key=lambda x: x['timestamp'])
        weights = np.array([m['weight'] for m in sorted_new])

        # Update measurement count
        profile.measurement_count += len(new_measurements)

        # Calculate recent changes
        for i in range(1, len(sorted_new)):
            time_diff = (sorted_new[i]['timestamp'] - sorted_new[i-1]['timestamp']).total_seconds() / 86400
            if 0.5 <= time_diff <= 2:
                change = abs(weights[i] - weights[i-1])
                profile.recent_changes.append(change)

        # Keep only recent changes (last 30 measurements)
        profile.recent_changes = profile.recent_changes[-30:]

        # Update statistics with exponential moving average
        if profile.recent_changes:
            alpha = 0.1  # Learning rate
            new_typical = np.median(profile.recent_changes)
            profile.typical_daily_variation = (1 - alpha) * profile.typical_daily_variation + alpha * new_typical

            # Update percentiles
            profile.p10_daily_change = (1 - alpha) * profile.p10_daily_change + alpha * np.percentile(profile.recent_changes, 10)
            profile.p90_daily_change = (1 - alpha) * profile.p90_daily_change + alpha * np.percentile(profile.recent_changes, 90)
            profile.p95_daily_change = (1 - alpha) * profile.p95_daily_change + alpha * np.percentile(profile.recent_changes, 95)

        # Update adaptive thresholds
        self._update_adaptive_thresholds(profile)

        # Update confidence
        profile.confidence_score = self._calculate_confidence(profile)
        profile.last_updated = datetime.now()

        return profile

    def _update_adaptive_thresholds(self, profile: UserVariationProfile):
        """Update adaptive processing parameters based on user's variation profile."""

        # Adjust outlier threshold based on typical variation
        if profile.typical_daily_variation < 0.5:  # Very stable user
            profile.outlier_threshold_multiplier = 2.0  # Strict
            profile.smoothing_window_size = 3
            profile.kalman_noise_adjustment = 0.5  # Reduce noise parameters

        elif profile.typical_daily_variation < 1.0:  # Moderately stable
            profile.outlier_threshold_multiplier = 2.5
            profile.smoothing_window_size = 3
            profile.kalman_noise_adjustment = 0.8

        elif profile.typical_daily_variation < 2.0:  # Normal variation
            profile.outlier_threshold_multiplier = 3.0
            profile.smoothing_window_size = 5
            profile.kalman_noise_adjustment = 1.0

        elif profile.typical_daily_variation < 3.0:  # High variation
            profile.outlier_threshold_multiplier = 4.0
            profile.smoothing_window_size = 5
            profile.kalman_noise_adjustment = 1.5

        else:  # Very high variation (possibly medical)
            profile.outlier_threshold_multiplier = 5.0
            profile.smoothing_window_size = 7
            profile.kalman_noise_adjustment = 2.0

        # Adjust for medical patterns
        if profile.has_medical_pattern:
            profile.outlier_threshold_multiplier *= 1.5  # More lenient
            profile.smoothing_window_size = max(3, profile.smoothing_window_size - 2)  # Less smoothing

    def _calculate_confidence(self, profile: UserVariationProfile) -> float:
        """Calculate confidence score based on data quantity and quality."""
        # Base confidence on measurement count
        count_score = min(1.0, profile.measurement_count / 50)

        # Adjust for data recency
        recency_score = 1.0
        if profile.last_updated:
            days_old = (datetime.now() - profile.last_updated).days
            recency_score = max(0.5, 1.0 - (days_old / 30))

        # Adjust for pattern stability
        if profile.recent_changes:
            cv = np.std(profile.recent_changes) / np.mean(profile.recent_changes) if np.mean(profile.recent_changes) > 0 else 1
            stability_score = max(0.3, 1.0 - cv)
        else:
            stability_score = 0.5

        return count_score * 0.5 + recency_score * 0.2 + stability_score * 0.3

    def _detect_patterns(self, profile: UserVariationProfile, measurements: List[Dict]):
        """Detect oscillation patterns and medical conditions."""
        if len(measurements) < 14:  # Need at least 2 weeks of data
            return

        weights = np.array([m['weight'] for m in measurements])
        timestamps = [m['timestamp'] for m in measurements]

        # Check for regular oscillations (dialysis, heart failure)
        # Simplified version - would use FFT in production
        daily_measurements = self._group_by_day(measurements)
        if len(daily_measurements) >= 7:
            daily_weights = [np.mean([m['weight'] for m in day]) for day in daily_measurements.values()]

            # Check for 2-3 day cycles (dialysis)
            cycle_detected = self._detect_cycle(daily_weights, period_range=(2, 4))
            if cycle_detected:
                profile.has_medical_pattern = True
                profile.oscillation_frequency = 1.0 / cycle_detected

            # Check for daily variation > 2kg (heart failure)
            daily_ranges = [max(day) - min(day) for day in daily_measurements.values()
                          if len(day) > 1]
            if daily_ranges and np.median(daily_ranges) > 2.0:
                profile.has_medical_pattern = True

    def _get_default_profile(self, user_id: str) -> UserVariationProfile:
        """Get conservative default profile for new users."""
        profile = UserVariationProfile(user_id=user_id)
        profile.typical_daily_variation = 1.0  # Conservative estimate
        profile.typical_weekly_range = 3.0
        profile.noise_floor = 0.3
        profile.p10_daily_change = 0.2
        profile.p90_daily_change = 2.0
        profile.p95_daily_change = 3.0
        profile.outlier_threshold_multiplier = 3.0  # Conservative
        profile.confidence_score = 0.1  # Low confidence for default
        return profile
```

### 3. Dynamic Range Validator

```python
class DynamicRangeValidator:
    """Validates measurements using user-specific dynamic ranges."""

    def __init__(self):
        self.profile_builder = UserProfileBuilder()
        self.profiles_cache = {}  # In-memory cache

    def validate_measurement(self, user_id: str, measurement: Dict,
                            profile: UserVariationProfile,
                            kalman_prediction: Optional[float] = None) -> Dict:
        """
        Validate measurement against user's dynamic range.

        Returns:
            Dict with validation results and adjusted parameters
        """
        weight = measurement['weight']

        # Calculate deviation from prediction or baseline
        if kalman_prediction is not None:
            deviation = abs(weight - kalman_prediction)
            relative_deviation = deviation / kalman_prediction
        else:
            deviation = abs(weight - profile.baseline_weight)
            relative_deviation = deviation / profile.baseline_weight if profile.baseline_weight > 0 else 0

        # Dynamic outlier threshold based on user's profile
        if profile.confidence_score > 0.5:  # Use profile if confident
            # Scale threshold by user's typical variation
            threshold = profile.p95_daily_change * profile.outlier_threshold_multiplier

            # Additional check against weekly range
            weekly_threshold = profile.typical_weekly_range * 1.5

            is_outlier = deviation > threshold or deviation > weekly_threshold

            # Calculate quality adjustment based on how well it fits the profile
            if deviation <= profile.typical_daily_variation:
                quality_multiplier = 1.0  # Normal for this user
            elif deviation <= profile.p90_daily_change:
                quality_multiplier = 0.9  # Slightly unusual
            elif deviation <= profile.p95_daily_change:
                quality_multiplier = 0.7  # Unusual but possible
            else:
                quality_multiplier = 0.5  # Very unusual

        else:  # Low confidence, use conservative defaults
            threshold = 3.0  # 3kg default
            is_outlier = deviation > threshold or relative_deviation > 0.05
            quality_multiplier = 1.0 if deviation < 1.0 else 0.8

        # Adjust for medical patterns
        if profile.has_medical_pattern and deviation <= profile.oscillation_amplitude * 1.5:
            is_outlier = False  # Expected variation
            quality_multiplier = min(1.0, quality_multiplier * 1.2)

        return {
            'is_outlier': is_outlier,
            'deviation': deviation,
            'threshold': threshold,
            'quality_multiplier': quality_multiplier,
            'confidence': profile.confidence_score,
            'user_typical_variation': profile.typical_daily_variation,
            'smoothing_recommended': deviation > profile.p90_daily_change
        }

    def get_adaptive_parameters(self, user_id: str,
                               profile: UserVariationProfile) -> Dict:
        """
        Get user-specific processing parameters.

        Returns:
            Dictionary with adaptive Kalman and smoothing parameters
        """
        params = {
            'kalman': {
                'process_noise': 0.1 * profile.kalman_noise_adjustment,
                'observation_noise': 1.0 * profile.kalman_noise_adjustment,
                'reset_threshold_days': 30 if not profile.has_medical_pattern else 45,
                'innovation_threshold': profile.p95_daily_change * 2
            },
            'smoothing': {
                'window_size': profile.smoothing_window_size,
                'apply_smoothing': profile.typical_daily_variation > 1.0,
                'smoothing_strength': min(0.7, profile.noise_floor / profile.typical_daily_variation) if profile.typical_daily_variation > 0 else 0.5
            },
            'outlier_detection': {
                'threshold_multiplier': profile.outlier_threshold_multiplier,
                'min_threshold_kg': profile.p95_daily_change,
                'max_threshold_kg': profile.typical_weekly_range * 2,
                'use_mad': profile.measurement_count > 30,  # Use MAD only with enough data
                'fallback_threshold': 3.0
            },
            'quality_scoring': {
                'source_weight': 0.3 if profile.confidence_score > 0.7 else 0.5,
                'consistency_weight': 0.4 if not profile.has_medical_pattern else 0.2,
                'plausibility_weight': 0.3
            }
        }

        return params
```

### 4. Integration with Main Processing Pipeline

```python
# Enhanced processor integration

class EnhancedWeightProcessor:
    def __init__(self, config):
        self.config = config
        self.profile_builder = UserProfileBuilder()
        self.range_validator = DynamicRangeValidator()
        self.profile_cache = {}  # Cache user profiles

    def process_measurements(self, user_id: str, measurements: List[Dict]) -> List[Dict]:
        """Process measurements with user-adaptive ranges."""

        # Get or build user profile
        profile = self._get_user_profile(user_id, measurements)

        # Get adaptive parameters for this user
        adaptive_params = self.range_validator.get_adaptive_parameters(user_id, profile)

        # Update Kalman filter with user-specific parameters
        kalman = AdaptiveKalmanFilter(
            process_noise=adaptive_params['kalman']['process_noise'],
            observation_noise=adaptive_params['kalman']['observation_noise']
        )

        processed = []
        for measurement in measurements:
            # Get Kalman prediction
            prediction = kalman.predict() if kalman.initialized else None

            # Validate against user's dynamic range
            validation = self.range_validator.validate_measurement(
                user_id, measurement, profile, prediction
            )

            # Apply user-specific outlier detection
            if validation['is_outlier'] and validation['confidence'] > 0.5:
                measurement['outlier'] = True
                measurement['outlier_reason'] = f"Exceeds user's typical range ({validation['user_typical_variation']:.1f}kg/day)"

            # Adjust quality score based on user's profile
            measurement['quality_score'] *= validation['quality_multiplier']

            # Apply smoothing if recommended
            if validation['smoothing_recommended'] and adaptive_params['smoothing']['apply_smoothing']:
                measurement['weight'] = self._apply_adaptive_smoothing(
                    measurement['weight'],
                    kalman.state if kalman.initialized else measurement['weight'],
                    adaptive_params['smoothing']['smoothing_strength']
                )
                measurement['smoothed'] = True

            # Update Kalman filter
            if not measurement.get('outlier'):
                kalman.update(measurement['weight'])

            processed.append(measurement)

        # Update profile with new measurements
        self._update_user_profile(user_id, profile, measurements)

        return processed

    def _get_user_profile(self, user_id: str, measurements: List[Dict]) -> UserVariationProfile:
        """Get cached profile or build new one."""
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]

        # Try to load from database
        profile = self.db.load_user_profile(user_id)
        if profile is None:
            # Build new profile from historical data
            historical = self.db.get_historical_measurements(user_id, days=90)
            all_measurements = historical + measurements if historical else measurements
            profile = self.profile_builder.build_profile(user_id, all_measurements)

        self.profile_cache[user_id] = profile
        return profile

    def _update_user_profile(self, user_id: str, profile: UserVariationProfile,
                           new_measurements: List[Dict]):
        """Update and persist user profile."""
        updated_profile = self.profile_builder.update_profile(profile, new_measurements)
        self.profile_cache[user_id] = updated_profile
        self.db.save_user_profile(updated_profile)
```

## Implementation Steps

### Phase 1: Profile System (Week 1)
1. Implement `UserVariationProfile` dataclass
2. Create profile persistence in database
3. Build profile builder with basic statistics
4. Add percentile calculations

### Phase 2: Dynamic Validation (Week 2)
1. Implement `DynamicRangeValidator`
2. Create adaptive threshold calculations
3. Add medical pattern detection
4. Build confidence scoring system

### Phase 3: Integration (Week 3)
1. Integrate with existing processor
2. Update Kalman filter with adaptive parameters
3. Modify outlier detection to use profiles
4. Update quality scoring with user context

### Phase 4: Learning System (Week 4)
1. Implement profile update mechanism
2. Add exponential moving average for statistics
3. Create feedback loop for continuous improvement
4. Build profile visualization tools

## Testing Strategy

```python
def test_stable_user_profile():
    """Test profile for stable weight user."""
    measurements = [
        {'weight': 70.0 + np.random.normal(0, 0.2),
         'timestamp': datetime.now() - timedelta(days=i)}
        for i in range(30)
    ]

    builder = UserProfileBuilder()
    profile = builder.build_profile('stable_user', measurements)

    assert profile.typical_daily_variation < 0.5
    assert profile.outlier_threshold_multiplier == 2.0  # Strict
    assert profile.kalman_noise_adjustment == 0.5  # Low noise

def test_oscillating_user_profile():
    """Test profile for oscillating weight user."""
    measurements = []
    for i in range(30):
        # 3-day oscillation pattern
        weight = 70 + 3 * np.sin(2 * np.pi * i / 3)
        measurements.append({
            'weight': weight,
            'timestamp': datetime.now() - timedelta(days=i)
        })

    builder = UserProfileBuilder()
    profile = builder.build_profile('oscillating_user', measurements)

    assert profile.has_medical_pattern == True
    assert profile.outlier_threshold_multiplier >= 4.0  # Lenient
    assert profile.typical_daily_variation > 2.0
```

## Configuration

```toml
[user_profiles]
enabled = true
min_measurements_for_profile = 20
profile_update_interval_days = 7
confidence_threshold = 0.5

[user_profiles.defaults]
# Used for new users with insufficient data
typical_daily_variation = 1.0
outlier_threshold_multiplier = 3.0
smoothing_window_size = 3

[user_profiles.learning]
# Learning rate for profile updates
alpha = 0.1
recent_measurements_buffer = 30
percentiles = [10, 90, 95]

[user_profiles.medical_detection]
dialysis_period_min = 2
dialysis_period_max = 4
heart_failure_daily_range = 2.0
```

## Success Metrics

1. **False positive reduction**: >40% reduction in incorrect outlier flags
2. **Pattern preservation**: >95% of medical patterns correctly identified
3. **Adaptation speed**: Profile converges within 20 measurements
4. **User satisfaction**: Fewer manual corrections needed

## Benefits Over Current System

1. **Personalized thresholds**: Each user has their own "normal" range
2. **Adaptive learning**: System improves over time for each user
3. **Medical pattern awareness**: Automatically detects and accommodates medical conditions
4. **Confidence-based processing**: More aggressive filtering when confident, conservative when uncertain
5. **Reduced false positives**: Fewer legitimate measurements flagged as outliers

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Slow initial learning | High | Conservative defaults, transfer learning from similar users |
| Profile drift | Medium | Regular profile validation, anomaly detection |
| Memory usage | Low | Profile caching with TTL, database persistence |
| Over-fitting to noise | Medium | Minimum sample requirements, robust statistics |

## Future Enhancements

1. **Cluster-based profiles**: Group similar users for better initial profiles
2. **Seasonal adjustments**: Account for seasonal weight patterns
3. **Cross-user learning**: Use population statistics to improve individual profiles
4. **Anomaly alerts**: Notify when user deviates from their normal pattern
5. **Profile explanations**: Show users why certain measurements were flagged