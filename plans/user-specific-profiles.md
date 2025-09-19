# Plan: User-Specific Profiles

## Problem Statement

Analysis of 15,701 users revealed significant variation in measurement patterns:
- Some users measure daily with <1kg variation
- Others measure monthly with 5-10kg normal fluctuations
- Medical conditions cause different variation patterns (dialysis, heart failure, etc.)
- Current system uses same thresholds for all users

Fixed thresholds cause:
- Over-filtering for users with natural high variation
- Under-filtering for users with stable patterns
- Inappropriate reset triggers based on individual patterns

## Objectives

1. Learn individual user measurement patterns
2. Adapt thresholds to user-specific baselines
3. Account for medical conditions affecting weight
4. Detect pattern changes over time
5. Provide personalized quality scoring

## Implementation Design

### 1. User Profile Builder

```python
# src/processing/user_profile.py

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass, field

@dataclass
class UserProfile:
    """Comprehensive profile of user's measurement patterns."""
    user_id: str
    created_at: datetime
    updated_at: datetime

    # Basic statistics
    mean_weight: float = 0.0
    median_weight: float = 0.0
    std_deviation: float = 0.0
    iqr: float = 0.0

    # Pattern characteristics
    typical_variation: float = 0.0  # Normal day-to-day variation
    max_safe_change: float = 0.0    # Maximum observed safe change
    measurement_frequency: str = ""  # daily, weekly, monthly, sporadic
    avg_gap_days: float = 0.0

    # Variation patterns
    morning_evening_diff: Optional[float] = None
    weekday_weekend_diff: Optional[float] = None
    seasonal_pattern: Dict[str, float] = field(default_factory=dict)

    # Medical indicators
    has_oscillating_pattern: bool = False
    has_dialysis_pattern: bool = False
    has_rapid_fluctuations: bool = False

    # Source preferences
    primary_source: Optional[str] = None
    source_distribution: Dict[str, float] = field(default_factory=dict)

    # Quality metrics
    measurement_consistency: float = 0.5
    data_quality_score: float = 0.5

    # Adaptive thresholds
    outlier_threshold: float = 0.15  # Percentage deviation
    reset_gap_days: float = 30.0
    min_quality_score: float = 0.5


class UserProfileBuilder:
    """Builds and maintains user-specific profiles."""

    def __init__(self, db, config=None):
        self.db = db
        self.config = config or {}

        # Configuration
        self.min_measurements = config.get('min_measurements_for_profile', 20)
        self.profile_update_frequency = config.get('update_frequency_days', 7)
        self.learning_rate = config.get('learning_rate', 0.1)

    def build_profile(self, user_id: str, force_rebuild: bool = False) -> UserProfile:
        """
        Build or update user profile from historical data.

        Args:
            user_id: User identifier
            force_rebuild: Force complete rebuild instead of update

        Returns:
            UserProfile object
        """
        # Check for existing profile
        if not force_rebuild:
            existing = self.db.get_user_profile(user_id)
            if existing and self._should_update(existing):
                return self._update_profile(existing)
            elif existing:
                return existing

        # Get historical measurements
        history = self.db.get_measurement_history(user_id, days=365)

        if len(history) < self.min_measurements:
            return self._create_default_profile(user_id)

        # Build new profile
        profile = UserProfile(
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Calculate basic statistics
        weights = [m['weight'] for m in history]
        profile.mean_weight = np.mean(weights)
        profile.median_weight = np.median(weights)
        profile.std_deviation = np.std(weights)

        q1, q3 = np.percentile(weights, [25, 75])
        profile.iqr = q3 - q1

        # Analyze variation patterns
        self._analyze_variation_patterns(profile, history)

        # Detect medical patterns
        self._detect_medical_patterns(profile, history)

        # Analyze measurement frequency
        self._analyze_frequency(profile, history)

        # Calculate adaptive thresholds
        self._calculate_adaptive_thresholds(profile, history)

        # Save profile
        self.db.save_user_profile(profile)

        return profile

    def _analyze_variation_patterns(self, profile: UserProfile, history: List[Dict]):
        """Analyze patterns of weight variation."""
        if len(history) < 10:
            return

        # Calculate day-to-day variations
        variations = []
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]

            time_diff = (curr['timestamp'] - prev['timestamp']).days

            if time_diff <= 2:  # Within 2 days
                weight_diff = abs(curr['weight'] - prev['weight'])
                variations.append(weight_diff / prev['weight'])

        if variations:
            profile.typical_variation = np.percentile(variations, 75)
            profile.max_safe_change = np.percentile(variations, 95)

        # Time-of-day patterns
        self._analyze_time_patterns(profile, history)

        # Seasonal patterns
        self._analyze_seasonal_patterns(profile, history)

    def _analyze_time_patterns(self, profile: UserProfile, history: List[Dict]):
        """Analyze time-based weight patterns."""
        morning_weights = []  # 6am - 12pm
        evening_weights = []  # 6pm - 12am

        for m in history:
            hour = m['timestamp'].hour
            if 6 <= hour < 12:
                morning_weights.append(m['weight'])
            elif 18 <= hour < 24:
                evening_weights.append(m['weight'])

        if len(morning_weights) > 5 and len(evening_weights) > 5:
            profile.morning_evening_diff = np.median(evening_weights) - np.median(morning_weights)

        # Weekday vs weekend
        weekday_weights = []
        weekend_weights = []

        for m in history:
            if m['timestamp'].weekday() < 5:  # Monday-Friday
                weekday_weights.append(m['weight'])
            else:
                weekend_weights.append(m['weight'])

        if len(weekday_weights) > 10 and len(weekend_weights) > 5:
            profile.weekday_weekend_diff = np.median(weekend_weights) - np.median(weekday_weights)

    def _detect_medical_patterns(self, profile: UserProfile, history: List[Dict]):
        """Detect medical condition patterns."""

        # Dialysis pattern: Regular large drops followed by gradual increases
        dialysis_pattern_score = self._detect_dialysis_pattern(history)
        profile.has_dialysis_pattern = dialysis_pattern_score > 0.7

        # Oscillating pattern: Frequent direction changes
        oscillation_score = self._calculate_oscillation_score(history)
        profile.has_oscillating_pattern = oscillation_score > 0.6

        # Rapid fluctuations: Large changes in short periods
        rapid_score = self._detect_rapid_fluctuations(history)
        profile.has_rapid_fluctuations = rapid_score > 0.5

    def _detect_dialysis_pattern(self, history: List[Dict]) -> float:
        """Detect dialysis-like patterns (regular drops and recoveries)."""
        if len(history) < 20:
            return 0.0

        # Look for pattern: drop of 2-5kg followed by gradual increase
        pattern_matches = 0
        pattern_checks = 0

        sorted_history = sorted(history, key=lambda x: x['timestamp'])

        i = 0
        while i < len(sorted_history) - 3:
            # Check for significant drop
            if i + 1 < len(sorted_history):
                drop = sorted_history[i]['weight'] - sorted_history[i+1]['weight']
                if 2 <= drop <= 5:  # Typical dialysis fluid removal
                    pattern_checks += 1

                    # Check for gradual recovery in next measurements
                    recovery_found = False
                    for j in range(i+2, min(i+5, len(sorted_history))):
                        if sorted_history[j]['weight'] > sorted_history[i+1]['weight']:
                            recovery_found = True
                            break

                    if recovery_found:
                        pattern_matches += 1

                    i += 3  # Skip ahead to avoid overlapping patterns
                else:
                    i += 1
            else:
                i += 1

        if pattern_checks > 0:
            return pattern_matches / pattern_checks
        return 0.0

    def _calculate_oscillation_score(self, history: List[Dict]) -> float:
        """Calculate how much weight oscillates."""
        if len(history) < 5:
            return 0.0

        sorted_history = sorted(history, key=lambda x: x['timestamp'])
        weights = [m['weight'] for m in sorted_history]

        direction_changes = 0
        for i in range(2, len(weights)):
            prev_direction = weights[i-1] - weights[i-2]
            curr_direction = weights[i] - weights[i-1]

            if prev_direction * curr_direction < 0:  # Sign change
                direction_changes += 1

        max_possible_changes = len(weights) - 2
        return direction_changes / max_possible_changes if max_possible_changes > 0 else 0.0

    def _calculate_adaptive_thresholds(self, profile: UserProfile, history: List[Dict]):
        """Calculate user-specific thresholds."""

        # Outlier threshold based on historical variation
        if profile.typical_variation > 0:
            # Users with high variation get more lenient thresholds
            profile.outlier_threshold = min(0.3, profile.typical_variation * 3)
        else:
            profile.outlier_threshold = 0.15

        # Reset gap based on measurement frequency
        gaps = []
        sorted_history = sorted(history, key=lambda x: x['timestamp'])
        for i in range(1, len(sorted_history)):
            gap = (sorted_history[i]['timestamp'] - sorted_history[i-1]['timestamp']).days
            if gap > 0:
                gaps.append(gap)

        if gaps:
            # Use 95th percentile of gaps as reset threshold
            profile.reset_gap_days = min(90, np.percentile(gaps, 95) * 2)
        else:
            profile.reset_gap_days = 30

        # Quality score threshold based on data consistency
        if profile.measurement_consistency > 0.8:
            profile.min_quality_score = 0.6  # Higher standard for consistent users
        else:
            profile.min_quality_score = 0.4  # More lenient for inconsistent patterns
```

### 2. Adaptive Threshold Manager

```python
class AdaptiveThresholdManager:
    """Manages user-specific adaptive thresholds."""

    def __init__(self, db, config=None):
        self.db = db
        self.config = config or {}
        self.profile_builder = UserProfileBuilder(db, config)

    def get_user_thresholds(self, user_id: str) -> Dict[str, Any]:
        """Get current thresholds for a user."""
        profile = self.profile_builder.build_profile(user_id)

        return {
            'outlier_threshold': profile.outlier_threshold,
            'reset_gap_days': profile.reset_gap_days,
            'min_quality_score': profile.min_quality_score,
            'typical_variation': profile.typical_variation,
            'max_safe_change': profile.max_safe_change,
            'has_medical_pattern': (
                profile.has_dialysis_pattern or
                profile.has_oscillating_pattern or
                profile.has_rapid_fluctuations
            )
        }

    def should_accept_measurement(self, user_id: str, measurement: Dict[str, Any],
                                 context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if measurement should be accepted based on user profile.

        Args:
            user_id: User identifier
            measurement: Current measurement
            context: Additional context (last weight, quality score, etc.)

        Returns:
            Tuple of (should_accept, reason)
        """
        profile = self.profile_builder.build_profile(user_id)

        weight = measurement['weight']
        quality_score = context.get('quality_score', 0.5)

        # Check against user's normal range
        if profile.mean_weight > 0:
            deviation = abs(weight - profile.mean_weight) / profile.mean_weight

            # Account for medical patterns
            if profile.has_medical_pattern:
                threshold = profile.outlier_threshold * 1.5  # More lenient
            else:
                threshold = profile.outlier_threshold

            if deviation > threshold:
                if quality_score < profile.min_quality_score + 0.2:  # Need higher quality
                    return False, f'exceeds_user_threshold_{deviation:.1%}'

        # Check day-to-day change
        if context.get('last_weight'):
            change = abs(weight - context['last_weight']) / context['last_weight']

            if change > profile.max_safe_change:
                # Check if this is expected based on patterns
                if not self._is_expected_change(profile, measurement, context):
                    return False, f'exceeds_safe_change_{change:.1%}'

        # Quality score check with user-specific threshold
        if quality_score < profile.min_quality_score:
            return False, f'below_user_quality_threshold_{quality_score:.2f}'

        return True, 'within_user_profile'

    def _is_expected_change(self, profile: UserProfile, measurement: Dict[str, Any],
                           context: Dict[str, Any]) -> bool:
        """Check if change is expected based on user patterns."""

        # Time-based expectations
        if profile.morning_evening_diff is not None:
            last_time = context.get('last_timestamp')
            curr_time = measurement['timestamp']

            if last_time and curr_time:
                last_hour = last_time.hour
                curr_hour = curr_time.hour

                # Morning to evening
                if last_hour < 12 and curr_hour >= 18:
                    expected_diff = profile.morning_evening_diff
                    actual_diff = measurement['weight'] - context['last_weight']

                    if abs(actual_diff - expected_diff) < profile.typical_variation:
                        return True

        # Dialysis pattern
        if profile.has_dialysis_pattern:
            weight_drop = context['last_weight'] - measurement['weight']
            if 2 <= weight_drop <= 5:  # Typical dialysis removal
                return True

        return False
```

### 3. Profile-Based Quality Scorer

```python
class ProfileBasedQualityScorer:
    """Scores measurement quality based on user profile."""

    def __init__(self, db):
        self.db = db
        self.profile_builder = UserProfileBuilder(db)

    def calculate_quality_score(self, user_id: str, measurement: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality score using user profile.

        Returns:
            Dict with component scores and final score
        """
        profile = self.profile_builder.build_profile(user_id)

        scores = {}

        # Deviation from user's normal
        weight = measurement['weight']
        if profile.mean_weight > 0:
            deviation = abs(weight - profile.mean_weight) / profile.mean_weight
            scores['profile_deviation'] = max(0, 1.0 - deviation / profile.outlier_threshold)
        else:
            scores['profile_deviation'] = 0.5

        # Consistency with user's patterns
        scores['pattern_consistency'] = self._calculate_pattern_consistency(
            profile, measurement, context
        )

        # Source reliability for this user
        source = measurement['source']
        if source == profile.primary_source:
            scores['source_reliability'] = 0.9
        elif source in profile.source_distribution:
            scores['source_reliability'] = 0.7 + 0.2 * profile.source_distribution[source]
        else:
            scores['source_reliability'] = 0.5

        # Temporal consistency
        scores['temporal_consistency'] = self._calculate_temporal_consistency(
            profile, measurement, context
        )

        # Calculate weighted final score
        weights = {
            'profile_deviation': 0.35,
            'pattern_consistency': 0.25,
            'source_reliability': 0.20,
            'temporal_consistency': 0.20
        }

        final_score = sum(scores[k] * weights[k] for k in weights)

        # Boost score for medical patterns
        if profile.has_medical_pattern:
            if self._matches_medical_pattern(profile, measurement, context):
                final_score = min(1.0, final_score * 1.2)

        return {
            'final_score': final_score,
            'components': scores,
            'profile_match': final_score > 0.7
        }
```

## Implementation Steps

### Phase 1: Profile Building (Week 1)
1. Create `UserProfile` dataclass
2. Implement `UserProfileBuilder` with basic statistics
3. Add variation pattern analysis
4. Create database schema for profiles

### Phase 2: Pattern Detection (Week 2)
1. Implement medical pattern detection
2. Add time-based pattern analysis
3. Create seasonal pattern detection
4. Build oscillation scoring

### Phase 3: Adaptive Thresholds (Week 3)
1. Implement `AdaptiveThresholdManager`
2. Create user-specific threshold calculation
3. Add expected change detection
4. Build profile-based acceptance logic

### Phase 4: Integration (Week 4)
1. Create `ProfileBasedQualityScorer`
2. Integrate with main processor
3. Add profile update scheduling
4. Create profile visualization tools

## Testing Strategy

### Unit Tests
```python
def test_dialysis_pattern_detection():
    """Test detection of dialysis patterns."""
    builder = UserProfileBuilder(mock_db)

    # Create dialysis-like pattern
    history = []
    base_weight = 70.0
    for week in range(4):
        # Pre-dialysis (high)
        history.append({'weight': base_weight + 3, 'timestamp': datetime.now() + timedelta(days=week*7)})
        # Post-dialysis (low)
        history.append({'weight': base_weight, 'timestamp': datetime.now() + timedelta(days=week*7+1)})
        # Recovery
        history.append({'weight': base_weight + 1.5, 'timestamp': datetime.now() + timedelta(days=week*7+3)})

    score = builder._detect_dialysis_pattern(history)
    assert score > 0.7

def test_user_specific_thresholds():
    """Test calculation of user-specific thresholds."""
    manager = AdaptiveThresholdManager(mock_db)

    # User with high variation
    high_var_profile = UserProfile(
        user_id='user1',
        typical_variation=0.05,  # 5% typical
        max_safe_change=0.10      # 10% max
    )

    thresholds = manager.get_user_thresholds('user1')
    assert thresholds['outlier_threshold'] > 0.15  # More lenient than default
```

## Configuration

```toml
[user_profiles]
enabled = true
min_measurements_for_profile = 20
update_frequency_days = 7
learning_rate = 0.1

[user_profiles.pattern_detection]
dialysis_drop_range = [2.0, 5.0]  # kg
oscillation_threshold = 0.6
rapid_fluctuation_window_hours = 24

[user_profiles.adaptive_thresholds]
default_outlier_threshold = 0.15
max_outlier_threshold = 0.30
default_reset_gap_days = 30
max_reset_gap_days = 90
```

## Success Metrics

1. **Acceptance accuracy**: >92% correct accept/reject decisions
2. **Pattern detection**: >85% medical patterns correctly identified
3. **Threshold stability**: Thresholds stabilize within 30 measurements
4. **False rejection reduction**: <3% for users with established profiles

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to noise | High | Minimum measurement requirements, robust statistics |
| Profile drift over time | Medium | Regular updates, drift detection |
| New user cold start | High | Conservative defaults, quick learning |
| Medical condition changes | High | Pattern change detection, alerts |

## Future Enhancements

1. Machine learning for pattern prediction
2. Integration with medical records
3. Collaborative filtering across similar users
4. Automatic anomaly explanations
5. Profile sharing between healthcare providers