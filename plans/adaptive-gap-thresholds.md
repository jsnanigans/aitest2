# Plan: Adaptive Gap Thresholds

## Problem Statement

Current system uses fixed 30-day threshold for hard resets. Analysis shows:
- Some users measure daily (30 days is too long)
- Others measure monthly (30 days is too short)
- Gap patterns vary by user behavior and medical needs
- Fixed threshold causes inappropriate resets or missed resets

Example: User with 3094-day gap (8.5 years) vs user with weekly measurements - both use same 30-day threshold.

## Objectives

1. Calculate user-specific gap thresholds based on history
2. Adapt thresholds as patterns change
3. Consider measurement regularity and variation
4. Account for expected gaps (vacations, hospitalizations)
5. Provide sensible defaults for new users

## Implementation Design

### 1. Gap Pattern Analyzer

```python
# src/processing/gap_analyzer.py

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from scipy import stats

class GapPatternAnalyzer:
    """Analyzes measurement gaps to determine adaptive thresholds."""

    def __init__(self, config=None):
        self.config = config or {}

        # Configuration
        self.min_history_days = config.get('min_history_days', 90)
        self.min_measurements = config.get('min_measurements', 10)
        self.outlier_percentile = config.get('outlier_percentile', 95)

        # Default thresholds by user category
        self.category_defaults = {
            'frequent': 14,    # Daily/weekly measurers
            'regular': 30,     # Weekly/biweekly
            'sporadic': 60,    # Monthly or less
            'inactive': 90,    # Very rare measurements
            'new': 30          # New users (default)
        }

    def calculate_adaptive_threshold(self, user_id: str, measurement_history: List[Dict]) -> Dict[str, Any]:
        """
        Calculate adaptive gap threshold for a user.

        Args:
            user_id: User identifier
            measurement_history: List of historical measurements

        Returns:
            Dict with threshold and analysis details
        """
        if not measurement_history or len(measurement_history) < self.min_measurements:
            return {
                'threshold_days': self.category_defaults['new'],
                'category': 'new',
                'confidence': 0.3,
                'reason': 'insufficient_history'
            }

        # Calculate gaps between consecutive measurements
        gaps = self._calculate_gaps(measurement_history)

        if not gaps:
            return {
                'threshold_days': self.category_defaults['new'],
                'category': 'new',
                'confidence': 0.3,
                'reason': 'no_gaps_found'
            }

        # Analyze gap distribution
        analysis = self._analyze_gap_distribution(gaps)

        # Determine user category
        category = self._categorize_user(analysis)

        # Calculate adaptive threshold
        threshold = self._calculate_threshold(analysis, category)

        return {
            'threshold_days': threshold,
            'category': category,
            'confidence': analysis['confidence'],
            'median_gap': analysis['median'],
            'typical_range': analysis['typical_range'],
            'measurement_frequency': analysis['frequency'],
            'reason': analysis['pattern_description']
        }

    def _calculate_gaps(self, measurements: List[Dict]) -> List[float]:
        """Calculate gaps in days between consecutive measurements."""
        # Sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])

        gaps = []
        for i in range(1, len(sorted_measurements)):
            prev_time = sorted_measurements[i-1]['timestamp']
            curr_time = sorted_measurements[i]['timestamp']

            if isinstance(prev_time, str):
                prev_time = datetime.fromisoformat(prev_time)
            if isinstance(curr_time, str):
                curr_time = datetime.fromisoformat(curr_time)

            gap_days = (curr_time - prev_time).total_seconds() / 86400
            gaps.append(gap_days)

        return gaps

    def _analyze_gap_distribution(self, gaps: List[float]) -> Dict[str, Any]:
        """Analyze statistical distribution of gaps."""
        gaps_array = np.array(gaps)

        # Remove extreme outliers for analysis
        q1 = np.percentile(gaps_array, 25)
        q3 = np.percentile(gaps_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # More lenient for gaps
        upper_bound = q3 + 3 * iqr

        typical_gaps = gaps_array[(gaps_array >= lower_bound) & (gaps_array <= upper_bound)]

        if len(typical_gaps) == 0:
            typical_gaps = gaps_array

        analysis = {
            'mean': np.mean(typical_gaps),
            'median': np.median(typical_gaps),
            'std': np.std(typical_gaps),
            'min': np.min(gaps_array),
            'max': np.max(gaps_array),
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'typical_range': (np.percentile(typical_gaps, 10), np.percentile(typical_gaps, 90)),
            'outlier_count': len(gaps_array) - len(typical_gaps),
            'total_gaps': len(gaps_array)
        }

        # Calculate frequency category
        median_gap = analysis['median']
        if median_gap <= 2:
            analysis['frequency'] = 'daily'
        elif median_gap <= 7:
            analysis['frequency'] = 'weekly'
        elif median_gap <= 14:
            analysis['frequency'] = 'biweekly'
        elif median_gap <= 30:
            analysis['frequency'] = 'monthly'
        else:
            analysis['frequency'] = 'sporadic'

        # Calculate confidence based on consistency
        cv = analysis['std'] / analysis['mean'] if analysis['mean'] > 0 else 1
        analysis['confidence'] = max(0.3, min(0.95, 1.0 - cv / 2))

        # Pattern description
        if cv < 0.3:
            analysis['pattern_description'] = 'highly_regular'
        elif cv < 0.6:
            analysis['pattern_description'] = 'moderately_regular'
        elif cv < 1.0:
            analysis['pattern_description'] = 'somewhat_irregular'
        else:
            analysis['pattern_description'] = 'highly_irregular'

        return analysis

    def _categorize_user(self, analysis: Dict[str, Any]) -> str:
        """Categorize user based on measurement patterns."""
        median_gap = analysis['median']

        if median_gap <= 3:
            return 'frequent'
        elif median_gap <= 14:
            return 'regular'
        elif median_gap <= 45:
            return 'sporadic'
        else:
            return 'inactive'

    def _calculate_threshold(self, analysis: Dict[str, Any], category: str) -> float:
        """
        Calculate adaptive threshold based on analysis.

        Formula considers:
        - Typical gap range (90th percentile)
        - Variability (standard deviation)
        - Category baseline
        - Confidence factor
        """
        # Start with category baseline
        baseline = self.category_defaults[category]

        # Calculate based on user's actual patterns
        # Use 95th percentile of typical gaps as base
        typical_gaps = analysis.get('typical_range', (baseline, baseline * 2))
        p95_gap = typical_gaps[1] * 1.5  # Some buffer above 90th percentile

        # Adjust for variability
        std_factor = 1.0 + (analysis['std'] / analysis['mean']) if analysis['mean'] > 0 else 1.5
        adjusted_threshold = p95_gap * std_factor

        # Apply confidence weighting
        confidence = analysis['confidence']
        weighted_threshold = (adjusted_threshold * confidence + baseline * (1 - confidence))

        # Apply bounds based on category
        if category == 'frequent':
            min_threshold, max_threshold = 7, 21
        elif category == 'regular':
            min_threshold, max_threshold = 14, 45
        elif category == 'sporadic':
            min_threshold, max_threshold = 30, 90
        else:  # inactive
            min_threshold, max_threshold = 60, 180

        return max(min_threshold, min(max_threshold, weighted_threshold))
```

### 2. Dynamic Threshold Updater

```python
class DynamicThresholdUpdater:
    """Updates thresholds as new patterns emerge."""

    def __init__(self, db, config=None):
        self.db = db
        self.config = config or {}
        self.analyzer = GapPatternAnalyzer(config)

        # Update frequency
        self.update_interval_days = config.get('update_interval_days', 30)
        self.min_new_measurements = config.get('min_new_measurements', 5)

    def should_update_threshold(self, user_id: str) -> bool:
        """Check if threshold should be updated."""
        last_update = self.db.get_threshold_last_update(user_id)

        if not last_update:
            return True

        days_since_update = (datetime.now() - last_update).days
        if days_since_update < self.update_interval_days:
            return False

        # Check if enough new measurements
        new_count = self.db.count_measurements_since(user_id, last_update)
        return new_count >= self.min_new_measurements

    def update_threshold(self, user_id: str) -> Dict[str, Any]:
        """Update threshold for a user."""
        # Get recent history
        history = self.db.get_measurement_history(user_id, days=180)

        # Calculate new threshold
        result = self.analyzer.calculate_adaptive_threshold(user_id, history)

        # Compare with current
        current = self.db.get_gap_threshold(user_id)
        change_percent = abs(result['threshold_days'] - current) / current if current else 0

        # Apply smoothing to avoid drastic changes
        if current and change_percent > 0.5:  # More than 50% change
            # Smooth the transition
            result['threshold_days'] = current * 0.7 + result['threshold_days'] * 0.3
            result['smoothed'] = True

        # Save to database
        self.db.update_gap_threshold(user_id, result)

        return result

    def batch_update_thresholds(self, user_ids: List[str] = None):
        """Update thresholds for multiple users."""
        if not user_ids:
            user_ids = self.db.get_users_needing_threshold_update()

        results = {}
        for user_id in user_ids:
            if self.should_update_threshold(user_id):
                results[user_id] = self.update_threshold(user_id)

        return results
```

### 3. Intelligent Reset Decision

```python
class AdaptiveResetManager:
    """Makes reset decisions using adaptive thresholds."""

    def __init__(self, db, config=None):
        self.db = db
        self.config = config or {}
        self.threshold_updater = DynamicThresholdUpdater(db, config)

    def should_trigger_reset(self, user_id: str, current_timestamp: datetime,
                            last_timestamp: datetime) -> Tuple[bool, str]:
        """
        Determine if reset should trigger based on adaptive threshold.

        Args:
            user_id: User identifier
            current_timestamp: Current measurement time
            last_timestamp: Last measurement time

        Returns:
            Tuple of (should_reset, reason)
        """
        # Calculate gap
        gap_days = (current_timestamp - last_timestamp).total_seconds() / 86400

        # Get adaptive threshold
        threshold_data = self.db.get_gap_threshold(user_id)

        if not threshold_data:
            # Calculate on the fly
            history = self.db.get_measurement_history(user_id)
            analyzer = GapPatternAnalyzer(self.config)
            threshold_data = analyzer.calculate_adaptive_threshold(user_id, history)

        threshold = threshold_data['threshold_days']

        # Check if gap exceeds threshold
        if gap_days >= threshold:
            # Additional checks for edge cases
            if threshold_data['category'] == 'frequent' and gap_days > 60:
                # Frequent measurer with huge gap - definitely reset
                return True, f'extreme_gap_for_frequent_user_{gap_days:.0f}d'

            if threshold_data['category'] == 'sporadic' and gap_days < threshold * 1.5:
                # Sporadic measurer - be more lenient
                if threshold_data['confidence'] < 0.7:
                    return False, 'sporadic_user_normal_variation'

            return True, f'gap_exceeds_adaptive_threshold_{gap_days:.0f}d>{threshold:.0f}d'

        return False, None

    def get_reset_parameters(self, user_id: str, gap_days: float) -> Dict[str, Any]:
        """Get reset parameters based on user pattern and gap size."""
        threshold_data = self.db.get_gap_threshold(user_id)
        category = threshold_data.get('category', 'new')

        # Adjust reset aggressiveness based on category and gap
        if category == 'frequent':
            # Frequent measurers need aggressive reset for gaps
            return {
                'type': 'hard',
                'variance_multiplier': 10,
                'adaptation_days': 14,
                'confidence_penalty': 0.7
            }
        elif category == 'sporadic':
            # Sporadic measurers need gentle reset
            return {
                'type': 'soft',
                'variance_multiplier': 3,
                'adaptation_days': 7,
                'confidence_penalty': 0.9
            }
        else:
            # Standard reset
            return {
                'type': 'hard' if gap_days > 60 else 'soft',
                'variance_multiplier': 5,
                'adaptation_days': 10,
                'confidence_penalty': 0.8
            }
```

## Implementation Steps

### Phase 1: Analysis Engine (Week 1)
1. Create `GapPatternAnalyzer` class
2. Implement gap calculation and distribution analysis
3. Create user categorization logic
4. Build threshold calculation algorithm

### Phase 2: Dynamic Updates (Week 2)
1. Implement `DynamicThresholdUpdater`
2. Add database schema for threshold storage
3. Create update scheduling logic
4. Implement smoothing for gradual changes

### Phase 3: Integration (Week 3)
1. Create `AdaptiveResetManager`
2. Integrate with existing reset logic
3. Update processor to use adaptive thresholds
4. Add fallback mechanisms

### Phase 4: Optimization (Week 4)
1. Performance tuning for batch updates
2. Add caching for frequently accessed thresholds
3. Create monitoring dashboards
4. Implement A/B testing framework

## Testing Strategy

### Unit Tests
```python
def test_daily_user_threshold():
    """Test threshold for daily measurement user."""
    analyzer = GapPatternAnalyzer()
    history = [
        {'timestamp': datetime(2024, 1, i), 'weight': 70}
        for i in range(1, 31)  # Daily for 30 days
    ]
    result = analyzer.calculate_adaptive_threshold('user1', history)
    assert result['category'] == 'frequent'
    assert 7 <= result['threshold_days'] <= 14

def test_sporadic_user_threshold():
    """Test threshold for sporadic user."""
    analyzer = GapPatternAnalyzer()
    history = [
        {'timestamp': datetime(2024, i, 15), 'weight': 70}
        for i in range(1, 7)  # Monthly for 6 months
    ]
    result = analyzer.calculate_adaptive_threshold('user2', history)
    assert result['category'] == 'sporadic'
    assert 45 <= result['threshold_days'] <= 90
```

## Configuration

```toml
[adaptive_gaps]
enabled = true
min_history_days = 90
min_measurements = 10
update_interval_days = 30

[adaptive_gaps.categories]
frequent_max_days = 21
regular_max_days = 45
sporadic_max_days = 90
inactive_max_days = 180

[adaptive_gaps.smoothing]
enabled = true
max_change_percent = 0.5
smoothing_factor = 0.3
```

## Success Metrics

1. **Reset accuracy**: >90% of resets occur at appropriate times
2. **False reset reduction**: <5% unnecessary resets
3. **User satisfaction**: Reduced data loss from inappropriate resets
4. **Adaptation speed**: Thresholds stabilize within 10 measurements

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Threshold oscillation | Medium | Smoothing, minimum update interval |
| Outlier gaps affecting threshold | High | Robust statistics, outlier removal |
| New user poor defaults | Medium | Conservative defaults, quick adaptation |
| Performance impact | Low | Caching, batch updates |

## Future Enhancements

1. Machine learning for pattern prediction
2. Seasonal adjustment (holidays, vacations)
3. Cross-user pattern learning
4. Integration with calendar/events
5. Predictive gap alerts