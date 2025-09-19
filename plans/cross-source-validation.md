# Plan: Cross-Source Validation

## Problem Statement

Real data analysis revealed conflicting measurements from different sources:
- **iglucose.com**: Produced 33.5kg reading followed by 118kg within 49 seconds
- Multiple sources reporting vastly different weights for same user
- No mechanism to validate measurements across sources
- Single-source extreme values accepted without corroboration

Current system trusts sources independently, missing opportunities to detect errors through cross-validation.

## Objectives

1. Validate extreme changes against multiple sources
2. Weight sources by historical reliability
3. Require corroboration for suspicious values
4. Detect and flag single-source anomalies
5. Learn source agreement patterns over time

## Implementation Design

### 1. Multi-Source Validator

```python
# src/processing/multi_source_validator.py

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class MultiSourceValidator:
    """Validates measurements by comparing across multiple sources."""

    def __init__(self, db, config=None):
        self.db = db
        self.config = config or {}

        # Configuration
        self.time_window = timedelta(hours=config.get('correlation_window_hours', 24))
        self.min_sources = config.get('min_sources_for_validation', 2)
        self.agreement_threshold = config.get('agreement_threshold', 0.10)  # 10% difference
        self.source_weights = config.get('source_weights', {})

        # Default source reliability weights
        if not self.source_weights:
            self.source_weights = {
                'care-team-upload': 0.95,
                'patient-device': 0.85,
                'patient-upload': 0.75,
                'questionnaire': 0.70,
                'connectivehealth.io': 0.65,
                'iglucose.com': 0.50
            }

    def validate_measurement(self, user_id: str, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a measurement against other sources.

        Args:
            user_id: User identifier
            measurement: Current measurement to validate

        Returns:
            Validation result with confidence and recommendations
        """
        # Get recent measurements from other sources
        recent_measurements = self._get_recent_measurements(
            user_id,
            measurement['timestamp'],
            exclude_source=measurement['source']
        )

        if len(recent_measurements) == 0:
            # No other sources to validate against
            return {
                'validated': False,
                'reason': 'no_other_sources',
                'confidence': self.source_weights.get(measurement['source'], 0.5),
                'requires_corroboration': True
            }

        # Calculate agreement with other sources
        agreement_score = self._calculate_agreement(measurement, recent_measurements)

        # Check if multiple sources agree
        if len(recent_measurements) >= self.min_sources - 1:
            if agreement_score >= 0.7:
                return {
                    'validated': True,
                    'confidence': min(0.95, agreement_score),
                    'corroborating_sources': len(recent_measurements),
                    'agreement_score': agreement_score
                }
            else:
                # Multiple sources disagree
                return {
                    'validated': False,
                    'reason': 'source_disagreement',
                    'confidence': agreement_score,
                    'conflicting_values': self._get_conflicting_values(measurement, recent_measurements),
                    'suggested_value': self._calculate_consensus_value(recent_measurements)
                }

        # Single other source - need stronger agreement
        if agreement_score >= 0.8:
            return {
                'validated': True,
                'confidence': agreement_score * 0.9,  # Slightly lower confidence
                'corroborating_sources': 1,
                'warning': 'single_source_corroboration'
            }

        return {
            'validated': False,
            'reason': 'insufficient_agreement',
            'confidence': agreement_score,
            'requires_additional_sources': True
        }

    def _get_recent_measurements(self, user_id: str, timestamp: datetime,
                                exclude_source: str = None) -> List[Dict[str, Any]]:
        """Get recent measurements from other sources."""
        start_time = timestamp - self.time_window
        end_time = timestamp + self.time_window

        measurements = self.db.get_measurements_in_range(
            user_id, start_time, end_time
        )

        # Filter out same source and group by source
        filtered = []
        sources_seen = set()

        for m in measurements:
            if m['source'] != exclude_source and m['source'] not in sources_seen:
                filtered.append(m)
                sources_seen.add(m['source'])

        return filtered

    def _calculate_agreement(self, measurement: Dict[str, Any],
                            other_measurements: List[Dict[str, Any]]) -> float:
        """Calculate agreement score between measurement and others."""
        if not other_measurements:
            return 0.0

        weight = measurement['weight']
        agreements = []

        for other in other_measurements:
            # Calculate relative difference
            diff = abs(weight - other['weight']) / max(weight, other['weight'])

            # Convert to agreement score
            if diff <= self.agreement_threshold:
                agreement = 1.0 - (diff / self.agreement_threshold)
            else:
                agreement = max(0, 1.0 - diff)

            # Weight by source reliability and time proximity
            source_weight = self.source_weights.get(other['source'], 0.5)
            time_diff = abs((measurement['timestamp'] - other['timestamp']).total_seconds())
            time_weight = max(0.3, 1.0 - time_diff / (self.time_window.total_seconds() * 2))

            weighted_agreement = agreement * source_weight * time_weight
            agreements.append(weighted_agreement)

        return sum(agreements) / len(agreements)

    def _calculate_consensus_value(self, measurements: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate weighted consensus value from multiple sources."""
        if not measurements:
            return None

        weights = []
        values = []

        for m in measurements:
            weight = self.source_weights.get(m['source'], 0.5)
            weights.append(weight)
            values.append(m['weight'])

        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(v * w for v, w in zip(values, weights)) / total_weight

        return np.median(values)
```

### 2. Source Agreement Learner

```python
class SourceAgreementLearner:
    """Learns patterns of agreement between sources over time."""

    def __init__(self, db):
        self.db = db
        self.agreement_matrix = {}  # source_pair -> agreement_stats

    def update_agreement(self, source1: str, source2: str,
                        value1: float, value2: float, timestamp: datetime):
        """Update agreement statistics between two sources."""
        pair = tuple(sorted([source1, source2]))

        if pair not in self.agreement_matrix:
            self.agreement_matrix[pair] = {
                'total_comparisons': 0,
                'agreements': 0,
                'total_diff': 0,
                'max_diff': 0,
                'typical_bias': []
            }

        stats = self.agreement_matrix[pair]
        stats['total_comparisons'] += 1

        diff = abs(value1 - value2) / max(value1, value2)
        stats['total_diff'] += diff
        stats['max_diff'] = max(stats['max_diff'], diff)

        if diff <= 0.10:  # Within 10%
            stats['agreements'] += 1

        # Track directional bias
        if source1 < source2:  # Consistent ordering
            bias = value1 - value2
        else:
            bias = value2 - value1

        stats['typical_bias'].append(bias)
        if len(stats['typical_bias']) > 100:
            stats['typical_bias'] = stats['typical_bias'][-100:]  # Keep last 100

        # Save to database periodically
        if stats['total_comparisons'] % 10 == 0:
            self.db.save_source_agreement(pair, stats)

    def get_expected_difference(self, source1: str, source2: str) -> Dict[str, float]:
        """Get expected difference between two sources."""
        pair = tuple(sorted([source1, source2]))

        if pair not in self.agreement_matrix:
            # Check database
            stats = self.db.get_source_agreement(pair)
            if stats:
                self.agreement_matrix[pair] = stats
            else:
                return {'expected_diff': 0.10, 'confidence': 0.3}

        stats = self.agreement_matrix[pair]

        if stats['total_comparisons'] < 10:
            return {'expected_diff': 0.10, 'confidence': 0.3}

        avg_diff = stats['total_diff'] / stats['total_comparisons']
        agreement_rate = stats['agreements'] / stats['total_comparisons']

        # Calculate typical bias
        if stats['typical_bias']:
            median_bias = np.median(stats['typical_bias'])
        else:
            median_bias = 0

        return {
            'expected_diff': avg_diff,
            'agreement_rate': agreement_rate,
            'median_bias': median_bias,
            'confidence': min(0.95, agreement_rate)
        }
```

### 3. Anomaly Detection Across Sources

```python
class CrossSourceAnomalyDetector:
    """Detects anomalies by comparing patterns across sources."""

    def __init__(self, config=None):
        self.config = config or {}
        self.anomaly_threshold = config.get('anomaly_threshold', 3.0)  # Standard deviations

    def detect_source_anomaly(self, user_id: str, measurement: Dict[str, Any],
                              source_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Detect if a measurement is anomalous compared to other sources.

        Args:
            user_id: User identifier
            measurement: Current measurement
            source_history: Recent values from each source

        Returns:
            Anomaly detection result
        """
        current_source = measurement['source']
        current_weight = measurement['weight']

        # Calculate statistics from other sources
        other_values = []
        for source, values in source_history.items():
            if source != current_source and values:
                other_values.extend(values[-5:])  # Last 5 from each source

        if len(other_values) < 3:
            return {'is_anomaly': False, 'reason': 'insufficient_data'}

        # Calculate robust statistics
        median = np.median(other_values)
        mad = np.median(np.abs(other_values - median))

        if mad == 0:
            # All values are identical
            if abs(current_weight - median) > median * 0.01:
                return {
                    'is_anomaly': True,
                    'reason': 'deviation_from_consensus',
                    'expected': median,
                    'deviation': abs(current_weight - median)
                }
            return {'is_anomaly': False}

        # Modified Z-score
        z_score = 0.6745 * (current_weight - median) / mad

        if abs(z_score) > self.anomaly_threshold:
            return {
                'is_anomaly': True,
                'reason': 'statistical_outlier',
                'z_score': z_score,
                'expected_range': (
                    median - self.anomaly_threshold * mad / 0.6745,
                    median + self.anomaly_threshold * mad / 0.6745
                )
            }

        # Check for source-specific patterns
        if current_source in source_history and len(source_history[current_source]) > 3:
            source_median = np.median(source_history[current_source])
            source_mad = np.median(np.abs(source_history[current_source] - source_median))

            if source_mad > 0:
                source_z = 0.6745 * (current_weight - source_median) / source_mad

                if abs(source_z) > self.anomaly_threshold * 1.5:  # More lenient for same source
                    return {
                        'is_anomaly': True,
                        'reason': 'source_pattern_deviation',
                        'source_z_score': source_z,
                        'typical_range': (
                            source_median - self.anomaly_threshold * 1.5 * source_mad / 0.6745,
                            source_median + self.anomaly_threshold * 1.5 * source_mad / 0.6745
                        )
                    }

        return {'is_anomaly': False}
```

### 4. Integration with Main Processing

```python
# Enhanced processor.py integration

def process_measurement_with_cross_validation(user_id, weight, timestamp, source, config, unit, db):
    """Process measurement with cross-source validation."""

    # Initialize validators
    multi_validator = MultiSourceValidator(db, config)
    agreement_learner = SourceAgreementLearner(db)
    anomaly_detector = CrossSourceAnomalyDetector(config)

    measurement = {
        'weight': weight,
        'timestamp': timestamp,
        'source': source,
        'unit': unit
    }

    # Step 1: Cross-source validation
    validation_result = multi_validator.validate_measurement(user_id, measurement)

    if not validation_result['validated']:
        if validation_result.get('requires_corroboration'):
            # Mark as provisional
            return {
                'accepted': True,
                'provisional': True,
                'reason': 'awaiting_corroboration',
                'confidence': validation_result['confidence']
            }

        if validation_result.get('suggested_value'):
            # Consider auto-correction
            suggested = validation_result['suggested_value']
            if abs(weight - suggested) / weight < 0.20:  # Within 20%
                weight = suggested
                metadata['auto_corrected'] = True
                metadata['correction_source'] = 'multi_source_consensus'

    # Step 2: Check for cross-source anomalies
    source_history = db.get_source_history(user_id, days=30)
    anomaly_result = anomaly_detector.detect_source_anomaly(
        user_id, measurement, source_history
    )

    if anomaly_result['is_anomaly']:
        quality_score *= 0.7  # Reduce quality score
        metadata['anomaly_detected'] = anomaly_result['reason']

    # Step 3: Update source agreement patterns
    recent = db.get_recent_measurements(user_id, hours=1)
    for other in recent:
        if other['source'] != source:
            agreement_learner.update_agreement(
                source, other['source'],
                weight, other['weight'],
                timestamp
            )

    # Continue with normal processing...
```

## Implementation Steps

### Phase 1: Core Framework (Week 1)
1. Create `MultiSourceValidator` class
2. Implement time-window based correlation
3. Add agreement scoring algorithm
4. Build consensus value calculation

### Phase 2: Learning System (Week 2)
1. Implement `SourceAgreementLearner`
2. Create agreement matrix tracking
3. Add bias detection between sources
4. Build database persistence layer

### Phase 3: Anomaly Detection (Week 3)
1. Create `CrossSourceAnomalyDetector`
2. Implement MAD-based detection
3. Add source-specific pattern analysis
4. Build anomaly flagging system

### Phase 4: Integration (Week 4)
1. Integrate with main processor
2. Add provisional acceptance for uncorroborated values
3. Implement auto-correction with consensus
4. Create monitoring dashboards

## Testing Strategy

### Unit Tests
```python
def test_multi_source_agreement():
    """Test agreement between multiple sources."""
    validator = MultiSourceValidator(mock_db)

    # Setup: Multiple sources agree
    measurement = {'weight': 120.0, 'source': 'patient-device', 'timestamp': now}
    recent = [
        {'weight': 119.5, 'source': 'care-team-upload'},
        {'weight': 120.8, 'source': 'questionnaire'}
    ]

    result = validator._calculate_agreement(measurement, recent)
    assert result > 0.8  # High agreement

def test_source_conflict_detection():
    """Test detection of conflicting sources."""
    validator = MultiSourceValidator(mock_db)

    # Setup: Sources disagree significantly
    measurement = {'weight': 120.0, 'source': 'iglucose.com', 'timestamp': now}
    recent = [
        {'weight': 33.5, 'source': 'iglucose.com'},  # Clear error
        {'weight': 118.0, 'source': 'patient-device'}
    ]

    result = validator.validate_measurement('user1', measurement)
    assert not result['validated']
    assert result['reason'] == 'source_disagreement'
```

## Configuration

```toml
[cross_source_validation]
enabled = true
correlation_window_hours = 24
min_sources_for_validation = 2
agreement_threshold = 0.10

[cross_source_validation.source_weights]
care-team-upload = 0.95
patient-device = 0.85
patient-upload = 0.75
questionnaire = 0.70
connectivehealth.io = 0.65
iglucose.com = 0.50

[cross_source_validation.anomaly_detection]
threshold_std_deviations = 3.0
min_historical_values = 5
```

## Success Metrics

1. **False positive reduction**: <3% valid measurements flagged incorrectly
2. **Error detection rate**: >95% of known source errors caught
3. **Consensus accuracy**: Consensus values within 5% of true weight
4. **Source agreement learning**: Patterns stabilize within 50 comparisons

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Delayed processing waiting for corroboration | High | Time limits, provisional acceptance |
| Source collusion (multiple bad sources) | Medium | Weighted voting, historical reliability |
| Sparse data (few sources available) | High | Graceful degradation, single-source fallback |
| Performance with many sources | Low | Caching, async validation |

## Future Enhancements

1. Machine learning for source reliability prediction
2. Temporal patterns in source agreement
3. User notification for source conflicts
4. Automatic source quality reporting
5. Integration with device calibration data