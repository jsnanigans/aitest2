# Plan: Source-Specific Validation Rules

## Problem Statement

Analysis of 15,701 users revealed that certain data sources produce systematic errors:
- **iglucose.com**: 33.5kg when actual weight ~120kg (74% error)
- **patient-upload**: Often has unit confusion or decimal errors
- **questionnaire**: May have typos or misunderstandings

Current system treats all sources equally except for noise multipliers, missing source-specific error patterns.

## Objectives

1. Implement source-specific validation rules
2. Detect common error patterns per source
3. Apply stricter validation for problematic sources
4. Track source reliability over time
5. Auto-adjust trust levels based on history

## Implementation Design

### 1. Source Validator Component

```python
# src/processing/source_validator.py

class SourceValidator:
    """Validates measurements based on source-specific patterns."""

    def __init__(self):
        self.source_rules = {
            'iglucose.com': {
                'min_weight': 40.0,
                'max_weight': 200.0,
                'max_change_rate': 0.10,  # 10% max change from last
                'common_errors': ['decimal_shift', 'unit_confusion'],
                'requires_confirmation': True,
                'confirmation_window_hours': 24,
                'historical_error_rate': 0.15
            },
            'patient-upload': {
                'min_weight': 35.0,
                'max_weight': 250.0,
                'max_change_rate': 0.15,
                'common_errors': ['typo', 'unit_confusion'],
                'requires_confirmation': False,
                'historical_error_rate': 0.08
            },
            'questionnaire': {
                'min_weight': 30.0,
                'max_weight': 300.0,
                'max_change_rate': 0.20,
                'common_errors': ['typo', 'misunderstanding'],
                'requires_confirmation': False,
                'historical_error_rate': 0.05
            },
            'patient-device': {
                'min_weight': 30.0,
                'max_weight': 300.0,
                'max_change_rate': 0.05,  # Most reliable
                'common_errors': ['calibration'],
                'requires_confirmation': False,
                'historical_error_rate': 0.02
            }
        }

    def validate(self, measurement, user_context):
        """
        Validate a measurement against source-specific rules.

        Returns:
            dict: Validation result with confidence and warnings
        """
        source = measurement['source']
        weight = measurement['weight']

        # Get rules for source
        rules = self._get_rules_for_source(source)

        # Check absolute bounds
        if weight < rules['min_weight'] or weight > rules['max_weight']:
            return {
                'valid': False,
                'reason': 'out_of_bounds',
                'confidence': 0.0,
                'suggestion': self._suggest_correction(weight, rules)
            }

        # Check rate of change
        if user_context.get('last_accepted_weight'):
            change_rate = abs(weight - user_context['last_accepted_weight']) / user_context['last_accepted_weight']
            if change_rate > rules['max_change_rate']:
                return {
                    'valid': False,
                    'reason': 'excessive_change',
                    'confidence': 1.0 - (change_rate / 0.5),  # Decay confidence
                    'requires_confirmation': rules['requires_confirmation']
                }

        # Check for common errors
        error_detected = self._check_common_errors(measurement, rules['common_errors'])
        if error_detected:
            return {
                'valid': False,
                'reason': f'likely_{error_detected}',
                'confidence': 0.3,
                'suggested_correction': self._correct_error(weight, error_detected)
            }

        return {
            'valid': True,
            'confidence': 1.0 - rules['historical_error_rate'],
            'source_reliability': 1.0 - rules['historical_error_rate']
        }
```

### 2. Error Pattern Detection

```python
class ErrorPatternDetector:
    """Detects common error patterns in measurements."""

    def detect_decimal_shift(self, weight, expected_range):
        """Detect if decimal point is misplaced."""
        # Check if weight * 10 or weight / 10 falls in expected range
        if weight * 10 >= expected_range[0] and weight * 10 <= expected_range[1]:
            return {'likely': True, 'correction': weight * 10}
        if weight / 10 >= expected_range[0] and weight / 10 <= expected_range[1]:
            return {'likely': True, 'correction': weight / 10}
        return {'likely': False}

    def detect_unit_confusion(self, weight, unit, expected_range):
        """Detect pounds vs kilograms confusion."""
        # Check if converting units brings weight into range
        if unit == 'kg':
            pounds_to_kg = weight / 2.20462
            if pounds_to_kg >= expected_range[0] and pounds_to_kg <= expected_range[1]:
                return {'likely': True, 'correction': pounds_to_kg, 'assumed_unit': 'lbs'}
        elif unit == 'lbs':
            kg_to_pounds = weight * 2.20462
            if kg_to_pounds >= expected_range[0] and kg_to_pounds <= expected_range[1]:
                return {'likely': True, 'correction': weight / 2.20462, 'assumed_unit': 'kg'}
        return {'likely': False}

    def detect_typo(self, weight, last_weight, keyboard_layout='qwerty'):
        """Detect likely keyboard typos."""
        # Check if digits are adjacent on keyboard
        weight_str = str(weight)
        last_str = str(last_weight)

        # Common typos: adjacent keys, repeated digits, transposition
        if self._is_likely_typo(weight_str, last_str):
            return {'likely': True, 'confidence': 0.7}
        return {'likely': False}
```

### 3. Source Reliability Tracking

```python
class SourceReliabilityTracker:
    """Tracks and updates source reliability over time."""

    def __init__(self, db):
        self.db = db
        self.reliability_scores = {}

    def update_reliability(self, source, measurement_outcome):
        """Update reliability score based on measurement outcome."""
        if source not in self.reliability_scores:
            self.reliability_scores[source] = {
                'total': 0,
                'accepted': 0,
                'rejected': 0,
                'corrected': 0,
                'score': 0.5  # Start neutral
            }

        stats = self.reliability_scores[source]
        stats['total'] += 1

        if measurement_outcome['accepted']:
            stats['accepted'] += 1
            # Increase reliability
            stats['score'] = min(1.0, stats['score'] * 1.02)
        elif measurement_outcome['corrected']:
            stats['corrected'] += 1
            # Slight decrease
            stats['score'] = max(0.1, stats['score'] * 0.98)
        else:
            stats['rejected'] += 1
            # Larger decrease
            stats['score'] = max(0.1, stats['score'] * 0.95)

        # Save to database
        self.db.save_source_reliability(source, stats)

    def get_reliability_score(self, source):
        """Get current reliability score for a source."""
        if source in self.reliability_scores:
            return self.reliability_scores[source]['score']

        # Check database
        db_score = self.db.get_source_reliability(source)
        if db_score:
            return db_score['score']

        # Default scores for known sources
        defaults = {
            'care-team-upload': 0.95,
            'patient-device': 0.85,
            'patient-upload': 0.75,
            'questionnaire': 0.70,
            'iglucose.com': 0.60
        }
        return defaults.get(source, 0.5)
```

### 4. Integration with Main Processing

```python
# In processor.py

def process_measurement(user_id, weight, timestamp, source, config, unit, db):
    """Enhanced processing with source validation."""

    # Get source validator
    validator = SourceValidator()

    # Get user context
    user_context = {
        'last_accepted_weight': db.get_last_accepted_weight(user_id),
        'expected_range': db.get_expected_range(user_id),
        'measurement_history': db.get_recent_measurements(user_id, days=30)
    }

    # Validate measurement
    validation_result = validator.validate(
        {'source': source, 'weight': weight, 'unit': unit},
        user_context
    )

    if not validation_result['valid']:
        # Check if we can auto-correct
        if validation_result.get('suggested_correction'):
            weight = validation_result['suggested_correction']
            metadata['auto_corrected'] = True
            metadata['correction_reason'] = validation_result['reason']
        elif validation_result['confidence'] < 0.3:
            # Reject outright
            return {
                'accepted': False,
                'reason': f"source_validation_failed: {validation_result['reason']}",
                'confidence': validation_result['confidence']
            }
        elif validation_result.get('requires_confirmation'):
            # Mark as provisional
            metadata['provisional'] = True
            metadata['requires_confirmation'] = True

    # Continue with normal processing...
```

## Implementation Steps

### Phase 1: Core Validation (Week 1)
1. Create `SourceValidator` class with basic rules
2. Implement bounds checking and rate validation
3. Add source-specific configuration
4. Write unit tests for validator

### Phase 2: Error Detection (Week 2)
1. Implement `ErrorPatternDetector` class
2. Add decimal shift detection
3. Add unit confusion detection
4. Add typo detection algorithms
5. Test with real problematic cases

### Phase 3: Reliability Tracking (Week 3)
1. Create `SourceReliabilityTracker` class
2. Add database schema for reliability scores
3. Implement score update logic
4. Add historical analysis tools

### Phase 4: Integration (Week 4)
1. Integrate validator into main processor
2. Add configuration options
3. Update quality scoring to use source reliability
4. Add monitoring and alerts

## Testing Strategy

### Unit Tests
```python
def test_iglucose_extreme_outlier():
    """Test that iglucose 33.5kg error is caught."""
    validator = SourceValidator()
    result = validator.validate(
        {'source': 'iglucose.com', 'weight': 33.5, 'unit': 'kg'},
        {'last_accepted_weight': 120.0}
    )
    assert not result['valid']
    assert result['reason'] == 'excessive_change'

def test_decimal_shift_correction():
    """Test decimal shift detection and correction."""
    detector = ErrorPatternDetector()
    result = detector.detect_decimal_shift(12.0, (100, 140))
    assert result['likely']
    assert result['correction'] == 120.0
```

### Integration Tests
- Process real problematic user data
- Verify iglucose errors are handled
- Test auto-correction functionality
- Validate provisional acceptance flow

## Success Metrics

1. **Error Detection Rate**: >90% of known errors caught
2. **False Positive Rate**: <5% valid measurements flagged
3. **Auto-Correction Success**: >70% of decimal/unit errors corrected
4. **Source Reliability Convergence**: Scores stabilize within 100 measurements

## Configuration

```toml
[source_validation]
enabled = true

[source_validation.iglucose]
min_weight = 40.0
max_weight = 200.0
max_change_rate = 0.10
requires_confirmation = true
error_patterns = ["decimal_shift", "unit_confusion"]

[source_validation.patient_device]
min_weight = 30.0
max_weight = 300.0
max_change_rate = 0.05
requires_confirmation = false
error_patterns = ["calibration"]
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-filtering valid data | High | Conservative thresholds, user-specific ranges |
| Source bias | Medium | Regular reliability recalibration |
| Auto-correction errors | High | Require confirmation for large corrections |
| Performance impact | Low | Cache validation rules, async processing |

## Future Enhancements

1. Machine learning for error pattern detection
2. Source-specific time-of-day patterns
3. Cross-source correlation analysis
4. Automatic rule adjustment based on outcomes
5. User feedback integration for corrections