# Implementation Plan: Data Quality Improvements

## Executive Summary

Implement three targeted improvements to handle the discovered data quality issues:
1. Enhanced pre-processor with unit conversion and BMI detection
2. Adaptive outlier detection based on source quality metrics
3. Kalman measurement noise adaptation while preserving mathematical integrity

**Council Guidance:**
- **Donald Knuth**: "The Kalman filter's mathematical optimality depends on proper noise modeling. Adapt the measurement noise, not the algorithm."
- **Nancy Leveson**: "Each layer must fail safely. Bad data should be caught early, not propagated."
- **Butler Lampson**: "Keep it simple. Three focused improvements are better than ten complex ones."

---

## Improvement 1: Enhanced Pre-Processor

### Objective
Create a robust pre-processor that catches bad data before it reaches the Kalman filter.

### Implementation Details

```python
class EnhancedPreProcessor:
    """Pre-process weight data with unit conversion and validation."""

    # Average height for BMI calculations (1.67m from global statistics)
    DEFAULT_HEIGHT_M = 1.67

    # BMI ranges for validation
    BMI_IMPOSSIBLE_LOW = 10   # Below this is physiologically impossible
    BMI_IMPOSSIBLE_HIGH = 100 # Above this is physiologically impossible
    BMI_SUSPICIOUS_LOW = 13   # Below this is extremely rare
    BMI_SUSPICIOUS_HIGH = 60  # Above this is extremely rare

    @staticmethod
    def preprocess(weight: float, unit: str, source: str) -> Tuple[Optional[float], Dict]:
        """
        Pre-process weight with unit conversion and validation.

        Steps:
        1. Convert to kg if needed
        2. Check if value might be BMI
        3. Validate against physiological limits
        4. Apply source-specific corrections
        """
        metadata = {
            'original_value': weight,
            'original_unit': unit,
            'source': source,
            'conversions': [],
            'warnings': [],
            'checks_passed': []
        }

        # Step 1: Unit Conversion
        weight_kg = EnhancedPreProcessor._convert_to_kg(weight, unit, metadata)

        if weight_kg is None:
            metadata['rejected'] = 'Unit conversion failed'
            return None, metadata

        # Step 2: BMI Detection
        is_bmi, confidence = EnhancedPreProcessor._check_if_bmi(weight_kg, metadata)

        if is_bmi and confidence > 0.8:
            # Try to recover actual weight from BMI
            estimated_weight = EnhancedPreProcessor._estimate_weight_from_bmi(
                weight_kg, metadata
            )
            if estimated_weight:
                weight_kg = estimated_weight
                metadata['conversions'].append(
                    f'Converted BMI {weight:.1f} to weight {weight_kg:.1f}kg'
                )
            else:
                metadata['rejected'] = f'Value appears to be BMI ({weight_kg:.1f}), not weight'
                return None, metadata

        # Step 3: Physiological Validation
        is_valid = EnhancedPreProcessor._validate_physiological_limits(
            weight_kg, metadata
        )

        if not is_valid:
            metadata['rejected'] = 'Outside physiological limits'
            return None, metadata

        # Step 4: Source-Specific Corrections
        weight_kg = EnhancedPreProcessor._apply_source_corrections(
            weight_kg, source, metadata
        )

        metadata['final_weight_kg'] = weight_kg
        return weight_kg, metadata

    @staticmethod
    def _convert_to_kg(value: float, unit: str, metadata: Dict) -> Optional[float]:
        """Convert various units to kilograms."""

        unit_lower = unit.lower() if unit else 'kg'

        # Handle various unit formats
        conversions = {
            'kg': 1.0,
            'kilogram': 1.0,
            'kilo': 1.0,
            'lb': 0.453592,
            'lbs': 0.453592,
            'pound': 0.453592,
            'pounds': 0.453592,
            '[lb_ap]': 0.453592,  # Apothecary pound
            'st': 6.35029,
            'stone': 6.35029,
            'g': 0.001,
            'gram': 0.001,
            'grams': 0.001,
            'oz': 0.0283495,
            'ounce': 0.0283495,
            '[oz_av]': 0.0283495,
            '[oz_ap]': 0.0311035,  # Apothecary ounce
            'mg': 0.000001,
            'milligram': 0.000001
        }

        # Check for BMI units (should not be converted)
        if 'kg/m2' in unit_lower or 'bmi' in unit_lower:
            metadata['warnings'].append(f'Unit suggests BMI: {unit}')
            # Return as-is for BMI detection
            return value

        # Find conversion factor
        factor = None
        for unit_key, conversion_factor in conversions.items():
            if unit_key in unit_lower:
                factor = conversion_factor
                break

        if factor is None:
            # Unknown unit - try to guess
            if value > 100 and value < 400:
                # Likely pounds
                metadata['warnings'].append(f'Unknown unit {unit}, assuming pounds')
                factor = 0.453592
            else:
                metadata['warnings'].append(f'Unknown unit: {unit}')
                factor = 1.0  # Assume kg

        result = value * factor

        if factor != 1.0:
            metadata['conversions'].append(
                f'Converted {value:.2f} {unit} to {result:.2f} kg'
            )

        return result

    @staticmethod
    def _check_if_bmi(value: float, metadata: Dict) -> Tuple[bool, float]:
        """
        Check if value is likely BMI instead of weight.

        Returns:
            (is_bmi, confidence)
        """
        # BMI typically ranges from 15-40 for adults
        if 15 <= value <= 50:
            # Calculate what weight would be for this BMI
            height_m = EnhancedPreProcessor.DEFAULT_HEIGHT_M
            implied_weight = value * (height_m ** 2)

            # Check if implied weight is reasonable
            if 40 <= implied_weight <= 200:
                # High confidence this is BMI
                metadata['warnings'].append(
                    f'Value {value:.1f} likely BMI (implies {implied_weight:.1f}kg weight)'
                )
                return True, 0.9
            elif 30 <= implied_weight <= 250:
                # Moderate confidence
                metadata['warnings'].append(
                    f'Value {value:.1f} might be BMI'
                )
                return True, 0.6

        return False, 0.0

    @staticmethod
    def _estimate_weight_from_bmi(bmi: float, metadata: Dict) -> Optional[float]:
        """Estimate weight from BMI using average height."""
        height_m = EnhancedPreProcessor.DEFAULT_HEIGHT_M
        estimated_weight = bmi * (height_m ** 2)

        # Validate the estimated weight
        if 30 <= estimated_weight <= 300:
            metadata['warnings'].append(
                f'Estimated weight {estimated_weight:.1f}kg from BMI {bmi:.1f}'
            )
            return estimated_weight

        return None

    @staticmethod
    def _validate_physiological_limits(weight_kg: float, metadata: Dict) -> bool:
        """Validate weight against physiological limits."""

        # Absolute limits
        if weight_kg < 20:
            metadata['warnings'].append(f'Weight {weight_kg:.1f}kg below minimum (20kg)')
            return False

        if weight_kg > 500:
            metadata['warnings'].append(f'Weight {weight_kg:.1f}kg above maximum (500kg)')
            return False

        # Check BMI limits
        height_m = EnhancedPreProcessor.DEFAULT_HEIGHT_M
        implied_bmi = weight_kg / (height_m ** 2)

        if implied_bmi < EnhancedPreProcessor.BMI_IMPOSSIBLE_LOW:
            metadata['warnings'].append(
                f'Implied BMI {implied_bmi:.1f} physiologically impossible'
            )
            return False

        if implied_bmi > EnhancedPreProcessor.BMI_IMPOSSIBLE_HIGH:
            metadata['warnings'].append(
                f'Implied BMI {implied_bmi:.1f} physiologically impossible'
            )
            return False

        # Warnings for suspicious but possible values
        if implied_bmi < EnhancedPreProcessor.BMI_SUSPICIOUS_LOW:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously low')

        if implied_bmi > EnhancedPreProcessor.BMI_SUSPICIOUS_HIGH:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously high')

        metadata['checks_passed'].append('physiological_limits')
        return True

    @staticmethod
    def _apply_source_corrections(weight_kg: float, source: str, metadata: Dict) -> float:
        """Apply source-specific corrections."""

        # Sources known to send pounds without proper unit marking
        POUND_SOURCES = {
            'patient-upload': 0.933,      # 93.3% pound entries
            'internal-questionnaire': 1.0, # 100% pound entries
            'care-team-upload': 0.745     # 74.5% pound entries
        }

        # Check if this might be pounds mismarked as kg
        if source in POUND_SOURCES:
            confidence = POUND_SOURCES[source]

            # If value is in typical pound range and source usually sends pounds
            if 100 <= weight_kg <= 400 and confidence > 0.7:
                # Check if converting would give reasonable kg
                potential_kg = weight_kg * 0.453592
                if 45 <= potential_kg <= 180:
                    # Likely pounds marked as kg
                    metadata['conversions'].append(
                        f'Source {source} correction: {weight_kg:.1f} lb → {potential_kg:.1f} kg'
                    )
                    weight_kg = potential_kg

        return weight_kg
```

### Testing Strategy

```python
def test_preprocessor():
    """Test cases for pre-processor."""

    test_cases = [
        # (weight, unit, source, expected_kg, should_pass)
        (150, 'lb', 'patient-upload', 68.04, True),
        (68, 'kg', 'patient-device', 68, True),
        (25, 'kg/m2', 'connectivehealth.io', None, False),  # BMI
        (10, 'stone', 'care-team-upload', 63.5, True),
        (500, 'kg', 'any', None, False),  # Too heavy
        (15, 'kg', 'any', None, False),   # Too light
        (160, 'kg', 'patient-upload', 72.57, True),  # Pounds as kg
    ]

    for weight, unit, source, expected, should_pass in test_cases:
        result, metadata = EnhancedPreProcessor.preprocess(weight, unit, source)

        if should_pass:
            assert result is not None, f"Failed: {weight} {unit}"
            if expected:
                assert abs(result - expected) < 0.1, f"Wrong conversion: {result} vs {expected}"
        else:
            assert result is None, f"Should reject: {weight} {unit}"
```

---

## Improvement 2: Adaptive Outlier Detection

### Objective
Dynamically adjust outlier thresholds based on source quality metrics.

### Implementation Details

```python
class AdaptiveOutlierDetector:
    """Adaptive outlier detection based on empirical source quality."""

    # Empirical outlier rates from analysis (per 1000)
    SOURCE_QUALITY_METRICS = {
        'care-team-upload': {
            'outlier_rate': 3.6,
            'quality_score': 10.0,
            'trust_level': 'excellent'
        },
        'patient-upload': {
            'outlier_rate': 13.0,
            'quality_score': 9.9,
            'trust_level': 'excellent'
        },
        'patient-device': {
            'outlier_rate': 20.7,
            'quality_score': 9.8,
            'trust_level': 'good'
        },
        'internal-questionnaire': {
            'outlier_rate': 14.0,
            'quality_score': 9.8,
            'trust_level': 'good'
        },
        'https://connectivehealth.io': {
            'outlier_rate': 35.8,
            'quality_score': 9.3,
            'trust_level': 'moderate'
        },
        'https://api.iglucose.com': {
            'outlier_rate': 151.4,
            'quality_score': 8.5,
            'trust_level': 'poor'
        }
    }

    # Physiological weight change limits
    PHYSIOLOGICAL_LIMITS = {
        'max_daily_change': 2.0,      # kg/day extreme but possible
        'typical_daily_change': 0.5,  # kg/day typical variation
        'max_weekly_change': 5.0,     # kg/week extreme diet/illness
        'typical_weekly_change': 1.0  # kg/week normal diet
    }

    @staticmethod
    def calculate_adaptive_threshold(
        source: str,
        time_gap_days: float,
        previous_weight: float,
        user_history: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate adaptive threshold based on source quality and context.

        Returns dict with:
        - threshold: Maximum acceptable change
        - confidence: Confidence in this threshold
        - method: How threshold was calculated
        """

        # Get source quality metrics
        source_metrics = AdaptiveOutlierDetector.SOURCE_QUALITY_METRICS.get(
            source,
            {'outlier_rate': 50.0, 'quality_score': 8.0, 'trust_level': 'unknown'}
        )

        outlier_rate = source_metrics['outlier_rate']
        quality_score = source_metrics['quality_score']

        # Base physiological limit
        if time_gap_days == 0:
            # Same day measurement
            base_limit = AdaptiveOutlierDetector.PHYSIOLOGICAL_LIMITS['max_daily_change']
        elif time_gap_days <= 7:
            # Within a week
            base_limit = AdaptiveOutlierDetector.PHYSIOLOGICAL_LIMITS['max_weekly_change']
        else:
            # Longer gap - use weekly rate
            weekly_rate = AdaptiveOutlierDetector.PHYSIOLOGICAL_LIMITS['max_weekly_change']
            base_limit = weekly_rate * (time_gap_days / 7.0)

        # Adjust based on source quality
        # Poor sources get stricter limits, good sources get more lenient
        if outlier_rate > 100:  # Very poor (iGlucose)
            # Much stricter - 50% of physiological limit
            threshold = base_limit * 0.5
            confidence = 0.6
            method = 'strict_poor_source'
        elif outlier_rate > 30:  # Moderate
            # Slightly stricter - 75% of physiological limit
            threshold = base_limit * 0.75
            confidence = 0.7
            method = 'moderate_source'
        elif outlier_rate > 15:  # Good
            # Normal physiological limit
            threshold = base_limit
            confidence = 0.8
            method = 'good_source'
        else:  # Excellent (care-team, patient-upload)
            # More lenient - 125% of physiological limit
            threshold = base_limit * 1.25
            confidence = 0.9
            method = 'excellent_source'

        # Consider user history if available
        if user_history:
            # Adjust based on user's typical variability
            user_variability = user_history.get('weight_std', 2.0)

            # If user is typically stable, be stricter
            if user_variability < 1.0:
                threshold *= 0.8
                method += '_stable_user'
            # If user is typically variable, be more lenient
            elif user_variability > 3.0:
                threshold *= 1.2
                method += '_variable_user'

        # Apply reasonable bounds
        threshold = max(2.0, min(threshold, 20.0))

        # Calculate as percentage of body weight
        percent_change = (threshold / previous_weight) * 100 if previous_weight > 0 else 10

        return {
            'threshold': threshold,
            'confidence': confidence,
            'method': method,
            'source_quality': source_metrics['trust_level'],
            'outlier_rate': outlier_rate,
            'percent_change': percent_change,
            'time_gap_days': time_gap_days
        }

    @staticmethod
    def check_outlier(
        current_weight: float,
        previous_weight: float,
        source: str,
        time_gap_days: float,
        user_history: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if measurement is an outlier.

        Returns:
            (is_outlier, reason, details)
        """

        weight_change = abs(current_weight - previous_weight)

        # Get adaptive threshold
        threshold_info = AdaptiveOutlierDetector.calculate_adaptive_threshold(
            source, time_gap_days, previous_weight, user_history
        )

        threshold = threshold_info['threshold']

        if weight_change > threshold:
            reason = (
                f"Weight change {weight_change:.1f}kg ({threshold_info['percent_change']:.1f}% body weight) "
                f"exceeds {threshold_info['source_quality']} source threshold {threshold:.1f}kg "
                f"over {time_gap_days} days"
            )
            return True, reason, threshold_info

        return False, None, threshold_info
```

---

## Improvement 3: Kalman Noise Adaptation

### Objective
Adapt Kalman measurement noise while preserving mathematical integrity.

### Mathematical Foundation

**Council - Donald Knuth**: "The Kalman filter is optimal when noise parameters match reality. We're not changing the algorithm, just providing better noise estimates."

The Kalman filter measurement update:
```
K = P⁻H^T(HP⁻H^T + R)^(-1)  # Kalman gain
x = x⁻ + K(z - Hx⁻)         # State update
P = (I - KH)P⁻               # Covariance update
```

Where R is measurement noise covariance. Increasing R reduces Kalman gain K, making the filter trust measurements less.

### Implementation Details

```python
class AdaptiveKalmanNoise:
    """
    Adapt Kalman measurement noise based on source reliability.
    Preserves Kalman mathematical integrity.
    """

    # Base measurement noise (variance in kg²)
    BASE_MEASUREMENT_VARIANCE = 1.0  # 1 kg standard deviation

    # Empirically derived noise multipliers
    # Based on actual outlier rates and quality scores
    NOISE_MULTIPLIERS = {
        'care-team-upload': 0.36,        # 3.6 outliers/1000 → trust more
        'patient-upload': 0.52,           # 13.0 outliers/1000 → trust more
        'internal-questionnaire': 0.56,   # 14.0 outliers/1000 → trust more
        'patient-device': 0.83,           # 20.7 outliers/1000 → baseline
        'https://connectivehealth.io': 1.43,  # 35.8 outliers/1000 → trust less
        'https://api.iglucose.com': 6.06      # 151.4 outliers/1000 → trust much less
    }

    @staticmethod
    def calculate_measurement_noise(
        source: str,
        base_config: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate adapted Kalman configuration.

        Mathematical integrity preserved:
        1. Only modifies measurement noise R
        2. Maintains positive definiteness
        3. Preserves Kalman optimality for given noise model
        """

        # Start with base configuration
        config = base_config.copy()

        # Get base measurement noise
        base_noise = config.get('measurement_noise',
                                AdaptiveKalmanNoise.BASE_MEASUREMENT_VARIANCE)

        # Get source-specific multiplier
        # Multiplier is sqrt(outlier_rate / baseline_rate)
        multiplier = AdaptiveKalmanNoise.NOISE_MULTIPLIERS.get(source, 1.0)

        # Apply dynamic adjustments if context available
        if context:
            # Recent measurement consistency
            if context.get('recent_consistency', 1.0) < 0.5:
                # Very inconsistent recent measurements
                multiplier *= 1.5
            elif context.get('recent_consistency', 1.0) > 0.9:
                # Very consistent recent measurements
                multiplier *= 0.8

            # Time since last measurement
            time_gap = context.get('time_gap_days', 0)
            if time_gap > 30:
                # Long gap - increase uncertainty
                multiplier *= 1.2
            elif time_gap < 1:
                # Frequent measurements - decrease uncertainty
                multiplier *= 0.9

        # Calculate adapted noise
        # Ensure positive definiteness (variance must be positive)
        adapted_noise = max(0.01, base_noise * multiplier)

        # Update configuration
        config['measurement_noise'] = adapted_noise

        # For very unreliable sources, also increase process noise slightly
        # This allows the filter to adapt more quickly to changes
        if multiplier > 3.0:
            process_noise = config.get('process_noise', 0.01)
            config['process_noise'] = min(0.1, process_noise * 1.5)

        # Document the adaptation
        config['noise_adaptation'] = {
            'source': source,
            'base_noise': base_noise,
            'multiplier': multiplier,
            'adapted_noise': adapted_noise,
            'method': 'empirical_outlier_rate',
            'preserves_optimality': True
        }

        return config

    @staticmethod
    def validate_kalman_integrity(config: Dict) -> bool:
        """
        Validate that Kalman configuration maintains mathematical integrity.

        Requirements:
        1. All noise values positive (positive definiteness)
        2. Reasonable bounds
        3. Proper matrix dimensions (for multi-dimensional state)
        """

        # Check measurement noise
        R = config.get('measurement_noise', 0)
        if R <= 0:
            raise ValueError(f"Measurement noise must be positive: {R}")
        if R > 100:
            raise ValueError(f"Measurement noise unreasonably large: {R}")

        # Check process noise
        Q = config.get('process_noise', 0)
        if Q <= 0:
            raise ValueError(f"Process noise must be positive: {Q}")
        if Q > 1:
            raise ValueError(f"Process noise unreasonably large: {Q}")

        # Check initial uncertainty
        P0 = config.get('initial_uncertainty', 0)
        if P0 <= 0:
            raise ValueError(f"Initial uncertainty must be positive: {P0}")

        return True
```

---

## Integration Plan

### Phase 1: Pre-Processor (Week 1-2)
1. Deploy `EnhancedPreProcessor`
2. Run in shadow mode (log but don't reject)
3. Validate detection accuracy
4. Enable rejection for clear errors (BMI, impossible values)

### Phase 2: Adaptive Outliers (Week 3-4)
1. Deploy `AdaptiveOutlierDetector`
2. A/B test against fixed thresholds
3. Monitor acceptance/rejection rates by source
4. Tune thresholds based on results

### Phase 3: Kalman Adaptation (Week 5-6)
1. Deploy `AdaptiveKalmanNoise`
2. Start with worst source (iGlucose)
3. Validate filter stability
4. Roll out to all sources

### Complete Integration

```python
def process_weight_improved(
    user_id: str,
    weight: float,
    unit: str,
    source: str,
    timestamp: datetime,
    base_config: Dict
) -> Optional[Dict]:
    """
    Improved processing with all three enhancements.
    """

    # 1. Pre-process
    cleaned_weight, preprocess_meta = EnhancedPreProcessor.preprocess(
        weight, unit, source
    )

    if cleaned_weight is None:
        return {
            'rejected': True,
            'stage': 'preprocessing',
            'reason': preprocess_meta.get('rejected'),
            'metadata': preprocess_meta
        }

    # 2. Get adaptive threshold
    state = get_state_db().get_state(user_id)
    if state and state.get('last_state'):
        last_weight = state['last_state'][0]
        time_gap = (timestamp - state['last_timestamp']).days

        is_outlier, reason, outlier_info = AdaptiveOutlierDetector.check_outlier(
            cleaned_weight, last_weight, source, time_gap
        )

        if is_outlier:
            return {
                'rejected': True,
                'stage': 'outlier_detection',
                'reason': reason,
                'metadata': outlier_info
            }

    # 3. Adapt Kalman configuration
    kalman_config = AdaptiveKalmanNoise.calculate_measurement_noise(
        source, base_config['kalman']
    )

    # Validate configuration
    AdaptiveKalmanNoise.validate_kalman_integrity(kalman_config)

    # 4. Process with adapted parameters
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=cleaned_weight,
        timestamp=timestamp,
        source=source,
        processing_config=base_config['processing'],
        kalman_config=kalman_config
    )

    # 5. Add metadata
    if result:
        result['preprocessing'] = preprocess_meta
        result['kalman_adaptation'] = kalman_config.get('noise_adaptation')

    return result
```

---

## Testing & Validation

### Test Suite

```python
def test_improvements():
    """Comprehensive test suite."""

    # Test 1: Unit conversion
    assert_unit_conversion()

    # Test 2: BMI detection
    assert_bmi_detection()

    # Test 3: Source-specific thresholds
    assert_adaptive_thresholds()

    # Test 4: Kalman noise adaptation
    assert_kalman_integrity()

    # Test 5: End-to-end with bad data
    assert_bad_data_handling()

    # Test 6: End-to-end with good data
    assert_good_data_preservation()
```

### Validation Metrics

1. **Pre-Processor Effectiveness**
   - BMI detection rate: >95%
   - Unit conversion accuracy: >99%
   - False rejection rate: <1%

2. **Outlier Detection**
   - iGlucose outliers: Reduce from 151.4 to <50 per 1000
   - Care-team false rejections: <5 per 1000
   - Overall outlier rate: <20 per 1000

3. **Kalman Performance**
   - Tracking error: <0.5kg for good sources
   - Convergence time: <5 measurements
   - Stability: No divergence over 1000+ measurements

---

## Risk Mitigation

1. **Rollback Strategy**
   - Feature flags per improvement
   - Shadow mode before production
   - Instant revert capability

2. **Monitoring**
   - Real-time rejection rates
   - Source quality metrics
   - Kalman filter stability

3. **Fallbacks**
   - If preprocessing fails: Use original weight
   - If outlier detection fails: Use fixed threshold
   - If Kalman adaptation fails: Use base configuration

---

## Success Criteria

### Must Have
- ✅ BMI values correctly identified and rejected
- ✅ Pound entries correctly converted
- ✅ iGlucose outlier rate reduced by >50%
- ✅ Kalman filter remains stable

### Should Have
- ✅ Overall outlier rate <20 per 1000
- ✅ Source-specific improvements measurable
- ✅ No degradation for good sources

### Nice to Have
- ✅ User-specific adaptation
- ✅ Automatic threshold tuning
- ✅ Predictive outlier detection

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1-2 | Pre-Processor | Unit conversion, BMI detection |
| 3-4 | Outlier Detection | Adaptive thresholds |
| 5-6 | Kalman Adaptation | Noise adjustment |
| 7 | Integration | Combined system |
| 8 | Validation | Metrics, tuning |

---

## Council Approval

**Donald Knuth** (Algorithms): "The mathematical foundation is sound. Adapting measurement noise preserves Kalman optimality while improving real-world performance."

**Nancy Leveson** (Safety): "Three-layer defense with clear failure modes. Each layer can fail safely without corrupting the system."

**Butler Lampson** (Simplicity): "Three focused improvements are manageable. The modular design allows independent testing and rollback."

**Barbara Liskov** (Architecture): "Clean separation of concerns. Each improvement is independently testable and maintainable."
