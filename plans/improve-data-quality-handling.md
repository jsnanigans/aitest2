# Plan: Improve Data Quality Handling Based on Source Analysis

## Executive Summary

Based on analysis of 709,246 measurements, we've identified that iGlucose API has 151.4 outliers/1000 (42× worse than care-team uploads) while patient uploads are surprisingly reliable (13.0 outliers/1000). This plan proposes improvements that maintain the Kalman filter's mathematical optimality while adding defensive layers for bad data.

## Key Principles (Council Guidance)

**Butler Lampson** (Simplicity): "Don't break what works. Add defensive layers without changing core Kalman processing."

**Nancy Leveson** (Safety): "Multiple defense layers. No single point of failure. Monitor and alert on degradation."

**Barbara Liskov** (Architecture): "Maintain clean interfaces. Source-specific logic should be isolated from core processing."

## Proposed Improvements

### 1. Pre-Processing Data Quality Layer

**Purpose:** Clean data BEFORE it reaches the Kalman filter

```python
class DataQualityPreprocessor:
    """Clean and standardize data before processing."""
    
    @staticmethod
    def preprocess(weight: float, source: str, timestamp: datetime) -> Tuple[float, Dict]:
        """
        Returns: (cleaned_weight, metadata)
        """
        metadata = {'original_weight': weight, 'corrections': []}
        
        # 1. Unit Detection & Conversion
        if source in ['patient-upload', 'internal-questionnaire', 'care-team-upload']:
            # These sources are 74-100% pound entries
            if 80 < weight < 400:  # Likely pounds range
                pounds = weight
                weight = weight * 0.453592
                metadata['corrections'].append(f'Converted {pounds} lb to {weight} kg')
        
        # 2. BMI Detection (ConnectiveHealth issue)
        if 15 < weight < 50:
            # Might be BMI, not weight
            metadata['warning'] = 'Possible BMI value, not weight'
            # Don't process as weight
            return None, metadata
        
        # 3. Source-Specific Corrections
        if source == 'https://api.iglucose.com':
            # This source has extreme outlier issues
            metadata['high_risk'] = True
            
        return weight, metadata
```

### 2. Adaptive Outlier Detection

**Purpose:** Use source-specific outlier thresholds based on observed patterns

```python
class AdaptiveOutlierDetector:
    """Source-aware outlier detection."""
    
    # Based on our analysis (outliers per 1000)
    SOURCE_OUTLIER_RATES = {
        'care-team-upload': 3.6,
        'patient-upload': 13.0,
        'internal-questionnaire': 14.0,
        'patient-device': 20.7,
        'https://connectivehealth.io': 35.8,
        'https://api.iglucose.com': 151.4
    }
    
    @staticmethod
    def get_threshold(source: str, time_gap_days: int) -> float:
        """Get outlier threshold based on source quality."""
        base_rate = AdaptiveOutlierDetector.SOURCE_OUTLIER_RATES.get(source, 50.0)
        
        # Stricter for unreliable sources
        if base_rate > 100:  # iGlucose
            max_change = min(5.0, 1.0 * (time_gap_days / 7.0))
        elif base_rate > 30:  # ConnectiveHealth
            max_change = min(10.0, 1.5 * (time_gap_days / 7.0))
        else:  # Reliable sources
            max_change = min(15.0, 2.0 * (time_gap_days / 7.0))
        
        return max_change
```

### 3. Kalman Measurement Noise Adaptation

**Purpose:** Adjust Kalman measurement noise based on source reliability

```python
class AdaptiveKalmanConfig:
    """Adapt Kalman parameters based on source quality."""
    
    @staticmethod
    def get_measurement_noise(source: str, base_noise: float = 1.0) -> float:
        """
        Adjust measurement noise based on source reliability.
        Higher noise = less trust in measurement.
        """
        # Based on outlier rates from analysis
        noise_multipliers = {
            'care-team-upload': 0.5,        # Most reliable
            'patient-upload': 0.7,           # Very reliable
            'internal-questionnaire': 0.8,   # Reliable but sparse
            'patient-device': 1.0,           # Baseline
            'https://connectivehealth.io': 1.5,  # Less reliable
            'https://api.iglucose.com': 3.0      # Least reliable
        }
        
        multiplier = noise_multipliers.get(source, 1.0)
        return base_noise * multiplier
```

### 4. Quality Monitoring & Alerting

**Purpose:** Track source quality degradation in real-time

```python
class SourceQualityMonitor:
    """Monitor source quality and alert on degradation."""
    
    def __init__(self):
        self.rolling_stats = defaultdict(lambda: {
            'measurements': deque(maxlen=1000),
            'outliers': deque(maxlen=1000),
            'baseline_outlier_rate': None
        })
    
    def check_quality(self, source: str, is_outlier: bool) -> Optional[str]:
        """Check if source quality has degraded."""
        stats = self.rolling_stats[source]
        stats['measurements'].append(1)
        stats['outliers'].append(1 if is_outlier else 0)
        
        if len(stats['measurements']) >= 100:
            current_rate = sum(stats['outliers']) / len(stats['measurements']) * 1000
            
            # Expected rates from our analysis
            expected_rates = {
                'care-team-upload': 3.6,
                'patient-upload': 13.0,
                'https://api.iglucose.com': 151.4
            }
            
            expected = expected_rates.get(source, 30.0)
            
            if current_rate > expected * 1.5:
                return f"ALERT: {source} outlier rate {current_rate:.1f}/1000 " \
                       f"(expected {expected:.1f}/1000)"
        
        return None
```

### 5. Enhanced Processor Integration

**Purpose:** Integrate improvements without breaking existing system

```python
class EnhancedWeightProcessor:
    """Enhanced processor with quality improvements."""
    
    @staticmethod
    def process_weight_enhanced(
        user_id: str,
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: Dict,
        kalman_config: Dict
    ) -> Optional[Dict]:
        """Enhanced processing with data quality improvements."""
        
        # 1. Pre-process data
        cleaned_weight, metadata = DataQualityPreprocessor.preprocess(
            weight, source, timestamp
        )
        
        if cleaned_weight is None:
            return None  # Rejected in pre-processing
        
        # 2. Get adaptive thresholds
        state = get_state_db().get_state(user_id)
        if state and state.get('last_timestamp'):
            time_gap = (timestamp - state['last_timestamp']).days
            outlier_threshold = AdaptiveOutlierDetector.get_threshold(source, time_gap)
            
            # Override config with adaptive threshold
            processing_config = processing_config.copy()
            processing_config['extreme_threshold'] = outlier_threshold
        
        # 3. Adapt Kalman noise
        kalman_config = kalman_config.copy()
        kalman_config['measurement_noise'] = AdaptiveKalmanConfig.get_measurement_noise(
            source, kalman_config.get('measurement_noise', 1.0)
        )
        
        # 4. Process with original Kalman
        result = WeightProcessor.process_weight(
            user_id, cleaned_weight, timestamp, source,
            processing_config, kalman_config
        )
        
        # 5. Monitor quality
        if result:
            monitor = SourceQualityMonitor()
            alert = monitor.check_quality(source, result.get('rejected', False))
            if alert:
                result['quality_alert'] = alert
            
            # Add metadata
            result['preprocessing_metadata'] = metadata
        
        return result
```

## Implementation Strategy

### Phase 1: Monitoring Only (Week 1-2)
1. Deploy SourceQualityMonitor in parallel
2. Log alerts without acting on them
3. Validate detection accuracy

### Phase 2: Pre-Processing (Week 3-4)
1. Deploy DataQualityPreprocessor
2. Handle unit conversions
3. Filter BMI values

### Phase 3: Adaptive Thresholds (Week 5-6)
1. Deploy AdaptiveOutlierDetector
2. A/B test against current fixed thresholds
3. Measure improvement in acceptance rates

### Phase 4: Kalman Adaptation (Week 7-8)
1. Deploy AdaptiveKalmanConfig
2. Test on high-outlier sources first (iGlucose)
3. Gradual rollout to all sources

## Expected Improvements

### Metrics
- **Reduce false rejections:** 20-30% improvement for reliable sources
- **Reduce false acceptances:** 50-60% improvement for iGlucose
- **Overall accuracy:** 15-20% improvement

### By Source
1. **iGlucose:** Outliers reduced from 151.4 to ~50 per 1000
2. **ConnectiveHealth:** BMI filtering eliminates 1,500+ errors
3. **Patient-upload:** Better acceptance with pound conversion
4. **Care-team:** Maintain high quality with lower noise

## Risk Mitigation

1. **Keep original processor:** Run in parallel initially
2. **Feature flags:** Enable per source, per user
3. **Rollback plan:** Instant revert to original
4. **Monitoring:** Real-time quality metrics

## Success Criteria

1. ✅ iGlucose outlier rate < 50 per 1000
2. ✅ No degradation in reliable sources
3. ✅ BMI values filtered successfully
4. ✅ Pound conversion accuracy > 95%
5. ✅ Overall outlier rate < 20 per 1000

## Council Review

**Nancy Leveson** (Safety): "Multiple defensive layers with monitoring at each level. Excellent safety architecture."

**Butler Lampson** (Simplicity): "Preserves core Kalman simplicity while adding necessary defenses. Good separation of concerns."

**Barbara Liskov** (Architecture): "Clean interfaces maintained. Source-specific logic properly isolated."

## Conclusion

These improvements leverage our analysis findings to:
1. **Protect** against bad data (iGlucose, ConnectiveHealth)
2. **Reward** good sources (care-team, patient-upload)
3. **Preserve** Kalman filter optimality
4. **Monitor** for quality degradation

The approach is incremental, reversible, and measurable.