# Plan: Improved Oscillation Handling

## Problem Statement

Real data analysis revealed 313 users with highly oscillating weight patterns:
- User 07d08dd8: 8 direction changes with 21.9kg range
- 994 users with >20% variation within a week
- Current system may over-filter legitimate oscillations or under-filter noise

Challenges with oscillating patterns:
- Statistical methods flag all values as outliers
- Kalman filter becomes unstable with frequent direction changes
- Difficult to distinguish between:
  - Medical conditions (heart failure, kidney disease)
  - Measurement noise
  - Real weight fluctuations

## Objectives

1. Distinguish between noise and legitimate oscillations
2. Stabilize Kalman filter for oscillating patterns
3. Apply appropriate smoothing without losing real signals
4. Detect and classify oscillation types
5. Adapt processing based on oscillation characteristics

## Implementation Design

### 1. Oscillation Pattern Analyzer

```python
# src/processing/oscillation_analyzer.py

import numpy as np
from scipy import signal, fft
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

class OscillationType(Enum):
    """Types of oscillation patterns."""
    NOISE = "noise"                    # Random fluctuations
    PERIODIC = "periodic"               # Regular oscillations (e.g., dialysis)
    CHAOTIC = "chaotic"                 # Irregular but bounded
    TRENDING = "trending"               # Oscillation with underlying trend
    MEDICAL = "medical"                 # Medical condition pattern

@dataclass
class OscillationProfile:
    """Profile of oscillation characteristics."""
    oscillation_type: OscillationType
    frequency: Optional[float]          # Dominant frequency if periodic
    amplitude: float                     # Average amplitude
    regularity: float                    # 0-1 score of pattern regularity
    trend_component: Optional[float]    # Underlying trend if present
    confidence: float                    # Confidence in classification

class OscillationAnalyzer:
    """Analyzes and classifies oscillation patterns in weight data."""

    def __init__(self, config=None):
        self.config = config or {}

        # Analysis parameters
        self.min_data_points = config.get('min_data_points', 10)
        self.frequency_threshold = config.get('frequency_threshold', 0.1)
        self.regularity_threshold = config.get('regularity_threshold', 0.6)

    def analyze_oscillation(self, measurements: List[Dict[str, Any]]) -> OscillationProfile:
        """
        Analyze oscillation pattern in measurements.

        Args:
            measurements: List of weight measurements

        Returns:
            OscillationProfile describing the pattern
        """
        if len(measurements) < self.min_data_points:
            return OscillationProfile(
                oscillation_type=OscillationType.NOISE,
                frequency=None,
                amplitude=0,
                regularity=0,
                trend_component=None,
                confidence=0.3
            )

        # Extract time series
        weights, timestamps = self._extract_time_series(measurements)

        # Detrend the data
        detrended, trend = self._detrend_data(weights)

        # Analyze frequency components
        freq_analysis = self._frequency_analysis(detrended, timestamps)

        # Calculate oscillation metrics
        metrics = self._calculate_metrics(detrended, weights)

        # Classify oscillation type
        oscillation_type = self._classify_oscillation(freq_analysis, metrics, trend)

        # Check for medical patterns
        if self._is_medical_pattern(measurements, metrics):
            oscillation_type = OscillationType.MEDICAL

        return OscillationProfile(
            oscillation_type=oscillation_type,
            frequency=freq_analysis.get('dominant_frequency'),
            amplitude=metrics['amplitude'],
            regularity=metrics['regularity'],
            trend_component=trend if abs(trend) > 0.01 else None,
            confidence=self._calculate_confidence(freq_analysis, metrics)
        )

    def _extract_time_series(self, measurements: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract weight and time arrays from measurements."""
        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])

        weights = np.array([m['weight'] for m in sorted_measurements])
        timestamps = np.array([m['timestamp'].timestamp() for m in sorted_measurements])

        # Convert to days from start
        timestamps = (timestamps - timestamps[0]) / 86400

        return weights, timestamps

    def _detrend_data(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Remove linear trend from data."""
        # Fit linear trend
        x = np.arange(len(weights))
        coeffs = np.polyfit(x, weights, 1)
        trend_line = np.polyval(coeffs, x)

        detrended = weights - trend_line
        trend_per_point = coeffs[0]

        return detrended, trend_per_point

    def _frequency_analysis(self, detrended: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Perform frequency analysis using FFT."""
        if len(detrended) < 4:
            return {'dominant_frequency': None, 'power': 0}

        # Interpolate to regular sampling
        regular_time = np.linspace(timestamps[0], timestamps[-1], len(timestamps))
        regular_weights = np.interp(regular_time, timestamps, detrended)

        # Apply FFT
        fft_result = fft.fft(regular_weights)
        frequencies = fft.fftfreq(len(regular_weights), d=np.mean(np.diff(regular_time)))

        # Get power spectrum
        power = np.abs(fft_result) ** 2

        # Find dominant frequency (exclude DC component)
        positive_freq_idx = frequencies > 0
        if np.any(positive_freq_idx):
            max_power_idx = np.argmax(power[positive_freq_idx])
            dominant_freq = frequencies[positive_freq_idx][max_power_idx]
            dominant_power = power[positive_freq_idx][max_power_idx]

            return {
                'dominant_frequency': dominant_freq,
                'power': dominant_power,
                'spectral_entropy': self._calculate_spectral_entropy(power[positive_freq_idx])
            }

        return {'dominant_frequency': None, 'power': 0}

    def _calculate_metrics(self, detrended: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Calculate oscillation metrics."""
        # Direction changes
        direction_changes = 0
        for i in range(2, len(detrended)):
            if (detrended[i] - detrended[i-1]) * (detrended[i-1] - detrended[i-2]) < 0:
                direction_changes += 1

        # Amplitude (using detrended data)
        amplitude = np.std(detrended)

        # Regularity (using autocorrelation)
        if len(detrended) > 10:
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            if len(peaks) > 0:
                # Check if peaks are regularly spaced
                peak_diffs = np.diff(peaks)
                if len(peak_diffs) > 0:
                    regularity = 1.0 - (np.std(peak_diffs) / np.mean(peak_diffs))
                else:
                    regularity = 0.0
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        # Range relative to median
        range_ratio = (np.max(original) - np.min(original)) / np.median(original)

        return {
            'direction_changes': direction_changes,
            'amplitude': amplitude,
            'regularity': max(0, min(1, regularity)),
            'range_ratio': range_ratio,
            'oscillation_rate': direction_changes / (len(detrended) - 2) if len(detrended) > 2 else 0
        }

    def _classify_oscillation(self, freq_analysis: Dict, metrics: Dict, trend: float) -> OscillationType:
        """Classify the type of oscillation."""

        # Strong trend with oscillation
        if abs(trend) > 0.5:  # Significant trend
            return OscillationType.TRENDING

        # Periodic pattern
        if freq_analysis.get('dominant_frequency') and metrics['regularity'] > self.regularity_threshold:
            return OscillationType.PERIODIC

        # High frequency random changes
        if metrics['oscillation_rate'] > 0.7 and metrics['regularity'] < 0.3:
            return OscillationType.NOISE

        # Bounded but irregular
        if metrics['range_ratio'] < 0.3 and metrics['regularity'] < self.regularity_threshold:
            return OscillationType.CHAOTIC

        return OscillationType.NOISE

    def _is_medical_pattern(self, measurements: List[Dict], metrics: Dict) -> bool:
        """Check if oscillation matches known medical patterns."""

        # Heart failure pattern: Daily variations of 1-3kg
        if metrics['amplitude'] >= 1 and metrics['amplitude'] <= 3:
            # Check for daily pattern
            if metrics.get('dominant_frequency', 0) > 0.8 and metrics.get('dominant_frequency', 0) < 1.2:
                return True

        # Dialysis pattern: 2-3 times per week drops
        if metrics['amplitude'] >= 2:
            # Check for 2-3 day cycle
            freq = metrics.get('dominant_frequency', 0)
            if 0.3 < freq < 0.5:  # 2-3 day period
                return True

        return False
```

### 2. Adaptive Smoothing Filter

```python
class AdaptiveSmoothingFilter:
    """Applies adaptive smoothing based on oscillation characteristics."""

    def __init__(self, config=None):
        self.config = config or {}
        self.analyzer = OscillationAnalyzer(config)

    def smooth_measurements(self, measurements: List[Dict[str, Any]],
                           profile: OscillationProfile) -> List[Dict[str, Any]]:
        """
        Apply appropriate smoothing based on oscillation type.

        Args:
            measurements: Raw measurements
            profile: Oscillation profile

        Returns:
            Smoothed measurements
        """
        if len(measurements) < 3:
            return measurements

        weights = np.array([m['weight'] for m in measurements])

        # Select smoothing method based on oscillation type
        if profile.oscillation_type == OscillationType.NOISE:
            smoothed = self._apply_median_filter(weights, window_size=3)

        elif profile.oscillation_type == OscillationType.PERIODIC:
            smoothed = self._apply_savitzky_golay(weights, profile)

        elif profile.oscillation_type == OscillationType.CHAOTIC:
            smoothed = self._apply_exponential_smoothing(weights, alpha=0.3)

        elif profile.oscillation_type == OscillationType.TRENDING:
            smoothed = self._apply_trend_preserving_filter(weights, profile)

        elif profile.oscillation_type == OscillationType.MEDICAL:
            # Minimal smoothing for medical patterns
            smoothed = self._apply_gentle_smoothing(weights)

        else:
            smoothed = weights

        # Create smoothed measurements
        smoothed_measurements = []
        for i, m in enumerate(measurements):
            smoothed_m = m.copy()
            smoothed_m['weight'] = smoothed[i]
            smoothed_m['smoothed'] = True
            smoothed_m['original_weight'] = m['weight']
            smoothed_measurements.append(smoothed_m)

        return smoothed_measurements

    def _apply_median_filter(self, weights: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Apply median filter for noise reduction."""
        return signal.medfilt(weights, kernel_size=window_size)

    def _apply_savitzky_golay(self, weights: np.ndarray, profile: OscillationProfile) -> np.ndarray:
        """Apply Savitzky-Golay filter for periodic patterns."""
        # Window size based on period
        if profile.frequency:
            window = max(5, int(1 / profile.frequency))
        else:
            window = 7

        # Ensure odd window size
        if window % 2 == 0:
            window += 1

        # Ensure window doesn't exceed data length
        window = min(window, len(weights))
        if window % 2 == 0:
            window -= 1

        if window >= 5:
            return signal.savgol_filter(weights, window, 3)
        return weights

    def _apply_exponential_smoothing(self, weights: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Apply exponential smoothing."""
        smoothed = np.zeros_like(weights)
        smoothed[0] = weights[0]

        for i in range(1, len(weights)):
            smoothed[i] = alpha * weights[i] + (1 - alpha) * smoothed[i-1]

        return smoothed

    def _apply_trend_preserving_filter(self, weights: np.ndarray,
                                      profile: OscillationProfile) -> np.ndarray:
        """Apply filter that preserves trend while reducing oscillation."""
        # Separate trend and oscillation
        x = np.arange(len(weights))
        trend_coeffs = np.polyfit(x, weights, 1)
        trend = np.polyval(trend_coeffs, x)

        # Smooth the oscillation component
        oscillation = weights - trend
        smoothed_oscillation = self._apply_median_filter(oscillation, window_size=3)

        # Recombine
        return trend + smoothed_oscillation * 0.7  # Reduce oscillation amplitude

    def _apply_gentle_smoothing(self, weights: np.ndarray) -> np.ndarray:
        """Apply minimal smoothing for medical patterns."""
        # Very gentle moving average
        window = 3
        smoothed = np.convolve(weights, np.ones(window)/window, mode='same')

        # Preserve endpoints
        smoothed[:window//2] = weights[:window//2]
        smoothed[-(window//2):] = weights[-(window//2):]

        return smoothed
```

### 3. Oscillation-Aware Kalman Filter

```python
class OscillationAwareKalmanFilter:
    """Kalman filter that adapts to oscillating patterns."""

    def __init__(self, config=None):
        self.config = config or {}
        self.analyzer = OscillationAnalyzer(config)

    def adapt_kalman_parameters(self, profile: OscillationProfile,
                               base_params: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt Kalman parameters based on oscillation profile.

        Args:
            profile: Oscillation profile
            base_params: Base Kalman parameters

        Returns:
            Adapted parameters
        """
        params = base_params.copy()

        if profile.oscillation_type == OscillationType.NOISE:
            # Increase observation noise for noisy data
            params['observation_noise'] *= 2.0
            params['process_noise'] *= 0.5

        elif profile.oscillation_type == OscillationType.PERIODIC:
            # Adjust for periodic patterns
            if profile.frequency:
                # Faster adaptation for known period
                params['process_noise'] *= (1 + profile.frequency)
            params['observation_noise'] *= (1.5 - profile.regularity * 0.5)

        elif profile.oscillation_type == OscillationType.MEDICAL:
            # Allow for legitimate variation
            params['process_noise'] *= 1.5
            params['observation_noise'] *= 0.8

        elif profile.oscillation_type == OscillationType.TRENDING:
            # Focus on trend
            params['process_noise'] *= 0.7
            params['observation_noise'] *= 1.2

        elif profile.oscillation_type == OscillationType.CHAOTIC:
            # Balanced approach
            params['process_noise'] *= 1.2
            params['observation_noise'] *= 1.2

        return params

    def should_reset_filter(self, profile: OscillationProfile,
                           innovation: float, current_uncertainty: float) -> bool:
        """
        Determine if Kalman filter should reset based on oscillation.

        Args:
            profile: Oscillation profile
            innovation: Current Kalman innovation
            current_uncertainty: Current state uncertainty

        Returns:
            Whether to reset the filter
        """
        # Don't reset for known medical patterns
        if profile.oscillation_type == OscillationType.MEDICAL:
            return False

        # Don't reset for regular periodic patterns
        if profile.oscillation_type == OscillationType.PERIODIC and profile.regularity > 0.7:
            return False

        # Reset if innovation exceeds threshold adjusted for oscillation
        threshold = 3 * current_uncertainty

        if profile.oscillation_type == OscillationType.NOISE:
            threshold *= 2  # More lenient for noisy data

        return abs(innovation) > threshold
```

### 4. Integration with Main Processing

```python
# Enhanced processor integration

def process_with_oscillation_handling(user_id, measurements, kalman_filter, db):
    """Process measurements with oscillation-aware handling."""

    # Analyze oscillation pattern
    analyzer = OscillationAnalyzer()
    profile = analyzer.analyze_oscillation(measurements)

    # Store profile for user
    db.save_oscillation_profile(user_id, profile)

    # Apply adaptive smoothing if needed
    if profile.oscillation_type in [OscillationType.NOISE, OscillationType.CHAOTIC]:
        smoother = AdaptiveSmoothingFilter()
        measurements = smoother.smooth_measurements(measurements, profile)

    # Adapt Kalman parameters
    kalman_adapter = OscillationAwareKalmanFilter()
    base_params = kalman_filter.get_parameters()
    adapted_params = kalman_adapter.adapt_kalman_parameters(profile, base_params)
    kalman_filter.update_parameters(adapted_params)

    # Process measurements with adapted filter
    for measurement in measurements:
        # Check if reset needed (oscillation-aware)
        innovation = kalman_filter.calculate_innovation(measurement['weight'])
        uncertainty = kalman_filter.get_uncertainty()

        if kalman_adapter.should_reset_filter(profile, innovation, uncertainty):
            kalman_filter.reset(measurement['weight'])

        # Process measurement
        kalman_filter.update(measurement['weight'])

        # Adjust quality score based on oscillation
        if profile.oscillation_type == OscillationType.MEDICAL:
            measurement['quality_score'] *= 1.1  # Boost for known patterns
        elif profile.oscillation_type == OscillationType.NOISE:
            measurement['quality_score'] *= 0.8  # Reduce for noise

    return measurements
```

## Implementation Steps

### Phase 1: Pattern Analysis (Week 1)
1. Create `OscillationAnalyzer` class
2. Implement frequency analysis with FFT
3. Add oscillation metrics calculation
4. Build pattern classification logic

### Phase 2: Adaptive Smoothing (Week 2)
1. Implement `AdaptiveSmoothingFilter`
2. Create type-specific smoothing methods
3. Add trend preservation logic
4. Build medical pattern protection

### Phase 3: Kalman Adaptation (Week 3)
1. Create `OscillationAwareKalmanFilter`
2. Implement parameter adaptation rules
3. Add oscillation-aware reset logic
4. Build stability monitoring

### Phase 4: Integration (Week 4)
1. Integrate with main processor
2. Add oscillation profile storage
3. Create visualization tools
4. Build monitoring dashboards

## Testing Strategy

### Unit Tests
```python
def test_periodic_oscillation_detection():
    """Test detection of periodic oscillations."""
    analyzer = OscillationAnalyzer()

    # Create periodic pattern (like dialysis)
    measurements = []
    for day in range(30):
        weight = 70 + 3 * np.sin(2 * np.pi * day / 3)  # 3-day period
        measurements.append({
            'weight': weight,
            'timestamp': datetime.now() + timedelta(days=day)
        })

    profile = analyzer.analyze_oscillation(measurements)
    assert profile.oscillation_type == OscillationType.PERIODIC
    assert 0.3 < profile.frequency < 0.4  # ~3 day period

def test_noise_vs_medical_pattern():
    """Test distinguishing noise from medical patterns."""
    analyzer = OscillationAnalyzer()

    # Medical pattern (heart failure - daily 2kg variation)
    medical = []
    for day in range(14):
        weight = 70 + 2 * np.sin(2 * np.pi * day)  # Daily oscillation
        medical.append({
            'weight': weight,
            'timestamp': datetime.now() + timedelta(days=day)
        })

    profile = analyzer.analyze_oscillation(medical)
    assert profile.oscillation_type == OscillationType.MEDICAL
```

## Configuration

```toml
[oscillation_handling]
enabled = true
min_data_points = 10
frequency_threshold = 0.1
regularity_threshold = 0.6

[oscillation_handling.smoothing]
noise_window_size = 3
periodic_polyorder = 3
exponential_alpha = 0.3
trend_reduction = 0.7

[oscillation_handling.kalman_adaptation]
noise_observation_multiplier = 2.0
periodic_process_multiplier = 1.5
medical_process_multiplier = 1.5
```

## Success Metrics

1. **Pattern classification accuracy**: >85% correct classification
2. **Smoothing effectiveness**: >30% noise reduction without signal loss
3. **Kalman stability**: <10% unnecessary resets for oscillating patterns
4. **Medical pattern preservation**: >95% medical patterns unchanged

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-smoothing legitimate variation | High | Medical pattern detection, minimal smoothing option |
| Misclassification of patterns | Medium | Confidence scoring, manual override |
| Computational overhead | Low | Caching, selective analysis |
| Filter instability | High | Bounded parameter adaptation, reset limits |

## Future Enhancements

1. Machine learning for pattern classification
2. User feedback on oscillation handling
3. Integration with medical condition database
4. Real-time oscillation alerts
5. Predictive oscillation forecasting