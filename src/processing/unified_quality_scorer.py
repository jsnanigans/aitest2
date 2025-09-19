"""
Unified Kalman-centric quality scoring system.
Replaces dual validation with single Kalman-deviation-based quality scorer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

from ..feature_manager import FeatureManager

try:
    from ..constants import (
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        KALMAN_DEFAULTS,
        BMI_LIMITS
    )
except ImportError:
    from src.constants import (
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        KALMAN_DEFAULTS,
        BMI_LIMITS
    )


@dataclass
class QualityScore:
    """Container for quality score and its components."""

    overall: float
    components: Dict[str, float]
    threshold: float = 0.6
    accepted: bool = False
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.accepted = self.overall >= self.threshold
        if self.metadata is None:
            self.metadata = {}

        if not self.accepted and not self.rejection_reason:
            if self.components:
                min_component = min(self.components.items(), key=lambda x: x[1])
                self.rejection_reason = (
                    f"Quality score {self.overall:.2f} below threshold {self.threshold} "
                    f"(weakest: {min_component[0]}={min_component[1]:.2f})"
                )
            else:
                self.rejection_reason = (
                    f"Quality score {self.overall:.2f} below threshold {self.threshold} "
                    f"(no components calculated)"
                )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'overall': self.overall,
            'components': self.components,
            'threshold': self.threshold,
            'accepted': self.accepted,
            'rejection_reason': self.rejection_reason,
            'metadata': self.metadata
        }


class UnifiedQualityScorer:
    """
    Unified Kalman-centric quality scoring system.
    Primary signal is deviation from Kalman prediction.
    """

    # Default component weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        'kalman_fit': 0.40,      # Primary signal
        'temporal_consistency': 0.20,
        'anomaly_detection': 0.20,
        'source_reliability': 0.10,
        'trend_alignment': 0.10
    }

    # Time-based thresholds for temporal consistency
    TEMPORAL_THRESHOLDS = {
        '6h': 3.0,   # 3kg in 6 hours
        '24h': 2.0,  # 2kg in 24 hours
        'sustained': 2.0  # 2kg/day sustained
    }

    # Anomaly patterns
    UNIT_CONFUSION_FACTORS = [2.2, 0.454, 10.0, 0.1]  # kg/lbs, lbs/kg, decimal errors
    BMI_RANGE = (15.0, 50.0)  # Common BMI range that might be entered as weight

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional config overrides."""
        self.config = config or {}

        # Get component weights from config
        self.weights = self.config.get('component_weights', self.DEFAULT_WEIGHTS.copy())

        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}

        # Get thresholds
        self.threshold = self.config.get('threshold', 0.6)
        self.temporal_thresholds = self.config.get('temporal_thresholds', self.TEMPORAL_THRESHOLDS.copy())

        # Get feature manager
        self.feature_manager = config.get('feature_manager') if config else None
        if not self.feature_manager:
            self.feature_manager = FeatureManager(config)

    def calculate_quality_score(
        self,
        weight: float,
        source: str,
        kalman_state: Optional[Dict] = None,
        kalman_prediction: Optional[float] = None,
        innovation_covariance: Optional[float] = None,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None,
        recent_weights: Optional[List[float]] = None,
        recent_timestamps: Optional[List[datetime]] = None,
        user_height_m: Optional[float] = None
    ) -> QualityScore:
        """
        Calculate unified quality score with Kalman-centric approach.

        Args:
            weight: Current weight measurement
            source: Data source
            kalman_state: Full Kalman filter state
            kalman_prediction: Predicted weight from Kalman filter
            innovation_covariance: Innovation covariance from Kalman
            previous_weight: Previous weight value
            time_diff_hours: Hours since last measurement
            recent_weights: Recent weight measurements
            recent_timestamps: Recent measurement timestamps
            user_height_m: User height in meters

        Returns:
            QualityScore object with overall score and components
        """
        components = {}
        metadata = {}

        # Only calculate components with non-zero weights
        # 1. Kalman Fit Component
        if self.weights.get('kalman_fit', 0) > 0:
            kalman_score, kalman_meta = self.calculate_kalman_fit(
                weight, kalman_prediction, innovation_covariance, kalman_state
            )
            components['kalman_fit'] = kalman_score
            metadata['kalman_fit'] = kalman_meta

        # 2. Temporal Consistency
        if self.weights.get('temporal_consistency', 0) > 0:
            temporal_score, temporal_meta = self.calculate_temporal_consistency(
                weight, previous_weight, time_diff_hours, recent_weights, recent_timestamps
            )
            components['temporal_consistency'] = temporal_score
            metadata['temporal_consistency'] = temporal_meta

        # 3. Anomaly Detection
        if self.weights.get('anomaly_detection', 0) > 0:
            anomaly_score, anomaly_meta = self.calculate_anomaly_detection(
                weight, recent_weights, recent_timestamps, user_height_m
            )
            components['anomaly_detection'] = anomaly_score
            metadata['anomaly_detection'] = anomaly_meta

        # 4. Source Reliability - skip if weight is 0
        if self.weights.get('source_reliability', 0) > 0:
            source_score = self.calculate_source_reliability(source)
            components['source_reliability'] = source_score
            metadata['source_reliability'] = {'source': source, 'score': source_score}

        # 5. Trend Alignment - skip if weight is 0
        if self.weights.get('trend_alignment', 0) > 0:
            trend_score, trend_meta = self.calculate_trend_alignment(
                weight, kalman_state, recent_weights
            )
            components['trend_alignment'] = trend_score
            metadata['trend_alignment'] = trend_meta

        # Calculate overall score using configured mean type
        use_harmonic = self.config.get('use_harmonic_mean', False)
        if use_harmonic:
            overall = self._calculate_weighted_harmonic_mean(components)
        else:
            overall = self._calculate_weighted_geometric_mean(components)

        return QualityScore(
            overall=overall,
            components=components,
            threshold=self.threshold,
            metadata=metadata
        )

    def calculate_kalman_fit(
        self,
        weight: float,
        kalman_prediction: Optional[float],
        innovation_covariance: Optional[float],
        kalman_state: Optional[Dict]
    ) -> Tuple[float, Dict]:
        """
        Calculate how well measurement fits Kalman prediction.
        Uses Mahalanobis distance and chi-squared test.
        Applies time-based decay: importance decreases over time since last measurement.
        """
        metadata = {}

        # If no Kalman prediction available, return neutral score
        if kalman_prediction is None or innovation_covariance is None:
            metadata['reason'] = 'No Kalman prediction available'
            return 0.5, metadata

        # Calculate innovation (prediction error)
        innovation = weight - kalman_prediction
        metadata['innovation'] = innovation
        metadata['prediction'] = kalman_prediction

        # Handle zero or very small covariance
        if innovation_covariance <= 0:
            innovation_covariance = 1.0

        # Normalize innovation (Mahalanobis distance)
        normalized_innovation = abs(innovation) / np.sqrt(innovation_covariance)
        metadata['normalized_innovation'] = normalized_innovation

        # Chi-squared test (df=1 for univariate)
        chi_squared = normalized_innovation ** 2
        p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
        metadata['chi_squared'] = chi_squared
        metadata['p_value'] = p_value

        # Check for adaptive period (relax thresholds)
        in_adaptive_period = False
        if kalman_state:
            measurements_since_reset = kalman_state.get('measurements_since_reset', 100)
            reset_params = kalman_state.get('reset_parameters', {})
            adaptation_measurements = reset_params.get('adaptation_measurements', 10)
            if measurements_since_reset < adaptation_measurements:
                in_adaptive_period = True
                metadata['adaptive_period'] = True

        # Convert to quality score
        if in_adaptive_period:
            # More forgiving during adaptation
            score = np.exp(-0.2 * normalized_innovation)  # Slower decay
        else:
            # Standard scoring
            score = np.exp(-0.5 * normalized_innovation)  # Exponential decay

        # Apply time-based decay for gap tolerance
        # After gaps, Kalman predictions become less reliable
        days_since_last = 0
        if kalman_state and 'last_timestamp' in kalman_state:
            last_timestamp = kalman_state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            # Get current timestamp from state or use now as fallback
            current_timestamp = kalman_state.get('current_timestamp', datetime.now())
            if isinstance(current_timestamp, str):
                current_timestamp = datetime.fromisoformat(current_timestamp)
            days_since_last = (current_timestamp - last_timestamp).total_seconds() / 86400.0
            metadata['days_since_last'] = days_since_last

        # Apply decay factor based on time gap
        # Linear decay: at 30 days, Kalman fit doesn't matter (score approaches 1.0)
        # Formula: final_score = score + (1 - score) * min(1, days/30)
        if days_since_last > 0:
            decay_factor = min(1.0, days_since_last / 30.0)  # Linear decay over 30 days
            # Blend towards 1.0 (full acceptance) as time increases
            adjusted_score = score + (1.0 - score) * decay_factor
            metadata['decay_factor'] = decay_factor
            metadata['original_score'] = score
            score = adjusted_score

        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))
        metadata['score'] = score

        return score, metadata

    def calculate_temporal_consistency(
        self,
        weight: float,
        previous_weight: Optional[float],
        time_diff_hours: Optional[float],
        recent_weights: Optional[List[float]],
        recent_timestamps: Optional[List[datetime]]
    ) -> Tuple[float, Dict]:
        """
        Calculate temporal consistency based on rate of change.
        Thresholds: 3kg/6hr, 2kg/24hr, 2kg/day sustained.
        """
        metadata = {}

        # If no previous weight, return neutral score
        if previous_weight is None or time_diff_hours is None:
            metadata['reason'] = 'No previous weight for comparison'
            return 0.7, metadata

        # Calculate change rate
        weight_change = abs(weight - previous_weight)
        metadata['weight_change'] = weight_change
        metadata['time_diff_hours'] = time_diff_hours

        # Apply time-based thresholds
        score = 1.0

        # 6-hour threshold
        if time_diff_hours <= 6:
            threshold = self.temporal_thresholds['6h']
            if weight_change > threshold:
                score *= max(0.2, 1.0 - (weight_change - threshold) / threshold)
                metadata['violated_6h'] = True

        # 24-hour threshold
        elif time_diff_hours <= 24:
            threshold = self.temporal_thresholds['24h']
            if weight_change > threshold:
                score *= max(0.3, 1.0 - (weight_change - threshold) / threshold)
                metadata['violated_24h'] = True

        # Sustained change (daily rate)
        else:
            daily_rate = weight_change / (time_diff_hours / 24.0)
            metadata['daily_rate'] = daily_rate
            threshold = self.temporal_thresholds['sustained']
            if daily_rate > threshold:
                score *= max(0.4, 1.0 - (daily_rate - threshold) / threshold)
                metadata['violated_sustained'] = True

        # Check for measurement gaps (be more lenient after gaps)
        if time_diff_hours > 168:  # More than a week
            score = max(score, 0.6)  # Don't penalize too harshly
            metadata['gap_adjustment'] = True

        return score, metadata

    def calculate_anomaly_detection(
        self,
        weight: float,
        recent_weights: Optional[List[float]],
        recent_timestamps: Optional[List[datetime]],
        user_height_m: Optional[float]
    ) -> Tuple[float, Dict]:
        """
        Detect anomalies: different user, unit confusion, BMI entry.
        """
        metadata = {}
        score = 1.0

        # Check for BMI entry (15-50 range)
        if self.BMI_RANGE[0] <= weight <= self.BMI_RANGE[1]:
            metadata['possible_bmi'] = True
            score *= 0.5  # Suspicious but not conclusive

        # Check for unit confusion
        if recent_weights and len(recent_weights) >= 2:
            median_weight = np.median(recent_weights)

            for factor in self.UNIT_CONFUSION_FACTORS:
                if abs(weight - median_weight * factor) < 5.0:
                    metadata['possible_unit_confusion'] = factor
                    score *= 0.2
                    break
                elif abs(weight * factor - median_weight) < 5.0:
                    metadata['possible_unit_confusion'] = 1/factor
                    score *= 0.2
                    break

        # Check for different user pattern (A→B→A)
        if self._detect_different_user(weight, recent_weights, recent_timestamps):
            metadata['different_user_detected'] = True
            score *= 0.1

        return score, metadata

    def _detect_different_user(
        self,
        weight: float,
        recent_weights: Optional[List[float]],
        recent_timestamps: Optional[List[datetime]]
    ) -> bool:
        """
        Detect A→B→A pattern within 24 hours.
        Indicates different user on same scale.
        """
        if not recent_weights or len(recent_weights) < 3:
            return False

        # Look for pattern in recent weights
        # If timestamps provided, filter to last 24 hours
        if recent_timestamps and len(recent_timestamps) == len(recent_weights):
            now = datetime.now()
            last_24h_weights = []
            for w, t in zip(recent_weights[-10:], recent_timestamps[-10:]):
                if isinstance(t, datetime):
                    if (now - t).total_seconds() <= 86400:  # 24 hours
                        last_24h_weights.append(w)
        else:
            # Use recent weights directly if no timestamps
            last_24h_weights = list(recent_weights[-10:])

        if len(last_24h_weights) < 3:
            return False

        # Check for large jump and return pattern
        for i in range(len(last_24h_weights) - 2):
            w1, w2, w3 = last_24h_weights[i:i+3]

            # Check if middle weight is significantly different
            jump_out = abs(w2 - w1) > 10.0  # >10kg jump
            jump_back = abs(w3 - w2) > 10.0 and abs(w3 - w1) < 2.0  # Return to original

            if jump_out and jump_back:
                return True

        # Check current weight against pattern
        if len(last_24h_weights) >= 2:
            w1, w2 = last_24h_weights[-2:]
            jump_out = abs(w2 - w1) > 10.0
            jump_back = abs(weight - w2) > 10.0 and abs(weight - w1) < 2.0

            if jump_out and jump_back:
                return True

        return False

    def calculate_source_reliability(self, source: str) -> float:
        """
        Calculate source reliability based on SOURCE_PROFILES.
        """
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)

        # Convert noise_multiplier to reliability score
        # Lower noise multiplier = higher reliability
        noise_multiplier = profile.get('noise_multiplier', 1.0)

        # Invert and normalize to [0, 1]
        # noise_multiplier range: 0.5 (best) to 3.0 (worst)
        reliability = 1.0 - ((noise_multiplier - 0.5) / 2.5)
        reliability = max(0.2, min(1.0, reliability))  # Clamp to [0.2, 1.0]

        return reliability

    def calculate_trend_alignment(
        self,
        weight: float,
        kalman_state: Optional[Dict],
        recent_weights: Optional[List[float]]
    ) -> Tuple[float, Dict]:
        """
        Calculate alignment with established trend using linear regression.
        """
        metadata = {}

        # Need at least 5 measurements for trend
        if not recent_weights or len(recent_weights) < 5:
            metadata['reason'] = 'Insufficient data for trend'
            return 0.8, metadata  # Neutral-high score

        # Get recent Kalman states if available
        if kalman_state and 'measurement_history' in kalman_state:
            history = kalman_state['measurement_history']
            if isinstance(history, list) and len(history) >= 5:
                # Use Kalman filtered weights for trend
                kalman_weights = [h.get('filtered_weight', h.get('weight'))
                                for h in history[-10:] if 'weight' in h]
                if len(kalman_weights) >= 5:
                    recent_weights = kalman_weights

        # Perform linear regression on recent weights
        x = np.arange(len(recent_weights))
        y = np.array(recent_weights)

        # Calculate trend line
        slope, intercept = np.polyfit(x, y, 1)
        predicted_next = slope * len(recent_weights) + intercept

        metadata['trend_slope'] = slope
        metadata['predicted'] = predicted_next

        # Calculate deviation from trend
        deviation = abs(weight - predicted_next)

        # Expected variance around trend (use std of residuals)
        trend_line = slope * x + intercept
        residuals = y - trend_line
        std_dev = np.std(residuals)

        # Ensure minimum std_dev to avoid division by zero
        # Use 0.5 kg as minimum expected variation (configurable)
        trend_config = self.config.get('trend_alignment', {})
        min_std_dev = trend_config.get('trend_min_std_dev', 0.5)
        if std_dev < min_std_dev:
            std_dev = min_std_dev

        metadata['deviation'] = deviation
        metadata['std_dev'] = std_dev

        # Score based on deviation from trend
        normalized_deviation = deviation / std_dev

        # More gradual scoring: use exponential decay
        # Score = exp(-k * normalized_deviation)
        # k=0.3 gives ~0.74 at 1 std dev, ~0.55 at 2 std devs, ~0.40 at 3 std devs
        # k=0.2 gives ~0.82 at 1 std dev, ~0.67 at 2 std devs, ~0.55 at 3 std devs (more lenient)
        trend_config = self.config.get('trend_alignment', {})
        k = trend_config.get('trend_decay_constant', 0.3)  # Lower = more lenient
        score = np.exp(-k * normalized_deviation)

        # Ensure minimum score of 0.3 for reasonable deviations
        score = max(0.3, score)

        return score, metadata

    def _calculate_weighted_geometric_mean(self, components: Dict[str, float]) -> float:
        """
        Calculate weighted geometric mean of component scores.
        S = Π(c_i^w_i)^(1/Σw_i)
        """
        if not components:
            return 0.0

        # Ensure all scores are positive (avoid log of 0)
        epsilon = 1e-10

        product = 1.0
        weight_sum = 0.0

        for component_name, score in components.items():
            weight = self.weights.get(component_name, 0.0)
            if weight > 0:
                # Clamp score to avoid numerical issues
                score = max(epsilon, min(1.0, score))
                product *= (score ** weight)
                weight_sum += weight

        if weight_sum > 0:
            # Normalize by weight sum
            overall = product ** (1.0 / weight_sum)
        else:
            overall = 0.0

        return max(0.0, min(1.0, overall))

    def _calculate_weighted_harmonic_mean(self, components: Dict[str, float]) -> float:
        """
        Calculate weighted harmonic mean of component scores.
        More forgiving than geometric mean - doesn't penalize low scores as harshly.
        H = Σw_i / Σ(w_i / c_i)
        """
        if not components:
            return 0.0

        # Ensure all scores are positive (avoid division by 0)
        epsilon = 1e-10

        weighted_sum = 0.0
        weight_sum = 0.0

        for component_name, score in components.items():
            weight = self.weights.get(component_name, 0.0)
            if weight > 0:
                # Clamp score to avoid numerical issues
                score = max(epsilon, min(1.0, score))
                weighted_sum += weight / score
                weight_sum += weight

        if weighted_sum > 0:
            return weight_sum / weighted_sum
        else:
            return 0.0