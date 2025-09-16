"""
Unified Quality Scoring System for weight measurements.
STATELESS DESIGN - Processes one measurement at a time with minimal context.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import numpy as np

try:
    from ..constants import (
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
    )
except ImportError:
    from src.constants import (
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
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
            min_component = min(self.components.items(), key=lambda x: x[1])
            self.rejection_reason = (
                f"Quality score {self.overall:.2f} below threshold {self.threshold} "
                f"(weakest: {min_component[0]}={min_component[1]:.2f})"
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


class QualityScorer:
    """
    Calculates quality scores for weight measurements.
    STATELESS - Only needs previous weight and time difference.
    """
    
    COMPONENT_WEIGHTS = {
        'safety': 0.35,
        'plausibility': 0.25,
        'consistency': 0.25,
        'reliability': 0.15
    }
    
    SAFETY_CRITICAL_THRESHOLD = 0.3
    
    # Time-based thresholds (research-based: 2-3% daily variation is normal)
    HOURLY_MAX_KG = 3.0
    DAILY_MAX_KG = 2.0
    DAILY_RATE_MAX_KG = 2.0
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional config overrides."""
        self.config = config or {}
        self.weights = self.config.get('component_weights', self.COMPONENT_WEIGHTS)
        self.threshold = self.config.get('threshold', 0.6)
        self.use_harmonic_mean = self.config.get('use_harmonic_mean', True)
    
    def calculate_quality_score(
        self,
        weight: float,
        source: str,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None,
        recent_weights: Optional[List[float]] = None,
        user_height_m: float = 1.67
    ) -> QualityScore:
        """
        Calculate overall quality score.
        
        Args:
            weight: Current measurement in kg
            source: Data source identifier
            previous_weight: Last accepted weight
            time_diff_hours: Hours since last measurement
            recent_weights: Optional recent weights (not required)
            user_height_m: User height for BMI calculation
            
        Returns:
            QualityScore with overall and component scores
        """
        components = {}
        
        # Safety check (physiological limits)
        components['safety'] = self._calculate_safety(weight, user_height_m)
        
        if components['safety'] < self.SAFETY_CRITICAL_THRESHOLD:
            return QualityScore(
                overall=0.0,
                components=components,
                threshold=self.threshold,
                rejection_reason=f"Safety score {components['safety']:.2f} below critical threshold"
            )
        
        # Other components
        components['plausibility'] = self._calculate_plausibility(
            weight, recent_weights, previous_weight
        )
        components['consistency'] = self._calculate_consistency(
            weight, previous_weight, time_diff_hours
        )
        components['reliability'] = self._calculate_reliability(source)
        
        # Combine scores
        if self.use_harmonic_mean:
            overall = self._harmonic_mean(components, self.weights)
        else:
            overall = self._arithmetic_mean(components, self.weights)
        
        return QualityScore(
            overall=overall,
            components=components,
            threshold=self.threshold,
            metadata={
                'weight': weight,
                'source': source,
                'previous_weight': previous_weight,
                'time_diff_hours': time_diff_hours
            }
        )
    
    def _calculate_safety(self, weight: float, height_m: float) -> float:
        """Check if weight is within physiological limits."""
        abs_min = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MIN_WEIGHT']
        abs_max = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MAX_WEIGHT']
        sus_min = PHYSIOLOGICAL_LIMITS['SUSPICIOUS_MIN_WEIGHT']
        sus_max = PHYSIOLOGICAL_LIMITS['SUSPICIOUS_MAX_WEIGHT']
        
        # Hard limits
        if weight < abs_min or weight > abs_max:
            return 0.0
        
        # Safe range
        if sus_min <= weight <= sus_max:
            return 1.0
        
        # Exponential penalty approaching limits
        if weight < sus_min:
            distance_ratio = (sus_min - weight) / (sus_min - abs_min)
        else:
            distance_ratio = (weight - sus_max) / (abs_max - sus_max)
        
        score = np.exp(-3 * distance_ratio)
        
        # BMI check
        bmi = weight / (height_m ** 2)
        if bmi < 15 or bmi > 60:
            score *= 0.5
        elif bmi < 18 or bmi > 40:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _calculate_plausibility(
        self,
        weight: float,
        recent_weights: Optional[List[float]],
        previous_weight: Optional[float]
    ) -> float:
        """Check statistical plausibility with trend awareness."""
        # Use recent weights if available
        if recent_weights and len(recent_weights) >= 3:
            recent_array = np.array(recent_weights[-20:])
            mean = np.mean(recent_array)
            std = np.std(recent_array)
            
            # Detect and account for trends
            if len(recent_array) >= 4:
                slope, r_squared = self._calculate_trend(list(recent_array))
                
                # If there's a clear trend (R² > 0.5), adjust expectations
                if r_squared > 0.5:
                    # Project trend forward
                    expected_next = recent_array[-1] + slope
                    
                    # Use projected value for mean if trend is strong
                    if r_squared > 0.8:
                        mean = expected_next
                    else:
                        # Blend projection with historical mean
                        mean = (r_squared * expected_next) + ((1 - r_squared) * mean)
                
                # Adjust minimum std for trending data
                if abs(slope) > 0.1 and r_squared > 0.5:
                    # For strong trends, allow more variation
                    min_std = max(1.0, abs(slope) * 3)
                else:
                    min_std = 0.5
            else:
                min_std = 0.5
            
            std = max(std, min_std)
            
        # Fall back to previous weight
        elif previous_weight is not None:
            mean = previous_weight
            # Assume 2% standard deviation for body weight
            baseline = (weight + previous_weight) / 2
            std = max(baseline * 0.02, 0.5)
            
        else:
            # No history available
            return 0.8
        
        # Calculate z-score
        z_score = abs(weight - mean) / std
        
        # Score based on deviation
        if z_score <= 1:
            return 1.0
        elif z_score <= 2:
            return 0.9
        elif z_score <= 3:
            return 0.7
        else:
            return max(0.0, min(0.5, np.exp(-0.5 * (z_score - 3))))
    
    def _calculate_consistency(
        self,
        weight: float,
        previous_weight: Optional[float],
        time_diff_hours: Optional[float]
    ) -> float:
        """Check consistency with previous measurement."""
        if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
            return 0.8
        
        weight_diff = abs(weight - previous_weight)
        
        # Time-based thresholds
        if time_diff_hours < 6:
            # Within 6 hours: very lenient
            if weight_diff <= self.HOURLY_MAX_KG:
                return 1.0
            else:
                excess = (weight_diff - self.HOURLY_MAX_KG) / self.HOURLY_MAX_KG
                return max(0.0, 0.7 * np.exp(-2 * excess))
                
        elif time_diff_hours < 24:
            # Within a day: moderate thresholds
            if weight_diff <= self.DAILY_MAX_KG:
                return 1.0
            elif weight_diff <= 4.0:
                return 1.0 - 0.1 * (weight_diff - self.DAILY_MAX_KG)
            else:
                # Use percentage for large changes
                baseline = (weight + previous_weight) / 2
                percent = (weight_diff / baseline) * 100
                if percent <= 5.0:
                    return 0.7
                else:
                    excess = (percent - 5.0) / 5.0
                    return max(0.0, 0.5 * np.exp(-2 * excess))
        
        else:
            # Longer periods: daily rate
            daily_rate = weight_diff / (time_diff_hours / 24)
            
            if daily_rate <= self.DAILY_RATE_MAX_KG:
                return 1.0
            elif daily_rate <= 4.0:
                return 1.0 - 0.25 * (daily_rate - self.DAILY_RATE_MAX_KG)
            elif daily_rate <= 6.44:  # Physiological max
                return 0.5 - 0.3 * ((daily_rate - 4.0) / 2.44)
            else:
                excess = (daily_rate - 6.44) / 6.44
                return max(0.0, 0.2 * np.exp(-2 * excess))
    
    def _calculate_reliability(self, source: str) -> float:
        """Score based on data source reliability."""
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        reliability = profile.get('reliability', 'unknown')
        
        # Base scores
        scores = {
            'excellent': 1.0,
            'good': 0.85,
            'moderate': 0.7,
            'poor': 0.5,
            'unknown': 0.6
        }
        
        base_score = scores.get(reliability, 0.6)
        
        # Adjust for outlier rate
        outlier_rate = profile.get('outlier_rate', 20.0)
        if outlier_rate < 5:
            multiplier = 1.0
        elif outlier_rate < 20:
            multiplier = 0.95
        elif outlier_rate < 50:
            multiplier = 0.9
        else:
            multiplier = 0.8
        
        return base_score * multiplier
    
    def _harmonic_mean(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted harmonic mean (penalizes low scores)."""
        total_weight = sum(weights.get(k, 0) for k in components)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = 0
        for key, score in components.items():
            weight = weights.get(key, 0)
            if score > 0:
                weighted_sum += weight / score
            else:
                return 0.0
        
        return total_weight / weighted_sum if weighted_sum > 0 else 0.0
    
    def _arithmetic_mean(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted arithmetic mean."""
        total_weight = sum(weights.get(k, 0) for k in components)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            components.get(k, 0) * weights.get(k, 0)
            for k in components
        )
        
        return weighted_sum / total_weight
    
    def _calculate_trend(self, weights: List[float]) -> tuple[float, float]:
        """Calculate linear trend in weights.
        
        Returns:
            (slope, r_squared) where slope is per measurement and r_squared is fit quality
        """
        if len(weights) < 2:
            return 0.0, 0.0
        
        x = np.arange(len(weights))
        y = np.array(weights)
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, r_squared
    
    def calculate_consistency_score(
        self,
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None
    ) -> float:
        """Public method for backward compatibility."""
        return self._calculate_consistency(weight, previous_weight, time_diff_hours)
    
    def calculate_plausibility_score(
        self,
        weight: float,
        recent_weights: Optional[List[float]] = None
    ) -> float:
        """Public method for backward compatibility."""
        return self._calculate_plausibility(weight, recent_weights, None)
    
    def calculate_safety_score(self, weight: float, height_m: float) -> float:
        """Public method for backward compatibility."""
        return self._calculate_safety(weight, height_m)
    
    def calculate_reliability_score(self, source: str) -> float:
        """Public method for backward compatibility."""
        return self._calculate_reliability(source)
    
    def explain_score(self, quality_score: QualityScore) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Quality Score: {quality_score.overall:.2f}/{quality_score.threshold:.1f}",
            f"Status: {'ACCEPTED' if quality_score.accepted else 'REJECTED'}",
            "",
            "Component Scores:"
        ]
        
        for component, score in quality_score.components.items():
            status = "✓" if score >= 0.7 else "⚠" if score >= 0.4 else "✗"
            lines.append(f"  {status} {component.capitalize()}: {score:.2f}")
        
        if quality_score.rejection_reason:
            lines.append("")
            lines.append(f"Rejection Reason: {quality_score.rejection_reason}")
        
        return "\n".join(lines)
    
    def _weighted_harmonic_mean(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Backward compatibility alias for tests."""
        return self._harmonic_mean(components, weights)
    
    def _weighted_arithmetic_mean(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Backward compatibility alias for tests."""
        return self._arithmetic_mean(components, weights)


class MeasurementHistory:
    """
    Test utility for maintaining measurement history.
    NOT used in production (processor is stateless).
    """
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.weights: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
        self.quality_scores: deque = deque(maxlen=max_size)
    
    def add(self, weight: float, timestamp: datetime, quality_score: float):
        """Add a measurement to history."""
        self.weights.append(weight)
        self.timestamps.append(timestamp)
        self.quality_scores.append(quality_score)
    
    def get_recent_weights(self, min_quality: float = 0.6) -> List[float]:
        """Get recent weights above quality threshold."""
        return [
            w for w, q in zip(self.weights, self.quality_scores)
            if q >= min_quality
        ]
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics for recent measurements."""
        if not self.weights:
            return {}
        
        weights_array = np.array(list(self.weights))
        return {
            'mean': np.mean(weights_array),
            'std': np.std(weights_array),
            'median': np.median(weights_array),
            'min': np.min(weights_array),
            'max': np.max(weights_array),
            'count': len(weights_array)
        }

    def _weighted_harmonic_mean(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Public alias for backward compatibility with tests."""
        return self._harmonic_mean(components, weights)
    
    def _weighted_arithmetic_mean(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Public alias for backward compatibility with tests."""
        return self._arithmetic_mean(components, weights)
