"""
Unified Quality Scoring System for weight measurements.
Combines multiple validation checks into a single quality score.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from collections import deque

try:
    from .constants import (
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        get_source_reliability
    )
except ImportError:
    from constants import (
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        get_source_reliability
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
            self.rejection_reason = f"Quality score {self.overall:.2f} below threshold {self.threshold} (weakest: {min_component[0]}={min_component[1]:.2f})"
    
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
    """Calculates unified quality scores for weight measurements."""
    
    COMPONENT_WEIGHTS = {
        'safety': 0.35,
        'plausibility': 0.25,
        'consistency': 0.25,
        'reliability': 0.15
    }
    
    SAFETY_CRITICAL_THRESHOLD = 0.3
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the quality scorer.
        
        Args:
            config: Optional configuration overrides
        """
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
        Calculate overall quality score for a weight measurement.
        
        Args:
            weight: The weight measurement in kg
            source: Data source identifier
            previous_weight: Previous weight measurement
            time_diff_hours: Hours since previous measurement
            recent_weights: List of recent accepted weights for statistics
            user_height_m: User's height in meters
            
        Returns:
            QualityScore object with overall and component scores
        """
        components = {}
        
        components['safety'] = self.calculate_safety_score(weight, user_height_m)
        
        if components['safety'] < self.SAFETY_CRITICAL_THRESHOLD:
            return QualityScore(
                overall=0.0,
                components=components,
                threshold=self.threshold,
                rejection_reason=f"Safety score {components['safety']:.2f} below critical threshold"
            )
        
        components['plausibility'] = self.calculate_plausibility_score(
            weight, recent_weights
        )
        
        components['consistency'] = self.calculate_consistency_score(
            weight, previous_weight, time_diff_hours
        )
        
        components['reliability'] = self.calculate_reliability_score(source)
        
        if self.use_harmonic_mean:
            overall = self._weighted_harmonic_mean(components, self.weights)
        else:
            overall = self._weighted_arithmetic_mean(components, self.weights)
        
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
    
    def calculate_safety_score(self, weight: float, height_m: float) -> float:
        """
        Calculate safety score based on physiological limits.
        
        Uses exponential penalty for values approaching limits.
        
        Args:
            weight: Weight in kg
            height_m: Height in meters
            
        Returns:
            Score from 0.0 (unsafe) to 1.0 (safe)
        """
        abs_min = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MIN_WEIGHT']
        abs_max = PHYSIOLOGICAL_LIMITS['ABSOLUTE_MAX_WEIGHT']
        sus_min = PHYSIOLOGICAL_LIMITS['SUSPICIOUS_MIN_WEIGHT']
        sus_max = PHYSIOLOGICAL_LIMITS['SUSPICIOUS_MAX_WEIGHT']
        
        if weight < abs_min or weight > abs_max:
            return 0.0
        
        if sus_min <= weight <= sus_max:
            return 1.0
        
        if weight < sus_min:
            distance_ratio = (sus_min - weight) / (sus_min - abs_min)
        else:
            distance_ratio = (weight - sus_max) / (abs_max - sus_max)
        
        score = np.exp(-3 * distance_ratio)
        
        bmi = weight / (height_m ** 2)
        if bmi < 15 or bmi > 60:
            score *= 0.5
        elif bmi < 18 or bmi > 40:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def calculate_plausibility_score(
        self,
        weight: float,
        recent_weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate plausibility based on statistical deviation.
        
        Uses z-score with Gaussian decay.
        
        Args:
            weight: Current weight measurement
            recent_weights: List of recent accepted weights
            
        Returns:
            Score from 0.0 (implausible) to 1.0 (plausible)
        """
        if not recent_weights or len(recent_weights) < 3:
            return 0.8
        
        recent_array = np.array(recent_weights[-20:])
        mean = np.mean(recent_array)
        std = np.std(recent_array)
        
        if std < 0.5:
            std = 0.5
        
        z_score = abs(weight - mean) / std
        
        if z_score <= 1:
            return 1.0
        elif z_score <= 2:
            return 0.9
        elif z_score <= 3:
            return 0.7
        else:
            score = np.exp(-0.5 * (z_score - 3))
            return max(0.0, min(0.5, score))
    
    def calculate_consistency_score(
        self,
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None
    ) -> float:
        """
        Calculate consistency based on rate of change.
        
        Args:
            weight: Current weight
            previous_weight: Previous weight
            time_diff_hours: Time since previous measurement
            
        Returns:
            Score from 0.0 (inconsistent) to 1.0 (consistent)
        """
        if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
            return 0.8
        
        weight_diff = abs(weight - previous_weight)
        daily_rate = (weight_diff / time_diff_hours) * 24
        
        max_daily = PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']
        typical_daily = PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']
        
        if daily_rate <= typical_daily:
            return 1.0
        elif daily_rate <= max_daily:
            ratio = (daily_rate - typical_daily) / (max_daily - typical_daily)
            return 1.0 - (0.5 * ratio)
        else:
            excess_ratio = (daily_rate - max_daily) / max_daily
            score = 0.5 * np.exp(-2 * excess_ratio)
            return max(0.0, score)
    
    def calculate_reliability_score(self, source: str) -> float:
        """
        Calculate reliability score based on source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Score from 0.0 (unreliable) to 1.0 (reliable)
        """
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        reliability = profile.get('reliability', 'unknown')
        
        reliability_scores = {
            'excellent': 1.0,
            'good': 0.85,
            'moderate': 0.7,
            'poor': 0.5,
            'unknown': 0.6
        }
        
        base_score = reliability_scores.get(reliability, 0.6)
        
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
    
    def _weighted_harmonic_mean(
        self,
        components: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted harmonic mean of component scores.
        
        Harmonic mean penalizes low scores more than arithmetic mean.
        
        Args:
            components: Component scores
            weights: Component weights
            
        Returns:
            Weighted harmonic mean
        """
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
        
        if weighted_sum == 0:
            return 0.0
        
        return total_weight / weighted_sum
    
    def _weighted_arithmetic_mean(
        self,
        components: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted arithmetic mean of component scores.
        
        Args:
            components: Component scores
            weights: Component weights
            
        Returns:
            Weighted arithmetic mean
        """
        total_weight = sum(weights.get(k, 0) for k in components)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            components.get(k, 0) * weights.get(k, 0)
            for k in components
        )
        
        return weighted_sum / total_weight
    
    def explain_score(self, quality_score: QualityScore) -> str:
        """
        Generate human-readable explanation of the quality score.
        
        Args:
            quality_score: QualityScore object
            
        Returns:
            Explanation string
        """
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


class MeasurementHistory:
    """Maintains rolling window of recent measurements for statistical analysis."""
    
    def __init__(self, max_size: int = 20):
        """
        Initialize measurement history.
        
        Args:
            max_size: Maximum number of measurements to keep
        """
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
        """Get recent weights above minimum quality threshold."""
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