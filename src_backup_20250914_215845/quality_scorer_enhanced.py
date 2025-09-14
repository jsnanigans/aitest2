"""
Enhanced Quality Scoring System incorporating research insights.
Based on clinical research showing:
- Daily fluctuations of 1-2 kg (2-3% of body weight) are normal
- Weekly patterns with 0.35% variation
- Dynamic, personalized thresholds are essential
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
class EnhancedQualityScore:
    """Enhanced quality score with research-based components."""
    
    overall: float
    components: Dict[str, float]
    threshold: float = 0.6
    accepted: bool = False
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
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
            'metadata': self.metadata,
            'confidence_interval': self.confidence_interval
        }


class EnhancedQualityScorer:
    """
    Enhanced quality scorer incorporating research findings:
    - Percentage-based thresholds that scale with body weight
    - Time-aware consistency checks
    - Weekly pattern recognition
    - Adaptive thresholds based on user history
    """
    
    # Research-based constants
    DAILY_FLUCTUATION_PERCENT = 2.5  # 2-3% daily variation is normal
    WEEKLY_VARIATION_PERCENT = 0.35  # Per day within a week
    HOURLY_VARIATION_PERCENT = 1.5   # Within 6 hours
    
    # Component weights (adjusted based on research)
    COMPONENT_WEIGHTS = {
        'safety': 0.30,      # Reduced from 0.35
        'plausibility': 0.25,
        'consistency': 0.30,  # Increased from 0.25
        'reliability': 0.15
    }
    
    SAFETY_CRITICAL_THRESHOLD = 0.3
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced quality scorer."""
        self.config = config or {}
        self.weights = self.config.get('component_weights', self.COMPONENT_WEIGHTS)
        self.threshold = self.config.get('threshold', 0.6)
        self.use_harmonic_mean = self.config.get('use_harmonic_mean', True)
        self.enable_adaptive = self.config.get('enable_adaptive', True)
    
    def calculate_quality_score(
        self,
        weight: float,
        source: str,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None,
        recent_weights: Optional[List[float]] = None,
        user_height_m: float = 1.67,
        timestamp: Optional[datetime] = None
    ) -> EnhancedQualityScore:
        """
        Calculate enhanced quality score with research-based improvements.
        """
        components = {}
        
        # Estimate baseline weight for percentage calculations
        baseline_weight = self._estimate_baseline_weight(
            weight, previous_weight, recent_weights
        )
        
        # Calculate component scores
        components['safety'] = self.calculate_safety_score(weight, user_height_m)
        
        if components['safety'] < self.SAFETY_CRITICAL_THRESHOLD:
            return EnhancedQualityScore(
                overall=0.0,
                components=components,
                threshold=self.threshold,
                rejection_reason=f"Safety score {components['safety']:.2f} below critical threshold"
            )
        
        components['plausibility'] = self.calculate_plausibility_score_enhanced(
            weight, recent_weights, baseline_weight
        )
        
        components['consistency'] = self.calculate_consistency_score_enhanced(
            weight, previous_weight, time_diff_hours, baseline_weight
        )
        
        components['reliability'] = self.calculate_reliability_score(source)
        
        # Apply weekly pattern adjustment if timestamp provided
        if timestamp and self.enable_adaptive:
            weekly_adjustment = self._get_weekly_pattern_adjustment(timestamp)
            components['consistency'] = min(1.0, components['consistency'] * weekly_adjustment)
        
        # Calculate overall score
        if self.use_harmonic_mean:
            overall = self._weighted_harmonic_mean(components, self.weights)
        else:
            overall = self._weighted_arithmetic_mean(components, self.weights)
        
        # Calculate confidence interval if we have enough data
        confidence_interval = None
        if recent_weights and len(recent_weights) >= 5:
            confidence_interval = self._calculate_confidence_interval(
                baseline_weight, recent_weights
            )
        
        return EnhancedQualityScore(
            overall=overall,
            components=components,
            threshold=self.threshold,
            metadata={
                'weight': weight,
                'source': source,
                'previous_weight': previous_weight,
                'time_diff_hours': time_diff_hours,
                'baseline_weight': baseline_weight,
                'timestamp': timestamp.isoformat() if timestamp else None
            },
            confidence_interval=confidence_interval
        )
    
    def _estimate_baseline_weight(
        self,
        current: float,
        previous: Optional[float],
        recent: Optional[List[float]]
    ) -> float:
        """Estimate baseline weight for percentage calculations."""
        if recent and len(recent) >= 5:
            # Use robust median of recent weights
            return np.median(recent[-10:])
        elif previous is not None:
            # Use average of current and previous
            return (current + previous) / 2
        else:
            # Use current weight as fallback
            return current
    
    def calculate_safety_score(self, weight: float, height_m: float) -> float:
        """Calculate safety score based on physiological limits."""
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
        
        # BMI check
        bmi = weight / (height_m ** 2)
        if bmi < 15 or bmi > 60:
            score *= 0.5
        elif bmi < 18 or bmi > 40:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def calculate_plausibility_score_enhanced(
        self,
        weight: float,
        recent_weights: Optional[List[float]],
        baseline_weight: float
    ) -> float:
        """
        Enhanced plausibility scoring using MAD (Median Absolute Deviation)
        as recommended in the research for robustness.
        """
        if not recent_weights or len(recent_weights) < 3:
            return 0.8
        
        recent_array = np.array(recent_weights[-20:])
        median_weight = np.median(recent_array)
        
        # Use MAD for robust variance estimation (research-recommended)
        mad = np.median(np.abs(recent_array - median_weight))
        # Scale factor for normal distribution
        robust_std = 1.4826 * mad if mad > 0 else 0.5
        
        # Calculate deviation
        deviation = abs(weight - median_weight)
        
        # Use percentage-based thresholds
        deviation_percent = (deviation / baseline_weight) * 100
        
        if deviation_percent <= 2.0:  # Within 2% is highly plausible
            return 1.0
        elif deviation_percent <= 3.0:  # 2-3% is normal daily variation
            return 0.9
        elif deviation_percent <= 5.0:  # Up to 5% is possible
            return 0.7
        else:
            # Use robust z-score for extreme deviations
            z_score = deviation / robust_std
            score = np.exp(-0.5 * (z_score - 3))
            return max(0.0, min(0.5, score))
    
    def calculate_consistency_score_enhanced(
        self,
        weight: float,
        previous_weight: Optional[float],
        time_diff_hours: Optional[float],
        baseline_weight: float
    ) -> float:
        """
        Enhanced consistency scoring based on research findings about
        physiological weight fluctuation patterns.
        """
        if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
            return 0.8
        
        weight_diff = abs(weight - previous_weight)
        weight_diff_percent = (weight_diff / baseline_weight) * 100
        
        # Time-aware thresholds based on research
        if time_diff_hours < 6:
            # Within 6 hours: up to 1.5% is normal (meals, hydration)
            typical_percent = 0.75
            max_percent = self.HOURLY_VARIATION_PERCENT
            
            if weight_diff_percent <= typical_percent:
                return 1.0
            elif weight_diff_percent <= max_percent:
                ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
                return 1.0 - (0.2 * ratio)  # Gentle penalty
            else:
                excess = (weight_diff_percent - max_percent) / max_percent
                return 0.8 * np.exp(-2 * excess)
                
        elif time_diff_hours < 24:
            # Within a day: 2-3% variation is documented as normal
            hours_ratio = time_diff_hours / 24
            typical_percent = 1.0 + (self.DAILY_FLUCTUATION_PERCENT - 1.0) * hours_ratio
            max_percent = typical_percent * 1.5
            
            if weight_diff_percent <= typical_percent:
                return 1.0
            elif weight_diff_percent <= max_percent:
                ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
                return 1.0 - (0.3 * ratio)
            else:
                excess = (weight_diff_percent - max_percent) / max_percent
                return 0.7 * np.exp(-2 * excess)
                
        elif time_diff_hours < 168:  # Within a week
            # Weekly variation: ~0.35% per day
            days = time_diff_hours / 24
            typical_percent = self.WEEKLY_VARIATION_PERCENT * days
            max_percent = typical_percent + self.DAILY_FLUCTUATION_PERCENT
            
            if weight_diff_percent <= typical_percent:
                return 1.0
            elif weight_diff_percent <= max_percent:
                ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
                return 1.0 - (0.4 * ratio)
            else:
                excess = (weight_diff_percent - max_percent) / max_percent
                return 0.6 * np.exp(-2 * excess)
                
        else:
            # Long-term: use conservative daily rate
            days = time_diff_hours / 24
            daily_rate_percent = weight_diff_percent / days
            
            if daily_rate_percent <= 0.2:  # Gradual change
                return 1.0
            elif daily_rate_percent <= self.WEEKLY_VARIATION_PERCENT:
                ratio = (daily_rate_percent - 0.2) / (self.WEEKLY_VARIATION_PERCENT - 0.2)
                return 1.0 - (0.5 * ratio)
            else:
                excess = (daily_rate_percent - self.WEEKLY_VARIATION_PERCENT) / self.WEEKLY_VARIATION_PERCENT
                return 0.5 * np.exp(-2 * excess)
    
    def calculate_reliability_score(self, source: str) -> float:
        """Calculate reliability score based on source."""
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
        
        # Adjust based on outlier rate
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
    
    def _get_weekly_pattern_adjustment(self, timestamp: datetime) -> float:
        """
        Adjust consistency score based on weekly patterns.
        Research shows weight peaks on Sunday/Monday.
        """
        day_of_week = timestamp.weekday()
        
        # Monday (0) and Sunday (6) have higher expected variation
        if day_of_week in [0, 6]:
            return 1.1  # Allow 10% more variation
        elif day_of_week in [5]:  # Saturday
            return 1.05  # Allow 5% more variation
        else:
            return 1.0  # Weekdays have normal variation
    
    def _calculate_confidence_interval(
        self,
        baseline: float,
        recent_weights: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for expected weight range.
        Uses robust statistics (MAD) as recommended in research.
        """
        recent_array = np.array(recent_weights[-20:])
        median_weight = np.median(recent_array)
        
        # Use MAD for robust estimation
        mad = np.median(np.abs(recent_array - median_weight))
        robust_std = 1.4826 * mad if mad > 0 else 0.5
        
        # Z-score for confidence level
        z_score = 1.96 if confidence_level == 0.95 else 2.58
        
        # Calculate interval
        margin = z_score * robust_std
        lower = median_weight - margin
        upper = median_weight + margin
        
        return (lower, upper)
    
    def _weighted_harmonic_mean(
        self,
        components: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted harmonic mean."""
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
        """Calculate weighted arithmetic mean."""
        total_weight = sum(weights.get(k, 0) for k in components)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            components.get(k, 0) * weights.get(k, 0)
            for k in components
        )
        
        return weighted_sum / total_weight
    
    def explain_score(self, quality_score: EnhancedQualityScore) -> str:
        """Generate human-readable explanation of the quality score."""
        lines = [
            f"Quality Score: {quality_score.overall:.2f}/{quality_score.threshold:.1f}",
            f"Status: {'ACCEPTED' if quality_score.accepted else 'REJECTED'}",
            "",
            "Component Scores:"
        ]
        
        for component, score in quality_score.components.items():
            status = "✓" if score >= 0.7 else "⚠" if score >= 0.4 else "✗"
            lines.append(f"  {status} {component.capitalize()}: {score:.2f}")
        
        if quality_score.confidence_interval:
            lower, upper = quality_score.confidence_interval
            lines.append("")
            lines.append(f"Expected Range: {lower:.1f} - {upper:.1f} kg")
        
        if quality_score.rejection_reason:
            lines.append("")
            lines.append(f"Rejection Reason: {quality_score.rejection_reason}")
        
        return "\n".join(lines)
