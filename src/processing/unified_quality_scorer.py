"""
Unified Kalman-centric quality scoring system.
Replaces dual validation with single Kalman-deviation-based quality scorer.
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


try:
    from ..constants import (
        BMI_LIMITS,
        DEFAULT_PROFILE,
        KALMAN_DEFAULTS,
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
    )
except ImportError:
    from src.constants import (
        BMI_LIMITS,
        DEFAULT_PROFILE,
        KALMAN_DEFAULTS,
        PHYSIOLOGICAL_LIMITS,
        SOURCE_PROFILES,
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
            "overall": self.overall,
            "components": self.components,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


class UnifiedQualityScorer:
    """
    Unified Kalman-centric quality scoring system.
    Primary signal is deviation from Kalman prediction.
    """

    # Default component weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "kalman_fit": 0.40,  # Primary signal
        "temporal_consistency": 0.20,
        "anomaly_detection": 0.20,
        "source_reliability": 0.10,
        "trend_alignment": 0.10,
    }

    # Time-based thresholds for temporal consistency
    TEMPORAL_THRESHOLDS = {
        "6h": 3.0,  # 3kg in 6 hours
        "24h": 2.0,  # 2kg in 24 hours
        "sustained": 2.0,  # 2kg/day sustained
    }

    # Anomaly patterns
    UNIT_CONFUSION_FACTORS = [2.2, 0.454, 10.0, 0.1]  # kg/lbs, lbs/kg, decimal errors
    BMI_RANGE = (15.0, 50.0)  # Common BMI range that might be entered as weight

    # Rapid measurement detection thresholds
    DUPLICATE_THRESHOLD_SECONDS = 5  # Only reject if < 5 seconds (true duplicates)
    RAPID_THRESHOLD_MINUTES = 5  # Measurements within 5 minutes need special handling
    BURST_WINDOW_MINUTES = 30  # Window to detect burst patterns
    BURST_COUNT_THRESHOLD = 5  # Increased from 3 to be less aggressive
    MAX_1MIN_CHANGE_KG = 0.5  # Increased from 0.1 to allow scale variance
    MAX_5MIN_CHANGE_KG = 1.0  # Increased from 0.3 to allow water/bathroom

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional config overrides."""
        self.config = config or {}

        # Get component weights from config
        self.weights = self.config.get("component_weights", self.DEFAULT_WEIGHTS.copy())

        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

        # Get thresholds
        self.threshold = self.config.get("threshold", 0.6)
        self.temporal_thresholds = self.config.get(
            "temporal_thresholds", self.TEMPORAL_THRESHOLDS.copy()
        )


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
        user_height_m: Optional[float] = None,
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

        # Store current source for use in anomaly detection
        self.current_source = source

        # Only calculate components with non-zero weights
        # 1. Kalman Fit Component
        if self.weights.get("kalman_fit", 0) > 0:
            kalman_score, kalman_meta = self.calculate_kalman_fit(
                weight, kalman_prediction, innovation_covariance, kalman_state
            )
            components["kalman_fit"] = kalman_score
            metadata["kalman_fit"] = kalman_meta

        # 2. Temporal Consistency
        if self.weights.get("temporal_consistency", 0) > 0:
            temporal_score, temporal_meta = self.calculate_temporal_consistency(
                weight,
                previous_weight,
                time_diff_hours,
                recent_weights,
                recent_timestamps,
            )
            components["temporal_consistency"] = temporal_score
            metadata["temporal_consistency"] = temporal_meta

        # 3. Anomaly Detection
        if self.weights.get("anomaly_detection", 0) > 0:
            # Try to get current timestamp from kalman_state or recent data
            current_ts = None
            if kalman_state and "current_timestamp" in kalman_state:
                current_ts = kalman_state["current_timestamp"]

            anomaly_score, anomaly_meta = self.calculate_anomaly_detection(
                weight, recent_weights, recent_timestamps, user_height_m, current_ts
            )
            components["anomaly_detection"] = anomaly_score
            metadata["anomaly_detection"] = anomaly_meta

        # 4. Source Reliability - skip if weight is 0
        if self.weights.get("source_reliability", 0) > 0:
            source_score = self.calculate_source_reliability(source)
            components["source_reliability"] = source_score
            metadata["source_reliability"] = {"source": source, "score": source_score}

        # 5. Trend Alignment - skip if weight is 0
        if self.weights.get("trend_alignment", 0) > 0:
            trend_score, trend_meta = self.calculate_trend_alignment(
                weight, kalman_state, recent_weights
            )
            components["trend_alignment"] = trend_score
            metadata["trend_alignment"] = trend_meta

        # Calculate overall score using configured mean type
        use_harmonic = self.config.get("use_harmonic_mean", False)
        if use_harmonic:
            overall = self._calculate_weighted_harmonic_mean(components)
        else:
            overall = self._calculate_weighted_geometric_mean(components)

        return QualityScore(
            overall=overall,
            components=components,
            threshold=self.threshold,
            metadata=metadata,
        )

    def calculate_kalman_fit(
        self,
        weight: float,
        kalman_prediction: Optional[float],
        innovation_covariance: Optional[float],
        kalman_state: Optional[Dict],
    ) -> Tuple[float, Dict]:
        """
        Calculate how well measurement fits Kalman prediction.
        Uses Mahalanobis distance and chi-squared test.
        Applies time-based decay: importance decreases over time since last measurement.
        """
        metadata = {}

        # If no Kalman prediction available, return neutral score
        if kalman_prediction is None or innovation_covariance is None:
            metadata["reason"] = "No Kalman prediction available"
            return 0.5, metadata

        # Calculate innovation (prediction error)
        innovation = weight - kalman_prediction
        metadata["innovation"] = innovation
        metadata["prediction"] = kalman_prediction

        # Handle zero or very small covariance
        if innovation_covariance <= 0:
            innovation_covariance = 1.0

        # Normalize innovation (Mahalanobis distance)
        normalized_innovation = abs(innovation) / np.sqrt(innovation_covariance)
        metadata["normalized_innovation"] = normalized_innovation

        # Chi-squared test (df=1 for univariate)
        chi_squared = normalized_innovation**2
        p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
        metadata["chi_squared"] = chi_squared
        metadata["p_value"] = p_value

        # Check for adaptive period (relax thresholds)
        in_adaptive_period = False
        if kalman_state:
            measurements_since_reset = kalman_state.get("measurements_since_reset", 100)
            reset_params = kalman_state.get("reset_parameters", {})
            adaptation_measurements = reset_params.get("adaptation_measurements", 10)
            if measurements_since_reset < adaptation_measurements:
                in_adaptive_period = True
                metadata["adaptive_period"] = True

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
        if kalman_state and "last_timestamp" in kalman_state:
            last_timestamp = kalman_state["last_timestamp"]
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            # Get current timestamp from state or use now as fallback
            current_timestamp = kalman_state.get("current_timestamp", datetime.now())
            if isinstance(current_timestamp, str):
                current_timestamp = datetime.fromisoformat(current_timestamp)
            days_since_last = (
                current_timestamp - last_timestamp
            ).total_seconds() / 86400.0
            metadata["days_since_last"] = days_since_last

        # Apply decay factor based on time gap
        # Linear decay: at 30 days, Kalman fit doesn't matter (score approaches 1.0)
        # Formula: final_score = score + (1 - score) * min(1, days/30)
        if days_since_last > 0:
            decay_factor = min(1.0, days_since_last / 30.0)  # Linear decay over 30 days
            # Blend towards 1.0 (full acceptance) as time increases
            adjusted_score = score + (1.0 - score) * decay_factor
            metadata["decay_factor"] = decay_factor
            metadata["original_score"] = score
            score = adjusted_score

        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))
        metadata["score"] = score

        return score, metadata

    def calculate_temporal_consistency(
        self,
        weight: float,
        previous_weight: Optional[float],
        time_diff_hours: Optional[float],
        recent_weights: Optional[List[float]],
        recent_timestamps: Optional[List[datetime]],
    ) -> Tuple[float, Dict]:
        """
        Calculate temporal consistency using continuous exponential function.
        Eliminates step functions that cause artificial cycles.
        """
        metadata = {}

        # If no previous weight, return neutral score
        if previous_weight is None or time_diff_hours is None:
            metadata["reason"] = "No previous weight for comparison"
            return 0.7, metadata

        weight_change = abs(weight - previous_weight)

        # Exponential growth of acceptable change over time
        # Starts at 0.5kg for immediate, grows to ~5kg at 7 days
        max_acceptable_change = 0.5 + 4.5 * (1 - math.exp(-time_diff_hours / 48))

        metadata["max_acceptable_change"] = max_acceptable_change
        metadata["actual_change"] = weight_change
        metadata["time_diff_hours"] = time_diff_hours

        # Smooth scoring based on deviation from acceptable
        if weight_change <= max_acceptable_change:
            # Within acceptable range: high score with smooth decay
            score = 0.8 + 0.2 * math.exp(-weight_change / max_acceptable_change)
        else:
            # Beyond acceptable: exponential penalty
            excess_ratio = (
                weight_change - max_acceptable_change
            ) / max_acceptable_change
            score = 0.8 * math.exp(-excess_ratio)

        # Check for adaptive period from kalman state (more lenient during adaptation)
        # This maintains backward compatibility with existing adaptive period handling
        if time_diff_hours > 168:  # More than a week gap
            score = max(score, 0.4)
            metadata["gap_adjustment"] = True

        # Clamp between 0.2 and 1.0
        score = max(0.2, min(1.0, score))

        return score, metadata

    def calculate_anomaly_detection(
        self,
        weight: float,
        recent_weights: Optional[List[float]],
        recent_timestamps: Optional[List[datetime]],
        user_height_m: Optional[float],
        current_timestamp: Optional[datetime] = None,
    ) -> Tuple[float, Dict]:
        """
        Enhanced anomaly detection with time-aware physiological limits.
        - Short-term fluctuations (hours): water/food intake, exercise
        - Medium-term changes (days): diet, illness, medication
        - Long-term trends (weeks): sustainable weight loss/gain
        """
        metadata = {}
        score = 1.0

        # 1. Check absolute physiological bounds
        if weight < PHYSIOLOGICAL_LIMITS["ABSOLUTE_MIN_WEIGHT"]:
            metadata["outside_absolute_min"] = True
            return 0.0, metadata  # Reject outright

        if weight > PHYSIOLOGICAL_LIMITS["ABSOLUTE_MAX_WEIGHT"]:
            metadata["outside_absolute_max"] = True
            return 0.0, metadata  # Reject outright

        # Check suspicious bounds (softer penalty)
        if weight < PHYSIOLOGICAL_LIMITS["SUSPICIOUS_MIN_WEIGHT"]:
            metadata["below_suspicious_min"] = True
            score *= 0.3
        elif weight > PHYSIOLOGICAL_LIMITS["SUSPICIOUS_MAX_WEIGHT"]:
            metadata["above_suspicious_max"] = True
            score *= 0.3

        # 2. Time-aware change detection
        if recent_weights and recent_timestamps:
            # Ensure we have matching lengths
            min_len = min(len(recent_weights), len(recent_timestamps))
            recent_weights = recent_weights[-min_len:]
            recent_timestamps = recent_timestamps[-min_len:]

            if len(recent_weights) > 0:
                previous_weight = float(recent_weights[-1])
                weight_change = abs(weight - previous_weight)

                # Calculate time difference
                if len(recent_timestamps) >= 1:
                    # Use provided timestamp or fall back to now (for real-time processing)
                    if current_timestamp is None:
                        current_timestamp = datetime.now()
                    elif isinstance(current_timestamp, str):
                        current_timestamp = datetime.fromisoformat(current_timestamp)

                    prev_timestamp = recent_timestamps[-1]
                    if isinstance(prev_timestamp, str):
                        prev_timestamp = datetime.fromisoformat(prev_timestamp)

                    # Calculate time differences
                    time_diff_seconds = (current_timestamp - prev_timestamp).total_seconds()
                    time_diff_minutes = time_diff_seconds / 60.0
                    time_diff_hours = time_diff_seconds / 3600.0

                    # Check for minute-level precision (likely from manual entry)
                    has_minute_precision = (
                        current_timestamp.second == 0
                        and current_timestamp.microsecond == 0
                    ) and (
                        prev_timestamp.second == 0 and prev_timestamp.microsecond == 0
                    )

                    # Enhanced rapid-fire measurement detection
                    # Only reject true duplicates (same weight within 5 seconds)
                    if time_diff_seconds < self.DUPLICATE_THRESHOLD_SECONDS:
                        # Check if weight is essentially the same (within 50g)
                        if weight_change < 0.05:
                            metadata["rejected_reason"] = "duplicate_measurement"
                            metadata["time_diff_seconds"] = time_diff_seconds
                            metadata["threshold_seconds"] = self.DUPLICATE_THRESHOLD_SECONDS
                            return 0.0, metadata  # Reject as duplicate
                        # Allow small variations (scale noise) even in rapid succession
                        elif weight_change < 0.2:
                            score *= 0.8  # Minor penalty for rapid but different reading
                            metadata["rapid_but_different"] = True

                    elif time_diff_minutes < self.RAPID_THRESHOLD_MINUTES:
                        # Calculate adaptive threshold based on time and source
                        # More lenient for device measurements (scale variance)
                        source_factor = 1.0
                        if hasattr(self, 'current_source'):
                            if 'device' in self.current_source.lower():
                                source_factor = 1.5  # 50% more lenient for devices
                            elif 'manual' in self.current_source.lower() or 'upload' in self.current_source.lower():
                                source_factor = 1.2  # 20% more lenient for manual

                        # Smooth exponential growth of allowed change
                        # Starts at 0.5kg at 0 min, grows to 1.0kg at 5 min
                        max_allowed = 0.5 + 0.5 * (1 - math.exp(-time_diff_minutes / 2))
                        max_allowed *= source_factor

                        if weight_change > max_allowed * 2:  # Only reject if WAY over (2x)
                            metadata["rejected_reason"] = "rapid_impossible_change"
                            metadata["time_diff_minutes"] = time_diff_minutes
                            metadata["change_kg"] = weight_change
                            metadata["max_allowed_change"] = max_allowed
                            return 0.0, metadata  # Reject as impossible

                        elif weight_change > max_allowed:
                            # Over threshold but not impossible - apply gradual penalty
                            excess_ratio = (weight_change - max_allowed) / max_allowed
                            rapid_penalty = math.exp(-excess_ratio)  # Smoother penalty
                            score *= rapid_penalty
                            metadata["rapid_measurement_penalty"] = rapid_penalty
                            metadata["time_diff_minutes"] = time_diff_minutes
                            metadata["exceeded_soft_threshold"] = True
                        else:
                            # Within acceptable range for short-term change
                            # Small penalty that decreases as time increases
                            time_penalty = 0.9 + 0.1 * (time_diff_minutes / self.RAPID_THRESHOLD_MINUTES)
                            score *= time_penalty
                            metadata["minor_time_penalty"] = time_penalty

                    # Additional check: Look for burst patterns (multiple measurements in short period)
                    if len(recent_timestamps) >= self.BURST_COUNT_THRESHOLD:
                        # Check if we have multiple measurements within burst window
                        burst_count = 1  # Start with current measurement
                        for ts in recent_timestamps[-(self.BURST_COUNT_THRESHOLD + 2):]:  # Look at recent measurements
                            if isinstance(ts, str):
                                ts = datetime.fromisoformat(ts)
                            if (current_timestamp - ts).total_seconds() / 60.0 <= self.BURST_WINDOW_MINUTES:
                                burst_count += 1

                        if burst_count >= self.BURST_COUNT_THRESHOLD:
                            # Multiple rapid measurements detected - could be intentional averaging
                            metadata["burst_pattern_detected"] = True
                            metadata["burst_count"] = burst_count
                            metadata["burst_window_minutes"] = self.BURST_WINDOW_MINUTES

                            # Less aggressive penalty - users often take multiple readings
                            # 5 measurements = 0.8, 6 = 0.7, 7+ = 0.6
                            burst_penalty = max(0.6, 1.0 - (burst_count - 4) * 0.1)
                            score *= burst_penalty
                            metadata["burst_penalty"] = burst_penalty

                    metadata["time_diff_hours"] = time_diff_hours

                    # Time-based physiological limits
                    max_change = self._calculate_max_physiological_change(
                        time_diff_hours
                    )
                    metadata["max_physiological_change"] = max_change
                    metadata["actual_change"] = weight_change

                    # Apply penalty based on deviation from max allowed
                    if weight_change > max_change:
                        # Calculate severity of violation
                        excess_ratio = (weight_change - max_change) / max_change
                        metadata["excess_ratio"] = excess_ratio

                        if excess_ratio > 1.0:  # More than double the max
                            score *= 0.0  # Impossible change
                            metadata["impossible_change"] = True
                        elif excess_ratio > 0.5:  # 50% over max
                            score *= 0.1  # Very unlikely
                            metadata["very_unlikely_change"] = True
                        else:
                            score *= 0.5 - excess_ratio * 0.4  # Gradual penalty
                            metadata["unlikely_change"] = True

                    # 3. Check for percentage-based changes (catch weight doubling etc.)
                    # Only apply percentage checks for periods > 3 days where percentage matters more
                    if time_diff_hours > 72 and time_diff_hours <= 720:  # Between 3-30 days
                        percent_change = (weight_change / previous_weight) * 100
                        max_monthly_percent = PHYSIOLOGICAL_LIMITS.get("MAX_MONTHLY_PERCENT", 15)

                        # Scale the allowed percentage based on actual time elapsed
                        # But with a minimum of 3 days worth to avoid being too strict on short periods
                        time_factor = max(0.1, min(1.0, time_diff_hours / 720))  # At least 3 days worth
                        allowed_percent = max_monthly_percent * time_factor

                        metadata["percent_change"] = percent_change
                        metadata["allowed_percent"] = allowed_percent

                        if percent_change > allowed_percent:
                            excess_percent_ratio = (percent_change - allowed_percent) / allowed_percent
                            metadata["excess_percent_ratio"] = excess_percent_ratio

                            if excess_percent_ratio > 2.0:  # More than 3x the allowed percentage
                                score *= 0.0
                                metadata["impossible_percent_change"] = True
                            elif excess_percent_ratio > 1.0:  # More than 2x the allowed percentage
                                score *= 0.05
                                metadata["extreme_percent_change"] = True
                            elif excess_percent_ratio > 0.5:  # More than 1.5x the allowed percentage
                                score *= 0.1
                                metadata["high_percent_change"] = True
                            else:
                                score *= max(0.2, 0.5 - excess_percent_ratio * 0.6)
                                metadata["suspicious_percent_change"] = True

                    # 4. Check for sustained vs. fluctuation patterns
                    if len(recent_weights) >= 3:
                        sustained_score = self._check_sustained_pattern(
                            weight, recent_weights, recent_timestamps
                        )
                        metadata["sustained_pattern_score"] = sustained_score
                        score *= sustained_score

        return max(0.0, min(1.0, score)), metadata

    def _calculate_max_physiological_change(self, time_hours: float) -> float:
        """
        Calculate maximum physiological weight change based on time elapsed.
        Uses strict, realistic limits to prevent accepting impossible changes.
        """
        if time_hours <= 0:
            return 0.0

        # Ultra short-term (< 1 minute): Scale variance + positioning
        if time_hours < 0.0167:  # 1 minute
            return PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_1MIN", 0.5)  # Increased from 0.1

        # Very short-term (< 5 minutes): Scale variance + water/bathroom
        elif time_hours < 0.0833:  # 5 minutes
            max_1min = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_1MIN", 0.5)  # Increased
            max_5min = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_5MIN", 1.0)  # Increased
            # Linear interpolation
            minutes = time_hours * 60
            return max_1min + (max_5min - max_1min) * (minutes - 1) / 4

        # Short-term (< 1 hour): Limited by water/food intake
        elif time_hours < 1:
            max_5min = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_5MIN", 0.3)
            max_1h = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_1H", 1.0)
            minutes = time_hours * 60
            # Use smooth curve from 5 min to 60 min
            # At 5 min: 0.3kg, at 60 min: 1.0kg
            if minutes <= 5:
                return max_5min
            else:
                # Logarithmic growth from 5 min to 1 hour
                return max_5min + (max_1h - max_5min) * math.log(minutes / 5) / math.log(12)

        # Hours (1-6 hours): Water, food, exercise effects
        elif time_hours <= 6:
            base_change = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_1H", 1.0)
            max_6h = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_6H", 3.0)
            # Smooth interpolation from 1h to 6h
            # At 2h: ~1.6kg, 3h: ~2.0kg, 4h: ~2.4kg, 6h: 3.0kg
            additional = (max_6h - base_change) * math.log(1 + (time_hours - 1)) / math.log(6)
            return base_change + additional

        # Day (6-24 hours): Full daily fluctuation
        elif time_hours <= 24:
            base_change = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_6H", 3.0)
            max_24h = PHYSIOLOGICAL_LIMITS.get("MAX_CHANGE_24H", 4.0)
            # Logarithmic growth
            additional = (max_24h - base_change) * math.log(1 + (time_hours - 6) / 6) / math.log(4)
            return base_change + additional

        # Week (1-7 days): Compound changes with realistic limits
        elif time_hours <= 168:  # 7 days
            days = time_hours / 24
            daily_max = PHYSIOLOGICAL_LIMITS.get("MAX_DAILY_CHANGE_KG", 2.0)
            weekly_max = PHYSIOLOGICAL_LIMITS.get("MAX_WEEKLY_CHANGE_KG", 3.5)
            # Use square root for realistic accumulation
            # This gives ~2.8kg for 2 days, ~3.5kg for 3 days, ~4kg for 4 days, capped at weekly max
            return min(weekly_max, daily_max * math.sqrt(days))

        # Long-term (> 1 week): Sustainable rates only
        else:
            days = time_hours / 24
            weekly_max = PHYSIOLOGICAL_LIMITS.get("MAX_WEEKLY_CHANGE_KG", 3.5)
            sustained_daily = PHYSIOLOGICAL_LIMITS.get("MAX_SUSTAINED_DAILY_KG", 0.5)

            # First week at aggressive rate, then sustainable rate
            if days <= 7:
                return weekly_max
            else:
                # Additional sustainable change after first week
                return weekly_max + (days - 7) * sustained_daily

    def _check_sustained_pattern(
        self,
        current_weight: float,
        recent_weights: List[float],
        recent_timestamps: List[datetime],
    ) -> float:
        """
        Check if changes follow a sustained pattern vs. erratic fluctuations.
        Sustained patterns are more believable than sudden jumps.
        """
        if len(recent_weights) < 3:
            return 1.0  # Not enough data

        # Look at last 5 measurements or available data
        lookback = min(5, len(recent_weights))
        weights = recent_weights[-lookback:] + [current_weight]

        # Calculate successive differences
        differences = [weights[i + 1] - weights[i] for i in range(len(weights) - 1)]

        # Check consistency of direction (all gains or all losses)
        positive = sum(1 for d in differences if d > 0.1)
        negative = sum(1 for d in differences if d < -0.1)

        # Consistent direction is more believable
        if positive == len(differences) or negative == len(differences):
            consistency_score = 1.0  # Perfectly consistent
        else:
            # Mixed directions - calculate variance
            variance = np.var(differences)
            mean_abs_change = np.mean([abs(d) for d in differences])

            if mean_abs_change > 0:
                # Coefficient of variation
                cv = math.sqrt(variance) / mean_abs_change
                # Lower CV = more consistent
                consistency_score = math.exp(-cv * 0.5)
            else:
                consistency_score = 1.0

        return max(0.3, min(1.0, consistency_score))

    def _calculate_mad_score(self, weight: float, recent_weights: List[float]) -> float:
        """
        Calculate outlier score using Median Absolute Deviation (MAD).
        More robust than standard deviation for outlier detection.
        """
        if len(recent_weights) < 3:
            return 1.0  # Not enough data

        # Use last 10 measurements or available
        lookback_weights = (
            recent_weights[-10:] if len(recent_weights) >= 10 else recent_weights
        )

        # Calculate median and MAD
        median = np.median(lookback_weights)
        mad = np.median([abs(w - median) for w in lookback_weights])

        # Avoid division by zero
        if mad < 0.5:  # Less than 0.5kg variation
            mad = 0.5

        # Calculate z-score equivalent using MAD
        z_mad = abs(weight - median) / (
            1.4826 * mad
        )  # 1.4826 makes MAD comparable to std dev

        # Convert to score (higher z = lower score)
        if z_mad <= 2:  # Within 2 MAD - very likely
            score = 1.0
        elif z_mad <= 3:  # 2-3 MAD - possible
            score = 0.8 - (z_mad - 2) * 0.3
        elif z_mad <= 4:  # 3-4 MAD - unlikely
            score = 0.5 - (z_mad - 3) * 0.3
        else:  # > 4 MAD - very unlikely
            score = 0.2 * math.exp(-(z_mad - 4) * 0.5)

        return max(0.1, min(1.0, score))

    def calculate_source_reliability(self, source: str) -> float:
        """
        Calculate source reliability based on SOURCE_PROFILES.
        """
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)

        # Convert noise_multiplier to reliability score
        # Lower noise multiplier = higher reliability
        noise_multiplier = profile.get("noise_multiplier", 1.0)

        # Invert and normalize to [0, 1]
        # noise_multiplier range: 0.5 (best) to 3.0 (worst)
        reliability = 1.0 - ((noise_multiplier - 0.5) / 2.5)
        reliability = max(0.2, min(1.0, reliability))  # Clamp to [0.2, 1.0]

        return reliability

    def calculate_trend_alignment(
        self,
        weight: float,
        kalman_state: Optional[Dict],
        recent_weights: Optional[List[float]],
    ) -> Tuple[float, Dict]:
        """
        Calculate alignment with established trend using linear regression.
        """
        metadata = {}

        # Need at least 5 measurements for trend
        if not recent_weights or len(recent_weights) < 5:
            metadata["reason"] = "Insufficient data for trend"
            return 0.8, metadata  # Neutral-high score

        # Get recent Kalman states if available
        if kalman_state and "measurement_history" in kalman_state:
            history = kalman_state["measurement_history"]
            if isinstance(history, list) and len(history) >= 5:
                # Use Kalman filtered weights for trend
                kalman_weights = [
                    h.get("filtered_weight", h.get("weight"))
                    for h in history[-10:]
                    if "weight" in h
                ]
                if len(kalman_weights) >= 5:
                    recent_weights = kalman_weights

        # Perform linear regression on recent weights
        x = np.arange(len(recent_weights))
        y = np.array(recent_weights)

        # Calculate trend line
        slope, intercept = np.polyfit(x, y, 1)
        predicted_next = slope * len(recent_weights) + intercept

        metadata["trend_slope"] = slope
        metadata["predicted"] = predicted_next

        # Calculate deviation from trend
        deviation = abs(weight - predicted_next)

        # Expected variance around trend (use std of residuals)
        trend_line = slope * x + intercept
        residuals = y - trend_line
        std_dev = np.std(residuals)

        # Ensure minimum std_dev to avoid division by zero
        # Use 0.5 kg as minimum expected variation (configurable)
        trend_config = self.config.get("trend_alignment", {})
        min_std_dev = trend_config.get("trend_min_std_dev", 0.5)
        if std_dev < min_std_dev:
            std_dev = min_std_dev

        metadata["deviation"] = deviation
        metadata["std_dev"] = std_dev

        # Score based on deviation from trend
        normalized_deviation = deviation / std_dev

        # More gradual scoring: use exponential decay
        # Score = exp(-k * normalized_deviation)
        # k=0.3 gives ~0.74 at 1 std dev, ~0.55 at 2 std devs, ~0.40 at 3 std devs
        # k=0.2 gives ~0.82 at 1 std dev, ~0.67 at 2 std devs, ~0.55 at 3 std devs (more lenient)
        trend_config = self.config.get("trend_alignment", {})
        k = trend_config.get("trend_decay_constant", 0.3)  # Lower = more lenient
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
                product *= score**weight
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

    def update_temporal_baseline(
        self, state: Dict, weight: float, timestamp: datetime
    ) -> Dict:
        """
        Update rolling temporal baseline for continuity across measurements.
        """
        baseline = state.get("temporal_baseline", {})

        if baseline.get("last_weight") and baseline.get("last_timestamp"):
            last_ts = baseline["last_timestamp"]
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts)

            time_diff = (timestamp - last_ts).total_seconds() / 3600
            if time_diff > 0:
                weight_change = abs(weight - baseline["last_weight"])
                daily_rate = weight_change / max(time_diff / 24, 0.1)

                # Exponential moving average with α=0.3
                prev_rate = baseline.get("rolling_avg_change_rate", daily_rate)
                baseline["rolling_avg_change_rate"] = 0.3 * daily_rate + 0.7 * prev_rate

        baseline["last_weight"] = weight
        baseline["last_timestamp"] = (
            timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        )

        state["temporal_baseline"] = baseline
        return state


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
