"""
Simplified Temporal Consistency Analyzer for Replay Processing

Replaces EnhancedReplayAnalyzer + OutlierDetector with single-purpose checker:
Validates that weight changes are physiologically plausible given time elapsed.

Design: Physics-based filtering only - if a change violates temporal limits, it's out.
No statistical methods (IQR, Z-score) - those can flag valid outliers like post-surgery.
"""

import logging
import math
from typing import List, Dict, Any, Set, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TemporalConsistencyAnalyzer:
    """
    Single-purpose analyzer: checks if weight changes are temporally plausible.
    Merges functionality from EnhancedReplayAnalyzer + OutlierDetector.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer with physiological limits.

        Args:
            config: Optional configuration for thresholds
        """
        self.config = config or {}

        # Physiological change limits (from PHYSIOLOGICAL_LIMITS constants)
        self.limits = {
            "MAX_CHANGE_1MIN": 0.5,      # Scale variance + positioning
            "MAX_CHANGE_5MIN": 1.0,      # Water/bathroom
            "MAX_CHANGE_1H": 1.0,        # Food/drink
            "MAX_CHANGE_6H": 3.0,        # Daily fluctuation
            "MAX_CHANGE_24H": 4.0,       # Full day cycle
            "MAX_WEEKLY_CHANGE_KG": 3.5, # Weekly maximum
            "MAX_SUSTAINED_DAILY_KG": 0.5, # Long-term rate
        }

        # Allow config overrides
        self.limits.update(self.config.get("physiological_limits", {}))

        # Minimum measurements needed for analysis
        self.min_measurements = self.config.get("min_measurements_for_analysis", 3)

    def analyze_and_filter(
        self,
        measurements: List[Dict[str, Any]],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze measurements and identify temporally impossible changes.

        Args:
            measurements: List of buffered measurements with 'weight', 'timestamp'
            user_id: User identifier (for logging)

        Returns:
            Analysis result with clean_measurements and outlier_indices
        """
        if len(measurements) < self.min_measurements:
            return {
                "success": True,
                "clean_measurements": measurements,
                "outlier_indices": set(),
                "reason": f"Insufficient measurements ({len(measurements)} < {self.min_measurements})",
                "total_measurements": len(measurements),
                "outliers_found": 0,
            }

        # Sort chronologically
        sorted_measurements = sorted(measurements, key=lambda m: m["timestamp"])

        # Check each consecutive pair for temporal plausibility
        outlier_indices = set()
        analysis_details = []

        for i in range(1, len(sorted_measurements)):
            prev = sorted_measurements[i - 1]
            curr = sorted_measurements[i]

            # Check temporal consistency
            is_outlier, details = self._check_temporal_consistency(
                prev_weight=prev["weight"],
                prev_timestamp=prev["timestamp"],
                curr_weight=curr["weight"],
                curr_timestamp=curr["timestamp"],
                index=i,
            )

            analysis_details.append(details)

            if is_outlier:
                outlier_indices.add(i)
                logger.info(
                    f"Temporal outlier detected for {user_id}: "
                    f"{details['weight_change']:.2f}kg in {details['time_hours']:.2f}h "
                    f"(max: {details['max_allowed']:.2f}kg) - {details['reason']}"
                )

        # Build clean measurements list
        clean_measurements = [
            m for i, m in enumerate(sorted_measurements) if i not in outlier_indices
        ]

        return {
            "success": True,
            "clean_measurements": clean_measurements,
            "outlier_indices": outlier_indices,
            "total_measurements": len(measurements),
            "outliers_found": len(outlier_indices),
            "analysis_details": analysis_details,
            "user_id": user_id,
        }

    def _check_temporal_consistency(
        self,
        prev_weight: float,
        prev_timestamp: datetime,
        curr_weight: float,
        curr_timestamp: datetime,
        index: int,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if weight change is temporally plausible.

        Args:
            prev_weight: Previous weight
            prev_timestamp: Previous timestamp
            curr_weight: Current weight
            curr_timestamp: Current timestamp
            index: Index of current measurement

        Returns:
            Tuple of (is_outlier, details_dict)
        """
        weight_change = abs(curr_weight - prev_weight)
        time_diff = curr_timestamp - prev_timestamp
        time_hours = time_diff.total_seconds() / 3600.0

        # Handle edge cases
        if time_hours <= 0:
            return True, {
                "index": index,
                "weight_change": weight_change,
                "time_hours": time_hours,
                "max_allowed": 0.0,
                "is_outlier": True,
                "reason": "Non-positive time difference",
            }

        if prev_weight <= 0:
            return True, {
                "index": index,
                "weight_change": weight_change,
                "time_hours": time_hours,
                "max_allowed": 0.0,
                "is_outlier": True,
                "reason": "Invalid previous weight",
            }

        # Calculate maximum physiologically plausible change for this time period
        max_allowed = self._calculate_max_physiological_change(time_hours)

        # Check if change exceeds limit
        is_outlier = weight_change > max_allowed

        # Build details
        details = {
            "index": index,
            "weight_change": weight_change,
            "time_hours": time_hours,
            "max_allowed": max_allowed,
            "is_outlier": is_outlier,
            "prev_weight": prev_weight,
            "curr_weight": curr_weight,
        }

        if is_outlier:
            excess = weight_change - max_allowed
            details["reason"] = (
                f"Change of {weight_change:.2f}kg in {time_hours:.2f}h "
                f"exceeds physiological limit of {max_allowed:.2f}kg "
                f"(excess: {excess:.2f}kg)"
            )
        else:
            details["reason"] = "Within physiological limits"

        return is_outlier, details

    def _calculate_max_physiological_change(self, time_hours: float) -> float:
        """
        Calculate maximum physiological weight change based on time elapsed.
        Uses continuous functions (no step changes) from unified_quality_scorer.

        Args:
            time_hours: Time elapsed in hours

        Returns:
            Maximum plausible change in kg
        """
        if time_hours <= 0:
            return 0.0

        # Ultra short-term (< 1 minute): Scale variance + positioning
        if time_hours < 0.0167:  # 1 minute
            return self.limits["MAX_CHANGE_1MIN"]

        # Very short-term (< 5 minutes): Scale variance + water/bathroom
        elif time_hours < 0.0833:  # 5 minutes
            max_1min = self.limits["MAX_CHANGE_1MIN"]
            max_5min = self.limits["MAX_CHANGE_5MIN"]
            # Linear interpolation
            minutes = time_hours * 60
            return max_1min + (max_5min - max_1min) * (minutes - 1) / 4

        # Short-term (< 1 hour): Limited by water/food intake
        elif time_hours < 1:
            max_5min = self.limits["MAX_CHANGE_5MIN"]
            max_1h = self.limits["MAX_CHANGE_1H"]
            minutes = time_hours * 60
            # Logarithmic growth from 5 min to 60 min
            if minutes <= 5:
                return max_5min
            else:
                return max_5min + (max_1h - max_5min) * math.log(
                    minutes / 5
                ) / math.log(12)

        # Hours (1-6 hours): Water, food, exercise effects
        elif time_hours <= 6:
            base_change = self.limits["MAX_CHANGE_1H"]
            max_6h = self.limits["MAX_CHANGE_6H"]
            # Smooth interpolation
            additional = (max_6h - base_change) * math.log(1 + (time_hours - 1)) / math.log(6)
            return base_change + additional

        # Day (6-24 hours): Full daily fluctuation
        elif time_hours <= 24:
            base_change = self.limits["MAX_CHANGE_6H"]
            max_24h = self.limits["MAX_CHANGE_24H"]
            # Logarithmic growth
            additional = (max_24h - base_change) * math.log(
                1 + (time_hours - 6) / 6
            ) / math.log(4)
            return base_change + additional

        # Week (1-7 days): Compound changes with realistic limits
        elif time_hours <= 168:  # 7 days
            days = time_hours / 24
            daily_max = 2.0  # Conservative daily maximum
            weekly_max = self.limits["MAX_WEEKLY_CHANGE_KG"]
            # Use square root for realistic accumulation
            return min(weekly_max, daily_max * math.sqrt(days))

        # Long-term (> 1 week): Sustainable rates only
        else:
            days = time_hours / 24
            weekly_max = self.limits["MAX_WEEKLY_CHANGE_KG"]
            sustained_daily = self.limits["MAX_SUSTAINED_DAILY_KG"]

            # First week at aggressive rate, then sustainable rate
            if days <= 7:
                return weekly_max
            else:
                # Additional sustainable change after first week
                return weekly_max + (days - 7) * sustained_daily

    def get_statistics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistical summary of analysis results.

        Args:
            analysis_result: Result from analyze_and_filter()

        Returns:
            Statistics dictionary
        """
        if not analysis_result.get("analysis_details"):
            return {}

        details = analysis_result["analysis_details"]

        # Filter for outliers only
        outliers = [d for d in details if d["is_outlier"]]

        if not outliers:
            return {
                "total_checked": len(details),
                "outliers_found": 0,
                "outlier_rate": 0.0,
            }

        return {
            "total_checked": len(details),
            "outliers_found": len(outliers),
            "outlier_rate": len(outliers) / len(details),
            "avg_excess_kg": sum(
                d["weight_change"] - d["max_allowed"] for d in outliers
            ) / len(outliers),
            "max_excess_kg": max(
                d["weight_change"] - d["max_allowed"] for d in outliers
            ),
            "outlier_time_ranges": {
                "under_1h": sum(1 for d in outliers if d["time_hours"] < 1),
                "1h_to_6h": sum(1 for d in outliers if 1 <= d["time_hours"] < 6),
                "6h_to_24h": sum(1 for d in outliers if 6 <= d["time_hours"] < 24),
                "over_24h": sum(1 for d in outliers if d["time_hours"] >= 24),
            },
        }
