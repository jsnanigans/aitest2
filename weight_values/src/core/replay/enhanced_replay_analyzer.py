"""
Enhanced Replay Analyzer - Focuses on Kalman-based decision making and reset re-evaluation.

Key improvements:
1. Prioritizes Kalman predictions and trajectory over statistical outlier detection
2. Can re-evaluate and change reset anchor points
3. Considers temporal proximity to previous accepted values
4. Implements sliding window analysis
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MeasurementScore:
    """Score for a measurement based on multiple factors."""

    index: int
    weight: float
    timestamp: datetime
    kalman_similarity: float  # How close to Kalman prediction (0-1, higher is better)
    temporal_consistency: float  # How consistent with recent trend (0-1)
    previous_similarity: float  # How close to previous accepted value (0-1)
    quality_score: float  # Original quality score
    reset_context_score: float  # How well it fits reset context (0-1)
    total_score: float  # Weighted combination
    is_outlier: bool  # Final decision
    reason: str  # Why it was marked as outlier or accepted


class EnhancedReplayAnalyzer:
    """
    Enhanced replay analyzer that prioritizes Kalman predictions and can re-evaluate resets.
    """

    def __init__(self, db, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer with database and configuration.

        Args:
            db: Database instance for accessing states and snapshots
            config: Configuration dictionary
        """
        self.db = db
        self.config = config or {}

        # Scoring weights - prioritize Kalman and temporal consistency
        self.weights = {
            "kalman_similarity": 0.35,  # Most important - trajectory fit
            "temporal_consistency": 0.25,  # Trend consistency
            "previous_similarity": 0.20,  # Proximity to last accepted
            "quality_score": 0.10,  # Original quality assessment
            "reset_context": 0.10,  # Reset-specific scoring
        }

        # Thresholds
        self.kalman_deviation_threshold = config.get(
            "kalman_deviation_threshold", 0.10
        )  # 10% max deviation
        self.temporal_change_threshold = config.get(
            "temporal_change_threshold", 0.05
        )  # 5% per day max
        self.outlier_score_threshold = config.get(
            "outlier_score_threshold", 0.4
        )  # Min score to accept
        self.reset_reevaluation_threshold = config.get(
            "reset_reevaluation_threshold", 0.6
        )  # Score to change reset

    def analyze_measurements_with_reset_context(
        self,
        measurements: List[Dict[str, Any]],
        user_id: str,
        buffer_start_time: datetime,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze measurements considering reset context and Kalman predictions.
        Can re-evaluate reset decisions.

        Args:
            measurements: List of buffered measurements
            user_id: User identifier
            buffer_start_time: Start of the buffer window

        Returns:
            Tuple of (clean_measurements, analysis_report)
        """
        if not measurements:
            return [], {"error": "No measurements to analyze"}

        # Sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda x: x["timestamp"])

        # Get user state and history
        user_state = self.db.get_state(user_id)
        if not user_state:
            return sorted_measurements, {"error": "No user state available"}

        # Check for resets in the buffer window
        reset_events = self._find_resets_in_window(
            user_state, buffer_start_time, sorted_measurements[-1]["timestamp"]
        )

        # Score all measurements
        scores = self._score_measurements(sorted_measurements, user_state, reset_events)

        # Check if any reset needs re-evaluation
        reset_changes = self._evaluate_reset_decisions(scores, reset_events, user_state)

        # Filter measurements based on scores
        clean_measurements = []
        outlier_indices = set()

        for i, score in enumerate(scores):
            if not score.is_outlier:
                # Include measurement, potentially with adjusted context
                measurement = sorted_measurements[i].copy()

                # If this was a reset point that we're changing, mark it
                if reset_changes and i in reset_changes.get("changed_indices", []):
                    measurement["reset_changed"] = True
                    measurement["new_reset_anchor"] = reset_changes.get("new_anchor")

                clean_measurements.append(measurement)
            else:
                outlier_indices.add(i)

        # Create analysis report
        analysis = {
            "total_measurements": len(measurements),
            "clean_measurements": len(clean_measurements),
            "outliers_found": len(outlier_indices),
            "outlier_indices": list(outlier_indices),
            "reset_events_found": len(reset_events),
            "reset_changes": reset_changes,
            "scores": [self._score_to_dict(s) for s in scores],
            "recommendation": self._generate_recommendation(scores, reset_changes),
        }

        return clean_measurements, analysis

    def _score_measurements(
        self,
        measurements: List[Dict[str, Any]],
        user_state: Dict[str, Any],
        reset_events: List[Dict[str, Any]],
    ) -> List[MeasurementScore]:
        """
        Score each measurement based on multiple factors.
        """
        scores = []

        # Get Kalman predictions for each measurement
        kalman_predictions = self._get_kalman_predictions(measurements, user_state)

        # Track the last accepted weight for similarity scoring
        last_accepted_weight = self._get_last_accepted_weight(user_state)

        for i, measurement in enumerate(measurements):
            weight = measurement["weight"]
            timestamp = measurement["timestamp"]

            # 1. Kalman similarity score
            kalman_score = 0.0
            if kalman_predictions and i < len(kalman_predictions):
                predicted = kalman_predictions[i]
                if predicted > 0:
                    deviation = abs(weight - predicted) / predicted
                    kalman_score = max(
                        0, 1.0 - (deviation / self.kalman_deviation_threshold)
                    )

            # 2. Temporal consistency score
            temporal_score = self._calculate_temporal_consistency(
                i, measurements, last_accepted_weight
            )

            # 3. Previous similarity score
            previous_score = 0.0
            if i > 0:
                prev_weight = measurements[i - 1]["weight"]
                if prev_weight > 0:
                    change = abs(weight - prev_weight) / prev_weight
                    previous_score = max(0, 1.0 - change / 0.3)  # 30% max change
            elif last_accepted_weight and last_accepted_weight > 0:
                change = abs(weight - last_accepted_weight) / last_accepted_weight
                previous_score = max(0, 1.0 - change / 0.3)

            # 4. Quality score (from original processing)
            quality_score = measurement.get("metadata", {}).get("quality_score", 0.5)
            if quality_score is None:
                quality_score = 0.5

            # 5. Reset context score
            reset_score = self._calculate_reset_context_score(
                i, measurement, reset_events, measurements
            )

            # Ensure all scores are valid floats (defensive programming)
            kalman_score = float(kalman_score) if kalman_score is not None else 0.5
            temporal_score = (
                float(temporal_score) if temporal_score is not None else 0.5
            )
            previous_score = (
                float(previous_score) if previous_score is not None else 0.0
            )
            quality_score = float(quality_score) if quality_score is not None else 0.5
            reset_score = float(reset_score) if reset_score is not None else 0.5

            # Calculate total score
            total_score = (
                self.weights["kalman_similarity"] * kalman_score
                + self.weights["temporal_consistency"] * temporal_score
                + self.weights["previous_similarity"] * previous_score
                + self.weights["quality_score"] * quality_score
                + self.weights["reset_context"] * reset_score
            )

            # Determine if outlier
            is_outlier = total_score < self.outlier_score_threshold

            # Generate reason
            if is_outlier:
                reasons = []
                if kalman_score < 0.3:
                    reasons.append(f"deviates from Kalman ({kalman_score:.2f})")
                if temporal_score < 0.3:
                    reasons.append(f"temporal inconsistency ({temporal_score:.2f})")
                if previous_score < 0.3:
                    reasons.append(f"large jump ({previous_score:.2f})")
                reason = (
                    "Outlier: " + ", ".join(reasons) if reasons else "Low total score"
                )
            else:
                reason = f"Accepted: score {total_score:.2f}"

            scores.append(
                MeasurementScore(
                    index=i,
                    weight=weight,
                    timestamp=timestamp,
                    kalman_similarity=kalman_score,
                    temporal_consistency=temporal_score,
                    previous_similarity=previous_score,
                    quality_score=quality_score,
                    reset_context_score=reset_score,
                    total_score=total_score,
                    is_outlier=is_outlier,
                    reason=reason,
                )
            )

        # Update last accepted for next iteration
        if scores:
            for score in scores:
                if not score.is_outlier:
                    last_accepted_weight = score.weight

        return scores

    def _find_resets_in_window(
        self, user_state: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Find reset events within the time window."""
        reset_events = []

        # Check state for reset events
        all_resets = user_state.get("reset_events", [])
        for reset in all_resets:
            reset_time = reset.get("timestamp")
            if isinstance(reset_time, str):
                reset_time = datetime.fromisoformat(reset_time)

            if start_time <= reset_time <= end_time:
                reset_events.append(reset)

        # Also check for reset_timestamp (single reset)
        reset_timestamp = user_state.get("reset_timestamp")
        if reset_timestamp:
            if isinstance(reset_timestamp, str):
                reset_timestamp = datetime.fromisoformat(reset_timestamp)

            if start_time <= reset_timestamp <= end_time:
                # Create reset event from state
                reset_events.append(
                    {
                        "timestamp": reset_timestamp,
                        "type": user_state.get("reset_type", "unknown"),
                        "weight": user_state.get("last_raw_weight"),
                    }
                )

        return reset_events

    def _evaluate_reset_decisions(
        self,
        scores: List[MeasurementScore],
        reset_events: List[Dict[str, Any]],
        user_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate if any reset decision should be changed based on subsequent measurements.

        Returns:
            Dict with reset change recommendations or None
        """
        if not scores:
            return None

        # If no reset events in buffer, check if first measurement looks like a reset
        if not reset_events:
            # Check if any measurement has unusually low score that might be a bad reset
            for i, score in enumerate(scores):
                if (
                    score.total_score < 0.3 and i == 0
                ):  # First measurement with very low score
                    # Look for better anchor in subsequent measurements
                    post_scores = scores[1 : min(5, len(scores))]
                    if post_scores:
                        best_alternative = max(post_scores, key=lambda s: s.total_score)
                        if (
                            best_alternative.total_score
                            > self.reset_reevaluation_threshold
                        ):
                            return {
                                "should_change": True,
                                "original_reset": {
                                    "index": 0,
                                    "weight": score.weight,
                                    "timestamp": score.timestamp,
                                    "score": score.total_score,
                                },
                                "new_anchor": {
                                    "index": best_alternative.index,
                                    "weight": best_alternative.weight,
                                    "timestamp": best_alternative.timestamp,
                                    "score": best_alternative.total_score,
                                },
                                "changed_indices": [0],
                                "reason": f"First measurement appears to be outlier reset: {score.weight:.1f}kg "
                                f"(score {score.total_score:.2f}), better anchor: {best_alternative.weight:.1f}kg "
                                f"(score {best_alternative.total_score:.2f})",
                            }
            return None

        changes = {}

        for reset in reset_events:
            reset_time = reset["timestamp"]
            if isinstance(reset_time, str):
                reset_time = datetime.fromisoformat(reset_time)

            # Find measurements around the reset
            reset_idx = None
            for i, score in enumerate(scores):
                # More lenient time matching - within 1 hour
                if abs((score.timestamp - reset_time).total_seconds()) < 3600:
                    reset_idx = i
                    break

            if reset_idx is None:
                continue

            # Look at measurements after the reset
            post_reset_scores = scores[reset_idx + 1 : min(reset_idx + 10, len(scores))]
            if not post_reset_scores:
                continue

            # Check if post-reset measurements are more consistent with pre-reset trajectory
            pre_reset_weight = self._get_pre_reset_weight(user_state, reset_time)
            if pre_reset_weight:
                # Calculate which measurement would be a better reset anchor
                reset_score = scores[reset_idx]
                better_anchor = None
                best_score = reset_score.total_score

                for post_score in post_reset_scores:
                    # Check if this measurement is closer to pre-reset trajectory
                    proximity_to_pre = (
                        abs(post_score.weight - pre_reset_weight) / pre_reset_weight
                    )

                    # If it's much closer to pre-reset and has good score
                    if proximity_to_pre < 0.05 and post_score.total_score > best_score:
                        better_anchor = post_score
                        best_score = post_score.total_score

                # If we found a better anchor point
                if better_anchor and best_score > self.reset_reevaluation_threshold:
                    changes = {
                        "should_change": True,
                        "original_reset": {
                            "index": reset_idx,
                            "weight": reset_score.weight,
                            "timestamp": reset_score.timestamp,
                            "score": reset_score.total_score,
                        },
                        "new_anchor": {
                            "index": better_anchor.index,
                            "weight": better_anchor.weight,
                            "timestamp": better_anchor.timestamp,
                            "score": better_anchor.total_score,
                        },
                        "changed_indices": [reset_idx],
                        "reason": f"Found better reset anchor: {better_anchor.weight:.1f}kg "
                        f"(score {better_anchor.total_score:.2f}) vs original "
                        f"{reset_score.weight:.1f}kg (score {reset_score.total_score:.2f})",
                    }

        return changes if changes else None

    def _get_kalman_predictions(
        self, measurements: List[Dict[str, Any]], user_state: Dict[str, Any]
    ) -> List[float]:
        """Get Kalman predictions for each measurement timestamp."""
        predictions = []

        # Get state history if available
        state_history = user_state.get("state_history", [])
        last_state = user_state.get("last_state")

        # Ensure last_state is a numpy array if it's a list
        if isinstance(last_state, list):
            last_state = np.array(last_state)

        if last_state is None or (
            isinstance(last_state, np.ndarray) and last_state.size == 0
        ):
            return [0.0] * len(measurements)

        # Extract weight from state (first component)
        last_weight = 0.0
        if isinstance(last_state, (list, tuple, np.ndarray)):
            try:
                if isinstance(last_state, np.ndarray):
                    # Handle numpy array
                    if last_state.ndim > 1:
                        # 2D array: extract first element of first row
                        if last_state.shape[0] > 0 and last_state.shape[1] > 0:
                            last_weight = float(last_state[0, 0])
                    else:
                        # 1D array
                        if len(last_state) > 0:
                            last_weight = float(last_state[0])
                elif isinstance(last_state, (list, tuple)) and len(last_state) > 0:
                    # Handle list/tuple - check if nested
                    if isinstance(last_state[0], (list, tuple, np.ndarray)):
                        # Nested format: [[weight]] or [[weight, velocity]]
                        if len(last_state[0]) > 0:
                            last_weight = float(last_state[0][0])
                    else:
                        # Flat format: [weight] or [weight, velocity]
                        last_weight = float(last_state[0])
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(
                    f"Could not extract weight from last_state: {last_state}, error: {e}"
                )
                last_weight = 0.0

        # For each measurement, find the best prediction
        for measurement in measurements:
            timestamp = measurement["timestamp"]

            # Look for closest state snapshot before this measurement
            best_prediction = last_weight  # Default to last known

            if state_history:
                for snapshot in reversed(state_history):
                    snap_time = snapshot.get("timestamp")
                    if snap_time and snap_time < timestamp:
                        snap_state = snapshot.get("state")
                        if snap_state:
                            try:
                                if isinstance(snap_state, np.ndarray):
                                    # Handle numpy array
                                    if snap_state.ndim > 1:
                                        if (
                                            snap_state.shape[0] > 0
                                            and snap_state.shape[1] > 0
                                        ):
                                            best_prediction = float(snap_state[0, 0])
                                    else:
                                        if len(snap_state) > 0:
                                            best_prediction = float(snap_state[0])
                                elif (
                                    isinstance(snap_state, (list, tuple))
                                    and len(snap_state) > 0
                                ):
                                    # Handle list/tuple - check if nested
                                    if isinstance(
                                        snap_state[0], (list, tuple, np.ndarray)
                                    ):
                                        # Nested format
                                        if len(snap_state[0]) > 0:
                                            best_prediction = float(snap_state[0][0])
                                    else:
                                        # Flat format
                                        best_prediction = float(snap_state[0])
                            except (TypeError, ValueError, IndexError) as e:
                                logger.debug(
                                    f"Could not extract weight from snap_state: {e}"
                                )
                            break

            predictions.append(best_prediction)

        return predictions

    def _calculate_temporal_consistency(
        self,
        index: int,
        measurements: List[Dict[str, Any]],
        last_accepted_weight: Optional[float],
    ) -> float:
        """Calculate how consistent a measurement is with temporal expectations."""
        if index == 0:
            # First measurement - check against last accepted if available
            if last_accepted_weight:
                weight = measurements[0]["weight"]
                change = abs(weight - last_accepted_weight) / last_accepted_weight
                # Penalize large changes
                return max(0, 1.0 - change / self.temporal_change_threshold)
            return 0.5  # Neutral score

        # Calculate rate of change
        curr = measurements[index]
        prev = measurements[index - 1]

        # Check for None timestamps
        if curr["timestamp"] is None or prev["timestamp"] is None:
            return 0.5  # Can't calculate without timestamps

        time_diff = (
            curr["timestamp"] - prev["timestamp"]
        ).total_seconds() / 86400.0  # Days
        if time_diff <= 0:
            return 0.5  # Can't calculate

        weight_change = abs(curr["weight"] - prev["weight"]) / prev["weight"]
        daily_change = weight_change / time_diff

        # Score based on daily change rate (expect < 5% per day)
        return max(0, 1.0 - daily_change / self.temporal_change_threshold)

    def _calculate_reset_context_score(
        self,
        index: int,
        measurement: Dict[str, Any],
        reset_events: List[Dict[str, Any]],
        all_measurements: List[Dict[str, Any]],
    ) -> float:
        """Calculate how well a measurement fits in reset context."""
        if not reset_events:
            return 0.5  # Neutral if no resets

        # Find nearest reset
        nearest_reset = None
        min_time_diff = float("inf")

        for reset in reset_events:
            reset_time = reset["timestamp"]
            if isinstance(reset_time, str):
                reset_time = datetime.fromisoformat(reset_time)

            # Check for None timestamps
            if measurement["timestamp"] is None or reset_time is None:
                continue

            time_diff = abs((measurement["timestamp"] - reset_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                nearest_reset = reset

        if not nearest_reset:
            return 0.5

        # Score based on proximity to reset and consistency
        reset_weight = nearest_reset.get("weight")
        if reset_weight:
            # Measurements close to reset should be similar to reset weight
            hours_from_reset = min_time_diff / 3600
            if hours_from_reset < 24:  # Within a day of reset
                weight_diff = abs(measurement["weight"] - reset_weight) / reset_weight
                # Expect measurements near reset to be close to reset value
                return max(0, 1.0 - weight_diff / 0.1)  # 10% tolerance
            elif hours_from_reset < 168:  # Within a week
                # More tolerance as time passes
                weight_diff = abs(measurement["weight"] - reset_weight) / reset_weight
                return max(0, 1.0 - weight_diff / 0.2)  # 20% tolerance

        return 0.5  # Neutral for measurements far from resets

    def _get_last_accepted_weight(self, user_state: Dict[str, Any]) -> Optional[float]:
        """Get the last accepted weight from user state."""
        # Try multiple sources
        last_weight = user_state.get("last_accepted_weight")
        if last_weight:
            return float(last_weight)

        last_weight = user_state.get("last_raw_weight")
        if last_weight:
            return float(last_weight)

        last_state = user_state.get("last_state")
        if last_state is not None:
            if isinstance(last_state, (list, np.ndarray)) and len(last_state) > 0:
                if isinstance(last_state, np.ndarray):
                    return (
                        last_state[0].item()
                        if hasattr(last_state[0], "item")
                        else float(last_state[0])
                    )
                else:
                    return float(last_state[0])

        return None

    def _get_pre_reset_weight(
        self, user_state: Dict[str, Any], reset_time: datetime
    ) -> Optional[float]:
        """Get the weight just before a reset event."""
        # Look in state history
        state_history = user_state.get("state_history", [])

        for snapshot in reversed(state_history):
            snap_time = snapshot.get("timestamp")
            if snap_time and snap_time < reset_time:
                state = snapshot.get("state")
                if state and len(state) > 0:
                    if isinstance(state, np.ndarray):
                        return (
                            state[0].item()
                            if hasattr(state[0], "item")
                            else float(state[0])
                        )
                    else:
                        return float(state[0])

        # Fallback to last state if before reset
        last_timestamp = user_state.get("last_timestamp")
        if last_timestamp:
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)

            if last_timestamp < reset_time:
                last_state = user_state.get("last_state")
                if last_state and len(last_state) > 0:
                    if isinstance(last_state, np.ndarray):
                        return (
                            last_state[0].item()
                            if hasattr(last_state[0], "item")
                            else float(last_state[0])
                        )
                    else:
                        return float(last_state[0])

        return None

    def _score_to_dict(self, score: MeasurementScore) -> Dict[str, Any]:
        """Convert MeasurementScore to dictionary for reporting."""
        return {
            "index": score.index,
            "weight": score.weight,
            "timestamp": score.timestamp.isoformat()
            if hasattr(score.timestamp, "isoformat")
            else str(score.timestamp),
            "scores": {
                "kalman": round(score.kalman_similarity, 3),
                "temporal": round(score.temporal_consistency, 3),
                "previous": round(score.previous_similarity, 3),
                "quality": round(score.quality_score, 3),
                "reset_context": round(score.reset_context_score, 3),
                "total": round(score.total_score, 3),
            },
            "is_outlier": score.is_outlier,
            "reason": score.reason,
        }

    def _generate_recommendation(
        self, scores: List[MeasurementScore], reset_changes: Optional[Dict[str, Any]]
    ) -> str:
        """Generate actionable recommendation based on analysis."""
        outliers = sum(1 for s in scores if s.is_outlier)
        total = len(scores)

        if outliers == 0:
            recommendation = (
                f"All {total} measurements appear valid. No corrections needed."
            )
        else:
            recommendation = (
                f"Found {outliers} outlier(s) out of {total} measurements. "
            )
            recommendation += "Recommend reprocessing without these measurements."

        if reset_changes and reset_changes.get("should_change"):
            recommendation += f"\n\nRESET CHANGE RECOMMENDED: {reset_changes['reason']}"

        return recommendation
