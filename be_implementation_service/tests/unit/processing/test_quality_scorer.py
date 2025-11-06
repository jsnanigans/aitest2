"""Unit tests for UnifiedQualityScorer.

Tests quality scoring which determines accept/reject for measurements.
CRITICAL for data quality.
"""

import pytest
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from src.core.processing.unified_quality_scorer import UnifiedQualityScorer, QualityScore


class TestQualityScoring:
    """Tests for quality scoring components."""

    @pytest.fixture
    def scorer(self, base_config):
        """Create quality scorer with default config."""
        quality_config = base_config.get("quality", {})
        return UnifiedQualityScorer(config=quality_config)

    def test_overall_quality_score_weighted_geometric_mean(self, scorer, base_timestamp):
        """Test overall quality score calculation using weighted geometric mean.

        Components weighted: kalman_fit(40%), temporal(30%), anomaly(20%), source(5%), trend(5%).
        Formula: S = Π(c_i^w_i)^(1/Σw_i)

        Expected behavior:
        - Score in [0, 1]
        - Perfect scores (all 1.0) → 1.0
        - Mixed scores → weighted combination
        """
        # Test 1: Perfect scores
        result = scorer.calculate_quality_score(
            weight=70.5,
            source="manual",
            kalman_prediction=70.5,  # Perfect match
            innovation_covariance=1.0,
            previous_weight=70.0,
            time_diff_hours=24.0,
            recent_weights=[70.0],
            user_height_m=1.75,
        )

        assert 0.0 <= result.overall <= 1.0, "Score should be in [0, 1]"
        assert result.overall > 0.8, "Perfect prediction should have high score"
        assert "kalman_fit" in result.components
        assert "temporal_consistency" in result.components

    def test_kalman_fit_perfect_prediction(self, scorer, base_timestamp):
        """Test Kalman fit component with perfect prediction (score ~1.0).

        Perfect match: measurement == prediction
        Innovation normalized by uncertainty (innovation_covariance)

        Expected behavior:
        - Innovation = 0
        - Normalized innovation = 0
        - Score ~1.0
        """
        kalman_score, metadata = scorer.calculate_kalman_fit(
            weight=70.0,
            kalman_prediction=70.0,  # Perfect prediction
            innovation_covariance=1.0,
            kalman_state={"measurements_since_reset": 100},  # Not in adaptive period
        )

        assert kalman_score >= 0.95, "Perfect prediction should score ~1.0"
        assert metadata["innovation"] == pytest.approx(0.0, abs=0.01)
        assert metadata["normalized_innovation"] == pytest.approx(0.0, abs=0.01)

    def test_kalman_fit_3_sigma_deviation(self, scorer):
        """Test Kalman fit with 3σ deviation (low score).

        3σ deviation is statistically rare (99.7% interval).
        Should result in low score.

        Expected behavior:
        - Normalized innovation = 3
        - Score < 0.3
        """
        # Prediction = 70, measurement = 71.8, σ = 0.6
        # Innovation = 1.8, normalized = 1.8/0.6 = 3.0
        kalman_score, metadata = scorer.calculate_kalman_fit(
            weight=71.8,
            kalman_prediction=70.0,
            innovation_covariance=0.36,  # σ² = 0.36, so σ = 0.6
            kalman_state={"measurements_since_reset": 100},
        )

        assert kalman_score < 0.3, f"3σ deviation should have low score, got {kalman_score}"
        assert metadata["normalized_innovation"] == pytest.approx(3.0, abs=0.1)

    def test_kalman_fit_time_decay_for_long_gaps(self, scorer, base_timestamp):
        """Test Kalman fit score increases for longer time gaps.

        After long gaps, Kalman predictions are less reliable.
        Score should decay toward 1.0 (acceptance).

        Expected behavior:
        - Short gap: normal scoring
        - 30-day gap: score approaches 1.0 regardless of deviation
        """
        kalman_state_recent = {
            "measurements_since_reset": 100,
            "last_timestamp": base_timestamp - timedelta(days=1),
            "current_timestamp": base_timestamp,
        }

        kalman_state_old = {
            "measurements_since_reset": 100,
            "last_timestamp": base_timestamp - timedelta(days=30),
            "current_timestamp": base_timestamp,
        }

        # Same poor fit, different gaps
        score_recent, meta_recent = scorer.calculate_kalman_fit(
            weight=72.0,
            kalman_prediction=70.0,
            innovation_covariance=1.0,
            kalman_state=kalman_state_recent,
        )

        score_old, meta_old = scorer.calculate_kalman_fit(
            weight=72.0,
            kalman_prediction=70.0,
            innovation_covariance=1.0,
            kalman_state=kalman_state_old,
        )

        assert score_old > score_recent, "Older gap should be more forgiving"
        assert "decay_factor" in meta_old

    def test_temporal_consistency_acceptable_change(self, scorer):
        """Test temporal consistency with acceptable change (1kg in 1 day).

        Normal day-to-day variation should score high.

        Expected behavior:
        - 1kg change in 24 hours is reasonable
        - Score > 0.8
        """
        temporal_score, metadata = scorer.calculate_temporal_consistency(
            weight=71.0,
            previous_weight=70.0,
            time_diff_hours=24.0,
            recent_weights=[70.0],
            recent_timestamps=None,
        )

        assert temporal_score > 0.8, f"1kg in 24h should score high, got {temporal_score}"
        assert metadata["actual_change"] == 1.0

    def test_temporal_consistency_excessive_change(self, scorer):
        """Test temporal consistency with excessive change (5kg in 1 hour).

        Rapid impossible change should score low.

        Expected behavior:
        - 5kg in 1 hour is physiologically impossible
        - Score < 0.3
        """
        temporal_score, metadata = scorer.calculate_temporal_consistency(
            weight=75.0,
            previous_weight=70.0,
            time_diff_hours=1.0,
            recent_weights=[70.0],
            recent_timestamps=None,
        )

        assert temporal_score < 0.3, f"5kg in 1h should score low, got {temporal_score}"
        assert metadata["actual_change"] == 5.0

    def test_anomaly_absolute_min_violation(self, scorer):
        """Test anomaly detection: weight < 30kg rejected.

        Hard safety limit.

        Expected behavior:
        - Weight < ABSOLUTE_MIN_WEIGHT → score = 0.0
        - Immediate rejection
        """
        anomaly_score, metadata = scorer.calculate_anomaly_detection(
            weight=25.0,  # Below 30kg minimum
            recent_weights=[],
            recent_timestamps=[],
            user_height_m=1.75,
        )

        assert anomaly_score == 0.0, "Below absolute minimum should score 0.0"
        assert "outside_absolute_min" in metadata

    def test_anomaly_absolute_max_violation(self, scorer):
        """Test anomaly detection: weight > 400kg rejected.

        Hard safety limit.

        Expected behavior:
        - Weight > ABSOLUTE_MAX_WEIGHT → score = 0.0
        - Immediate rejection
        """
        anomaly_score, metadata = scorer.calculate_anomaly_detection(
            weight=450.0,  # Above 400kg maximum
            recent_weights=[],
            recent_timestamps=[],
            user_height_m=1.75,
        )

        assert anomaly_score == 0.0, "Above absolute maximum should score 0.0"
        assert "outside_absolute_max" in metadata

    def test_anomaly_duplicate_detection_within_5_seconds(self, scorer, base_timestamp):
        """Test anomaly detection: duplicate within 5 seconds rejected.

        Detects accidental double-submissions.

        Expected behavior:
        - Same weight within 5 seconds → score = 0.0
        - Different weight within 5 seconds → minor penalty
        """
        # Test 1: True duplicate (same weight, 3 seconds apart)
        anomaly_score, metadata = scorer.calculate_anomaly_detection(
            weight=70.0,  # Same weight
            recent_weights=[70.0],
            recent_timestamps=[base_timestamp - timedelta(seconds=3)],
            user_height_m=1.75,
            current_timestamp=base_timestamp,
        )

        assert anomaly_score == 0.0, "True duplicate should score 0.0"
        assert metadata.get("rejected_reason") == "duplicate_measurement"

    def test_anomaly_burst_pattern_detection(self, scorer, base_timestamp):
        """Test anomaly detection: 5+ measurements in 30 minutes.

        Detects measurement spam or scale instability.

        Expected behavior:
        - >= 5 measurements in 30 min → burst penalty
        - Score reduced (not rejected outright)
        """
        # Create 5 measurements within 30 minutes
        recent_timestamps = [
            base_timestamp - timedelta(minutes=25),
            base_timestamp - timedelta(minutes=20),
            base_timestamp - timedelta(minutes=15),
            base_timestamp - timedelta(minutes=10),
            base_timestamp - timedelta(minutes=5),
        ]
        recent_weights = [70.0, 70.1, 70.2, 70.1, 70.0]

        anomaly_score, metadata = scorer.calculate_anomaly_detection(
            weight=70.1,
            recent_weights=recent_weights,
            recent_timestamps=recent_timestamps,
            user_height_m=1.75,
            current_timestamp=base_timestamp,
        )

        # Should apply burst penalty but not reject
        assert 0.3 < anomaly_score < 1.0, "Burst pattern should reduce score but not reject"
        assert metadata.get("burst_pattern_detected") is True

    def test_quality_scorer_with_no_previous_weight(self, scorer, base_timestamp):
        """Test quality scorer handles first measurement (no previous weight) correctly.

        This is CRITICAL: first measurement is common scenario, must not crash.
        When no previous weight exists, temporal consistency should default gracefully.

        Expected behavior:
        - No previous weight available
        - Temporal consistency returns neutral score (0.7)
        - Overall quality score can still be calculated
        - Measurement can be accepted based on other components
        """
        # Simulate first measurement: no previous weight, no Kalman prediction yet
        result = scorer.calculate_quality_score(
            weight=70.0,
            source="manual",
            kalman_prediction=None,  # No Kalman yet
            innovation_covariance=None,
            previous_weight=None,  # First measurement
            time_diff_hours=None,  # No previous timestamp
            recent_weights=[],  # No history
            user_height_m=1.75,
        )

        # Verify overall score is calculated
        assert 0.0 <= result.overall <= 1.0, "Overall score should be in [0, 1]"

        # Verify temporal consistency has neutral score
        assert "temporal_consistency" in result.components
        assert result.components["temporal_consistency"] == 0.7, "Should return neutral 0.7 for first measurement"

        # Verify Kalman fit has neutral score (no prediction available)
        assert "kalman_fit" in result.components
        assert result.components["kalman_fit"] == 0.5, "Should return neutral 0.5 when no Kalman prediction"

        # Verify measurement is not rejected (can accept first measurement)
        # With neutral scores, should still be acceptable
        assert result.overall > 0.4, "First measurement should have reasonable score"

    def test_quality_scorer_with_no_kalman_prediction(self, scorer, base_timestamp):
        """Test quality scorer handles missing Kalman prediction correctly.

        This is CRITICAL: edge case after reset, Kalman not initialized yet.
        When no Kalman prediction exists, should default gracefully.

        Expected behavior:
        - No Kalman prediction available
        - Kalman fit component returns neutral score (0.5)
        - Other components still calculated normally
        - Overall quality score can still be calculated
        """
        # Simulate post-reset scenario: have previous weight but no Kalman prediction yet
        result = scorer.calculate_quality_score(
            weight=70.5,
            source="manual",
            kalman_prediction=None,  # Kalman not initialized
            innovation_covariance=None,
            previous_weight=70.0,  # Have previous measurement
            time_diff_hours=24.0,  # 1 day gap
            recent_weights=[70.0],
            user_height_m=1.75,
        )

        # Verify overall score is calculated
        assert 0.0 <= result.overall <= 1.0, "Overall score should be in [0, 1]"

        # Verify Kalman fit has neutral score
        assert "kalman_fit" in result.components
        assert result.components["kalman_fit"] == 0.5, "Should return neutral 0.5 when no prediction"

        # Verify metadata explains why neutral score
        kalman_meta = result.metadata.get("kalman_fit", {})
        assert kalman_meta.get("reason") == "No Kalman prediction available"

        # Verify temporal consistency is calculated normally
        assert "temporal_consistency" in result.components
        temporal_score = result.components["temporal_consistency"]
        assert temporal_score > 0.7, "Temporal should be high for normal change"

        # Verify overall score is reasonable (not penalized too much)
        assert result.overall > 0.4, "Should have reasonable score without Kalman"
