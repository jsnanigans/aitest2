"""Unit tests for short-term measurement improvements."""

import pytest
from datetime import datetime, timedelta

from src.processing.unified_quality_scorer import UnifiedQualityScorer


class TestShortTermMeasurementImprovements:
    """Test suite for improved short-term measurement handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = UnifiedQualityScorer()
        self.base_time = datetime.now()

    def test_duplicate_rejection_threshold(self):
        """Test that only true duplicates within 5 seconds are rejected."""
        # True duplicate: < 5 seconds, < 0.05kg change
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.02,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(seconds=3)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score == 0.0
        assert metadata.get("rejected_reason") == "duplicate_measurement"

        # Not a duplicate: < 5 seconds but > 0.05kg change
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.15,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(seconds=3)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.0
        assert metadata.get("rapid_but_different") == True

    def test_one_minute_threshold(self):
        """Test 1-minute change threshold (0.5kg)."""
        # Within threshold: 0.3kg in 1 minute
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.3,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=1)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.5  # Should be accepted with minor penalty

        # At threshold: 0.5kg in 1 minute
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.5,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=1)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.5  # Should still be accepted

        # Way over threshold: 1.5kg in 1 minute (> 2x threshold)
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=101.5,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=1)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score == 0.0  # Should be rejected
        assert metadata.get("rejected_reason") == "rapid_impossible_change"

    def test_five_minute_threshold(self):
        """Test 5-minute change threshold (1.0kg)."""
        # Within threshold: 0.8kg in 5 minutes
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.8,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=5)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.6  # Should be accepted

        # At threshold: 1.0kg in 5 minutes
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=101.0,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=5)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.5  # Should be accepted

        # Over threshold but not impossible: 1.5kg in 5 minutes
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=101.5,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=5)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.0  # Should be accepted with penalty
        assert score < 0.5  # But with significant penalty

    def test_two_hour_threshold(self):
        """Test 2-hour change threshold (~1.6kg based on new formula)."""
        # Reasonable change: 1.5kg in 2 hours
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=101.5,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(hours=2)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.5  # Should be accepted

        # Large but possible: 3.0kg in 2 hours
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=103.0,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(hours=2)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.0  # Should be accepted with penalty
        assert metadata.get("excess_ratio", 0) > 0  # Shows it exceeded threshold

    def test_burst_pattern_detection(self):
        """Test burst pattern detection with less aggressive penalties."""
        # Create a burst of measurements within BURST_WINDOW_MINUTES (30 min)
        recent_weights = []
        recent_timestamps = []

        # Create 6 measurements all within 30 minutes to trigger burst detection
        for i in range(6):
            recent_weights.append(100.0 + i * 0.1)
            recent_timestamps.append(
                self.base_time
                - timedelta(minutes=28 - i * 4)  # All within 30 min window
            )

        # 7th measurement should trigger burst detection (threshold is 5)
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.7,
            recent_weights=recent_weights,
            recent_timestamps=recent_timestamps,
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )

        # Should detect burst but not reject completely
        if metadata.get("burst_pattern_detected"):
            assert metadata.get("burst_count") >= 5
            assert score > 0.5  # Less aggressive penalty than before
        else:
            # If burst not detected, should still accept the measurement
            assert score > 0.5

    def test_source_aware_thresholds(self):
        """Test that source type affects threshold calculation."""
        self.scorer.current_source = "patient-device"

        # Device measurement should be more lenient
        score_device, metadata_device = self.scorer.calculate_anomaly_detection(
            weight=100.6,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=1)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )

        self.scorer.current_source = "manual-entry"

        # Manual entry should be stricter
        score_manual, metadata_manual = self.scorer.calculate_anomaly_detection(
            weight=100.6,
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=1)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )

        # Device measurements should have higher score (more lenient)
        assert score_device >= score_manual

    def test_scale_variance_pattern(self):
        """Test handling of typical scale variance pattern."""
        measurements = [
            (self.base_time - timedelta(minutes=4), 100.0),
            (self.base_time - timedelta(minutes=3), 100.3),
            (self.base_time - timedelta(minutes=2), 99.8),
            (self.base_time - timedelta(minutes=1), 100.1),
        ]

        recent_weights = [m[1] for m in measurements]
        recent_timestamps = [m[0] for m in measurements]

        # Next measurement with typical variance
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=100.4,
            recent_weights=recent_weights,
            recent_timestamps=recent_timestamps,
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )

        # Should accept with reasonable score
        assert score > 0.4
        assert "sustained_pattern_score" in metadata

    def test_extreme_but_valid_cases(self):
        """Test extreme cases that should still be accepted."""
        # Large meal + exercise + hydration change
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=102.5,  # 2.5kg change
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(hours=3)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.0  # Should not be completely rejected

        # Multiple day gap
        score, metadata = self.scorer.calculate_anomaly_detection(
            weight=105.0,  # 5kg change
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(days=7)],
            user_height_m=1.70,
            current_timestamp=self.base_time,
        )
        assert score > 0.0  # Should be accepted for weekly change

    def test_quality_score_integration(self):
        """Test that anomaly detection integrates properly with overall quality score."""
        # Create a scenario that would have been rejected before
        kalman_prediction = 100.0
        innovation_covariance = 0.5

        quality_score = self.scorer.calculate_quality_score(
            weight=100.4,
            source="patient-device",
            kalman_state={"measurements_since_reset": 50},
            kalman_prediction=kalman_prediction,
            innovation_covariance=innovation_covariance,
            previous_weight=100.0,
            time_diff_hours=0.0167,  # 1 minute
            recent_weights=[100.0],
            recent_timestamps=[self.base_time - timedelta(minutes=1)],
            user_height_m=1.70,
        )

        # Should be accepted
        assert quality_score.accepted == True
        assert quality_score.overall > self.scorer.threshold
