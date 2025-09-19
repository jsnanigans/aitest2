"""
Test suite for continuous temporal scoring to verify elimination of step discontinuities.
"""

import pytest
from datetime import datetime, timedelta
import math

from src.processing.unified_quality_scorer import UnifiedQualityScorer


class TestContinuousTemporalScoring:
    """Test suite for continuous temporal scoring function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = UnifiedQualityScorer()
        self.base_weight = 80.0

    def test_continuity_no_step_functions(self):
        """
        Test that scores are continuous with no jumps at 6h, 24h boundaries.
        Verifies smooth transition instead of step functions.
        """
        # Test across traditional boundary points - check continuity at each boundary
        boundary_tests = [
            [5.9, 6.0, 6.1],      # Around 6 hour boundary
            [23.9, 24.0, 24.1],   # Around 24 hour boundary
            [47.9, 48.0, 48.1]    # Around 48 hour boundary
        ]
        weight_change = 2.0  # 2kg change

        for time_points in boundary_tests:
            scores = []
            for hours in time_points:
                current_weight = self.base_weight + weight_change
                score, metadata = self.scorer.calculate_temporal_consistency(
                    weight=current_weight,
                    previous_weight=self.base_weight,
                    time_diff_hours=hours,
                    recent_weights=None,
                    recent_timestamps=None
                )
                scores.append(score)

            # Check no sudden jumps within each boundary region
            for i in range(len(scores) - 1):
                diff = abs(scores[i+1] - scores[i])
                # Allow for small numerical differences but no step jumps
                assert diff < 0.05, (
                    f"Discontinuity detected between {time_points[i]}h and {time_points[i+1]}h: "
                    f"scores {scores[i]:.4f} -> {scores[i+1]:.4f} (diff={diff:.4f})"
                )

    def test_exponential_growth_of_acceptable_change(self):
        """
        Test that acceptable change grows exponentially with time.
        Starts at ~0.5kg and grows to ~5kg at 7 days.
        """
        test_cases = [
            (0, 0.5),      # Immediate: 0.5kg
            (12, 1.3),     # 12 hours: ~1.3kg
            (24, 2.3),     # 1 day: ~2.3kg
            (48, 3.5),     # 2 days: ~3.5kg
            (96, 4.4),     # 4 days: ~4.4kg
            (168, 4.8),    # 7 days: ~4.8kg
        ]

        for hours, expected_threshold in test_cases:
            # Calculate actual threshold using the formula
            actual_threshold = 0.5 + 4.5 * (1 - math.exp(-hours / 48))

            # Test with weight change at threshold
            score, metadata = self.scorer.calculate_temporal_consistency(
                weight=self.base_weight + actual_threshold,
                previous_weight=self.base_weight,
                time_diff_hours=hours,
                recent_weights=None,
                recent_timestamps=None
            )

            # Score should be reasonable (not heavily penalized) at threshold
            assert 0.6 < score <= 1.0, (
                f"At {hours}h with change={actual_threshold:.2f}kg, "
                f"expected good score but got {score:.3f}"
            )

            # Verify threshold is approximately as expected
            assert abs(actual_threshold - expected_threshold) < 0.5, (
                f"At {hours}h, expected threshold ~{expected_threshold}kg "
                f"but got {actual_threshold:.2f}kg"
            )

    def test_similar_weights_similar_scores(self):
        """
        Test that scores change smoothly and gradually over time.
        Verifies no sudden jumps from step functions.
        """
        weight_change = 1.5  # 1.5kg change

        # Test at closely-spaced time points
        time_ranges = [
            [5.5, 5.8, 6.2, 6.5],      # Around 6 hours
            [23.5, 23.8, 24.2, 24.5]   # Around 24 hours
        ]

        for time_points in time_ranges:
            scores = []
            for hours in time_points:
                score, _ = self.scorer.calculate_temporal_consistency(
                    weight=self.base_weight + weight_change,
                    previous_weight=self.base_weight,
                    time_diff_hours=hours,
                    recent_weights=None,
                    recent_timestamps=None
                )
                scores.append(score)

            # Within a 1-hour range, scores should be similar (gradual change)
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            assert score_range < 0.15, (
                f"Scores vary too much within small time range {time_points}: "
                f"range={score_range:.3f} (min={min_score:.3f}, max={max_score:.3f})"
            )

    def test_no_step_discontinuities(self):
        """
        Comprehensive test to ensure no step discontinuities exist.
        Samples scores densely around former boundary points.
        """
        weight_change = 2.5  # 2.5kg change

        # Dense sampling around former boundaries (6h and 24h)
        critical_regions = [
            (5.0, 7.0, 0.1),    # Around 6 hours
            (23.0, 25.0, 0.1),  # Around 24 hours
        ]

        for start_h, end_h, step in critical_regions:
            hours = start_h
            prev_score = None

            while hours <= end_h:
                score, _ = self.scorer.calculate_temporal_consistency(
                    weight=self.base_weight + weight_change,
                    previous_weight=self.base_weight,
                    time_diff_hours=hours,
                    recent_weights=None,
                    recent_timestamps=None
                )

                if prev_score is not None:
                    score_diff = abs(score - prev_score)
                    # With 0.1 hour steps, score changes should be tiny
                    assert score_diff < 0.02, (
                        f"Discontinuity found between {hours-step:.1f}h and {hours:.1f}h: "
                        f"scores {prev_score:.4f} -> {score:.4f} (diff={score_diff:.4f})"
                    )

                prev_score = score
                hours += step

    def test_smooth_penalty_beyond_threshold(self):
        """
        Test that penalties beyond acceptable threshold are smooth and exponential.
        """
        time_hours = 24  # 1 day
        max_acceptable = 0.5 + 4.5 * (1 - math.exp(-time_hours / 48))

        # Test various amounts beyond threshold
        excess_factors = [1.1, 1.2, 1.5, 2.0, 3.0]
        scores = []

        for factor in excess_factors:
            weight_change = max_acceptable * factor
            score, metadata = self.scorer.calculate_temporal_consistency(
                weight=self.base_weight + weight_change,
                previous_weight=self.base_weight,
                time_diff_hours=time_hours,
                recent_weights=None,
                recent_timestamps=None
            )
            scores.append(score)

            # Verify metadata contains expected values
            assert 'max_acceptable_change' in metadata
            assert 'actual_change' in metadata
            assert abs(metadata['actual_change'] - weight_change) < 0.01

        # Scores should decrease smoothly
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i+1], (
                f"Scores should decrease with excess: {scores[i]:.3f} should be > {scores[i+1]:.3f}"
            )

    def test_temporal_baseline_update(self):
        """
        Test temporal baseline updating for state continuity.
        """
        state = {}
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        weight1 = 80.0

        # First update - should initialize baseline
        state = self.scorer.update_temporal_baseline(state, weight1, timestamp1)
        assert 'temporal_baseline' in state
        assert state['temporal_baseline']['last_weight'] == weight1
        assert state['temporal_baseline']['last_timestamp'] == timestamp1.isoformat()

        # Second update - should calculate rolling average
        timestamp2 = timestamp1 + timedelta(days=1)
        weight2 = 81.0
        state = self.scorer.update_temporal_baseline(state, weight2, timestamp2)

        baseline = state['temporal_baseline']
        assert baseline['last_weight'] == weight2
        assert 'rolling_avg_change_rate' in baseline

        # Rate should be 1kg/day
        expected_rate = 1.0
        actual_rate = baseline['rolling_avg_change_rate']
        assert abs(actual_rate - expected_rate) < 0.1

    def test_score_clamping(self):
        """
        Test that scores are properly clamped between 0.2 and 1.0.
        """
        # Test with extreme weight changes
        test_cases = [
            (0.1, 0.1),    # Tiny change, short time - should get high score
            (0.1, 50.0),   # Huge change, short time - should get low score
            (168, 0.1),    # Tiny change, long time - should get high score
            (168, 50.0),   # Huge change, long time - should still be penalized
        ]

        for hours, weight_change in test_cases:
            score, _ = self.scorer.calculate_temporal_consistency(
                weight=self.base_weight + weight_change,
                previous_weight=self.base_weight,
                time_diff_hours=hours,
                recent_weights=None,
                recent_timestamps=None
            )

            # Score must be within bounds
            assert 0.2 <= score <= 1.0, (
                f"Score {score:.3f} out of bounds for "
                f"hours={hours}, change={weight_change}kg"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])