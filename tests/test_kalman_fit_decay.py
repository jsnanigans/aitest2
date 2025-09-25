"""Test Kalman fit time-based decay functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime, timedelta
import numpy as np

from src.processing.unified_quality_scorer import UnifiedQualityScorer


def test_kalman_fit_decay_over_time():
    """Test that Kalman fit score decays over time gaps."""
    scorer = UnifiedQualityScorer()

    # Test weight and prediction with significant deviation
    weight = 80.0
    kalman_prediction = 75.0  # 5kg difference
    innovation_covariance = 1.0

    # Test with different time gaps
    test_cases = [
        (0, "No gap"),  # No time gap
        (1, "1 day gap"),  # 1 day
        (7, "1 week gap"),  # 1 week
        (15, "15 days gap"),  # 15 days (halfway to 30)
        (30, "30 days gap"),  # 30 days (maximum decay)
        (60, "60 days gap"),  # Beyond 30 days (should cap at 30)
    ]

    base_timestamp = datetime(2024, 1, 1, 12, 0, 0)
    scores = []

    for days_gap, description in test_cases:
        current_timestamp = base_timestamp + timedelta(days=days_gap)
        last_timestamp = base_timestamp

        kalman_state = {
            "last_timestamp": last_timestamp,
            "current_timestamp": current_timestamp,
            "measurements_since_reset": 50,  # Not in adaptive period
        }

        score, metadata = scorer.calculate_kalman_fit(
            weight=weight,
            kalman_prediction=kalman_prediction,
            innovation_covariance=innovation_covariance,
            kalman_state=kalman_state,
        )

        scores.append((days_gap, score, metadata))
        print(
            f"{description}: score={score:.3f}, decay_factor={metadata.get('decay_factor', 0):.3f}"
        )

    # Verify that scores increase with time gap
    for i in range(1, len(scores)):
        days_prev, score_prev, _ = scores[i - 1]
        days_curr, score_curr, _ = scores[i]
        assert score_curr >= score_prev, (
            f"Score should increase with time gap: {days_prev}d={score_prev:.3f} vs {days_curr}d={score_curr:.3f}"
        )

    # Verify specific expectations
    _, score_0d, _ = scores[0]  # No gap
    _, score_30d, metadata_30d = scores[4]  # 30 days gap
    _, score_60d, metadata_60d = scores[5]  # 60 days gap

    # With no gap, score should be relatively low due to 5kg deviation
    assert score_0d < 0.5, (
        f"With 5kg deviation and no gap, score should be low: {score_0d:.3f}"
    )

    # At 30 days, score should be close to 1.0 (full acceptance)
    assert score_30d > 0.95, (
        f"At 30 days gap, score should be near 1.0: {score_30d:.3f}"
    )
    assert metadata_30d.get("decay_factor", 0) >= 0.99, (
        "Decay factor should be ~1.0 at 30 days"
    )

    # Beyond 30 days should cap at same as 30 days
    assert abs(score_60d - score_30d) < 0.01, (
        f"Score should cap at 30 days: 30d={score_30d:.3f} vs 60d={score_60d:.3f}"
    )


def test_kalman_fit_decay_gradual():
    """Test that decay is gradual and linear over 30 days."""
    scorer = UnifiedQualityScorer()

    weight = 80.0
    kalman_prediction = 75.0
    innovation_covariance = 1.0
    base_timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # Test linear progression
    decay_factors = []
    for days in [0, 5, 10, 15, 20, 25, 30]:
        kalman_state = {
            "last_timestamp": base_timestamp,
            "current_timestamp": base_timestamp + timedelta(days=days),
            "measurements_since_reset": 50,
        }

        _, metadata = scorer.calculate_kalman_fit(
            weight=weight,
            kalman_prediction=kalman_prediction,
            innovation_covariance=innovation_covariance,
            kalman_state=kalman_state,
        )

        decay_factor = metadata.get("decay_factor", 0)
        decay_factors.append(decay_factor)

        # Verify linear decay formula: days/30
        expected_decay = min(1.0, days / 30.0)
        assert abs(decay_factor - expected_decay) < 0.001, (
            f"Day {days}: expected decay={expected_decay:.3f}, actual={decay_factor:.3f}"
        )


def test_kalman_fit_no_decay_without_gap():
    """Test that there's no decay when there's no time gap."""
    scorer = UnifiedQualityScorer()

    weight = 80.0
    kalman_prediction = 80.0  # Perfect match
    innovation_covariance = 1.0

    # No time gap scenario
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    kalman_state = {
        "last_timestamp": timestamp,
        "current_timestamp": timestamp,
        "measurements_since_reset": 50,
    }

    score, metadata = scorer.calculate_kalman_fit(
        weight=weight,
        kalman_prediction=kalman_prediction,
        innovation_covariance=innovation_covariance,
        kalman_state=kalman_state,
    )

    # With perfect match and no gap, score should be very high
    assert score > 0.95, (
        f"Perfect match with no gap should have high score: {score:.3f}"
    )
    assert "decay_factor" not in metadata or metadata["decay_factor"] == 0, (
        "No decay factor should be applied with no time gap"
    )


def test_kalman_fit_adaptive_period_with_decay():
    """Test that adaptive period and time decay work together."""
    scorer = UnifiedQualityScorer()

    weight = 80.0
    kalman_prediction = 75.0
    innovation_covariance = 1.0
    base_timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # In adaptive period with 15-day gap
    kalman_state = {
        "last_timestamp": base_timestamp,
        "current_timestamp": base_timestamp + timedelta(days=15),
        "measurements_since_reset": 3,  # In adaptive period
        "reset_parameters": {"adaptation_measurements": 10},
    }

    score, metadata = scorer.calculate_kalman_fit(
        weight=weight,
        kalman_prediction=kalman_prediction,
        innovation_covariance=innovation_covariance,
        kalman_state=kalman_state,
    )

    # Should have both adaptive period flag and decay factor
    assert metadata.get("adaptive_period") == True, "Should be in adaptive period"
    assert metadata.get("decay_factor", 0) > 0, "Should have decay factor"
    assert 0.45 < metadata["decay_factor"] < 0.55, (
        f"15 days should give ~0.5 decay factor: {metadata['decay_factor']:.3f}"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    test_kalman_fit_decay_over_time()
    test_kalman_fit_decay_gradual()
    test_kalman_fit_no_decay_without_gap()
    test_kalman_fit_adaptive_period_with_decay()
    print("\nAll tests passed!")
