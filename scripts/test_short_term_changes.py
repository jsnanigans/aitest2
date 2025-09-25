#!/usr/bin/env python3
"""Test the improvements to short-term change handling."""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.unified_quality_scorer import UnifiedQualityScorer


def test_rapid_measurements():
    """Test handling of rapid measurements."""
    scorer = UnifiedQualityScorer()

    print("Testing Rapid Measurement Handling")
    print("=" * 50)

    # Test cases: (time_diff_seconds, weight_change, source, expected_behavior)
    test_cases = [
        # True duplicates (< 5 seconds, same weight)
        (3, 0.02, "patient-device", "reject_duplicate"),
        (4, 0.04, "patient-device", "reject_duplicate"),
        # Rapid but different (< 5 seconds, different weight)
        (3, 0.15, "patient-device", "accept_with_penalty"),
        (4, 0.25, "patient-device", "accept_with_penalty"),
        # 1 minute changes
        (60, 0.3, "patient-device", "accept"),
        (60, 0.5, "patient-device", "accept"),
        (60, 1.2, "patient-device", "reject_impossible"),  # Over 2x threshold
        # 2-5 minute changes
        (120, 0.7, "patient-device", "accept"),
        (180, 0.8, "patient-device", "accept"),
        (240, 0.9, "patient-device", "accept"),
        (300, 1.0, "patient-device", "accept"),
        (180, 2.5, "patient-device", "reject_impossible"),  # Way over threshold
        # Manual entries (should be slightly stricter)
        (60, 0.4, "patient-upload", "accept"),
        (60, 0.8, "patient-upload", "accept_with_penalty"),
        (60, 1.5, "patient-upload", "reject_impossible"),
    ]

    base_timestamp = datetime.now()

    for time_diff_sec, weight_change, source, expected in test_cases:
        # Set up recent data
        recent_weights = [100.0]
        recent_timestamps = [base_timestamp - timedelta(seconds=time_diff_sec)]
        current_weight = 100.0 + weight_change

        # Calculate score
        score_result = scorer.calculate_anomaly_detection(
            weight=current_weight,
            recent_weights=recent_weights,
            recent_timestamps=recent_timestamps,
            user_height_m=1.70,
            current_timestamp=base_timestamp,
        )

        score, metadata = score_result

        # Determine actual behavior
        if score == 0.0:
            if metadata.get("rejected_reason") == "duplicate_measurement":
                actual = "reject_duplicate"
            else:
                actual = "reject_impossible"
        elif score < 0.6:
            actual = "accept_with_penalty"
        else:
            actual = "accept"

        # Check if it matches expected
        status = "✓" if actual == expected else "✗"

        print(
            f"{status} {time_diff_sec:3d}s, {weight_change:4.2f}kg, {source:20s}: "
            f"Score={score:.2f}, Expected={expected:20s}, Actual={actual:20s}"
        )

        if actual != expected:
            print(f"   Metadata: {metadata}")

    print()


def test_user_scenario():
    """Test a real user scenario with multiple rapid measurements."""
    scorer = UnifiedQualityScorer()

    print("Testing Real User Scenario: Multiple Rapid Measurements")
    print("=" * 50)

    # Simulate user 39fce2da-03b2-4bce-8a3e-5622009a3287's pattern
    base_time = datetime.now()
    measurements = [
        (base_time, 103.15, "patient-device"),
        (
            base_time + timedelta(seconds=50),
            103.22,
            "patient-device",
        ),  # 50 seconds later, +0.07kg
        (base_time + timedelta(minutes=2), 103.18, "patient-device"),  # 2 min, -0.04kg
        (base_time + timedelta(minutes=3), 103.25, "patient-device"),  # 3 min, +0.07kg
        (base_time + timedelta(minutes=5), 103.20, "patient-device"),  # 5 min, -0.05kg
    ]

    recent_weights = []
    recent_timestamps = []

    for timestamp, weight, source in measurements:
        if len(recent_weights) > 0:
            score_result = scorer.calculate_anomaly_detection(
                weight=weight,
                recent_weights=recent_weights,
                recent_timestamps=recent_timestamps,
                user_height_m=1.70,
                current_timestamp=timestamp,
            )

            score, metadata = score_result

            time_diff = (timestamp - recent_timestamps[-1]).total_seconds()
            weight_diff = weight - recent_weights[-1]

            status = "ACCEPTED" if score > 0.0 else "REJECTED"
            print(
                f"{timestamp.strftime('%H:%M:%S')}: {weight:.2f}kg ({weight_diff:+.2f}kg in {time_diff:.0f}s)"
            )
            print(f"  Status: {status}, Score: {score:.3f}")
            if score == 0.0:
                print(
                    f"  Rejection reason: {metadata.get('rejected_reason', 'unknown')}"
                )
            print()

        recent_weights.append(weight)
        recent_timestamps.append(timestamp)


def test_extreme_cases():
    """Test extreme but valid cases."""
    scorer = UnifiedQualityScorer()

    print("Testing Extreme But Valid Cases")
    print("=" * 50)

    # User 1ff23e8b-75c8-4048-a087-86e334e61065 pattern (3kg variations)
    base_time = datetime.now()

    # Case 1: Large change after reasonable time
    recent_weights = [100.0]
    recent_timestamps = [base_time - timedelta(hours=2)]

    score, metadata = scorer.calculate_anomaly_detection(
        weight=103.0,  # 3kg change in 2 hours
        recent_weights=recent_weights,
        recent_timestamps=recent_timestamps,
        user_height_m=1.70,
        current_timestamp=base_time,
    )

    print(
        f"3kg change in 2 hours: Score={score:.3f} (should be accepted with some penalty)"
    )

    # Case 2: Multiple measurements with variance (scale noise)
    measurements = [
        (base_time, 100.0),
        (base_time + timedelta(minutes=1), 100.3),
        (base_time + timedelta(minutes=2), 99.8),
        (base_time + timedelta(minutes=3), 100.1),
        (base_time + timedelta(minutes=4), 100.4),
    ]

    print("\nScale variance pattern (±0.4kg):")
    for i, (ts, weight) in enumerate(measurements[1:], 1):
        recent_weights = [m[1] for m in measurements[:i]]
        recent_timestamps = [m[0] for m in measurements[:i]]

        score, metadata = scorer.calculate_anomaly_detection(
            weight=weight,
            recent_weights=recent_weights,
            recent_timestamps=recent_timestamps,
            user_height_m=1.70,
            current_timestamp=ts,
        )

        print(f"  Measurement {i + 1}: {weight:.1f}kg, Score={score:.3f}")


if __name__ == "__main__":
    test_rapid_measurements()
    test_user_scenario()
    test_extreme_cases()
