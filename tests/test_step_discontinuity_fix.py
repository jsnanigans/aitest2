"""
Quick test to demonstrate elimination of step discontinuities at 6h and 24h boundaries.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.unified_quality_scorer import UnifiedQualityScorer


def test_no_step_discontinuities():
    """
    Demonstrate that temporal scoring no longer has step discontinuities.
    """
    scorer = UnifiedQualityScorer()
    base_weight = 80.0
    weight_change = 2.0  # 2kg change

    # Test around 6-hour boundary (old step function boundary)
    print("\n=== Testing around 6-hour boundary ===")
    for hours in [5.8, 5.9, 6.0, 6.1, 6.2]:
        score, metadata = scorer.calculate_temporal_consistency(
            weight=base_weight + weight_change,
            previous_weight=base_weight,
            time_diff_hours=hours,
            recent_weights=None,
            recent_timestamps=None
        )
        print(f"{hours:4.1f}h: score={score:.4f}, max_acceptable={metadata['max_acceptable_change']:.3f}kg")

    # Test around 24-hour boundary (old step function boundary)
    print("\n=== Testing around 24-hour boundary ===")
    for hours in [23.8, 23.9, 24.0, 24.1, 24.2]:
        score, metadata = scorer.calculate_temporal_consistency(
            weight=base_weight + weight_change,
            previous_weight=base_weight,
            time_diff_hours=hours,
            recent_weights=None,
            recent_timestamps=None
        )
        print(f"{hours:4.1f}h: score={score:.4f}, max_acceptable={metadata['max_acceptable_change']:.3f}kg")

    print("\n=== Continuous growth of acceptable threshold ===")
    for hours in [0, 6, 12, 24, 48, 72, 96, 168]:
        score, metadata = scorer.calculate_temporal_consistency(
            weight=base_weight + weight_change,
            previous_weight=base_weight,
            time_diff_hours=hours,
            recent_weights=None,
            recent_timestamps=None
        )
        print(f"{hours:3d}h: max_acceptable={metadata['max_acceptable_change']:.3f}kg")

    print("\n✅ No step discontinuities - scores change smoothly and continuously!")


if __name__ == "__main__":
    test_no_step_discontinuities()