#!/usr/bin/env python3
"""
Test script to verify cliffing issue fix.
Compares results before and after temporal_consistency weight adjustment.
"""

import json
import sys
from pathlib import Path


def analyze_cliff_user(results_file):
    """Analyze the cliff user's data for rapid changes."""

    user_id = "678639d5-e18b-4014-8859-fa3f1b436a99"

    with open(results_file, "r") as f:
        data = json.load(f)

    if user_id not in data.get("users", {}):
        print(f"User {user_id} not found in results")
        return None

    measurements = data["users"][user_id]

    # Find rapid measurement sequences (< 1 hour apart)
    rapid_sequences = []
    cliff_events = []

    for i in range(1, len(measurements)):
        curr = measurements[i]
        prev = measurements[i - 1]

        if not curr.get("accepted") or not prev.get("accepted"):
            continue

        # Parse timestamps
        from datetime import datetime

        curr_time = datetime.fromisoformat(curr["timestamp"].replace(" ", "T"))
        prev_time = datetime.fromisoformat(prev["timestamp"].replace(" ", "T"))

        time_diff_hours = (curr_time - prev_time).total_seconds() / 3600

        if curr.get("filtered_weight") and prev.get("filtered_weight"):
            weight_change = abs(curr["filtered_weight"] - prev["filtered_weight"])

            # Detect rapid measurements
            if time_diff_hours < 1.0:
                rapid_sequences.append(
                    {
                        "timestamp": curr["timestamp"],
                        "time_diff_hours": time_diff_hours,
                        "weight_change_kg": weight_change,
                        "quality_score": curr.get("quality_score", 0),
                        "filtered_weight": curr["filtered_weight"],
                    }
                )

            # Detect cliff events (large change in short time)
            if time_diff_hours < 24 and weight_change > 1.0:
                rate_of_change = weight_change / max(time_diff_hours, 0.01)
                if rate_of_change > 0.5:  # > 0.5 kg/hour
                    cliff_events.append(
                        {
                            "timestamp": curr["timestamp"],
                            "time_diff_hours": time_diff_hours,
                            "weight_change_kg": weight_change,
                            "rate_kg_per_hour": rate_of_change,
                            "quality_score": curr.get("quality_score", 0),
                        }
                    )

    # Analyze March 13-24 period specifically (known problem period)
    march_measurements = [m for m in measurements if "2025-03" in m["timestamp"]]
    march_accepted = [m for m in march_measurements if m.get("accepted")]

    print(f"\n=== Analysis for User {user_id} ===")
    print(f"Total measurements: {len(measurements)}")
    print(f"Accepted measurements: {sum(1 for m in measurements if m.get('accepted'))}")
    print(f"\nRapid sequences (< 1 hour apart): {len(rapid_sequences)}")

    if rapid_sequences:
        avg_quality = sum(r["quality_score"] for r in rapid_sequences) / len(
            rapid_sequences
        )
        print(f"Average quality score for rapid measurements: {avg_quality:.3f}")

        # Show some examples
        print("\nExample rapid measurements:")
        for r in rapid_sequences[:5]:
            print(
                f"  {r['timestamp']}: {r['time_diff_hours']:.2f}h apart, "
                f"change={r['weight_change_kg']:.2f}kg, quality={r['quality_score']:.3f}"
            )

    print(f"\nCliff events (rapid large changes): {len(cliff_events)}")
    if cliff_events:
        print("Example cliff events:")
        for c in cliff_events[:3]:
            print(
                f"  {c['timestamp']}: {c['weight_change_kg']:.2f}kg in {c['time_diff_hours']:.1f}h "
                f"({c['rate_kg_per_hour']:.2f}kg/h), quality={c['quality_score']:.3f}"
            )

    print(f"\nMarch 2025 period analysis:")
    print(f"  Total measurements: {len(march_measurements)}")
    print(f"  Accepted: {len(march_accepted)}")
    print(
        f"  Acceptance rate: {len(march_accepted) / len(march_measurements) * 100:.1f}%"
    )

    # Calculate trajectory smoothness (variance of filtered weights)
    if march_accepted:
        filtered_weights = [
            m["filtered_weight"] for m in march_accepted if m.get("filtered_weight")
        ]
        if len(filtered_weights) > 1:
            import numpy as np

            weight_diffs = np.diff(filtered_weights)
            smoothness = np.std(weight_diffs)
            print(f"  Trajectory smoothness (lower=smoother): {smoothness:.3f}")

    return {
        "rapid_count": len(rapid_sequences),
        "cliff_count": len(cliff_events),
        "march_acceptance_rate": len(march_accepted) / len(march_measurements) * 100
        if march_measurements
        else 0,
    }


def main():
    # Check if results file exists
    results_file = Path("output/results_test_no_date.json")
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run: uv run python main.py data/weights.csv")
        sys.exit(1)

    print("Analyzing current results (with temporal_consistency weight = 0)...")
    current_stats = analyze_cliff_user(results_file)

    print("\n" + "=" * 60)
    print("RECOMMENDED FIX:")
    print("=" * 60)
    print("\nEdit config.toml and change:")
    print("\n[quality_scoring.component_weights]")
    print("kalman_fit = 0.40              # (was 0.65)")
    print("temporal_consistency = 0.30    # (was 0!!)")
    print("anomaly_detection = 0.20       # (was 0.25)")
    print("source_reliability = 0.05      # (was 0)")
    print("trend_alignment = 0.05         # (was 0.1)")

    print("\n" + "=" * 60)
    print("\nAfter making the config change, re-run:")
    print("  uv run python main.py data/weights.csv")
    print("  uv run python scripts/test_cliffing_fix.py")
    print("\nExpected improvements:")
    print("  - Fewer rapid sequences accepted")
    print("  - Lower quality scores for rapid measurements")
    print("  - Smoother trajectory (no cliffs)")
    print("  - Lower acceptance rate for March burst period")


if __name__ == "__main__":
    main()
