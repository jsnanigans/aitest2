#!/usr/bin/env python3
"""Analyze short-term weight changes and rejection patterns."""

import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.constants import REJECTION_REASONS


def analyze_user_rejections(processor, user_id, df):
    """Analyze rejections for a specific user."""
    user_data = df[df["user_id"] == user_id].sort_values("effectiveDateTime")
    if len(user_data) == 0:
        return None

    print(f"\n{'=' * 60}")
    print(f"User: {user_id}")
    print(f"Total measurements: {len(user_data)}")

    # Process the user's data
    processor.reset_user(user_id)
    results = []

    for idx, row in user_data.iterrows():
        result = processor.process_measurement(
            user_id=user_id,
            timestamp=row["effectiveDateTime"],
            weight=row["weight"],
            source=row["source_type"],
        )
        results.append(result)

    # Analyze rejections
    rejected = [r for r in results if not r["accepted"]]
    print(f"Rejected measurements: {len(rejected)}")

    # Analyze close-in-time rejections
    close_rejections = []
    for i, result in enumerate(results):
        if not result["accepted"] and i > 0:
            prev_result = results[i - 1]
            time_delta = (
                result["timestamp"] - prev_result["timestamp"]
            ).total_seconds() / 60
            if time_delta < 120:  # Within 2 hours
                weight_delta = abs(result["weight"] - prev_result["weight"])
                close_rejections.append(
                    {
                        "timestamp": result["timestamp"],
                        "weight": result["weight"],
                        "prev_weight": prev_result["weight"],
                        "weight_delta": weight_delta,
                        "time_delta_minutes": time_delta,
                        "reason": result.get("rejection_reason", "unknown"),
                        "quality_score": result.get("quality_score", 0),
                    }
                )

    if close_rejections:
        print(
            f"\nRejections within 2 hours of previous measurement: {len(close_rejections)}"
        )
        df_close = pd.DataFrame(close_rejections)

        # Group by time ranges
        time_ranges = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 60), (60, 120)]
        for min_time, max_time in time_ranges:
            range_data = df_close[
                (df_close["time_delta_minutes"] >= min_time)
                & (df_close["time_delta_minutes"] < max_time)
            ]
            if len(range_data) > 0:
                print(f"\n  {min_time}-{max_time} minutes:")
                print(f"    Count: {len(range_data)}")
                print(f"    Max weight delta: {range_data['weight_delta'].max():.2f}kg")
                print(
                    f"    Mean weight delta: {range_data['weight_delta'].mean():.2f}kg"
                )
                print(f"    Reasons: {range_data['reason'].value_counts().to_dict()}")

                # Show examples
                if len(range_data) <= 3:
                    for _, row in range_data.iterrows():
                        print(
                            f"      {row['timestamp']}: {row['weight']:.2f}kg (Δ{row['weight_delta']:.2f}kg, Q:{row['quality_score']:.2f})"
                        )

    # Analyze consecutive measurements (same timestamp or < 1 minute)
    consecutive_groups = []
    current_group = []

    for i, row in enumerate(user_data.itertuples()):
        if i > 0:
            prev_row = user_data.iloc[i - 1]
            time_delta = (
                row.effectiveDateTime - prev_row["effectiveDateTime"]
            ).total_seconds() / 60
            if time_delta < 1:
                if not current_group:
                    current_group.append(prev_row)
                current_group.append(user_data.iloc[i])
            elif current_group:
                consecutive_groups.append(current_group)
                current_group = []

    if current_group:
        consecutive_groups.append(current_group)

    if consecutive_groups:
        print(
            f"\nConsecutive measurement groups (<1 min apart): {len(consecutive_groups)}"
        )
        for group in consecutive_groups[:3]:  # Show first 3 groups
            weights = [r["weight"] for r in group]
            weight_range = max(weights) - min(weights)
            print(f"  Group at {group[0]['effectiveDateTime']}:")
            print(f"    Measurements: {len(group)}")
            print(
                f"    Weight range: {weight_range:.2f}kg ({min(weights):.2f} - {max(weights):.2f})"
            )

    return {
        "user_id": user_id,
        "total_measurements": len(user_data),
        "rejected": len(rejected),
        "close_rejections": close_rejections,
    }


def main():
    # Read data
    df = pd.read_csv("data/2025-09-05_nocon.csv")
    df["effectiveDateTime"] = pd.to_datetime(df["effectiveDateTime"])

    # Initialize processor
    processor = WeightProcessor(config_path="config.toml")

    # Users to analyze
    users = [
        "39fce2da-03b2-4bce-8a3e-5622009a3287",
        "8ad2a7f4-fd1a-4ac6-9bd0-4d12ec64e55b",
        "1401eed2-0ebf-4814-9710-244b4f309251",
        "678639d5-e18b-4014-8859-fa3f1b436a99",
        "1fe1e802-101b-4a5c-8480-16b4f00a638e",
        "e4b35fbc-3611-41fc-95da-9e279f3f4ace",
        "740c86a2-e358-46b2-8095-002001da8726",
        "1ff23e8b-75c8-4048-a087-86e334e61065",
    ]

    all_results = []
    for user_id in users:
        result = analyze_user_rejections(processor, user_id, df)
        if result:
            all_results.append(result)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY ANALYSIS")
    print("=" * 60)

    # Collect all close rejections
    all_close_rejections = []
    for result in all_results:
        all_close_rejections.extend(result["close_rejections"])

    if all_close_rejections:
        df_all = pd.DataFrame(all_close_rejections)

        print("\nWeight deltas by time range (all users):")
        time_ranges = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 60), (60, 120)]

        recommendations = []
        for min_time, max_time in time_ranges:
            range_data = df_all[
                (df_all["time_delta_minutes"] >= min_time)
                & (df_all["time_delta_minutes"] < max_time)
            ]
            if len(range_data) > 0:
                p95 = range_data["weight_delta"].quantile(0.95)
                p99 = range_data["weight_delta"].quantile(0.99)
                max_delta = range_data["weight_delta"].max()

                print(f"\n{min_time}-{max_time} minutes:")
                print(f"  Count: {len(range_data)}")
                print(f"  95th percentile: {p95:.2f}kg")
                print(f"  99th percentile: {p99:.2f}kg")
                print(f"  Max: {max_delta:.2f}kg")

                # Recommendation
                threshold = p99 * 1.2  # 20% margin above 99th percentile
                recommendations.append(
                    {
                        "min_minutes": min_time,
                        "max_minutes": max_time,
                        "threshold_kg": threshold,
                        "samples": len(range_data),
                    }
                )

        print("\n" + "=" * 60)
        print("RECOMMENDED THRESHOLDS")
        print("=" * 60)

        for rec in recommendations:
            if rec["samples"] > 5:  # Only if we have enough samples
                print(
                    f"{rec['min_minutes']}-{rec['max_minutes']} min: {rec['threshold_kg']:.2f}kg"
                )

        # Analyze rejection reasons
        print("\n" + "=" * 60)
        print("REJECTION REASONS")
        print("=" * 60)
        reason_counts = df_all["reason"].value_counts()
        for reason, count in reason_counts.items():
            print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
