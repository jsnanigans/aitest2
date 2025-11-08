#!/usr/bin/env python3
"""
Extract and compare quality component scores for divergent measurement
Uses service layer which returns quality_details
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "python_lib" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "be_implementation_service" / "src"))

from weight_processor_lib.core.database.memory_store import InMemoryStore
from aws.services.weight_processor_service import WeightProcessorService
from aws.api.models import Measurement


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats"""
    if not date_str:
        return datetime.now(timezone.utc)

    try:
        if "T" in date_str:
            if date_str.endswith("Z"):
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                return datetime.fromisoformat(date_str)
        elif " " in date_str:
            return datetime.fromisoformat(date_str.replace(" ", "T") + "+00:00")
        else:
            return datetime.fromisoformat(date_str + "T00:00:00+00:00")
    except Exception:
        return datetime.now(timezone.utc)


def get_default_config():
    """Get default configuration"""
    return {
        "database": {"backend": "memory"},
        "kalman": {
            "initial_variance": 0.364,
            "transition_covariance_weight": 0.018,
            "transition_covariance_trend": 0.00015,
            "observation_covariance": 3.49,
        },
        "quality_scoring": {
            "threshold": 0.5,
            "components": {
                "kalman_fit": {"weight": 0.3, "enabled": True},
                "temporal_consistency": {"weight": 0.25, "enabled": True},
                "anomaly_detection": {"weight": 0.25, "enabled": True},
                "source_reliability": {"weight": 0.1, "enabled": True},
                "trend_alignment": {"weight": 0.1, "enabled": True},
            },
        },
        "processing": {"enable_validation": True, "enable_quality_scoring": True},
        "reset": {"time_gap_days": 30, "weight_change_threshold_kg": 10},
        "snapshot": {"interval_hours": 24, "periodic_enabled": True},
        "adaptive_noise": {"enabled": True},
        "replay": {"buffered_replay_enabled": False},  # Disable replay for clean test
    }


def main():
    csv_path = "test_user.csv"
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    # Load CSV
    measurements = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurement_id = row.get("id") or row.get("measurement_id")
            if not measurement_id or row.get("user_id") != user_id:
                continue

            weight_str = (row.get("value_quantity") or row.get("weight") or "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue

            try:
                weight = float(weight_str)
                if weight <= 0 or weight > 1000:
                    continue
            except ValueError:
                continue

            date_str = (
                row.get("effective_date_time")
                or row.get("effectiveDateTime")
                or row.get("timestamp")
                or ""
            )
            source = row.get("source_type") or "unknown"
            unit = (row.get("unit") or "").strip()
            timestamp = parse_timestamp(date_str)

            measurements.append(
                {
                    "measurementId": measurement_id,
                    "weight": weight,
                    "unit": unit,
                    "timestamp": timestamp,
                    "source": source,
                }
            )

    # Sort chronologically
    measurements.sort(key=lambda m: m["timestamp"])

    print("\n=== Quality Component Comparison ===\n")

    # Initialize service
    state_store = InMemoryStore()
    config = get_default_config()
    service = WeightProcessorService(state_store, config)

    # Process first 2 measurements one at a time
    results = []

    for i in range(2):
        m_data = measurements[i]

        print(f"\n[{i}] Processing {m_data['measurementId'][:8]}...")
        print(f"    Timestamp: {m_data['timestamp'].isoformat()}")
        print(f"    Weight: {m_data['weight']} {m_data['unit']}")

        measurement = Measurement(
            uuid=m_data["measurementId"],
            weight=m_data["weight"],
            unit=m_data["unit"],
            effectiveDateTime=m_data["timestamp"],
            source=m_data["source"],
        )

        response = service.process_batch(user_id, [measurement])
        result = response.results[0]

        print(f"    Quality Score: {result.quality_score:.15f}")
        print(f"    Accepted: {result.accepted}")

        # Extract component scores
        if result.quality_components:
            components = result.quality_components
            weights = config["quality_scoring"]["components"]

            print(f"\n    Component Breakdown:")
            for name, score in components.items():
                weight = weights.get(name, {}).get("weight", 0)
                contribution = score * weight
                print(
                    f"      {name:<25} {score:.15f} × {weight:.2f} = {contribution:.15f}"
                )

            results.append(
                {
                    "index": i,
                    "measurementId": m_data["measurementId"],
                    "timestamp": m_data["timestamp"].isoformat(),
                    "weight": m_data["weight"],
                    "qualityScore": result.quality_score,
                    "accepted": result.accepted,
                    "componentScores": components,
                    "componentWeights": {
                        name: weights.get(name, {}).get("weight", 0)
                        for name in components.keys()
                    },
                    "componentContributions": {
                        name: score * weights.get(name, {}).get("weight", 0)
                        for name, score in components.items()
                    },
                }
            )
        else:
            print(f"    ⚠️  No quality_details in result")
            results.append(
                {
                    "index": i,
                    "measurementId": m_data["measurementId"],
                    "timestamp": m_data["timestamp"].isoformat(),
                    "weight": m_data["weight"],
                    "qualityScore": result.quality_score,
                    "accepted": result.accepted,
                    "error": "No quality_details available",
                }
            )

    # Focus on measurement #1 (the divergent one)
    print("\n\n" + "=" * 80)
    print("🎯 DIVERGENT MEASUREMENT #1 BREAKDOWN")
    print("=" * 80)

    divergent = results[1]
    print(f"\nMeasurement ID: {divergent['measurementId']}")
    print(f"Timestamp: {divergent['timestamp']}")
    print(f"Weight: {divergent['weight']} kg")
    print(f"\nPython Quality Score: {divergent['qualityScore']:.15f}")
    print(f"Expected TypeScript:   0.665926676041184 (31% lower!)")

    if "componentScores" in divergent:
        print(f"\nComponent Scores (Python):")
        print(f"{'Component':<25} {'Raw Score':<20} Weight  Contribution")
        print("=" * 80)

        for name, score in divergent["componentScores"].items():
            weight = divergent["componentWeights"][name]
            contribution = divergent["componentContributions"][name]
            print(
                f"{name:<25} {score:.15f}   {weight:.2f}    {contribution:.15f}"
            )

        # Calculate sum
        total = sum(divergent["componentContributions"].values())
        print("=" * 80)
        print(f"{'TOTAL':<25} {'':<20} {'':6} {total:.15f}")
        print(
            f"\n✅ Sum matches quality_score: {abs(total - divergent['qualityScore']) < 1e-10}"
        )

    # Write results
    with open("py_quality_components.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results written to py_quality_components.json")

    # Compare with TypeScript if available
    try:
        with open("ts_quality_components.json", "r") as f:
            ts_results = json.load(f)

        print("\n\n" + "=" * 80)
        print("📊 COMPONENT-BY-COMPONENT COMPARISON")
        print("=" * 80)

        ts_divergent = ts_results[1]
        py_divergent = results[1]

        if "componentScores" in ts_divergent and "componentScores" in py_divergent:
            print(f"\n{'Component':<25} {'TypeScript':<20} {'Python':<20} {'Difference':<15} Status")
            print("=" * 100)

            all_components = set(ts_divergent["componentScores"].keys()) | set(
                py_divergent["componentScores"].keys()
            )

            max_diff = 0
            max_diff_component = None

            for component in sorted(all_components):
                ts_score = ts_divergent["componentScores"].get(component, 0)
                py_score = py_divergent["componentScores"].get(component, 0)
                diff = abs(ts_score - py_score)

                if diff > max_diff:
                    max_diff = diff
                    max_diff_component = component

                status = "✓" if diff < 0.01 else "✗"
                print(
                    f"{component:<25} {ts_score:.15f}   {py_score:.15f}   {diff:.15f}  {status}"
                )

            print("=" * 100)
            print(f"\n🔴 LARGEST DIVERGENCE: {max_diff_component}")
            print(f"   Difference: {max_diff:.15f}")
            print(
                f"   This component is causing the quality score divergence!"
            )

    except FileNotFoundError:
        print(
            "\n⚠️  TypeScript results not found. Run test_quality_components.ts first."
        )


if __name__ == "__main__":
    main()
