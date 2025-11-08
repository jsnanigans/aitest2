#!/usr/bin/env python3
"""
Quality Component Breakdown for Divergent Measurement

Processes the first divergent measurement (#1) and extracts
detailed breakdown of each quality component
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
from weight_processor_lib.core.processing.processor import Processor


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
        "replay": {
            "buffered_replay_enabled": False,
            "buffer_hours": 24,
            "max_buffer_measurements": 100,
        },
    }


def main():
    csv_path = "test_user.csv"
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"
    target_measurement_id = "0bb4ca6c-d123-4461-8cae-a40297230843"  # Divergent measurement #1

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

    target_index = next(
        (i for i, m in enumerate(measurements) if m["measurementId"] == target_measurement_id),
        None,
    )

    print(f"\n=== Quality Component Breakdown ===")
    print(f"Target measurement: {target_measurement_id}")
    print(f"Processing first {target_index + 1} measurements...\n")

    # Initialize
    state_store = InMemoryStore()
    config = get_default_config()
    processor = Processor(user_id, state_store, config)

    # Process measurements up to and including target
    for i, m_data in enumerate(measurements[: target_index + 1]):
        print(f"\n[{i}] Processing {m_data['measurementId'][:8]}...")
        print(f"   Timestamp: {m_data['timestamp'].isoformat()}")
        print(f"   Weight: {m_data['weight']} {m_data['unit']}")

        result = processor.process_measurement(
            measurement_id=m_data["measurementId"],
            weight=m_data["weight"],
            unit=m_data["unit"],
            timestamp=m_data["timestamp"],
            source=m_data["source"],
        )

        print(f"   Accepted: {result.accepted}")
        print(f"   Quality Score: {result.quality_score:.15f}")

        # For the target measurement, get detailed breakdown
        if m_data["measurementId"] == target_measurement_id:
            print(f"\n🎯 DETAILED BREAKDOWN FOR DIVERGENT MEASUREMENT:")

            if result.quality_details:
                print(f"\n   Quality Components:")

                components = result.quality_details.get("component_scores", {})
                weights = config["quality_scoring"]["components"]

                for name, score in components.items():
                    weight = weights.get(name, {}).get("weight", 0)
                    contribution = score * weight
                    print(f"     {name}:")
                    print(f"       Raw score: {score:.15f}")
                    print(f"       Weight: {weight}")
                    print(f"       Contribution: {contribution:.15f}")

                print(f"\n   Final Quality Score: {result.quality_score:.15f}")
                print(f"   Threshold: 0.5")
                print(f"   Decision: {'ACCEPT' if result.accepted else 'REJECT'}")

                # Check if detailed metrics available
                if "metrics" in result.quality_details:
                    print(f"\n   Additional Metrics:")
                    print(f"   {json.dumps(result.quality_details['metrics'], indent=2)}")
            else:
                print(f"   ⚠️  No quality_details available in result")

            # Get Kalman state
            state = state_store.get_state(user_id)
            if state and "kalman_filter" in state:
                kf = state["kalman_filter"]
                print(f"\n   Kalman State:")
                print(f"     Weight: {kf['state'][0]:.15f}")
                print(f"     Velocity: {kf['state'][1]:.15f}")
                cov = kf["covariance"]
                print(f"     Covariance: [[{', '.join(f'{v:.6f}' for v in cov[0])}],")
                print(f"                  [{', '.join(f'{v:.6f}' for v in cov[1])}]]")

            # Write detailed results
            breakdown = {
                "measurementId": m_data["measurementId"],
                "measurementIndex": i,
                "timestamp": m_data["timestamp"].isoformat(),
                "weight": m_data["weight"],
                "unit": m_data["unit"],
                "source": m_data["source"],
                "qualityScore": result.quality_score,
                "accepted": result.accepted,
                "qualityDetails": result.quality_details,
                "kalmanState": (
                    {
                        "weight": float(state["kalman_filter"]["state"][0]),
                        "velocity": float(state["kalman_filter"]["state"][1]),
                        "covariance": state["kalman_filter"]["covariance"],
                    }
                    if state and "kalman_filter" in state
                    else None
                ),
            }

            with open("py_quality_breakdown.json", "w") as f:
                json.dump(breakdown, f, indent=2)

            print(f"\n✅ Detailed breakdown written to py_quality_breakdown.json")


if __name__ == "__main__":
    main()
