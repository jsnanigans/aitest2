#!/usr/bin/env python3
"""
Phase 1: Sequential Comparison - Find First Divergence Point

Processes measurements one-by-one and outputs Kalman state after each.
This helps identify EXACTLY where TS and Python first diverge.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "replay": {
            "buffered_replay_enabled": True,
            "buffer_hours": 24,
            "max_buffer_measurements": 100,
        },
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
                    "id": measurement_id,
                    "weight": weight,
                    "unit": unit,
                    "timestamp": timestamp,
                    "source": source,
                }
            )

    # Sort chronologically
    measurements.sort(key=lambda m: m["timestamp"])

    print(f"Loaded {len(measurements)} measurements for user {user_id[:12]}...")

    # Initialize service
    state_store = InMemoryStore()
    config = get_default_config()
    service = WeightProcessorService(state_store, config)

    # Process ONE measurement at a time and capture state
    snapshots = []

    for i, m_data in enumerate(measurements):
        measurement = Measurement(
            uuid=m_data["id"],
            weight=m_data["weight"],
            unit=m_data["unit"],
            effectiveDateTime=m_data["timestamp"],
            source=m_data["source"],
        )

        # Process single measurement
        response = service.process_batch(user_id, [measurement])
        result = response.results[0]

        # Get current Kalman state from store
        state = state_store.get_state(user_id)

        # Extract Kalman state if available
        kalman_state = None
        kalman_covariance = None
        process_noise = None

        if state and "kalman_filter" in state:
            kf = state["kalman_filter"]
            if "state" in kf:
                kalman_state = {
                    "weight": float(kf["state"][0]),
                    "velocity": float(kf["state"][1]),
                }
            if "covariance" in kf:
                kalman_covariance = kf["covariance"]
            if "process_noise" in kf:
                process_noise = kf["process_noise"]

        snapshot = {
            "measurementIndex": i,
            "measurementId": m_data["id"],
            "timestamp": m_data["timestamp"].isoformat(),
            "weight": m_data["weight"],
            "accepted": result.accepted,
            "qualityScore": result.quality_score,
            "kalmanState": kalman_state,
            "kalmanCovariance": kalman_covariance,
            "processNoise": process_noise,
        }

        snapshots.append(snapshot)

        # Log progress every 10 measurements
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(measurements)} measurements...")

    # Write results to JSON
    output_path = "py_sequential_states.json"
    with open(output_path, "w") as f:
        json.dump(snapshots, f, indent=2)

    print(f"\n✅ Sequential state snapshots written to: {output_path}")
    print(f"Total measurements processed: {len(snapshots)}")
    print(f"Accepted: {sum(1 for s in snapshots if s['accepted'])}")
    print(f"Rejected: {sum(1 for s in snapshots if not s['accepted'])}")


if __name__ == "__main__":
    main()
