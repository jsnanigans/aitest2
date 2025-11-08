#!/usr/bin/env python3
"""
Test disputed measurements to see exact quality scores
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

# Add python_lib to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "python_lib" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "be_implementation_service" / "src"))

from weight_processor_lib.core.database.memory_store import InMemoryStore
from aws.services.weight_processor_service import WeightProcessorService
from aws.api.models import Measurement

# IDs that differ between TS and Python
DISPUTED_IDS = [
    "5b022b9c-509e-4a9f-bd5c-7857733bf2f8",  # TS accepts, Python rejects
    "726b441f-eb43-47d9-8f3c-845d164e5a5b",  # TS accepts, Python rejects
    "d957b0de-58fc-4e96-b351-a81cfc10e54c",  # TS accepts, Python rejects
    "df8e3da2-5f5d-4177-b535-4e4fc8e59bd0",  # TS accepts, Python rejects
    "86233705-0332-44f1-bc69-bc796220f598",  # Python accepts, TS rejects
]


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats"""
    if not date_str:
        return datetime.now(timezone.utc)

    try:
        # Try ISO format first
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
    """Get default configuration matching TypeScript"""
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
        "processing": {
            "enable_validation": True,
            "enable_quality_scoring": True,
        },
        "reset": {
            "time_gap_days": 30,
            "weight_change_threshold_kg": 10,
        },
        "snapshot": {
            "interval_hours": 24,
            "periodic_enabled": True,
        },
        "adaptive_noise": {"enabled": True},
        "replay": {
            "buffered_replay_enabled": True,
            "buffer_hours": 24,
            "max_buffer_measurements": 100,
        },
    }


def main():
    csv_path = "test_user.csv"

    # Load all measurements
    measurements = []
    id_to_row = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurement_id = row.get("id") or row.get("measurement_id")
            if not measurement_id:
                continue

            id_to_row[measurement_id] = row

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
                Measurement(
                    uuid=measurement_id,
                    weight=weight,
                    unit=unit,
                    effectiveDateTime=timestamp,
                    source=source,
                )
            )

    # Sort chronologically
    measurements.sort(key=lambda m: m.measured_at)

    # Initialize service
    state_store = InMemoryStore()
    config = get_default_config()
    service = WeightProcessorService(state_store, config)

    # Process all measurements
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"
    response = service.process_batch(user_id, measurements)

    # Find disputed measurements
    print("\n=== Disputed Measurements Quality Scores ===\n")

    for disputed_id in DISPUTED_IDS:
        idx = next(
            (i for i, m in enumerate(measurements) if m.measurement_id == disputed_id),
            None,
        )
        if idx is None:
            print(f"{disputed_id}: NOT FOUND")
            continue

        result = response.results[idx]
        measurement = measurements[idx]

        print(f"ID: {disputed_id}")
        print(f"  Timestamp: {measurement.measured_at.isoformat()}")
        print(f"  Weight: {measurement.weight_value} {measurement.weight_unit}")
        print(f"  Quality Score: {result.quality_score:.15f}")
        print(f"  Accepted: {result.accepted}")
        print(f"  Threshold: 0.5")
        print(f"  Margin: {(result.quality_score - 0.5):.15f}")
        print()

    print("\n=== Summary ===")
    print(f"Total measurements processed: {response.measurements_processed}")
    print(f"Total accepted: {response.measurements_accepted}")
    print(f"Total rejected: {response.measurements_rejected}")


if __name__ == "__main__":
    main()
