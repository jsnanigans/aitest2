#!/usr/bin/env python3
"""Debug Kalman state evolution to find where divergence begins"""

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add paths for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python_lib" / "src"))
sys.path.insert(0, str(project_root / "be_implementation_service" / "src"))

# Import from be_implementation_service
from aws.api.models import Measurement
from aws.config.config_manager import ConfigManager

# Import from python_lib
from weight_processor_lib.core.database import InMemoryStore
from weight_processor_lib.core.processing.processor import process_measurement


def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        elif " " in date_str:
            for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse: {date_str}")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def main():
    csv_file = "/tmp/debug_user_full.csv"
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    print("=== Python State Evolution Debug ===")
    print("Disabling replay to see raw processing")
    print("")

    # Load CSV
    measurements = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurement = Measurement(
                uuid=row.get("id"),
                weight=float(row.get("value_quantity", "0")),
                unit=row.get("unit", "kg"),
                effectiveDateTime=parse_timestamp(row.get("effective_date_time") or row.get("timestamp")),
                source=row.get("source_type", "unknown"),
            )
            measurements.append(measurement)

    # Sort chronologically
    measurements = sorted(measurements, key=lambda m: m.measured_at)

    print(f"Loaded {len(measurements)} measurements")
    print("")

    # Initialize storage and config
    state_store = InMemoryStore()
    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    # Disable replay for clearer debugging
    config["replay"]["buffered_replay_enabled"] = False

    # Process one measurement at a time to track state
    measurement_count = 0

    for measurement in measurements:
        measurement_count += 1

        result = process_measurement(
            user_id=user_id,
            weight=measurement.weight_value,
            timestamp=measurement.measured_at,
            source=measurement.source,
            config=config,
            unit=measurement.weight_unit,
            db=state_store,
            user_height_m=None
        )

        # Print state for measurements around the problematic ones
        is_near_july11 = (datetime(2025, 7, 9, tzinfo=timezone.utc) <=
                          measurement.measured_at <=
                          datetime(2025, 7, 13, tzinfo=timezone.utc))
        is_near_july26 = (datetime(2025, 7, 24, tzinfo=timezone.utc) <=
                          measurement.measured_at <=
                          datetime(2025, 7, 28, tzinfo=timezone.utc))

        if is_near_july11 or is_near_july26:
            print(f"\n[{measurement_count}] {measurement.measurement_id}")
            print(f"  Date: {measurement.measured_at.isoformat()}")
            print(f"  Weight: {measurement.weight_value} kg")
            print(f"  Accepted: {result.get('accepted')}")
            quality_score = result.get('quality_score')
            if quality_score is not None:
                print(f"  Quality: {quality_score:.4f}")

            components = result.get('quality_components', {})
            if components:
                print(f"  Components:")
                if 'kalman_fit' in components:
                    print(f"    kalman_fit: {components['kalman_fit']:.4f}")
                if 'temporal_consistency' in components:
                    print(f"    temporal_consistency: {components['temporal_consistency']:.4f}")
                if 'trend_alignment' in components:
                    print(f"    trend_alignment: {components['trend_alignment']:.4f}")

            kalman_estimate = result.get('kalman_estimate')
            if kalman_estimate is not None:
                print(f"  Kalman estimate: {kalman_estimate:.4f}")

            trend = result.get('trend')
            if trend is not None:
                print(f"  Trend (velocity): {trend:.6f}")

    print(f"\nProcessed all {measurement_count} measurements")


if __name__ == "__main__":
    main()