#!/usr/bin/env python3
"""Debug temporal consistency inputs for Python implementation."""

import csv
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Enable verbose logging
os.environ["VERBOSE_LOGGING"] = "true"

# Add paths for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python_lib" / "src"))
sys.path.insert(0, str(project_root / "be_implementation_service" / "src"))

from aws.api.models import Measurement
from aws.services.weight_processor_service import WeightProcessorService
from aws.config.config_manager import ConfigManager
from weight_processor_lib.core.database import InMemoryStore

TARGET_ID = "726b441f-eb43-47d9-8f3c-845d164e5a5b"

def parse_timestamp(date_str: str) -> datetime:
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
    print(f"=== Debugging Temporal Consistency Inputs (Python) for {TARGET_ID} ===\n")

    # Load CSV
    csv_path = "test_user.csv"
    user_measurements = []
    target_index = -1

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurement_id = row.get("id") or row.get("measurement_id")
            user_id = row.get("user_id")
            weight_str = (row.get("value_quantity", "") or row.get("weight", "")).strip()

            if not user_id or not measurement_id or not weight_str:
                continue

            try:
                weight = float(weight_str)
                if weight <= 0 or weight > 1000:
                    continue
            except (ValueError, TypeError):
                continue

            date_str = row.get("effective_date_time", "") or row.get("effectiveDateTime", "")
            source = row.get("source_type", "unknown")
            unit = row.get("unit", "").strip()

            if not unit:
                continue

            timestamp = parse_timestamp(date_str) if date_str else datetime.now(timezone.utc)

            measurement = Measurement(
                uuid=measurement_id,
                weight=weight,
                unit=unit,
                effectiveDateTime=timestamp,
                source=source
            )

            if measurement_id == TARGET_ID:
                target_index = len(user_measurements)
                print(f"Found target measurement at index {target_index}:")
                print(f"  ID: {measurement_id}")
                print(f"  Timestamp: {timestamp.isoformat()}")
                print(f"  Weight: {weight} {unit}")
                print(f"  Source: {source}\n")

            user_measurements.append(measurement)

    if target_index == -1:
        print("Target measurement not found!")
        return

    # Sort by timestamp
    user_measurements.sort(key=lambda m: m.measured_at)

    # Find new index after sorting
    target_index = next(
        (i for i, m in enumerate(user_measurements) if m.measurement_id == TARGET_ID),
        -1
    )
    print(f"After sorting, target is at index {target_index}\n")

    # Show previous measurements
    print("Previous 3 measurements:")
    for i in range(max(0, target_index - 3), target_index):
        m = user_measurements[i]
        print(f"  [{i}] {m.measured_at.isoformat()} - {m.weight_value}kg - {m.source}")
    print()

    # Calculate expected inputs
    if target_index > 0:
        target_m = user_measurements[target_index]
        prev_m = user_measurements[target_index - 1]

        time_diff_ms = (target_m.measured_at - prev_m.measured_at).total_seconds() * 1000
        time_diff_hours = time_diff_ms / (1000 * 60 * 60)

        print("Expected inputs to temporal_consistency (if using raw previous):")
        print(f"  previousWeight: {prev_m.weight_value}")
        print(f"  timeDiffHours: {time_diff_hours:.2f}")
        print(f"  weight: {target_m.weight_value}")
        print(f"  weightChange: {abs(target_m.weight_value - prev_m.weight_value):.2f}\n")

    # Run processing
    print("Running full processing...\n")

    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    store = InMemoryStore()
    service = WeightProcessorService(state_store=store, config=config)

    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    response = service.process_batch(user_id, user_measurements)

    # Find result for target measurement
    result = response.results[target_index]

    print("\n\n=== Result for Target Measurement ===")
    print(f"Accepted: {result.accepted}")
    print(f"Quality Score: {result.quality_score:.6f}" if result.quality_score else "Quality Score: None")
    if result.quality_components:
        print("Quality Components:")
        for component, score in result.quality_components.items():
            print(f"  {component}: {score:.6f}")

if __name__ == "__main__":
    main()
