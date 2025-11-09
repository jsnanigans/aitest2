#!/usr/bin/env python3
"""
Debug script to trace processing of divergent measurements.
Compares TypeScript and Python behavior for specific measurement IDs.
"""

import csv
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add paths for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python_lib" / "src"))
sys.path.insert(0, str(project_root / "be_implementation_service" / "src"))

from aws.api.models import Measurement
from aws.services.weight_processor_service import WeightProcessorService
from aws.config.config_manager import ConfigManager
from weight_processor_lib.core.database import InMemoryStore

# Target measurement IDs that differ between implementations
DIVERGENT_IDS = [
    "726b441f-eb43-47d9-8f3c-845d164e5a5b",  # TS accepts, PY rejects
    "1a98b2c3-e023-4757-8d01-d35ef2fb363e",  # PY accepts, TS rejects
    "510977fa-9d3f-4b50-a667-e676a0cc0791",  # PY accepts, TS rejects
    "70d7918e-87d9-4968-84a4-b2bfec488e76",  # PY accepts, TS rejects
    "86233705-0332-44f1-bc69-bc796220f598",  # PY accepts, TS rejects
]

def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats and return timezone-aware datetime in UTC."""
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
            raise ValueError(f"Cannot parse space-separated date: {date_str}")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def main():
    print("=== Debugging Divergent Measurements (Python) ===\n")
    print("Target IDs:")
    for id_ in DIVERGENT_IDS:
        print(f"  {id_}")
    print()

    # Load CSV
    csv_path = "test_user.csv"
    user_measurements = []
    divergent_measurements = []

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
                source=source,
                metadata={"original_row": row}
            )

            user_measurements.append(measurement)

            if measurement_id in DIVERGENT_IDS:
                divergent_measurements.append((measurement, row))

    print(f"Loaded {len(user_measurements)} measurements for user")
    print(f"Found {len(divergent_measurements)} divergent measurements\n")

    # Sort by timestamp
    user_measurements.sort(key=lambda m: m.measured_at)

    # Initialize Python implementation
    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    store = InMemoryStore()
    service = WeightProcessorService(state_store=store, config=config)

    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    # Process all measurements
    print("Processing all measurements...\n")
    response = service.process_batch(user_id, user_measurements)

    print(f"Total processed: {response.measurements_processed}")
    print(f"Total accepted: {response.measurements_accepted}")
    print(f"Total rejected: {response.measurements_rejected}\n")

    # Analyze divergent measurements
    print("=== Divergent Measurement Analysis ===\n")

    for measurement, original_row in divergent_measurements:
        # Find the result for this measurement
        measurement_index = next(
            (i for i, m in enumerate(user_measurements) if m.measurement_id == measurement.measurement_id),
            -1
        )

        if 0 <= measurement_index < len(response.results):
            result = response.results[measurement_index]

            print(f"ID: {measurement.measurement_id}")
            print(f"  Timestamp: {measurement.measured_at.isoformat()}")
            print(f"  Source: {measurement.source}")
            print(f"  Weight: {measurement.weight_value} {measurement.weight_unit}")
            print(f"  Accepted: {result.accepted}")
            print(f"  Quality Score: {result.quality_score:.6f}" if result.quality_score else "  Quality Score: None")

            if result.quality_components:
                print(f"  Quality Components:")
                for component, score in result.quality_components.items():
                    print(f"    {component}: {score:.6f}")

            if not result.accepted:
                print(f"  Rejection Reason: {result.rejection_reason or 'Unknown'}")

            print(f"  Kalman Estimate: {result.kalman_estimate:.3f}" if result.kalman_estimate else "  Kalman Estimate: None")
            print(f"  Innovation: {result.innovation:.3f}" if result.innovation is not None else "  Innovation: None")
            print(f"  Normalized Innovation: {result.normalized_innovation:.3f}" if result.normalized_innovation is not None else "  Normalized Innovation: None")
            print()

if __name__ == "__main__":
    main()
