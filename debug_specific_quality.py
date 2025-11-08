#!/usr/bin/env python3
"""Debug specific quality scores for problematic measurements"""

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
from aws.services.weight_processor_service import WeightProcessorService
from aws.config.config_manager import ConfigManager

# Import from python_lib
from weight_processor_lib.core.database import InMemoryStore

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
    target_ids = [
        "726b441f-eb43-47d9-8f3c-845d164e5a5b",  # TS accepts, Py rejects?
        "1a98b2c3-e023-4757-8d01-d35ef2fb363e",  # Py accepts, TS rejects?
    ]

    print("=== Python Quality Score Debug ===")
    print(f"Target IDs: {', '.join(target_ids)}")
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

    print(f"Loaded {len(measurements)} measurements")

    # Sort chronologically
    measurements = sorted(measurements, key=lambda m: m.measured_at)

    # Initialize storage and service
    state_store = InMemoryStore()
    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    service = WeightProcessorService(state_store=state_store, config=config)

    # Process batch
    response = service.process_batch("ADC64C0B-CB46-41F9-BDA0-CC11A35942D7", measurements)

    print(f"\nProcessed {response.measurements_processed} measurements")
    print(f"Accepted: {response.measurements_accepted}, Rejected: {response.measurements_rejected}")
    print(f"Replays triggered: {len(response.replay_metadata) if response.replay_metadata else 0}")

    # Find our target measurements in results
    print("\n=== Target Measurement Results ===")
    for target_id in target_ids:
        # Find measurement index
        measurement_index = None
        for i, m in enumerate(measurements):
            if m.measurement_id == target_id:
                measurement_index = i
                break

        if measurement_index is not None:
            result = response.results[measurement_index]
            measurement = measurements[measurement_index]
            print(f"\nID: {target_id}")
            print(f"  Timestamp: {measurement.measured_at.isoformat()}")
            print(f"  Weight: {measurement.weight_value} {measurement.weight_unit}")
            print(f"  Accepted: {result.accepted}")
            print(f"  Quality Score: {result.quality_score}")
            print(f"  Threshold: 0.46")

            if result.quality_score is not None:
                diff = result.quality_score - 0.46
                status = "PASS" if result.quality_score >= 0.46 else "FAIL"
                print(f"  Score vs Threshold: {diff:.15f} ({status})")
            else:
                print(f"  Score vs Threshold: N/A (quality_score is None)")

            if result.quality_components:
                print(f"  Components:")
                for key, value in result.quality_components.items():
                    print(f"    {key}: {value}")

            if not result.accepted:
                reason = result.rejection_reason or getattr(result, 'reason', 'Unknown')
                print(f"  Rejection reason: {reason}")
        else:
            print(f"\n⚠️  {target_id} not found in measurements")


if __name__ == "__main__":
    main()