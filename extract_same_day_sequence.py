#!/usr/bin/env python3
"""
Extract measurements up to and including all from the same day as the divergent measurement.
Processes measurements incrementally and checks for divergence.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "python_lib" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "be_implementation_service" / "src"))

from weight_processor_lib.core.database.memory_store import InMemoryStore
from weight_processor_lib.core.processing.processor import process_measurement
from aws.services.weight_processor_service import WeightProcessorService
from aws.api.models import Measurement

CONFIG = {
    "kalman": {
        "process_noise_position": 0.01,
        "process_noise_velocity": 0.0001,
        "initial_position_variance": 1.0,
        "initial_velocity_variance": 0.01,
        "trend_limit_kg_per_week": 5.0,
    },
    "quality_weights": {
        "kalman_fit": 0.35,
        "temporal_consistency": 0.25,
        "plausibility": 0.25,
        "anomaly_detection": 0.15,
    },
    "quality_threshold": 0.55,
    "replay": {
        "buffer_hours": 24,
        "max_buffer_measurements": 100,
        "buffered_replay_enabled": True,
    },
}


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats."""
    if not date_str:
        return datetime.now(timezone.utc)

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


def load_measurements(csv_file: str):
    """Load measurements from CSV."""
    measurements = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                measurements.append({
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "timestamp": parse_timestamp(row["timestamp"]),
                    "weight": float(row["value_quantity"]),
                    "unit": row.get("unit", "kg"),
                    "source": row.get("source_type", "unknown"),
                })
            except (ValueError, KeyError):
                continue

    measurements.sort(key=lambda m: m["timestamp"])
    return measurements


def process_with_service(measurements, user_id, config):
    """Process measurements using the service (includes replay)."""
    store = InMemoryStore()
    service = WeightProcessorService(state_store=store, config=config)

    measurement_objs = [
        Measurement(
            measurement_id=m["id"],
            weight_value=m["weight"],
            weight_unit=m["unit"],
            measured_at=m["timestamp"],
            source=m["source"],
        )
        for m in measurements
    ]

    response = service.process_batch(user_id, measurement_objs)

    # Extract results
    results = []
    for i, result in enumerate(response.results):
        measurement_id = measurement_objs[i].measurement_id if i < len(measurement_objs) else f"unknown-{i}"
        results.append({
            "id": measurement_id,
            "accepted": result.accepted,
            "quality_score": result.quality_score,
        })

    return results


def process_typescript_service(measurements, user_id, config):
    """Process using TypeScript service."""
    input_data = {
        "measurements": [
            {
                "id": m["id"],
                "userId": user_id,
                "timestamp": m["timestamp"].isoformat(),
                "weight": m["weight"],
                "unit": m["unit"],
                "source": m["source"],
            }
            for m in measurements
        ],
        "config": config,
    }

    input_file = Path("/tmp/extract_same_day_ts.json")
    with open(input_file, 'w') as f:
        json.dump(input_data, f)

    result = subprocess.run(
        ["bun", "run", "extract_divergence_helper.ts", str(input_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    # Extract JSON from output
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        if line.strip().startswith('['):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def main():
    csv_file = "test_user.csv"
    print(f"Loading measurements from {csv_file}...")
    all_measurements = load_measurements(csv_file)
    user_id = all_measurements[0]["user_id"]
    print(f"Total measurements: {len(all_measurements)}\n")

    # The problematic measurement
    target_id = "4f07af66-cd5e-4a38-9403-80d6da1d1542"

    # Find the target measurement and its date
    target_index = None
    target_date = None
    for i, m in enumerate(all_measurements):
        if m["id"] == target_id:
            target_index = i
            target_date = m["timestamp"].date()
            break

    if target_index is None:
        print(f"ERROR: Could not find measurement {target_id}")
        return 1

    print(f"Target measurement found at index {target_index}")
    print(f"Date: {target_date}")
    print(f"Timestamp: {all_measurements[target_index]['timestamp']}")
    print(f"Weight: {all_measurements[target_index]['weight']} kg\n")

    # Find all measurements from the same day
    same_day_indices = []
    for i, m in enumerate(all_measurements):
        if m["timestamp"].date() == target_date:
            same_day_indices.append(i)

    print(f"Found {len(same_day_indices)} measurements from {target_date}:")
    for idx in same_day_indices:
        m = all_measurements[idx]
        marker = " <- TARGET" if m["id"] == target_id else ""
        print(f"  [{idx}] {m['id'][:8]} - {m['timestamp'].strftime('%H:%M:%S')} - {m['weight']}kg{marker}")

    # Get last measurement from that day
    last_same_day_index = max(same_day_indices)
    print(f"\nLast measurement from {target_date} is at index {last_same_day_index}")

    # Process all measurements up to and including the last same-day measurement
    sequence = all_measurements[:last_same_day_index + 1]
    print(f"\nProcessing sequence of {len(sequence)} measurements...\n")

    print("Python processing (with replay)...")
    py_results = process_with_service(sequence, user_id, CONFIG)

    print("TypeScript processing (with replay)...")
    ts_results = process_typescript_service(sequence, user_id, CONFIG)

    if ts_results is None:
        print("ERROR: TypeScript processing failed")
        return 1

    print(f"\nPython returned {len(py_results)} results")
    print(f"TypeScript returned {len(ts_results)} results")

    # Find the target result
    py_target = None
    ts_target = None

    for r in py_results:
        if r.get("id") == target_id:
            py_target = r
            break

    for r in ts_results:
        if r.get("id") == target_id:
            ts_target = r
            break

    if py_target is None or ts_target is None:
        print("ERROR: Could not find target in results")
        return 1

    py_q_str = f"{py_target['quality_score']:.6f}" if py_target['quality_score'] is not None else "None"
    ts_q_str = f"{ts_target['quality_score']:.6f}" if ts_target['quality_score'] is not None else "None"

    print("\n" + "=" * 60)
    print("RESULTS FOR TARGET MEASUREMENT")
    print("=" * 60)
    print(f"Measurement: {target_id[:16]}...")
    print(f"Weight: {all_measurements[target_index]['weight']} kg")
    print(f"\nPython:")
    print(f"  accepted={py_target['accepted']}")
    print(f"  quality_score={py_q_str}")
    print(f"\nTypeScript:")
    print(f"  accepted={ts_target['accepted']}")
    print(f"  quality_score={ts_q_str}")

    # Check for divergence
    py_q = py_target['quality_score']
    ts_q = ts_target['quality_score']

    if py_q is not None and ts_q is not None:
        diff = abs(py_q - ts_q)
        print(f"\nQuality score difference: {diff:.6f}")

        if py_target['accepted'] != ts_target['accepted']:
            print("\n🎯 DIVERGENCE DETECTED: Different acceptance decisions!")
        elif diff > 0.001:
            print("\n⚠️  DIVERGENCE DETECTED: Significant quality score difference!")
        else:
            print("\n✅ No significant divergence")

    # Save the sequence
    output_file = Path("test_fixtures/same_day_divergence_sequence.json")
    output_data = {
        "description": f"Sequence including all measurements from {target_date} (the day of divergent measurement)",
        "total_measurements": len(sequence),
        "target_measurement_id": target_id,
        "target_measurement_index": target_index,
        "target_date": str(target_date),
        "same_day_measurement_count": len(same_day_indices),
        "same_day_indices": same_day_indices,
        "measurements": [
            {
                "id": m["id"],
                "timestamp": m["timestamp"].isoformat(),
                "weight": m["weight"],
                "unit": m["unit"],
                "source": m["source"],
            }
            for m in sequence
        ],
        "config": CONFIG,
        "results": {
            "python": {
                "accepted": py_target["accepted"],
                "quality_score": py_target["quality_score"],
            },
            "typescript": {
                "accepted": ts_target["accepted"],
                "quality_score": ts_target["quality_score"],
            },
        },
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved sequence to: {output_file}")
    print(f"   Contains {len(sequence)} measurements up to and including all from {target_date}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
