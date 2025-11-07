#!/usr/bin/env python3
"""
Binary search to find the exact measurement count where divergence starts.
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

TARGET_ID = "4f07af66-cd5e-4a38-9403-80d6da1d1542"
TOLERANCE = 0.001  # Consider diverged if quality score differs by more than this


def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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
    """Process measurements using Python service."""
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

    results = []
    for i, result in enumerate(response.results):
        measurement_id = measurement_objs[i].measurement_id
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

    input_file = Path("/tmp/find_divergence_threshold_ts.json")
    with open(input_file, 'w') as f:
        json.dump(input_data, f)

    result = subprocess.run(
        ["bun", "run", "extract_divergence_helper.ts", str(input_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        if line.strip().startswith('['):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def test_sequence(measurements, user_id, target_id, count):
    """Test a sequence of measurements and return quality scores."""
    sequence = measurements[:count]

    py_results = process_with_service(sequence, user_id, CONFIG)
    ts_results = process_typescript_service(sequence, user_id, CONFIG)

    if ts_results is None:
        return None

    # Find target measurement
    py_target = next((r for r in py_results if r["id"] == target_id), None)
    ts_target = next((r for r in ts_results if r["id"] == target_id), None)

    if not py_target or not ts_target:
        return None

    return {
        "py_quality": py_target["quality_score"],
        "ts_quality": ts_target["quality_score"],
        "py_accepted": py_target["accepted"],
        "ts_accepted": ts_target["accepted"],
    }


def main():
    print("Loading measurements...")
    all_measurements = load_measurements("test_user.csv")
    user_id = all_measurements[0]["user_id"]

    # Find target index
    target_index = next(i for i, m in enumerate(all_measurements) if m["id"] == TARGET_ID)
    print(f"Target measurement at index {target_index}\n")

    # We know:
    # - 50 measurements: NO divergence
    # - 120 measurements: YES divergence
    # Binary search between 50 and 120

    left = 50
    right = 120

    print("=" * 60)
    print("BINARY SEARCH FOR DIVERGENCE THRESHOLD")
    print("=" * 60)

    # First confirm boundaries
    print(f"\nTesting boundary: {left} measurements...")
    result = test_sequence(all_measurements, user_id, TARGET_ID, left)
    if result:
        diff = abs(result["py_quality"] - result["ts_quality"])
        print(f"  Python:     quality={result['py_quality']:.6f}")
        print(f"  TypeScript: quality={result['ts_quality']:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Status: {'✅ MATCH' if diff <= TOLERANCE else '❌ DIVERGE'}")

    print(f"\nTesting boundary: {right} measurements...")
    result = test_sequence(all_measurements, user_id, TARGET_ID, right)
    if result:
        diff = abs(result["py_quality"] - result["ts_quality"])
        print(f"  Python:     quality={result['py_quality']:.6f}")
        print(f"  TypeScript: quality={result['ts_quality']:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Status: {'✅ MATCH' if diff <= TOLERANCE else '❌ DIVERGE'}")

    print("\n" + "=" * 60)
    print("Starting binary search...")
    print("=" * 60)

    last_match = left
    first_diverge = right

    while left < right - 1:
        mid = (left + right) // 2
        print(f"\nTesting {mid} measurements...")

        result = test_sequence(all_measurements, user_id, TARGET_ID, mid)
        if not result:
            print("  ERROR: Test failed")
            break

        diff = abs(result["py_quality"] - result["ts_quality"])
        diverged = diff > TOLERANCE

        print(f"  Python:     quality={result['py_quality']:.6f}")
        print(f"  TypeScript: quality={result['ts_quality']:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Status: {'❌ DIVERGE' if diverged else '✅ MATCH'}")

        if diverged:
            right = mid
            first_diverge = mid
        else:
            left = mid
            last_match = mid

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Last matching count: {last_match} measurements")
    print(f"First diverging count: {first_diverge} measurements")
    print(f"\nDivergence starts appearing between measurements {last_match} and {first_diverge}")

    # Show what's in that range
    print(f"\nMeasurements {last_match} to {first_diverge}:")
    for i in range(last_match, min(first_diverge, len(all_measurements))):
        m = all_measurements[i]
        print(f"  [{i}] {m['id'][:8]} - {m['timestamp']} - {m['weight']}kg")

    return 0


if __name__ == "__main__":
    sys.exit(main())
