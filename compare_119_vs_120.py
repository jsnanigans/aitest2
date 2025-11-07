#!/usr/bin/env python3
"""
Compare overall results between 119 and 120 measurements.
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

    input_file = Path("/tmp/compare_119_120_ts.json")
    with open(input_file, 'w') as f:
        json.dump(input_data, f)

    result = subprocess.run(
        ["bun", "run", "extract_divergence_helper.ts", str(input_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"TypeScript error: {result.stderr}")
        return None

    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        if line.strip().startswith('['):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def compare_results(py_results, ts_results, measurements, count):
    """Compare results between Python and TypeScript."""
    print(f"\n{'=' * 60}")
    print(f"PROCESSING {count} MEASUREMENTS")
    print('=' * 60)

    py_accepted = [r for r in py_results if r["accepted"]]
    ts_accepted = [r for r in ts_results if r["accepted"]]

    print(f"Python accepted: {len(py_accepted)}/{len(py_results)}")
    print(f"TypeScript accepted: {len(ts_accepted)}/{len(ts_results)}")

    # Find differences
    py_ids = {r["id"] for r in py_accepted}
    ts_ids = {r["id"] for r in ts_accepted}

    only_py = py_ids - ts_ids
    only_ts = ts_ids - py_ids

    if only_py or only_ts:
        print(f"\n❌ DIVERGENCE DETECTED")

        if only_py:
            print(f"\n{len(only_py)} measurements accepted only by Python:")
            for mid in list(only_py)[:5]:
                m = next((m for m in measurements if m["id"] == mid), None)
                py_r = next((r for r in py_results if r["id"] == mid), None)
                ts_r = next((r for r in ts_results if r["id"] == mid), None)
                if m and py_r and ts_r:
                    print(f"  {mid[:8]} - {m['weight']}kg - py_q={py_r['quality_score']:.3f} ts_q={ts_r['quality_score']:.3f}")

        if only_ts:
            print(f"\n{len(only_ts)} measurements accepted only by TypeScript:")
            for mid in list(only_ts)[:5]:
                m = next((m for m in measurements if m["id"] == mid), None)
                py_r = next((r for r in py_results if r["id"] == mid), None)
                ts_r = next((r for r in ts_results if r["id"] == mid), None)
                if m and py_r and ts_r:
                    print(f"  {mid[:8]} - {m['weight']}kg - py_q={py_r['quality_score']:.3f} ts_q={ts_r['quality_score']:.3f}")
    else:
        print(f"\n✅ PERFECT MATCH - All acceptance decisions identical")

    return len(only_py) + len(only_ts) > 0


def main():
    print("Loading measurements...")
    all_measurements = load_measurements("test_user.csv")
    user_id = all_measurements[0]["user_id"]
    print(f"Loaded {len(all_measurements)} measurements\n")

    # Test 119 measurements
    print("Processing 119 measurements...")
    sequence_119 = all_measurements[:119]
    py_119 = process_with_service(sequence_119, user_id, CONFIG)
    ts_119 = process_typescript_service(sequence_119, user_id, CONFIG)

    if ts_119 is None:
        print("ERROR: TypeScript processing failed for 119")
        return 1

    diverged_119 = compare_results(py_119, ts_119, sequence_119, 119)

    # Test 120 measurements
    print("\nProcessing 120 measurements...")
    sequence_120 = all_measurements[:120]
    py_120 = process_with_service(sequence_120, user_id, CONFIG)
    ts_120 = process_typescript_service(sequence_120, user_id, CONFIG)

    if ts_120 is None:
        print("ERROR: TypeScript processing failed for 120")
        return 1

    diverged_120 = compare_results(py_120, ts_120, sequence_120, 120)

    # Show the 120th measurement
    print(f"\n{'=' * 60}")
    print("THE 120TH MEASUREMENT")
    print('=' * 60)
    m120 = all_measurements[119]
    print(f"ID: {m120['id']}")
    print(f"Timestamp: {m120['timestamp']}")
    print(f"Weight: {m120['weight']}kg")

    py_120th = next((r for r in py_120 if r["id"] == m120["id"]), None)
    ts_120th = next((r for r in ts_120 if r["id"] == m120["id"]), None)

    if py_120th and ts_120th:
        print(f"\nPython:")
        print(f"  accepted={py_120th['accepted']}")
        print(f"  quality_score={py_120th['quality_score']:.6f}")
        print(f"\nTypeScript:")
        print(f"  accepted={ts_120th['accepted']}")
        print(f"  quality_score={ts_120th['quality_score']:.6f}")

        if py_120th['accepted'] != ts_120th['accepted']:
            print(f"\n🎯 The 120th measurement itself causes acceptance divergence!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
