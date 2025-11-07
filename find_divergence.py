#!/usr/bin/env python3
"""
Find the exact point where Python and TypeScript implementations diverge.
Processes measurements one-by-one and compares results.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add python_lib to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python_lib" / "src"))
sys.path.insert(0, str(project_root / "be_implementation_service" / "src"))

from weight_processor_lib.core.database.memory_store import InMemoryStore
from weight_processor_lib.core.processing.processor import process_measurement


def parse_timestamp(date_str: str) -> datetime:
    """Parse various timestamp formats and return timezone-aware datetime in UTC."""
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
        raise ValueError(f"Cannot parse space-separated date: {date_str}")
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
            except (ValueError, KeyError) as e:
                continue

    # Sort by timestamp
    measurements.sort(key=lambda m: m["timestamp"])
    return measurements


def process_python(measurements, config):
    """Process measurements with Python implementation."""
    store = InMemoryStore()
    results = []

    for measurement in measurements:
        result = process_measurement(
            user_id=measurement["user_id"],
            weight=measurement["weight"],
            timestamp=measurement["timestamp"],
            source=measurement["source"],
            config=config,
            unit=measurement["unit"],
            db=store,
            user_height_m=1.75,
        )
        results.append({
            "id": measurement["id"],
            "accepted": result["accepted"],
            "quality_score": result.get("quality_score"),
            "timestamp": measurement["timestamp"].isoformat(),
        })

    return results


def process_typescript(measurements, config):
    """Process measurements with TypeScript implementation and return results."""
    # Create a temporary input file
    input_data = {
        "measurements": [
            {
                "id": m["id"],
                "userId": m["user_id"],
                "timestamp": m["timestamp"].isoformat(),
                "weight": m["weight"],
                "unit": m["unit"],
                "source": m["source"],
            }
            for m in measurements
        ],
        "config": config,
    }

    input_file = Path("/tmp/divergence_test_input.json")
    with open(input_file, 'w') as f:
        json.dump(input_data, f)

    # Run TypeScript processor
    result = subprocess.run(
        ["bun", "run", "find_divergence_helper.ts", str(input_file)],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    if result.returncode != 0:
        print(f"TypeScript error: {result.stderr}", file=sys.stderr)
        return None

    if not result.stdout.strip():
        print(f"TypeScript returned empty output", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        return None

    # Extract JSON from output (last line should be the JSON array)
    lines = result.stdout.strip().split('\n')
    json_line = None

    for line in reversed(lines):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            json_line = line
            break

    if json_line is None:
        print(f"Could not find JSON in TypeScript output", file=sys.stderr)
        print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
        return None

    try:
        return json.loads(json_line)
    except json.JSONDecodeError as e:
        print(f"Failed to parse TypeScript JSON: {e}", file=sys.stderr)
        print(f"json_line: {json_line[:500]}", file=sys.stderr)
        return None


def compare_results(py_result, ts_result):
    """Compare Python and TypeScript results."""
    if py_result["id"] != ts_result["id"]:
        return False, "ID mismatch"

    if py_result["accepted"] != ts_result["accepted"]:
        return False, f"Acceptance mismatch: py={py_result['accepted']} vs ts={ts_result['accepted']}"

    py_score = py_result.get("quality_score")
    ts_score = ts_result.get("quality_score")

    if py_score is None and ts_score is None:
        return True, None

    if py_score is None or ts_score is None:
        return False, f"Quality score presence mismatch: py={py_score} vs ts={ts_score}"

    diff = abs(py_score - ts_score)
    if diff > 0.0001:  # Allow tiny floating point differences
        return False, f"Quality score diff: py={py_score:.6f} vs ts={ts_score:.6f} (diff={diff:.6f})"

    return True, None


def main():
    config = {
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
    }

    csv_file = "test_user.csv"
    print(f"Loading measurements from {csv_file}...")
    measurements = load_measurements(csv_file)
    print(f"Loaded {len(measurements)} measurements\n")

    # Process incrementally and compare at each step
    print("Processing measurements incrementally to find divergence point...\n")

    divergence_point = None
    batch_size = 10  # Start with batches of 10

    for i in range(batch_size, len(measurements) + 1, batch_size):
        batch = measurements[:i]
        print(f"Testing batch 1-{i} ({i} measurements)...", end=" ")

        py_results = process_python(batch, config)
        ts_results = process_typescript(batch, config)

        if ts_results is None:
            print("ERROR: TypeScript processing failed")
            return 1

        # Compare last result
        match, reason = compare_results(py_results[-1], ts_results[-1])

        if not match:
            print(f"❌ DIVERGENCE FOUND!")
            print(f"   Reason: {reason}")
            print(f"   Last measurement: {py_results[-1]['id'][:8]}")
            divergence_point = i
            break
        else:
            print(f"✅ Match")

    if divergence_point is None:
        print("\n✅ No divergence found!")
        return 0

    # Narrow down to exact measurement
    print(f"\nNarrowing down to exact divergence point (between {divergence_point - batch_size + 1} and {divergence_point})...")

    for i in range(max(1, divergence_point - batch_size + 1), divergence_point + 1):
        batch = measurements[:i]
        py_results = process_python(batch, config)
        ts_results = process_typescript(batch, config)

        if ts_results is None:
            continue

        match, reason = compare_results(py_results[-1], ts_results[-1])

        if not match:
            print(f"\n🎯 EXACT DIVERGENCE POINT: Measurement #{i}")
            print(f"   ID: {measurements[i-1]['id']}")
            print(f"   Timestamp: {measurements[i-1]['timestamp']}")
            print(f"   Weight: {measurements[i-1]['weight']} {measurements[i-1]['unit']}")
            print(f"   Reason: {reason}")

            print(f"\n   Python result:")
            print(f"      accepted={py_results[-1]['accepted']}")
            print(f"      quality_score={py_results[-1].get('quality_score')}")

            print(f"\n   TypeScript result:")
            print(f"      accepted={ts_results[-1]['accepted']}")
            print(f"      quality_score={ts_results[-1].get('quality_score')}")

            # Save minimal test case
            output_file = Path("test_fixtures/minimal_divergence.json")
            output_data = {
                "description": f"Minimal test case for divergence at measurement #{i}",
                "measurements": [
                    {
                        "id": m["id"],
                        "timestamp": m["timestamp"].isoformat(),
                        "weight": m["weight"],
                        "unit": m["unit"],
                        "source": m["source"],
                    }
                    for m in measurements[:i]
                ],
                "config": config,
                "divergence_measurement_index": i - 1,
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"\n   Saved minimal test case to: {output_file}")
            return 0

    print("\n❌ Could not pinpoint exact divergence")
    return 1


if __name__ == "__main__":
    sys.exit(main())
