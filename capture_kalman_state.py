#!/usr/bin/env python3
"""
Capture the exact Kalman state when measurement 4f07af66 gets replayed.
This will allow us to create an isolated test case with just that state + one measurement.
"""

import csv
import json
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


def capture_state_at_snapshot(store, user_id, target_ts):
    """Capture Kalman state from snapshot."""
    snapshots = store.snapshots.get(user_id, [])

    # Find snapshot closest to target timestamp
    closest = None
    for snapshot_ts, snapshot_state in snapshots:
        if snapshot_ts <= target_ts:
            if closest is None or snapshot_ts > closest[0]:
                closest = (snapshot_ts, snapshot_state)

    if closest:
        return closest[1]
    return None


def main():
    print("Loading measurements...")
    all_measurements = load_measurements("test_user.csv")
    user_id = all_measurements[0]["user_id"]

    # Find target measurement
    target_index = next(i for i, m in enumerate(all_measurements) if m["id"] == TARGET_ID)
    target_measurement = all_measurements[target_index]

    print(f"\nTarget measurement:")
    print(f"  Index: {target_index}")
    print(f"  ID: {TARGET_ID}")
    print(f"  Timestamp: {target_measurement['timestamp']}")
    print(f"  Weight: {target_measurement['weight']}kg")

    # Process all 120 measurements
    print(f"\nProcessing all 120 measurements...")
    store = InMemoryStore()
    service = WeightProcessorService(state_store=store, config=CONFIG)

    measurement_objs = [
        Measurement(
            measurement_id=m["id"],
            weight_value=m["weight"],
            weight_unit=m["unit"],
            measured_at=m["timestamp"],
            source=m["source"],
        )
        for m in all_measurements
    ]

    response = service.process_batch(user_id, measurement_objs)

    print(f"Processed: {response.measurements_processed}")
    print(f"Replays triggered: {len(response.replay_metadata) if response.replay_metadata else 0}")

    if response.replay_metadata:
        for i, replay in enumerate(response.replay_metadata):
            print(f"\n  Replay {i+1}:")
            print(f"    Trigger: {replay.get('trigger')}")
            print(f"    Buffer size: {replay.get('buffer_size')}")
            print(f"    From: {replay.get('replay_from')}")
            print(f"    To: {replay.get('replay_to')}")

    # Get final state
    final_state = store.get_state(user_id)

    print(f"\n{'='*60}")
    print("FINAL KALMAN STATE")
    print('='*60)
    print(f"Position (weight estimate): {final_state.get('kalman_position', 'N/A')}")
    print(f"Velocity: {final_state.get('kalman_velocity', 'N/A')}")
    print(f"Position variance: {final_state.get('position_variance', 'N/A')}")
    print(f"Velocity variance: {final_state.get('velocity_variance', 'N/A')}")
    print(f"Covariance: {final_state.get('covariance', 'N/A')}")
    print(f"Last processed: {final_state.get('last_processed_at', 'N/A')}")
    print(f"Last raw weight: {final_state.get('last_raw_weight', 'N/A')}")

    # Check snapshots - find the one used when replaying the target measurement
    print(f"\n{'='*60}")
    print("SNAPSHOTS")
    print('='*60)

    snapshots = store._snapshots.get(user_id, [])
    print(f"Total snapshots: {len(snapshots)}")

    if snapshots:
        # Find snapshot that would be used for replaying target measurement
        target_ts = target_measurement['timestamp']
        snapshot_for_target = None

        # Look for Replay 11 snapshot (the one that includes our target)
        # From the output: 2025-07-11T23:22:41.324000+00:00
        replay_11_start = parse_timestamp("2025-07-11T23:22:41.324000+00:00")

        for snapshot_ts, snapshot_state in snapshots:
            # Only show snapshots around our target time
            if abs((snapshot_ts - target_ts).total_seconds()) < 86400 * 7:  # Within 7 days
                print(f"\nSnapshot at {snapshot_ts}:")
                print(f"  Kalman position: {snapshot_state.get('kalman_position', 'N/A')}")
                print(f"  Kalman velocity: {snapshot_state.get('kalman_velocity', 'N/A')}")
                print(f"  Position variance: {snapshot_state.get('position_variance', 'N/A')}")

            # Check if this is the snapshot used for Replay 11
            if snapshot_ts == replay_11_start:
                snapshot_for_target = (snapshot_ts, snapshot_state)
                print(f"  ⭐ THIS IS THE REPLAY 11 SNAPSHOT")

        if snapshot_for_target:
            print(f"\n{'='*60}")
            print("STATE USED WHEN REPLAYING TARGET MEASUREMENT")
            print('='*60)
            print(f"Snapshot timestamp: {snapshot_for_target[0]}")

            state = snapshot_for_target[1]

            # Save this state for test fixture
            # Convert any non-serializable types
            state_for_json = {}
            for key, value in state.items():
                if key in ['last_timestamp', 'last_accepted_timestamp', 'last_processed_at', 'reset_timestamp']:
                    state_for_json[key] = value.isoformat() if value else None
                elif key == 'reset_events':
                    # Skip reset events (too complex)
                    continue
                elif key == 'last_state':
                    # Convert numpy array to list
                    state_for_json[key] = value.tolist() if hasattr(value, 'tolist') else value
                elif key == 'last_covariance':
                    # Convert numpy array to list
                    state_for_json[key] = value.tolist() if hasattr(value, 'tolist') else value
                else:
                    state_for_json[key] = value

            fixture = {
                "description": "Kalman state when measurement 4f07af66 gets replayed (after 120 measurements processed)",
                "target_measurement": {
                    "id": TARGET_ID,
                    "timestamp": target_measurement["timestamp"].isoformat(),
                    "weight": target_measurement["weight"],
                    "unit": target_measurement["unit"],
                    "source": target_measurement["source"],
                },
                "kalman_state": state_for_json,
                "config": CONFIG,
            }

            output_file = Path("test_fixtures/kalman_state_replay_divergence.json")
            with open(output_file, 'w') as f:
                json.dump(fixture, f, indent=2, default=str)

            print(f"\n✅ Saved Kalman state to: {output_file}")

            # Print the state
            for key, value in state.items():
                print(f"  {key}: {value}")

    # Find result for target measurement
    target_result = next((r for r in response.results if r.measurement_id == TARGET_ID), None)

    if target_result:
        print(f"\n{'='*60}")
        print("TARGET MEASUREMENT RESULT (AFTER 120 MEASUREMENTS)")
        print('='*60)
        print(f"Accepted: {target_result.accepted}")
        print(f"Quality score: {target_result.quality_score}")
        print(f"Kalman estimate: {target_result.kalman_estimate}")
        print(f"Rejection reason: {target_result.rejection_reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
