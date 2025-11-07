"""
Test to reproduce TS/PY divergence in the replay mechanism.

This test uses the SERVICE LAYER (WeightProcessorService.process_batch)
which includes buffered replay logic, not just core processMeasurement().
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent

from weight_processor_lib.core.database.memory_store import InMemoryStore
from aws.services.weight_processor_service import WeightProcessorService
from aws.api.models import Measurement


def load_fixture():
    """Load test fixture with replay sequence."""
    fixture_path = project_root / "test_fixtures" / "replay_divergence_12_measurements.json"
    with open(fixture_path) as f:
        return json.load(f)


def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp to UTC datetime."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def test_replay_divergence_python():
    """
    Test Python service layer replay mechanism.

    This test processes 12 measurements through WeightProcessorService.process_batch()
    which includes the buffered replay mechanism. We expect replay to be triggered
    due to time gaps in the sequence.

    Expected behavior (as of 2025-11-07):
    - Python service layer REJECTS measurement 4f07af66
    - TypeScript service layer ACCEPTS measurement 4f07af66

    This test captures detailed replay metadata to help debug the divergence.
    """
    fixture = load_fixture()
    measurements_data = fixture["measurements"]
    config = fixture["config"]
    user_id = fixture["user_id"]
    target_id = fixture["target_measurement_id"]

    print(f"\n{'='*70}")
    print(f"PYTHON REPLAY TEST - {len(measurements_data)} measurements")
    print(f"{'='*70}")

    # Initialize service with replay enabled
    store = InMemoryStore()
    service = WeightProcessorService(state_store=store, config=config)

    # Convert to Measurement objects
    measurements = []
    for m in measurements_data:
        timestamp = parse_timestamp(m["timestamp"])
        measurements.append(
            Measurement(
                measurement_id=m["id"],
                weight_value=m["weight"],
                weight_unit=m["unit"],
                measured_at=timestamp,
                source=m["source"],
            )
        )

    print(f"\nMeasurements to process:")
    for i, m in enumerate(measurements_data):
        print(f"  [{i+1:2d}] {m['id'][:8]} - {m['weight']:5.1f}kg @ {m['timestamp']}")
        if m["id"] == target_id:
            print(f"       ^ TARGET MEASUREMENT")

    # Process batch through service layer (includes replay)
    print(f"\nProcessing batch through service layer...")
    response = service.process_batch(user_id, measurements, user_height_m=1.75)

    # Analyze results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total measurements: {response.measurements_processed}")
    print(f"Accepted: {response.measurements_accepted}")
    print(f"Rejected: {response.measurements_rejected}")

    # Find target measurement result
    target_result = None
    target_accepted = None

    # Check individual results if available
    if hasattr(response, "results") and response.results:
        for result in response.results:
            if hasattr(result, "measurement_id") and result.measurement_id == target_id:
                target_result = result
                target_accepted = getattr(result, "accepted", False)
                break

    # Print replay metadata
    replay_metadata = getattr(response, "replay_metadata", []) or []
    print(f"\nReplay Events: {len(replay_metadata)}")
    for i, replay in enumerate(replay_metadata):
        print(f"\n  Replay {i+1}:")
        print(f"    Trigger: {getattr(replay, 'trigger', 'N/A')}")
        print(f"    Buffer size: {getattr(replay, 'buffer_size', 'N/A')}")
        print(f"    Replay from: {getattr(replay, 'replay_from', 'N/A')}")
        print(f"    Replay to: {getattr(replay, 'replay_to', 'N/A')}")
        print(f"    Measurements replayed: {getattr(replay, 'measurements_replayed', 'N/A')}")

    # Check target measurement
    print(f"\n{'='*70}")
    print(f"TARGET MEASUREMENT: {target_id[:8]}")
    print(f"{'='*70}")

    if target_result:
        print(f"Accepted: {getattr(target_result, 'accepted', 'N/A')}")
        print(f"Quality Score: {getattr(target_result, 'quality_score', 'N/A')}")
        print(f"Kalman Estimate: {getattr(target_result, 'kalman_estimate', 'N/A')}")
    else:
        # If not in individual results, check if it was accepted overall
        # by looking at the state
        state = store.get_state(user_id)
        if state and state.measurement_buffer:
            for buffered in state.measurement_buffer.measurements:
                if buffered.id == target_id:
                    print(f"Found in buffer - was processed")
                    target_accepted = True
                    break

        if target_accepted is None:
            print(f"⚠️  Could not determine if measurement was accepted")
            print(f"Response attributes: {dir(response)}")

    # Final verification
    print(f"\n{'='*70}")
    print(f"VERIFICATION")
    print(f"{'='*70}")

    # Count how many measurements were accepted
    accepted_count = response.measurements_accepted
    print(f"Total accepted: {accepted_count}")

    # We expect Python to reject the target measurement
    # The acceptance count gives us a hint
    if target_accepted is not None:
        if target_accepted:
            print(f"❌ Python ACCEPTED {target_id[:8]} (unexpected!)")
        else:
            print(f"✓ Python REJECTED {target_id[:8]} (expected)")
    else:
        print(f"⚠️  Could not determine acceptance status for {target_id[:8]}")

    print(f"\n{'='*70}\n")

    return {
        "measurements_processed": response.measurements_processed,
        "measurements_accepted": response.measurements_accepted,
        "measurements_rejected": response.measurements_rejected,
        "replay_count": len(replay_metadata),
        "target_accepted": target_accepted,
        "target_result": str(target_result) if target_result else None,
    }


if __name__ == "__main__":
    result = test_replay_divergence_python()
    print(f"\nTest completed:")
    print(json.dumps(result, indent=2, default=str))
