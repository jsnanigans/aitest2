#!/usr/bin/env python3
"""Test script to verify replay trigger logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from src.aws.api.models import Measurement
from src.aws.services.weight_processor_service import WeightProcessorService
from src.core.database import get_state_db

def test_replay_trigger():
    """Test that replay triggers correctly with time gap."""

    # Setup
    state_store = get_state_db()
    service = WeightProcessorService(state_store=state_store)
    user_id = "test-user-replay-001"

    # Clean state first
    print("Cleaning state...")
    state_store.delete_state(user_id)

    # Test measurements
    measurements = [
        Measurement(
            uuid="test-001",
            weight=99.0,
            unit="kg",
            effectiveDateTime="2024-01-01T10:00:00",
            source="MANUAL"
        ),
        Measurement(
            uuid="test-002",
            weight=106.0,
            unit="kg",
            effectiveDateTime="2024-01-20T15:00:00",
            source="MANUAL"
        ),
        Measurement(
            uuid="test-003",
            weight=100.0,
            unit="kg",
            effectiveDateTime="2024-01-20T15:20:00",
            source="MANUAL"
        ),
        Measurement(
            uuid="test-004",
            weight=98.0,
            unit="kg",
            effectiveDateTime="2024-01-25T12:00:00",
            source="MANUAL"
        ),
    ]

    print("Processing measurements...")
    print("-" * 80)

    try:
        result = service.process_measurements(user_id, measurements)

        print(f"\n✓ Processing completed successfully!")
        print(f"  Total processed: {result.measurements_processed}")
        print(f"  Accepted: {result.measurements_accepted}")
        print(f"  Rejected: {result.measurements_rejected}")

        if result.replay_metadata:
            print(f"\n✓ Replay triggered {len(result.replay_metadata)} time(s):")
            for i, replay in enumerate(result.replay_metadata, 1):
                print(f"  Replay {i}:")
                print(f"    Trigger: {replay['trigger']}")
                print(f"    Buffer size: {replay['buffer_size']}")
                print(f"    From: {replay['replay_from']}")
                print(f"    To: {replay['replay_to']}")
        else:
            print(f"\n✗ No replay triggered!")

        print(f"\nResults breakdown:")
        print("-" * 80)
        for r in result.results:
            status = "✓ ACCEPTED" if r.accepted else "✗ REJECTED"
            print(f"{r.measurement_id}: {status}")
            print(f"  Value: {r.value}{r.unit}")
            print(f"  Time: {r.effective_date_time}")
            print(f"  Quality: {r.quality_score:.6f}" if r.quality_score else "  Quality: None")
            print(f"  Stage: {r.processing_stage}")
            if r.rejection_reason:
                print(f"  Reason: {r.rejection_reason}")
            print()

        # Verify expected behavior
        print("Verification:")
        print("-" * 80)

        # Should have exactly 1 replay triggered (when M4 arrives)
        assert result.replay_metadata and len(result.replay_metadata) == 1, \
            f"Expected 1 replay, got {len(result.replay_metadata) if result.replay_metadata else 0}"
        print("✓ Correct number of replays (1)")

        # The replay should be triggered by time_gap
        assert result.replay_metadata[0]['trigger'] == 'time_gap', \
            f"Expected time_gap trigger, got {result.replay_metadata[0]['trigger']}"
        print("✓ Replay triggered by time_gap")

        # The replay should include M2 and M3 (buffer_size=2)
        assert result.replay_metadata[0]['buffer_size'] == 2, \
            f"Expected buffer_size=2, got {result.replay_metadata[0]['buffer_size']}"
        print("✓ Buffer contained 2 measurements (M2, M3)")

        # Find the results
        m1 = next(r for r in result.results if r.measurement_id == "test-001")
        m2 = next(r for r in result.results if r.measurement_id == "test-002")
        m3 = next(r for r in result.results if r.measurement_id == "test-003")
        m4 = next(r for r in result.results if r.measurement_id == "test-004")

        print(f"\nM1 (99kg): {'accepted' if m1.accepted else 'rejected'}")
        print(f"M2 (106kg - outlier): {'accepted' if m2.accepted else 'rejected'} - Expected: rejected")
        print(f"M3 (100kg): {'accepted' if m3.accepted else 'rejected'} - Expected: accepted")
        print(f"M4 (98kg): {'accepted' if m4.accepted else 'rejected'}")

        # M2 should be rejected as outlier after replay
        if not m2.accepted:
            print("\n✓ M2 correctly rejected as outlier!")
        else:
            print("\n⚠ Warning: M2 was accepted (expected rejection)")

        print("\n" + "=" * 80)
        print("TEST PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_replay_trigger()
