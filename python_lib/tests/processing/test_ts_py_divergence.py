"""
Test to reproduce TS/PY divergence on measurement 4f07af66.

This test processes the full sequence of measurements from test_user.csv
up to and including the divergent measurement.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "python_lib" / "src"))
sys.path.insert(0, str(project_root / "be_implementation_service" / "src"))

from weight_processor_lib.core.database.memory_store import InMemoryStore
from weight_processor_lib.core.processing.processor import process_measurement


def load_fixture():
    """Load test fixture with full measurement sequence (ALL 120 measurements)."""
    fixture_path = project_root / "test_fixtures" / "divergence_all_120_measurements.json"
    with open(fixture_path) as f:
        return json.load(f)


def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp to UTC datetime."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def test_divergent_measurement_python():
    """
    Test Python implementation on the divergent measurement.

    This test processes ALL 120 measurements from test_user.csv.
    The divergence only appears after processing the full sequence due to
    cumulative floating-point precision differences between Python and TypeScript.

    Expected behavior (as of 2025-11-07):
    - Python REJECTS measurement 4f07af66 (accepted=False)
    - TypeScript ACCEPTS measurement 4f07af66 (accepted=True)

    NOTE: Processing only the first 49 measurements shows BOTH implementations
    rejecting the measurement. The full 120-measurement sequence is required
    to reproduce the divergence.
    """
    fixture = load_fixture()
    measurements = fixture["measurements"]
    config = fixture["config"]
    user_id = fixture["user_id"]
    target_id = fixture["target_measurement_id"]

    print(f"\nProcessing {len(measurements)} measurements up to target...")

    store = InMemoryStore()
    target_result = None

    for i, m in enumerate(measurements):
        timestamp = parse_timestamp(m["timestamp"])
        weight = m["weight"]

        result = process_measurement(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=m["source"],
            config=config,
            unit=m["unit"],
            db=store,
            user_height_m=1.75,
        )

        if m["id"] == target_id:
            target_result = result
            print(f"\n[{i+1}] Target measurement {target_id[:8]}:")
            print(f"    Weight: {weight} kg")
            print(f"    Timestamp: {timestamp}")
            print(f"    Accepted: {result['accepted']}")
            print(f"    Quality Score: {result.get('quality_score', 'N/A')}")

    assert target_result is not None, "Target measurement not found"

    # Document the current behavior
    # Python rejects this measurement
    assert target_result["accepted"] is False, (
        "Python should reject this measurement. "
        "If this assertion fails, the divergence may have been fixed!"
    )

    # Quality score should be low
    quality = target_result.get("quality_score")
    assert quality is not None
    assert quality < 0.5, f"Quality score {quality} should be below threshold"

    print(f"\n✓ Python rejects measurement {target_id[:8]} as expected")
    print(f"  Quality score: {quality:.6f} (threshold: 0.5)")


if __name__ == "__main__":
    test_divergent_measurement_python()
