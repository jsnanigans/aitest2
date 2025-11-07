"""Test for July 11 replay scenario that diverges between Python and TypeScript."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path

from weight_processor_lib.core.database.memory_store import InMemoryStore
from weight_processor_lib.core.processing.processor import process_measurement


@pytest.fixture
def july11_fixture():
    """Load the July 11 scenario fixture."""
    # Path from test file: python_lib/tests/processing/test_july11_scenario.py
    # Up to project root: ../../../test_fixtures/
    fixture_path = Path(__file__).parent.parent.parent.parent / "test_fixtures" / "july11_replay_scenario.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def store():
    """Create an in-memory store."""
    return InMemoryStore()


@pytest.fixture
def base_config():
    """Base configuration for testing."""
    return {
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


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def test_july11_replay_scenario(july11_fixture, store, base_config):
    """
    Test the July 11 replay scenario in isolation.

    This test processes:
    1. 6 setup measurements to establish Kalman state (July 8-10)
    2. 3 test measurements after 49-hour gap (July 11)

    Note: Results differ from full batch because Kalman filter state
    depends on ALL prior measurements. This test is for comparing
    Python implementation consistency, not matching full batch results.
    """
    user_id = july11_fixture["user_id"]

    # Process setup measurements to establish state
    print("\n📊 Python - Setup measurements:")
    for measurement in july11_fixture["setup"]["measurements"]:
        timestamp = parse_timestamp(measurement["timestamp"])
        result = process_measurement(
            user_id=user_id,
            weight=measurement["weight"],
            timestamp=timestamp,
            source=measurement["source"],
            config=base_config,
            unit=measurement["unit"],
            db=store,
            user_height_m=1.75,  # Assumed height
        )
        quality = result.get('quality_score')
        quality_str = f"{quality:.4f}" if quality is not None else "None"
        print(f"  {timestamp} -> accepted={result['accepted']}, quality={quality_str}")

    # Create snapshot before test measurements (simulating replay mechanism)
    snapshot_timestamp = parse_timestamp(july11_fixture["test_measurements"]["measurements"][0]["timestamp"])
    store.save_state_snapshot(user_id, snapshot_timestamp)

    # Process the 3 problematic measurements
    print("\n📊 Python - Test measurements (after 49h gap):")
    results = []
    for measurement in july11_fixture["test_measurements"]["measurements"]:
        timestamp = parse_timestamp(measurement["timestamp"])
        result = process_measurement(
            user_id=user_id,
            weight=measurement["weight"],
            timestamp=timestamp,
            source=measurement["source"],
            config=base_config,
            unit=measurement["unit"],
            db=store,
            user_height_m=1.75,
        )
        results.append({
            "id": measurement["id"],
            "accepted": result["accepted"],
            "quality_score": result.get("quality_score"),
            "weight": measurement["weight"],
            "timestamp": timestamp.isoformat(),
        })
        quality = result.get('quality_score')
        quality_str = f"{quality:.4f}" if quality is not None else "None"
        print(f"  {measurement['id'][:8]} ({measurement['weight']}kg) -> "
              f"accepted={result['accepted']}, quality={quality_str}")

    # Document what Python produces in this isolated scenario
    expected_py_batch = july11_fixture["expected_results"]["python"]
    problematic_id = "4f07af66-cd5e-4a38-9403-80d6da1d1542"

    print(f"\n📊 ISOLATED SCENARIO RESULTS (Python):")
    print(f"   This differs from full batch because Kalman state depends on ALL prior measurements")
    score0 = f"{results[0]['quality_score']:.4f}" if results[0]['quality_score'] is not None else "None"
    score1 = f"{results[1]['quality_score']:.4f}" if results[1]['quality_score'] is not None else "None"
    score2 = f"{results[2]['quality_score']:.4f}" if results[2]['quality_score'] is not None else "None"
    print(f"   First:  {results[0]['id'][:8]} -> accepted={results[0]['accepted']}, score={score0}")
    print(f"   Middle: {results[1]['id'][:8]} -> accepted={results[1]['accepted']}, score={score1}")
    print(f"   Third:  {results[2]['id'][:8]} -> accepted={results[2]['accepted']}, score={score2}")
    print(f"\n   For comparison, Python in full batch produced:")
    print(f"   Middle: accepted={expected_py_batch[problematic_id]['accepted']}, score={expected_py_batch[problematic_id]['quality_score']:.4f}")

    # Basic sanity checks
    assert len(results) == 3
    assert results[0]["id"] == "52ec2c45-c6a8-4946-887b-e5e8907f19b9"
    assert results[1]["id"] == problematic_id
    assert results[2]["id"] == "726b441f-eb43-47d9-8f3c-845d164e5a5b"

    # All results should have quality scores
    for result in results:
        assert result["quality_score"] is not None, f"Missing quality score for {result['id']}"

    print("✅ Python isolated scenario processed successfully")


if __name__ == "__main__":
    # Run the test directly
    import asyncio

    fixture_path = Path(__file__).parent.parent.parent.parent / "test_fixtures" / "july11_replay_scenario.json"
    with open(fixture_path) as f:
        fixture = json.load(f)

    store = InMemoryStore()
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

    asyncio.run(test_july11_replay_scenario(fixture, store, config))
