#!/usr/bin/env python3
"""
Test processing a single measurement with pre-configured Kalman state.

This test sets up the exact Kalman state from when measurement 4f07af66
gets replayed (after 120 measurements), then processes just that one measurement.

This isolates the divergence to a single measurement with known state.
"""

import json
import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database.memory_store import InMemoryStore


@pytest.fixture
def kalman_state_fixture():
    """Load the Kalman state fixture."""
    # We're in python_lib/tests/processing/ - need to go up to project root
    fixture_path = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures" / "kalman_state_replay_divergence.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def store_with_state(kalman_state_fixture):
    """Create a store with the pre-configured Kalman state."""
    store = InMemoryStore()
    user_id = "test-user"

    # Set the state directly
    state = kalman_state_fixture["kalman_state"].copy()

    # Convert timestamp strings back to datetime objects
    if state.get("last_timestamp"):
        state["last_timestamp"] = datetime.fromisoformat(state["last_timestamp"])
    if state.get("last_accepted_timestamp"):
        state["last_accepted_timestamp"] = datetime.fromisoformat(state["last_accepted_timestamp"])
    if state.get("reset_timestamp"):
        state["reset_timestamp"] = datetime.fromisoformat(state["reset_timestamp"])

    # Convert arrays back to numpy arrays
    if state.get("last_state") and isinstance(state["last_state"], list):
        state["last_state"] = np.array(state["last_state"])
    if state.get("last_covariance") and isinstance(state["last_covariance"], list):
        state["last_covariance"] = np.array(state["last_covariance"])

    store.save_state(user_id, state)

    return store, user_id


def test_single_measurement_with_kalman_state(kalman_state_fixture, store_with_state):
    """
    Test processing the divergent measurement with exact Kalman state.

    This test:
    1. Sets up the exact Kalman state from when replay happens
    2. Processes just measurement 4f07af66 (58.4kg)
    3. Records the quality score and acceptance decision

    Expected behavior (from full 120-measurement run):
    - Python: accepted=False, quality_score=0.009308
    """
    store, user_id = store_with_state

    # Get measurement and config from fixture
    measurement = kalman_state_fixture["target_measurement"]
    config = kalman_state_fixture["config"]

    # Process the measurement
    result = process_measurement(
        user_id=user_id,
        weight=measurement["weight"],
        timestamp=datetime.fromisoformat(measurement["timestamp"]),
        source=measurement["source"],
        config=config,
        unit=measurement["unit"],
        db=store,
        user_height_m=1.75,
    )

    # Print results for comparison with FULL PRECISION
    print(f"\n{'='*60}")
    print(f"PYTHON - Single Measurement Test (FULL PRECISION)")
    print(f"{'='*60}")
    print(f"Measurement ID: {measurement['id'][:16]}...")
    print(f"Weight: {measurement['weight']}kg")
    print(f"Timestamp: {measurement['timestamp']}")

    print(f"\n{'='*60}")
    print(f"KALMAN STATE BEFORE PROCESSING")
    print(f"{'='*60}")
    initial_state = store.get_state(user_id)
    if initial_state:
        print(f"Last raw weight: {initial_state.get('last_raw_weight')}")
        if initial_state.get('last_state') is not None:
            print(f"Last state (position, velocity):")
            state_array = initial_state['last_state']
            if hasattr(state_array, 'shape'):
                for i, row in enumerate(state_array):
                    print(f"  Component {i}: {row}")
        if initial_state.get('last_covariance') is not None:
            print(f"Last covariance:")
            cov_array = initial_state['last_covariance']
            if hasattr(cov_array, 'shape'):
                for i, mat in enumerate(cov_array):
                    print(f"  Covariance {i}:")
                    for j, row in enumerate(mat):
                        print(f"    Row {j}: {row}")
        print(f"Measurements since reset: {initial_state.get('measurements_since_reset')}")

    print(f"\n{'='*60}")
    print(f"PROCESSING RESULT")
    print(f"{'='*60}")
    print(f"  Accepted: {result['accepted']}")

    # Print quality score with maximum precision
    if result.get('quality_score') is not None:
        print(f"  Quality score: {result['quality_score']:.18f}")
    else:
        print(f"  Quality score: None")

    # Print quality components if available
    if result.get('quality_components'):
        print(f"\n  Quality Components:")
        for component, value in result['quality_components'].items():
            if value is not None:
                print(f"    {component}: {value:.18f}")
            else:
                print(f"    {component}: None")

    # Print other result fields
    if result.get('kalman_estimate') is not None:
        print(f"  Kalman estimate: {result['kalman_estimate']:.18f}")
    else:
        print(f"  Kalman estimate: None")

    print(f"  Rejection reason: {result.get('rejection_reason', 'N/A')}")

    # Print final state after processing
    print(f"\n{'='*60}")
    print(f"KALMAN STATE AFTER PROCESSING")
    print(f"{'='*60}")
    final_state = store.get_state(user_id)
    if final_state:
        print(f"Last raw weight: {final_state.get('last_raw_weight')}")
        if final_state.get('last_state') is not None:
            print(f"Last state (position, velocity):")
            state_array = final_state['last_state']
            if hasattr(state_array, 'shape'):
                for i, row in enumerate(state_array):
                    print(f"  Component {i}: {row}")
        if final_state.get('last_covariance') is not None:
            print(f"Last covariance:")
            cov_array = final_state['last_covariance']
            if hasattr(cov_array, 'shape'):
                for i, mat in enumerate(cov_array):
                    print(f"  Covariance {i}:")
                    for j, row in enumerate(mat):
                        print(f"    Row {j}: {row}")

    print(f"{'='*60}")

    # Assertions - these should match the full 120-measurement run
    assert result is not None, "Processing should return a result"
    assert "accepted" in result, "Result should have 'accepted' field"
    assert "quality_score" in result, "Result should have 'quality_score' field"

    # Epsilon-based comparison to verify consistency with TypeScript implementation
    # within acceptable floating-point precision tolerance
    expected_quality_score = 0.009308080750420552  # Reference value
    quality_score = result.get("quality_score", 0)

    # Use a larger epsilon for accumulated floating-point errors
    test_epsilon = 1e-9  # Allow up to 1 billionth difference

    # Expected from full run: accepted=False, quality_score≈0.009308
    print(f"\nExpected (from full 120-measurement run):")
    print(f"  Accepted: False")
    print(f"  Quality score: 0.009308")
    print(f"\nEpsilon-based comparison:")
    print(f"  Expected:  {expected_quality_score:.18f}")
    print(f"  Actual:    {quality_score:.18f}")
    print(f"  Difference: {abs(quality_score - expected_quality_score):.3e}")
    print(f"  Within epsilon ({test_epsilon}): {abs(quality_score - expected_quality_score) < test_epsilon}")

    # Verify quality score matches within epsilon
    assert abs(quality_score - expected_quality_score) < test_epsilon, \
        f"Quality score {quality_score:.18f} differs from expected {expected_quality_score:.18f} by more than epsilon {test_epsilon}"

    # Store results for comparison
    test_result = {
        "accepted": result["accepted"],
        "quality_score": result.get("quality_score"),
        "kalman_estimate": result.get("kalman_estimate"),
    }

    # Save results to fixture for TypeScript comparison
    output_file = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures" / "python_single_measurement_result.json"
    with open(output_file, 'w') as f:
        json.dump({
            "description": "Python result for single measurement with Kalman state",
            "measurement_id": measurement["id"],
            "result": test_result
        }, f, indent=2)

    print(f"\n✅ Saved Python result to: {output_file}")


if __name__ == "__main__":
    # Load fixture
    fixture_path = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures" / "kalman_state_replay_divergence.json"
    with open(fixture_path, 'r') as f:
        fixture = json.load(f)

    # Create store with state
    store = InMemoryStore()
    user_id = "test-user"
    state = fixture["kalman_state"].copy()

    # Convert timestamp strings
    if state.get("last_timestamp"):
        state["last_timestamp"] = datetime.fromisoformat(state["last_timestamp"])
    if state.get("last_accepted_timestamp"):
        state["last_accepted_timestamp"] = datetime.fromisoformat(state["last_accepted_timestamp"])
    if state.get("reset_timestamp"):
        state["reset_timestamp"] = datetime.fromisoformat(state["reset_timestamp"])

    # Convert arrays back to numpy arrays
    if state.get("last_state") and isinstance(state["last_state"], list):
        state["last_state"] = np.array(state["last_state"])
    if state.get("last_covariance") and isinstance(state["last_covariance"], list):
        state["last_covariance"] = np.array(state["last_covariance"])

    store.save_state(user_id, state)

    # Get measurement and config
    measurement = fixture["target_measurement"]
    config = fixture["config"]

    # Process measurement
    result = process_measurement(
        user_id=user_id,
        weight=measurement["weight"],
        timestamp=datetime.fromisoformat(measurement["timestamp"]),
        source=measurement["source"],
        config=config,
        unit=measurement["unit"],
        db=store,
        user_height_m=1.75,
    )

    # Print results with full precision
    print(f"\n{'='*60}")
    print(f"PYTHON - Single Measurement Test (FULL PRECISION)")
    print(f"{'='*60}")
    print(f"Measurement ID: {measurement['id'][:16]}...")
    print(f"Weight: {measurement['weight']}kg")

    print(f"\n{'='*60}")
    print(f"KALMAN STATE BEFORE PROCESSING")
    print(f"{'='*60}")
    initial_state = store.get_state(user_id)
    if initial_state:
        print(f"Last raw weight: {initial_state.get('last_raw_weight')}")
        if initial_state.get('last_state') is not None:
            print(f"Last state (position, velocity):")
            state_array = initial_state['last_state']
            if hasattr(state_array, 'shape'):
                for i, row in enumerate(state_array):
                    print(f"  Component {i}: {row}")
        if initial_state.get('last_covariance') is not None:
            print(f"Last covariance:")
            cov_array = initial_state['last_covariance']
            if hasattr(cov_array, 'shape'):
                for i, mat in enumerate(cov_array):
                    print(f"  Covariance {i}:")
                    for j, row in enumerate(mat):
                        print(f"    Row {j}: {row}")
        print(f"Measurements since reset: {initial_state.get('measurements_since_reset')}")

    print(f"\n{'='*60}")
    print(f"PROCESSING RESULT")
    print(f"{'='*60}")
    print(f"  Accepted: {result['accepted']}")

    # Print quality score with maximum precision
    quality_score = result.get('quality_score')
    if quality_score is not None:
        print(f"  Quality score: {quality_score:.18f}")
    else:
        print(f"  Quality score: None")

    # Print quality components if available
    if result.get('quality_components'):
        print(f"\n  Quality Components:")
        for component, value in result['quality_components'].items():
            if value is not None:
                print(f"    {component}: {value:.18f}")
            else:
                print(f"    {component}: None")

    # Print other result fields
    if result.get('kalman_estimate') is not None:
        print(f"  Kalman estimate: {result['kalman_estimate']:.18f}")
    else:
        print(f"  Kalman estimate: None")

    print(f"  Rejection reason: {result.get('rejection_reason', 'N/A')}")

    # Print final state after processing
    print(f"\n{'='*60}")
    print(f"KALMAN STATE AFTER PROCESSING")
    print(f"{'='*60}")
    final_state = store.get_state(user_id)
    if final_state:
        print(f"Last raw weight: {final_state.get('last_raw_weight')}")
        if final_state.get('last_state') is not None:
            print(f"Last state (position, velocity):")
            state_array = final_state['last_state']
            if hasattr(state_array, 'shape'):
                for i, row in enumerate(state_array):
                    print(f"  Component {i}: {row}")
        if final_state.get('last_covariance') is not None:
            print(f"Last covariance:")
            cov_array = final_state['last_covariance']
            if hasattr(cov_array, 'shape'):
                for i, mat in enumerate(cov_array):
                    print(f"  Covariance {i}:")
                    for j, row in enumerate(mat):
                        print(f"    Row {j}: {row}")

    print(f"{'='*60}")
