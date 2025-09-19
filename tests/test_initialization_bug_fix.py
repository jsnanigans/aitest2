"""Test that initialization path correctly goes through quality validation."""

import pytest
from datetime import datetime
from src.processing.processor import process_measurement
from src.database.database import get_state_db


def test_initialization_quality_validation():
    """Test that the initialization path goes through quality validation.

    This test verifies the fix for the bug where Kalman state was persisted
    before quality validation during initialization.
    """
    # Setup
    db = get_state_db()
    user_id = "test_init_quality_user"
    timestamp = datetime(2024, 1, 1, 10, 0, 0)

    # Clear any existing state
    db.delete_state(user_id)

    # Config that enables quality scoring with strict validation
    config = {
        'processing': {
            'extreme_threshold': 0.20,
            'default_height_m': 1.75,
        },
        'kalman': {
            'observation_covariance': 3.49,
            'transition_covariance_weight': 0.001,
            'transition_covariance_trend': 0.0001,
            'initial_uncertainty': 100.0,
        },
        'quality_scoring': {
            'threshold': 0.5,  # Moderate threshold
            'component_weights': {
                'safety': 0.35,
                'plausibility': 0.25,
                'consistency': 0.25,
                'reliability': 0.15
            }
        },
        'adaptive_noise': {
            'enabled': True,
            'default_multiplier': 1.5
        }
    }

    # Test 1: First measurement with an implausible weight should be rejected
    # 500 kg is clearly implausible and should fail quality scoring
    result = process_measurement(
        user_id=user_id,
        weight=500.0,  # Implausible weight
        timestamp=timestamp,
        source="patient-device",
        config=config,
        unit='kg'
    )

    # Should be rejected (either by preprocessing or quality scoring)
    assert result['accepted'] == False
    # The important part is that it's rejected, not necessarily by which stage

    # Verify that state was NOT persisted (since measurement was rejected)
    state = db.get_state(user_id)
    assert state is None or not state.get('kalman_params'), "State should not be persisted for rejected initialization"

    # Test 2: A plausible weight should be accepted and state should be persisted
    result = process_measurement(
        user_id=user_id,
        weight=75.0,  # Plausible weight
        timestamp=timestamp,
        source="patient-device",
        config=config,
        unit='kg'
    )

    # Debug why it might be rejected
    if not result['accepted']:
        print(f"Result rejected: reason={result.get('reason')}, stage={result.get('stage')}")
        print(f"Quality score: {result.get('quality_score')}, components: {result.get('quality_components')}")
        print(f"Quality details: {result.get('quality_details')}")
    # Should be accepted
    assert result['accepted'] == True
    assert result['stage'] == 'initialization' or result['stage'] == 'accepted'

    # Verify that state was persisted after successful initialization
    state = db.get_state(user_id)
    assert state is not None
    assert state.get('kalman_params') is not None, "Kalman state should be persisted after successful initialization"
    assert state.get('last_accepted_timestamp') is not None

    # Clean up
    db.delete_state(user_id)


def test_initialization_with_high_quality_override():
    """Test that high-quality measurements during initialization can override outlier detection."""
    # Setup
    db = get_state_db()
    user_id = "test_init_override_user"
    timestamp = datetime(2024, 1, 1, 10, 0, 0)

    # Clear any existing state
    db.delete_state(user_id)

    # Config with quality scoring enabled
    config = {
        'processing': {
            'extreme_threshold': 0.20,
            'default_height_m': 1.75,
        },
        'kalman': {
            'observation_covariance': 3.49,
            'transition_covariance_weight': 0.001,
            'transition_covariance_trend': 0.0001,
            'initial_uncertainty': 100.0,
        },
        'quality_scoring': {
            'threshold': 0.4,  # Lower threshold for adaptation
            'component_weights': {
                'safety': 0.45,  # Higher safety weight during adaptation
                'plausibility': 0.10,
                'consistency': 0.15,
                'reliability': 0.30
            }
        },
        'adaptive_noise': {
            'enabled': True,
            'default_multiplier': 1.5
        }
    }

    # A weight from a highly reliable source should be accepted even if unusual
    result = process_measurement(
        user_id=user_id,
        weight=120.0,  # High but not impossible weight
        timestamp=timestamp,
        source="care-team-upload",  # Most reliable source
        config=config,
        unit='kg'
    )

    # Should be accepted due to high reliability of source
    assert result['accepted'] == True

    # Verify state was persisted
    state = db.get_state(user_id)
    assert state is not None
    assert state.get('kalman_params') is not None

    # Clean up
    db.delete_state(user_id)


if __name__ == "__main__":
    test_initialization_quality_validation()
    test_initialization_with_high_quality_override()
    print("All tests passed!")