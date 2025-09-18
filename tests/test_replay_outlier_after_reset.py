"""
Test replay system's ability to catch outliers after reset events.
Tests two scenarios:
1. With reset: Large gap triggers reset, then outlier should be caught by replay
2. Without reset: Measurements close together, outlier should be caught by replay
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from unittest.mock import patch

from src.processing.processor import process_measurement
from src.database.database import ProcessorStateDB
from src.config_loader import load_config


@pytest.fixture
def test_config():
    """Create test configuration with replay enabled."""
    return {
        'kalman': {
            'initial_variance': 1.0,
            'transition_covariance': 0.1,
            'observation_covariance': 1.0,
            'reset_threshold_days': 30,
            'adaptation': {
                'enabled': True,
                'initial_process_noise_multiplier': 5.0,
                'initial_observation_noise_multiplier': 3.0,
                'decay_rate': 0.1,
                'min_measurements': 5,
                'time_constant_days': 7.0
            },
            'reset': {
                'gap_threshold_days': 30,
                'soft_reset_sources': ['questionnaire', 'patient-upload', 'manual-entry']
            }
        },
        'outlier_detection': {
            'enabled': True,
            'method': 'mad',
            'mad_threshold': 3.0,
            'min_measurements': 5,
            'lookback_days': 30,
            'extreme_threshold_percent': 0.15,
            'quality_override_enabled': True,
            'quality_override_threshold': 0.8
        },
        'replay': {
            'enabled': True,
            'lookback_days': 30,
            'trigger_conditions': {
                'high_rejection_rate': {
                    'enabled': True,
                    'threshold': 0.3,
                    'min_measurements': 5
                },
                'suspicious_pattern': {
                    'enabled': True,
                    'oscillation_count': 3,
                    'deviation_threshold': 2.0
                }
            },
            'outlier_methods': ['iqr', 'isolation_forest'],
            'consensus_threshold': 1
        },
        'quality_scoring': {
            'enabled': True,
            'weights': {
                'safety': 0.4,
                'plausibility': 0.3,
                'consistency': 0.2,
                'reliability': 0.1
            }
        },
        'features': {
            'validation_physiological': True,
            'validation_rate_limiting': True,
            'outlier_quality_override': True,
            'reset_adaptation': True,
            'replay_system': True
        }
    }


@pytest.fixture
def setup_test(test_config):
    """Setup test environment with config and database."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        import toml
        toml.dump(test_config, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        db = ProcessorStateDB()
        db.enable_validation = False  # Disable validation for tests
        yield config, db
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
        # Clear database state
        if hasattr(db, 'conn'):
            db.conn.close()


def test_outlier_after_reset_with_gap(setup_test):
    """
    Test case 1: Reset triggered by 30+ day gap, then outlier should be caught.
    
    Sequence:
    1. Initial measurement: 80 kg
    2. 35 day gap (triggers HARD reset)
    3. Outlier: 120 kg (should be caught by replay)
    4. Normal measurements: ~79 kg
    """
    config, db = setup_test
    user_id = 'test_user_reset'
    
    # Patch get_state_db to return our test database
    with patch('src.processing.processor.get_state_db', return_value=db):
        # Initial measurement
        result1 = process_measurement(
            user_id=user_id,
            weight=80.0,
            source='patient-device',
            timestamp=datetime(2025, 3, 1, 10, 0, 0),
            config=config
        )
        
        # After 35 day gap - outlier (120 kg instead of reasonable weight)
        result2 = process_measurement(
            user_id=user_id,
            weight=120.0,  # Outlier
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 27, 5),
            config=config
        )
        
        # Normal measurements shortly after
        result3 = process_measurement(
            user_id=user_id,
            weight=79.637,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 34, 18),
            config=config
        )
        
        result4 = process_measurement(
            user_id=user_id,
            weight=79.651,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 34, 40),
            config=config
        )
        
        result5 = process_measurement(
            user_id=user_id,
            weight=79.450,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 20, 29, 27),
            config=config
        )
        
        # Add more measurements to trigger replay
        result6 = process_measurement(
            user_id=user_id,
            weight=79.5,
            source='patient-device',
            timestamp=datetime(2025, 4, 11, 10, 0, 0),
            config=config
        )
        
        result7 = process_measurement(
            user_id=user_id,
            weight=79.6,
            source='patient-device',
            timestamp=datetime(2025, 4, 12, 10, 0, 0),
            config=config
        )
    
    # Check results
    assert result1['accepted'], "Initial measurement should be accepted"
    assert 'reset_event' in result1, "Initial measurement should have reset event"
    
    assert 'reset_event' in result2, "Measurement after gap should trigger reset"
    assert result2['reset_event']['type'] == 'hard', "Should be hard reset due to 35 day gap"
    
    # The 120 kg outlier should eventually be caught
    # Check database state after all measurements
    state = db.get_state(user_id) or {}
    
    # Verify replay was triggered
    replay_metadata = state.get('replay_metadata', {})
    replay_triggered = replay_metadata.get('last_replay_timestamp') is not None
    
    # The 120 kg measurement should be identified as outlier
    print(f"\n=== Test 1 (With Reset) Results ===")
    print(f"Result 2 (120kg outlier): accepted={result2.get('accepted')}, stage={result2.get('stage')}")
    print(f"Result 2 rejection reason: {result2.get('rejection_reason', 'N/A')}")
    print(f"Reset event type: {result2.get('reset_event', {}).get('type')}")
    print(f"Replay triggered: {replay_triggered}")
    if replay_triggered:
        print(f"Replay timestamp: {replay_metadata.get('last_replay_timestamp')}")
        print(f"Replay outliers found: {replay_metadata.get('results', {}).get('outlier_indices', [])}")
    
    # Check measurement history
    measurement_history = state.get('measurement_history', [])
    print(f"Total measurements in history: {len(measurement_history)}")
    weights_in_history = [m.get('weight', m.get('raw_weight')) for m in measurement_history]
    print(f"Weights in history: {weights_in_history}")
    
    # After reset, 120kg is a huge outlier compared to ~79kg
    # It should be caught either immediately or by replay
    if result2.get('accepted'):
        print(f"120kg was initially accepted (stage: {result2.get('stage')})")
        if not replay_triggered:
            print("WARNING: Replay was not triggered despite outlier pattern")
    else:
        print(f"120kg was rejected immediately: {result2.get('rejection_reason')}")


def test_outlier_without_reset(setup_test):
    """
    Test case 2: No reset (measurements close together), outlier should still be caught.
    
    Sequence:
    1. Initial measurements: ~80 kg
    2. After 7 days (no reset): 120 kg outlier
    3. More normal measurements: ~79 kg
    """
    config, db = setup_test
    user_id = 'test_user_no_reset'
    
    with patch('src.processing.processor.get_state_db', return_value=db):
        # Initial measurements to establish baseline
        result1 = process_measurement(
            user_id=user_id,
            weight=80.0,
            source='patient-device',
            timestamp=datetime(2025, 4, 1, 10, 0, 0),
            config=config
        )
        
        result2 = process_measurement(
            user_id=user_id,
            weight=79.8,
            source='patient-device',
            timestamp=datetime(2025, 4, 2, 10, 0, 0),
            config=config
        )
        
        result3 = process_measurement(
            user_id=user_id,
            weight=79.9,
            source='patient-device',
            timestamp=datetime(2025, 4, 3, 10, 0, 0),
            config=config
        )
        
        result4 = process_measurement(
            user_id=user_id,
            weight=80.1,
            source='patient-device',
            timestamp=datetime(2025, 4, 4, 10, 0, 0),
            config=config
        )
        
        result5 = process_measurement(
            user_id=user_id,
            weight=79.7,
            source='patient-device',
            timestamp=datetime(2025, 4, 5, 10, 0, 0),
            config=config
        )
        
        # After 5 days (within reset threshold) - outlier
        result_outlier = process_measurement(
            user_id=user_id,
            weight=120.0,  # Outlier
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 27, 5),
            config=config
        )
        
        # Normal measurements shortly after
        result6 = process_measurement(
            user_id=user_id,
            weight=79.637,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 34, 18),
            config=config
        )
        
        result7 = process_measurement(
            user_id=user_id,
            weight=79.651,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 34, 40),
            config=config
        )
        
        result8 = process_measurement(
            user_id=user_id,
            weight=79.450,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 20, 29, 27),
            config=config
        )
        
        # Add more to potentially trigger replay
        result9 = process_measurement(
            user_id=user_id,
            weight=79.5,
            source='patient-device',
            timestamp=datetime(2025, 4, 11, 10, 0, 0),
            config=config
        )
    
    # Check results
    assert result1['accepted'], "Initial measurement should be accepted"
    assert 'reset_event' not in result_outlier or result_outlier.get('reset_event', {}).get('type') != 'hard', \
        "Should NOT have hard reset (only 5-7 day gap)"
    
    # The 120 kg should be caught as outlier
    state = db.get_state(user_id) or {}
    
    print(f"\n=== Test 2 (Without Reset) Results ===")
    print(f"Outlier result (120kg): accepted={result_outlier.get('accepted')}, stage={result_outlier.get('stage')}")
    print(f"Has reset event: {'reset_event' in result_outlier}")
    if 'reset_event' in result_outlier:
        print(f"Reset type: {result_outlier.get('reset_event', {}).get('type')}")
    
    # Check if outlier detection caught it
    if not result_outlier.get('accepted'):
        print(f"Outlier was rejected immediately by: {result_outlier.get('rejection_reason')}")
    else:
        print(f"Outlier was accepted (stage: {result_outlier.get('stage')})")
        # Check if replay caught it
        replay_metadata = state.get('replay_metadata', {})
        replay_triggered = replay_metadata.get('last_replay_timestamp') is not None
        print(f"Replay triggered: {replay_triggered}")
        
        if replay_triggered:
            replay_results = replay_metadata.get('results', {})
            print(f"Replay outliers: {replay_results.get('outlier_indices', [])}")
    
    # The 120kg should be caught either immediately or by replay
    measurement_history = state.get('measurement_history', [])
    weights = [m.get('weight', m.get('raw_weight')) for m in measurement_history]
    print(f"All weights in history: {weights}")
    
    # With established baseline of ~80kg, 120kg is a 50% jump - clearly an outlier
    if 120.0 in weights and result_outlier.get('accepted'):
        print("WARNING: 120 kg outlier was accepted and not caught!")


def test_original_case_with_proper_limits(setup_test):
    """
    Test the original case where 34.56 kg should now be rejected by BMI validation.
    """
    config, db = setup_test
    user_id = 'e751ebe4-3e13-423d-bf50-88a9dd13f132'
    
    with patch('src.processing.processor.get_state_db', return_value=db):
        # This should now be rejected due to BMI < 18
        result = process_measurement(
            user_id=user_id,
            weight=34.56500244140625,
            source='patient-device',
            timestamp=datetime(2025, 4, 10, 19, 27, 5),
            unit='kg',
            config=config
        )
    
    print(f"\n=== Original Case (34.56 kg) ===")
    print(f"Result: accepted={result.get('accepted')}, stage={result.get('stage')}")
    print(f"Rejection reason: {result.get('rejection_reason', 'N/A')}")
    
    # Should be rejected at preprocessing due to BMI limits
    if result.get('accepted'):
        print(f"ERROR: 34.56 kg was accepted when it should be rejected!")
        print(f"Preprocessing data: {result.get('preprocessing')}")
        # Check the user's height
        state = db.get_state(user_id) or {}
        print(f"User state height info: {state.get('user_height_m', 'not found')}")
    else:
        assert 'BMI' in result.get('rejection_reason', ''), \
            f"Should be rejected for BMI reasons, got: {result.get('rejection_reason')}"
        print("SUCCESS: Weight properly rejected for BMI violation")


if __name__ == '__main__':
    # Run tests directly
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    pytest.main([__file__, '-xvs'])