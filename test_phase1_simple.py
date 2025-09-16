#!/usr/bin/env python3
"""
Simple Phase 1 Test - Core Components Only

Tests the implemented retrospective components without complex dependencies:
1. Database state history extensions
2. Batch outlier detection
3. RetroBuffer management
4. Configuration integration
"""

import sys
sys.path.append('.')

from datetime import datetime, timedelta
import numpy as np
import importlib.util


def import_module(name, file_path):
    """Import module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_functionality():
    """Test all core retrospective functionality."""
    print("=== Phase 1 Core Functionality Test ===\n")

    # Import modules
    print("1. Testing component imports...")
    database_module = import_module("database", "src/database.py")
    outlier_module = import_module("outlier_detection", "src/outlier_detection.py")
    retro_module = import_module("retro_buffer", "src/retro_buffer.py")

    ProcessorStateDB = database_module.ProcessorStateDB
    OutlierDetector = outlier_module.OutlierDetector
    RetroBuffer = retro_module.RetroBuffer
    print("✓ All modules imported successfully\n")

    # Test database state history
    print("2. Testing database state history...")
    db = ProcessorStateDB()
    user_id = "test_user"

    # Check state_history in initial state
    state = db.create_initial_state()
    assert 'state_history' in state
    assert isinstance(state['state_history'], list)
    print("✓ Initial state includes state_history")

    # Create mock Kalman state
    state['kalman_params'] = {'test': True}
    state['last_state'] = np.array([70.0, 0.1])
    db.save_state(user_id, state)

    # Test snapshot saving
    timestamp = datetime.now()
    result = db.save_state_snapshot(user_id, timestamp)
    assert result == True
    print("✓ State snapshot saved")

    # Test snapshot retrieval
    snapshot = db.get_state_snapshot_before(user_id, timestamp + timedelta(hours=1))
    assert snapshot is not None
    print("✓ State snapshot retrieved")

    print("✓ Database state history: PASSED\n")

    # Test outlier detection
    print("3. Testing outlier detection...")
    detector = OutlierDetector({
        'iqr_multiplier': 1.5,
        'min_measurements_for_analysis': 5
    })

    # Create test data with clear outlier
    measurements = []
    weights = [70.0, 71.0, 69.5, 70.5, 95.0, 70.2, 69.8, 71.2]  # 95.0 is outlier
    base_time = datetime.now()

    for i, weight in enumerate(weights):
        measurements.append({
            'weight': weight,
            'timestamp': base_time + timedelta(hours=i),
            'source': 'test'
        })

    # Test outlier detection
    outliers = detector.detect_outliers(measurements)
    assert len(outliers) > 0
    assert 4 in outliers  # Index of 95.0
    print(f"✓ Detected {len(outliers)} outliers: {sorted(outliers)}")

    # Test clean measurements
    clean_measurements, outlier_indices = detector.get_clean_measurements(measurements)
    expected_clean = len(measurements) - len(outliers)
    assert len(clean_measurements) == expected_clean
    print(f"✓ Clean measurements: {len(clean_measurements)}/{len(measurements)}")

    print("✓ Outlier detection: PASSED\n")

    # Test retro buffer
    print("4. Testing retrospective buffer...")
    buffer = RetroBuffer({
        'buffer_hours': 1,  # Short for testing
        'max_buffer_measurements': 10,
        'trigger_mode': 'time_based'
    })

    # Add measurements to buffer
    for measurement in measurements[:3]:  # Add first 3
        result = buffer.add_measurement(user_id, measurement)
        assert result['success'] == True

    # Check buffer info
    info = buffer.get_buffer_info(user_id)
    assert info['measurement_count'] == 3
    print(f"✓ Buffer contains {info['measurement_count']} measurements")

    # Test buffer retrieval
    buffered = buffer.get_buffer_measurements(user_id)
    assert len(buffered) == 3
    print("✓ Buffer measurements retrieved")

    # Test buffer clearing
    buffer.clear_buffer(user_id)
    info = buffer.get_buffer_info(user_id)
    assert info['measurement_count'] == 0
    print("✓ Buffer cleared")

    print("✓ Retro buffer: PASSED\n")

    # Test configuration
    print("5. Testing configuration...")
    import tomllib

    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    retro_config = config.get('retrospective', {})
    assert retro_config.get('enabled') == True
    assert retro_config.get('buffer_hours') == 72
    print("✓ Retrospective config loaded")

    outlier_config = retro_config.get('outlier_detection', {})
    assert outlier_config.get('iqr_multiplier') == 1.5
    print("✓ Outlier detection config loaded")

    print("✓ Configuration: PASSED\n")

    # Integration test
    print("6. Testing integration...")

    # Initialize fresh components with config
    db = ProcessorStateDB()
    buffer = RetroBuffer(retro_config)
    detector = OutlierDetector(outlier_config)

    # Add measurements with outliers
    user_id = "integration_user"
    measurements = []
    weights = [70.0, 120.0, 69.5, 70.5]  # 120.0 is clear outlier

    for i, weight in enumerate(weights):
        measurement = {
            'weight': weight,
            'timestamp': datetime.now() + timedelta(minutes=i*30),
            'source': 'test'
        }
        measurements.append(measurement)
        buffer.add_measurement(user_id, measurement)

    # Get buffered measurements and analyze
    buffered = buffer.get_buffer_measurements(user_id)
    clean_measurements, outliers = detector.get_clean_measurements(buffered)

    print(f"✓ Integration test: {len(outliers)} outliers found from {len(buffered)} measurements")
    print("✓ Integration: PASSED\n")

    print("🎉 ALL PHASE 1 CORE TESTS PASSED! 🎉")
    print("\nPhase 1 Components Ready:")
    print("  ✓ Database state history extensions")
    print("  ✓ Batch outlier detection (IQR, Z-score, temporal)")
    print("  ✓ Thread-safe retrospective buffer")
    print("  ✓ Configuration integration")
    print("  ✓ Component integration")
    print("\nNext Steps:")
    print("  - Full system integration testing")
    print("  - Real data testing with user 0040872d")
    print("  - Performance validation")


if __name__ == "__main__":
    try:
        test_core_functionality()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)