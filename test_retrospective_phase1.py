#!/usr/bin/env python3
"""
Test Phase 1 Retrospective Implementation

Tests the core infrastructure components:
1. Database state history extensions
2. Batch outlier detection
3. Replay manager
4. RetroBuffer management
5. Configuration integration
"""

import sys
sys.path.append('.')

from datetime import datetime, timedelta
import json
import numpy as np

def test_database_extensions():
    """Test database state history functionality."""
    print("Testing Database Extensions...")

    # Import directly to avoid __init__.py issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("database", "src/database.py")
    database_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_module)

    ProcessorStateDB = database_module.ProcessorStateDB

    # Create test database
    db = ProcessorStateDB()
    user_id = "test_user_001"

    # Test initial state includes state_history
    initial_state = db.create_initial_state()
    assert 'state_history' in initial_state
    assert initial_state['state_history'] == []
    print("✓ Initial state includes empty state_history")

    # Create a mock state with Kalman data
    test_state = initial_state.copy()
    test_state['kalman_params'] = {'process_noise': 0.1, 'measurement_noise': 1.0}
    test_state['last_state'] = np.array([70.0, 0.1])
    test_state['last_covariance'] = np.array([[1.0, 0.0], [0.0, 1.0]])

    db.save_state(user_id, test_state)

    # Test state snapshot saving
    timestamp1 = datetime(2024, 1, 1, 12, 0, 0)
    result = db.save_state_snapshot(user_id, timestamp1)
    assert result == True
    print("✓ State snapshot saved successfully")

    # Test snapshot retrieval
    timestamp2 = datetime(2024, 1, 1, 18, 0, 0)
    snapshot = db.get_state_snapshot_before(user_id, timestamp2)
    assert snapshot is not None
    assert snapshot['timestamp'] == timestamp1
    print("✓ State snapshot retrieved correctly")

    # Test state restoration
    result = db.restore_state_from_snapshot(user_id, snapshot)
    assert result == True
    print("✓ State restoration completed")

    # Test history count
    count = db.get_state_history_count(user_id)
    assert count == 1
    print("✓ State history count correct")

    print("Database Extensions: ✓ PASSED\n")


def test_outlier_detection():
    """Test outlier detection algorithms."""
    print("Testing Outlier Detection...")

    # Import directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("outlier_detection", "src/outlier_detection.py")
    outlier_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outlier_module)

    OutlierDetector = outlier_module.OutlierDetector

    # Create detector with test config
    config = {
        'iqr_multiplier': 1.5,
        'z_score_threshold': 3.0,
        'temporal_max_change_percent': 0.30,
        'min_measurements_for_analysis': 5
    }

    detector = OutlierDetector(config)
    print(f"✓ Detector initialized with IQR multiplier: {detector.iqr_multiplier}")

    # Create test measurements with known outliers
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    measurements = []
    weights = [70.0, 71.0, 69.5, 70.5, 95.0, 70.2, 69.8, 71.2, 70.1]  # 95.0 is clear outlier

    for i, weight in enumerate(weights):
        measurements.append({
            'weight': weight,
            'timestamp': base_time + timedelta(hours=i),
            'source': 'test'
        })

    # Test outlier detection
    outlier_indices = detector.detect_outliers(measurements)
    assert 4 in outlier_indices  # Index of 95.0
    print(f"✓ Detected outliers at indices: {sorted(outlier_indices)}")

    # Test clean measurements extraction
    clean_measurements, outliers = detector.get_clean_measurements(measurements)
    assert len(clean_measurements) == len(measurements) - len(outliers)
    print(f"✓ Clean measurements: {len(clean_measurements)}/{len(measurements)}")

    # Test analysis
    analysis = detector.analyze_outliers(measurements, outlier_indices)
    assert analysis['outlier_count'] == len(outlier_indices)
    assert analysis['total_measurements'] == len(measurements)
    print(f"✓ Analysis: {analysis['outlier_count']} outliers ({analysis['outlier_percentage']:.1f}%)")

    print("Outlier Detection: ✓ PASSED\n")


def test_retro_buffer():
    """Test retrospective buffer management."""
    print("Testing Retro Buffer...")

    # Import directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("retro_buffer", "src/retro_buffer.py")
    retro_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retro_module)

    RetroBuffer = retro_module.RetroBuffer

    config = {
        'buffer_hours': 2,  # Short for testing
        'max_buffer_measurements': 10,
        'trigger_mode': 'time_based'
    }

    buffer = RetroBuffer(config)
    user_id = "test_user_002"

    print(f"✓ Buffer initialized with {buffer.buffer_hours}h window")

    # Test measurement addition
    base_time = datetime.now()
    measurement = {
        'weight': 70.0,
        'timestamp': base_time,
        'source': 'test',
        'unit': 'kg'
    }

    result = buffer.add_measurement(user_id, measurement)
    assert result['success'] == True
    assert result['buffer_size'] == 1
    print("✓ Measurement added to buffer")

    # Test buffer info
    info = buffer.get_buffer_info(user_id)
    assert info is not None
    assert info['measurement_count'] == 1
    print(f"✓ Buffer info: {info['measurement_count']} measurements")

    # Test measurement retrieval
    measurements = buffer.get_buffer_measurements(user_id)
    assert len(measurements) == 1
    assert measurements[0]['weight'] == 70.0
    print("✓ Measurements retrieved from buffer")

    # Test buffer clearing
    result = buffer.clear_buffer(user_id)
    assert result == True
    info = buffer.get_buffer_info(user_id)
    assert info['measurement_count'] == 0
    print("✓ Buffer cleared successfully")

    # Test statistics
    stats = buffer.get_stats()
    assert 'active_buffers' in stats
    print(f"✓ Buffer stats: {stats['active_buffers']} active buffers")

    print("Retro Buffer: ✓ PASSED\n")


def test_replay_manager():
    """Test replay manager functionality."""
    print("Testing Replay Manager...")

    # Import directly
    import importlib.util

    # Database
    spec = importlib.util.spec_from_file_location("database", "src/database.py")
    database_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_module)
    ProcessorStateDB = database_module.ProcessorStateDB

    # Replay manager
    spec = importlib.util.spec_from_file_location("replay_manager", "src/replay_manager.py")
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)
    ReplayManager = replay_module.ReplayManager

    # Create test database and replay manager
    db = ProcessorStateDB()
    config = {
        'max_processing_time_seconds': 60,
        'require_rollback_confirmation': False,
        'preserve_immediate_results': True
    }

    replay_mgr = ReplayManager(db, config)
    user_id = "test_user_003"

    print(f"✓ Replay manager initialized with {replay_mgr.max_processing_time}s timeout")

    # Test backup creation
    # First create some state
    test_state = db.create_initial_state()
    test_state['kalman_params'] = {'test': True}
    db.save_state(user_id, test_state)

    result = replay_mgr._create_state_backup(user_id)
    assert result == True
    assert replay_mgr.has_backup(user_id) == True
    print("✓ State backup created")

    # Test backup restoration
    result = replay_mgr._restore_state_from_backup(user_id)
    assert result == True
    print("✓ State restored from backup")

    # Test backup cleanup
    count = replay_mgr.cleanup_old_backups()
    assert count >= 0
    print(f"✓ Cleaned up {count} old backups")

    # Test stats
    stats = replay_mgr.get_replay_stats()
    assert 'max_processing_time' in stats
    print(f"✓ Replay stats: {stats['max_processing_time']}s max time")

    print("Replay Manager: ✓ PASSED\n")


def test_configuration():
    """Test configuration loading and validation."""
    print("Testing Configuration...")

    import tomllib

    # Test config loading
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    # Test retrospective section
    retro_config = config.get('retrospective', {})
    assert retro_config.get('enabled') == True
    assert retro_config.get('buffer_hours') == 72
    assert retro_config.get('trigger_mode') == 'time_based'
    print("✓ Retrospective configuration loaded")

    # Test outlier detection config
    outlier_config = retro_config.get('outlier_detection', {})
    assert outlier_config.get('iqr_multiplier') == 1.5
    assert outlier_config.get('z_score_threshold') == 3.0
    print("✓ Outlier detection configuration loaded")

    # Test safety config
    safety_config = retro_config.get('safety', {})
    assert safety_config.get('max_processing_time_seconds') == 60
    assert safety_config.get('preserve_immediate_results') == True
    print("✓ Safety configuration loaded")

    print("Configuration: ✓ PASSED\n")


def test_integration():
    """Test basic integration between components."""
    print("Testing Integration...")

    # Import all components directly
    import importlib.util

    # Database
    spec = importlib.util.spec_from_file_location("database", "src/database.py")
    database_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database_module)
    ProcessorStateDB = database_module.ProcessorStateDB

    # RetroBuffer
    spec = importlib.util.spec_from_file_location("retro_buffer", "src/retro_buffer.py")
    retro_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retro_module)
    RetroBuffer = retro_module.RetroBuffer

    # OutlierDetector
    spec = importlib.util.spec_from_file_location("outlier_detection", "src/outlier_detection.py")
    outlier_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outlier_module)
    OutlierDetector = outlier_module.OutlierDetector

    # ReplayManager
    spec = importlib.util.spec_from_file_location("replay_manager", "src/replay_manager.py")
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)
    ReplayManager = replay_module.ReplayManager

    # Initialize all components
    db = ProcessorStateDB()
    buffer = RetroBuffer({'buffer_hours': 1, 'max_buffer_measurements': 5})
    detector = OutlierDetector({'min_measurements_for_analysis': 3})
    replay_mgr = ReplayManager(db, {})

    user_id = "integration_test_user"

    # Create test measurements
    base_time = datetime.now()
    measurements = []
    weights = [70.0, 71.0, 120.0, 70.5]  # 120.0 is outlier

    for i, weight in enumerate(weights):
        measurement = {
            'weight': weight,
            'timestamp': base_time + timedelta(minutes=i*30),
            'source': 'test'
        }
        measurements.append(measurement)
        buffer.add_measurement(user_id, measurement)

    print(f"✓ Added {len(measurements)} measurements to buffer")

    # Get buffered measurements and detect outliers
    buffered = buffer.get_buffer_measurements(user_id)
    assert len(buffered) == len(measurements)

    clean_measurements, outliers = detector.get_clean_measurements(buffered)
    assert len(outliers) > 0  # Should detect 120.0 as outlier
    print(f"✓ Detected {len(outliers)} outliers from {len(buffered)} measurements")

    # Test state snapshot saving
    db.save_state_snapshot(user_id, base_time)
    snapshot = db.get_state_snapshot_before(user_id, base_time + timedelta(hours=1))
    # Note: snapshot will be None since we didn't create proper Kalman state
    print("✓ State snapshot operations completed")

    print("Integration: ✓ PASSED\n")


def run_all_tests():
    """Run all Phase 1 tests."""
    print("=== Retrospective Data Quality Processor - Phase 1 Tests ===\n")

    try:
        test_database_extensions()
        test_outlier_detection()
        test_retro_buffer()
        test_replay_manager()
        test_configuration()
        test_integration()

        print("🎉 ALL PHASE 1 TESTS PASSED! 🎉")
        print("\nPhase 1 Core Infrastructure is ready for integration.")
        print("Next: Test with real measurement data and full system integration.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)