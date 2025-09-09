#!/usr/bin/env python3
"""Test suite for weight stream reprocessor."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, date, timedelta
import numpy as np

from reprocessor import WeightReprocessor
from processor_database import ProcessorDatabase
from processor import WeightProcessor


def test_select_best_measurements():
    """Test measurement selection algorithm."""
    print("\nTest: Measurement Selection")
    print("-" * 40)
    
    measurements = [
        {'weight': 6226.0, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 6, 0, 0)},
        {'weight': 123.0, 'source': 'https://api.iglucose.com', 'timestamp': datetime(2025, 3, 6, 12, 52)},
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements)
    
    assert len(selected) == 1
    assert selected[0]['weight'] == 123.0
    print(f"✓ Extreme outlier rejected: 6226kg → 123kg selected")
    
    measurements = [
        {'weight': 85.2, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 6, 8, 0)},
        {'weight': 85.5, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 6, 12, 0)},
        {'weight': 85.3, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 6, 16, 0)},
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements)
    
    assert len(selected) == 3  # Now keeps all valid measurements
    assert all(85.0 <= s['weight'] <= 86.0 for s in selected)
    print(f"✓ Multiple valid readings: kept all {len(selected)} measurements")
    
    measurements = [
        {'weight': 80.0, 'source': 'internal-questionnaire', 'timestamp': datetime(2025, 3, 6, 0, 0)},
        {'weight': 75.0, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 6, 8, 0)},
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements)
    
    assert len(selected) == 2  # Keeps both measurements now
    assert selected[0]['weight'] == 80.0  # But questionnaire is first due to priority
    assert selected[0]['source'] == 'internal-questionnaire'
    print(f"✓ Source priority: questionnaire first, but both kept ({len(selected)} total)")
    
    measurements = [
        {'weight': 70, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 26)},
        {'weight': 85, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 28)},
        {'weight': 14, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 29)},
        {'weight': 77, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 31)},
        {'weight': 120.8, 'source': 'https://api.iglucose.com', 'timestamp': datetime(2025, 3, 22, 15, 25)},
    ]
    
    selected = WeightReprocessor.select_best_measurements(measurements, recent_weight=80.0)
    
    assert len(selected) >= 1  # Should keep valid measurements, reject outliers
    assert all(70 <= s['weight'] <= 85 for s in selected)
    print(f"✓ Multiple outliers filtered: {len(measurements)} → {len(selected)} selected")


def test_retroactive_detection():
    """Test detection of retroactive additions."""
    print("\nTest: Retroactive Detection")
    print("-" * 40)
    
    db = ProcessorDatabase()
    user_id = "test_retro_user"
    
    db.save_state(user_id, {
        'initialized': True,
        'last_timestamp': datetime(2025, 3, 10, 12, 0),
        'last_state': np.array([80.0, 0.0]),
    })
    
    new_measurement = {
        'weight': 79.5,
        'timestamp': datetime(2025, 3, 5, 8, 0),
        'source': 'internal-questionnaire'
    }
    
    is_retroactive = WeightReprocessor.detect_retroactive_addition(
        user_id, new_measurement, db
    )
    
    assert is_retroactive == True
    print(f"✓ Retroactive detected: March 5 < March 10 (last processed)")
    
    future_measurement = {
        'weight': 80.5,
        'timestamp': datetime(2025, 3, 15, 8, 0),
        'source': 'patient-device'
    }
    
    is_retroactive = WeightReprocessor.detect_retroactive_addition(
        user_id, future_measurement, db
    )
    
    assert is_retroactive == False
    print(f"✓ Not retroactive: March 15 > March 10")


def test_daily_batch_processing():
    """Test processing a day's batch of measurements."""
    print("\nTest: Daily Batch Processing")
    print("-" * 40)
    
    db = ProcessorDatabase()
    user_id = "test_batch_user"
    
    measurements = [
        {'weight': 70, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 26)},
        {'weight': 85, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 28)},
        {'weight': 14, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 29)},
        {'weight': 77, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 31)},
        {'weight': 103, 'source': 'patient-device', 'timestamp': datetime(2025, 3, 22, 13, 44)},
        {'weight': 120.8, 'source': 'https://api.iglucose.com', 'timestamp': datetime(2025, 3, 22, 15, 25)},
    ]
    
    processing_config = {
        'min_weight': 30.0,
        'max_weight': 400.0,
        'max_daily_change': 0.05,
        'extreme_threshold': 0.20,
        'min_init_readings': 10
    }
    
    kalman_config = {
        'initial_variance': 0.5,
        'transition_covariance_weight': 0.05,
        'transition_covariance_trend': 0.0005,
        'observation_covariance': 1.5,
        'reset_gap_days': 30
    }
    
    result = WeightReprocessor.process_daily_batch(
        user_id=user_id,
        batch_date=date(2025, 3, 22),
        measurements=measurements,
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=None  
    )
    
    assert len(result['selected']) >= 1  # Should keep valid measurements
    assert len(result['rejected']) >= 1  # Should reject some outliers
    assert len(result['selected']) + len(result['rejected']) == len(measurements)
    print(f"✓ Batch processing: {len(measurements)} → {len(result['selected'])} selected, {len(result['rejected'])} rejected")
    print(f"  Selected: {[m['weight'] for m in result['selected']]}")
    print(f"  Rejected: {[m['weight'] for m in result['rejected']]}")


def test_snapshot_and_restore():
    """Test snapshot creation and restoration."""
    print("\nTest: Snapshot & Restore")
    print("-" * 40)
    
    db = ProcessorDatabase()
    user_id = "test_snapshot_user"
    
    original_state = {
        'initialized': True,
        'last_timestamp': datetime(2025, 3, 10, 12, 0),
        'last_state': np.array([80.0, 0.0]),
        'last_covariance': np.array([[1.0, 0.0], [0.0, 0.01]]),
    }
    
    db.save_state(user_id, original_state)
    
    snapshot_id = db.create_snapshot(user_id, datetime.now())
    assert snapshot_id is not None
    print(f"✓ Snapshot created: {snapshot_id}")
    
    modified_state = {
        'initialized': True,
        'last_timestamp': datetime(2025, 3, 15, 12, 0),
        'last_state': np.array([85.0, 0.5]),
        'last_covariance': np.array([[2.0, 0.0], [0.0, 0.02]]),
    }
    db.save_state(user_id, modified_state)
    
    current = db.get_state(user_id)
    assert np.allclose(current['last_state'], [85.0, 0.5])
    print(f"✓ State modified: weight changed to 85.0kg")
    
    restored = db.restore_snapshot(user_id, snapshot_id)
    assert restored == True
    
    current = db.get_state(user_id)
    assert np.allclose(current['last_state'], [80.0, 0.0])
    print(f"✓ State restored: weight back to 80.0kg")


def test_validation():
    """Test reprocessing validation."""
    print("\nTest: Reprocessing Validation")
    print("-" * 40)
    
    db = ProcessorDatabase()
    user_id = "test_validation_user"
    
    db.save_state(user_id, {
        'initialized': True,
        'last_state': np.array([75.0, 0.0]),
    })
    
    after_results = {
        'measurements_processed': 10,
        'errors': []
    }
    
    is_valid, reason = WeightReprocessor.validate_reprocessing(
        user_id, "snapshot_123", after_results, db
    )
    
    assert is_valid == True
    print(f"✓ Valid reprocessing: {reason}")
    
    db.save_state(user_id, {
        'initialized': True,
        'last_state': np.array([500.0, 0.0]),  
    })
    
    is_valid, reason = WeightReprocessor.validate_reprocessing(
        user_id, "snapshot_123", after_results, db
    )
    
    assert is_valid == False
    assert "Invalid weight" in reason
    print(f"✓ Invalid weight detected: {reason}")
    
    db.save_state(user_id, {
        'initialized': True,
        'last_state': np.array([75.0, 0.0]),  
    })
    
    after_results_with_errors = {
        'measurements_processed': 10,
        'errors': [{'error': 'test'} for _ in range(6)]
    }
    
    is_valid, reason = WeightReprocessor.validate_reprocessing(
        user_id, "snapshot_123", after_results_with_errors, db
    )
    
    assert is_valid == False
    assert "High error rate" in reason
    print(f"✓ High error rate detected: {reason}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("REPROCESSOR TEST SUITE")
    print("=" * 50)
    
    test_select_best_measurements()
    test_retroactive_detection()
    test_daily_batch_processing()
    test_snapshot_and_restore()
    test_validation()
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()