#!/usr/bin/env python3

import csv
import tempfile
from datetime import datetime
from pathlib import Path

from src.database import get_state_db
from src.processor import process_weight_enhanced


def test_export_after_processing():
    """Test database export after processing some weight data."""
    db = get_state_db()
    
    # Process some measurements for a user
    config = {
        "processing": {
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20,
            "kalman_cleanup_threshold": 4.0
        },
        "kalman": {
            "initial_variance": 0.361,
            "transition_covariance_weight": 0.0160,
            "transition_covariance_trend": 0.0001,
            "observation_covariance": 3.490,
            "reset_gap_days": 30
        }
    }
    
    # Process several measurements
    weights = [75.0, 75.2, 74.8, 75.1, 74.9]
    base_time = datetime(2024, 1, 15, 10, 0)
    
    for i, weight in enumerate(weights):
        timestamp = datetime(2024, 1, 15 + i, 10, 0)
        result = process_weight_enhanced(
            user_id="test_user_001",
            weight=weight,
            timestamp=timestamp,
            source="scale",
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            unit="kg"
        )
        print(f"Processed weight {weight}: accepted={result['accepted']}")
    
    # Export to CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        users_exported = db.export_to_csv(csv_path)
        assert users_exported == 1, f"Expected 1 user, got {users_exported}"
        
        # Read and verify CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            
            row = rows[0]
            assert row['user_id'] == 'test_user_001'
            assert row['has_kalman_params'] == 'true'
            
            # Should have a last weight close to the last measurement
            if row['last_weight']:
                last_weight = float(row['last_weight'])
                assert 74.0 <= last_weight <= 76.0, f"Last weight {last_weight} out of expected range"
            
            # Should have a timestamp
            assert row['last_timestamp'] != ''
            
            print(f"Exported user data:")
            print(f"  User ID: {row['user_id']}")
            print(f"  Last timestamp: {row['last_timestamp']}")
            print(f"  Last weight: {row['last_weight']}")
            print(f"  Last trend: {row['last_trend']}")
            print(f"  Has Kalman params: {row['has_kalman_params']}")
            
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_export_multiple_users_after_processing():
    """Test exporting multiple users after processing."""
    # Clear any existing state from previous tests
    import src.database
    src.database._db_instance = None
    db = get_state_db()
    
    config = {
        "processing": {
            "min_weight": 30.0,
            "max_weight": 400.0,
            "max_daily_change": 0.05,
            "extreme_threshold": 0.20,
            "kalman_cleanup_threshold": 4.0
        },
        "kalman": {
            "initial_variance": 0.361,
            "transition_covariance_weight": 0.0160,
            "transition_covariance_trend": 0.0001,
            "observation_covariance": 3.490,
            "reset_gap_days": 30
        }
    }
    
    # Process measurements for multiple users
    users_data = {
        "user_alpha": [70.0, 70.2, 69.8],
        "user_beta": [85.0, 85.5, 84.9],
        "user_gamma": [62.0, 62.1, 62.3]
    }
    
    for user_id, weights in users_data.items():
        for i, weight in enumerate(weights):
            timestamp = datetime(2024, 1, 15 + i, 10, 0)
            process_weight_enhanced(
                user_id=user_id,
                weight=weight,
                timestamp=timestamp,
                source="scale",
                processing_config=config["processing"],
                kalman_config=config["kalman"],
                unit="kg"
            )
    
    # Export to CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        users_exported = db.export_to_csv(csv_path)
        assert users_exported == 3, f"Expected 3 users, got {users_exported}"
        
        # Read and verify CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            
            # Check all users are present (sorted)
            user_ids = [row['user_id'] for row in rows]
            assert user_ids == ['user_alpha', 'user_beta', 'user_gamma']
            
            # All should have Kalman params
            for row in rows:
                assert row['has_kalman_params'] == 'true'
                assert row['last_timestamp'] != ''
                
            print(f"Exported {len(rows)} users successfully")
            
    finally:
        Path(csv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("Testing database export after processing...")
    test_export_after_processing()
    print("✓ Export after processing test passed\n")
    
    print("Testing multiple users export after processing...")
    test_export_multiple_users_after_processing()
    print("✓ Multiple users export test passed\n")
    
    print("All integration tests passed!")