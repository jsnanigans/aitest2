#!/usr/bin/env python3

import csv
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from src.database import ProcessorStateDB


def test_export_empty_database():
    """Test exporting an empty database."""
    db = ProcessorStateDB()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        users_exported = db.export_to_csv(csv_path)
        assert users_exported == 0
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 0
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_export_single_user():
    """Test exporting a single user's state."""
    db = ProcessorStateDB()
    
    state = {
        'last_state': np.array([75.5, 0.02]),
        'last_covariance': np.eye(2),
        'last_timestamp': datetime(2024, 1, 15, 10, 30),
        'kalman_params': {
            'process_noise_cov': 0.01,
            'observation_cov': 1.0,
            'initial_state_cov': 100.0,
            'min_uncertainty': 0.1
        },
        'state_reset_count': 2,
        'last_reset_timestamp': datetime(2024, 1, 10, 8, 0)
    }
    
    db.save_state('user123', state)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        users_exported = db.export_to_csv(csv_path)
        assert users_exported == 1
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            
            row = rows[0]
            assert row['user_id'] == 'user123'
            assert row['last_timestamp'] == '2024-01-15T10:30:00'
            assert row['last_weight'] == '75.50'
            assert row['last_trend'] == '0.0200'
            assert row['has_kalman_params'] == 'true'
            assert row['process_noise'] == '0.01'
            assert row['measurement_noise'] == '1.0'
            assert row['state_reset_count'] == '2'
            assert row['last_reset_timestamp'] == '2024-01-10T08:00:00'
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_export_multiple_users():
    """Test exporting multiple users with various states."""
    db = ProcessorStateDB()
    
    # User with full state
    state1 = {
        'last_state': np.array([82.3, -0.01]),
        'last_covariance': np.eye(2),
        'last_timestamp': datetime(2024, 1, 20, 14, 15),
        'kalman_params': {
            'process_noise_cov': 0.02,
            'observation_cov': 2.0,
        },
        'adapted_params': {
            'process_noise': 0.015,
            'measurement_noise': 1.5
        }
    }
    db.save_state('user_alpha', state1)
    
    # User with minimal state
    state2 = {
        'last_state': None,
        'last_covariance': None,
        'last_timestamp': None,
        'kalman_params': None,
    }
    db.save_state('user_beta', state2)
    
    # User with partial state
    state3 = {
        'last_state': [68.9, 0.0],
        'last_covariance': None,
        'last_timestamp': datetime(2024, 1, 18, 9, 0),
        'kalman_params': None,
    }
    db.save_state('user_gamma', state3)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        users_exported = db.export_to_csv(csv_path)
        assert users_exported == 3
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            
            # Check sorted order
            assert rows[0]['user_id'] == 'user_alpha'
            assert rows[1]['user_id'] == 'user_beta'
            assert rows[2]['user_id'] == 'user_gamma'
            
            # Check user_alpha
            assert rows[0]['last_weight'] == '82.30'
            assert rows[0]['last_trend'] == '-0.0100'
            assert rows[0]['has_kalman_params'] == 'true'
            assert rows[0]['has_adapted_params'] == 'true'
            assert rows[0]['adapted_process_noise'] == '0.015'
            
            # Check user_beta (empty state)
            assert rows[1]['last_weight'] == ''
            assert rows[1]['last_trend'] == ''
            assert rows[1]['has_kalman_params'] == 'false'
            assert rows[1]['has_adapted_params'] == 'false'
            
            # Check user_gamma (partial state)
            assert rows[2]['last_weight'] == '68.90'
            assert rows[2]['last_trend'] == '0.0000'
            assert rows[2]['has_kalman_params'] == 'false'
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_export_creates_directory():
    """Test that export creates the output directory if it doesn't exist."""
    db = ProcessorStateDB()
    
    db.save_state('test_user', {
        'last_state': [70.0, 0.0],
        'last_timestamp': datetime.now(),
        'kalman_params': None
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'subdir' / 'export.csv'
        
        users_exported = db.export_to_csv(str(csv_path))
        assert users_exported == 1
        assert csv_path.exists()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]['user_id'] == 'test_user'


if __name__ == "__main__":
    test_export_empty_database()
    print("✓ Empty database export test passed")
    
    test_export_single_user()
    print("✓ Single user export test passed")
    
    test_export_multiple_users()
    print("✓ Multiple users export test passed")
    
    test_export_creates_directory()
    print("✓ Directory creation test passed")
    
    print("\nAll database export tests passed!")