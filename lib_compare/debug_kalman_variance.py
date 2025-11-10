"""
Debug script to trace kalman_variance calculation step-by-step in Python
to compare with TypeScript implementation.
"""

import sys
import os
from datetime import datetime, timezone

# Add python_lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_lib', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'be_implementation_service', 'src'))

from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database.memory_store import InMemoryStore
from aws.config.config_manager import ConfigManager

def debug_kalman_variance():
    print('=' * 80)
    print('DEBUG: Kalman Variance Calculation (Python)')
    print('=' * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'python_lib', 'config.toml')
    config = ConfigManager.load_config(config_path)

    # Create store
    store = InMemoryStore()
    user_id = 'debug-user:debug-device'

    # Fixed timestamps for reproducibility (match TypeScript)
    timestamp1 = datetime(2025, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
    timestamp2 = datetime(2025, 11, 11, 10, 0, 0, tzinfo=timezone.utc)  # 1 day later

    print('\n' + '=' * 80)
    print('MEASUREMENT 1: Initialize Kalman filter')
    print('=' * 80)
    print(f'Weight: 70.0 kg')
    print(f'Timestamp: {timestamp1.isoformat()}')
    print(f'Source: withings')

    result1 = process_measurement(
        user_id=user_id,
        weight=70.0,
        timestamp=timestamp1,
        source='withings',
        config=config,
        unit='kg',
        db=store,
        user_height_m=None
    )

    print('\nResult 1:')
    print(f'  filtered_weight: {result1.get("filtered_weight")}')
    print(f'  trend: {result1.get("trend")}')
    print(f'  kalman_variance: {result1.get("kalman_variance")}')
    print(f'  kalman_confidence_upper: {result1.get("kalman_confidence_upper")}')
    print(f'  kalman_confidence_lower: {result1.get("kalman_confidence_lower")}')

    # Get state after first measurement
    state1 = store.get_state(user_id)
    print('\nState after measurement 1:')
    print(f'  last_state shape: {state1["last_state"].shape}')
    print(f'  last_state[0]: {state1["last_state"][0]}')
    print(f'  last_state[1]: {state1["last_state"][1]}')
    print(f'  last_covariance shape: {state1["last_covariance"].shape}')
    print(f'  last_covariance[0]: ')
    print(f'    {state1["last_covariance"][0]}')
    print(f'  last_covariance[1]: ')
    print(f'    {state1["last_covariance"][1]}')
    print(f'  last_covariance[0][0,0]: {state1["last_covariance"][0][0,0]}')
    print(f'  last_covariance[1][0,0]: {state1["last_covariance"][1][0,0]}')

    if 'kalman_params' in state1:
        print('\nKalman params:')
        print(f'  observation_covariance: {state1["kalman_params"]["observation_covariance"]}')
        print(f'  transition_covariance: {state1["kalman_params"]["transition_covariance"]}')

    print('\n' + '=' * 80)
    print('MEASUREMENT 2: Update with prediction step (1 day later)')
    print('=' * 80)
    print(f'Weight: 70.1 kg')
    print(f'Timestamp: {timestamp2.isoformat()}')
    print(f'Time delta: 1 day')
    print(f'Source: withings')

    # Enable verbose logging
    os.environ['VERBOSE_LOGGING'] = 'true'

    result2 = process_measurement(
        user_id=user_id,
        weight=70.1,
        timestamp=timestamp2,
        source='withings',
        config=config,
        unit='kg',
        db=store,
        user_height_m=None
    )

    os.environ['VERBOSE_LOGGING'] = 'false'

    print('\nResult 2:')
    print(f'  filtered_weight: {result2.get("filtered_weight")}')
    print(f'  trend: {result2.get("trend")}')
    print(f'  kalman_variance: {result2.get("kalman_variance")}')
    print(f'  innovation: {result2.get("innovation")}')
    print(f'  normalized_innovation: {result2.get("normalized_innovation")}')
    print(f'  kalman_confidence_upper: {result2.get("kalman_confidence_upper")}')
    print(f'  kalman_confidence_lower: {result2.get("kalman_confidence_lower")}')

    # Get final state
    state2 = store.get_state(user_id)
    print('\nState after measurement 2:')
    print(f'  last_state[0]: {state2["last_state"][0]}')
    print(f'  last_state[1]: {state2["last_state"][1]}')
    print(f'  last_covariance[0]: ')
    print(f'    {state2["last_covariance"][0]}')
    print(f'  last_covariance[1]: ')
    print(f'    {state2["last_covariance"][1]}')
    print(f'  last_covariance[0][0,0]: {state2["last_covariance"][0][0,0]}')
    print(f'  last_covariance[1][0,0]: {state2["last_covariance"][1][0,0]}')

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Python kalman_variance (measurement 2): {result2.get("kalman_variance")}')
    print(f'Expected TypeScript value: 2.379729588093491')
    if result2.get('kalman_variance'):
        diff = result2.get('kalman_variance') - 2.379729588093491
        rel_err = (abs(diff) / result2.get('kalman_variance')) * 100
        print(f'Difference: {diff:.6f}')
        print(f'Relative error: {rel_err:.2f}%')

if __name__ == '__main__':
    debug_kalman_variance()
