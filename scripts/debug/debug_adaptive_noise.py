"""
Debug why adaptive noise isn't having any effect.
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.processor import WeightProcessor, process_weight_enhanced
from src.database import get_state_db
from src.models import KALMAN_DEFAULTS, SOURCE_PROFILES, PROCESSING_DEFAULTS

def test_single_measurement():
    """Test a single measurement to see how adaptive noise is applied."""
    
    # Clear state
    db = get_state_db()
    user_id = "debug_user"
    db.clear_state(user_id)
    
    timestamp = datetime(2024, 1, 1, 10, 0)
    weight = 75.0
    
    print("=" * 80)
    print("TESTING ADAPTIVE NOISE APPLICATION")
    print("=" * 80)
    
    # Test with different sources
    sources = ['care-team-upload', 'patient-device', 'https://api.iglucose.com']
    
    for source in sources:
        print(f"\n{source}:")
        profile = SOURCE_PROFILES.get(source, {'noise_multiplier': 1.0})
        print(f"  Expected noise multiplier: {profile['noise_multiplier']}")
        
        # Process with enhanced (adaptive)
        db.clear_state(user_id)
        result_adaptive = process_weight_enhanced(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=PROCESSING_DEFAULTS.copy(),
            kalman_config=KALMAN_DEFAULTS.copy()
        )
        
        # Process with fixed
        db.clear_state(user_id)
        result_fixed = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=PROCESSING_DEFAULTS.copy(),
            kalman_config=KALMAN_DEFAULTS.copy()
        )
        
        print(f"\n  Adaptive result:")
        if result_adaptive:
            print(f"    Accepted: {result_adaptive.get('accepted')}")
            print(f"    Confidence: {result_adaptive.get('confidence', 0):.3f}")
            print(f"    Innovation: {result_adaptive.get('innovation', 0):.3f}")
            print(f"    Noise used: {result_adaptive.get('measurement_noise_used', 'N/A')}")
            if 'adapted_params' in result_adaptive:
                print(f"    Adapted params: {result_adaptive['adapted_params']}")
        
        print(f"\n  Fixed result:")
        if result_fixed:
            print(f"    Accepted: {result_fixed.get('accepted')}")
            print(f"    Confidence: {result_fixed.get('confidence', 0):.3f}")
            print(f"    Innovation: {result_fixed.get('innovation', 0):.3f}")
        
        # Check the state to see Kalman parameters
        state = db.get_state(user_id)
        if state and 'kalman_params' in state:
            kalman_params = state['kalman_params']
            obs_cov = kalman_params.get('observation_covariance', [[0]])
            print(f"\n  State observation covariance: {obs_cov}")

def check_kalman_update():
    """Check if Kalman filter actually uses the adapted noise."""
    print("\n" + "=" * 80)
    print("CHECKING KALMAN FILTER UPDATE")
    print("=" * 80)
    
    from src.kalman import KalmanFilterManager
    import numpy as np
    
    # Create a simple state
    state = {
        'kalman_params': {
            'initial_state_mean': [75, 0],
            'initial_state_covariance': [[1, 0], [0, 0.001]],
            'transition_covariance': [[0.1, 0], [0, 0.001]],
            'observation_covariance': [[1.0]]  # Base noise
        },
        'last_state': np.array([[75, 0]]),
        'last_covariance': np.array([[[1, 0], [0, 0.001]]]),
        'last_timestamp': datetime(2024, 1, 1),
        'last_raw_weight': 75
    }
    
    # Update with same weight but different observation noise
    weight = 76.0
    timestamp = datetime(2024, 1, 2)
    
    # Test with base noise
    state1 = state.copy()
    state1['kalman_params'] = state['kalman_params'].copy()
    result1 = KalmanFilterManager.update_state(state1, weight, timestamp, 'test', {})
    res1 = KalmanFilterManager.create_result(result1, weight, timestamp, 'test', True)
    
    # Test with 3x noise
    state2 = state.copy()
    state2['kalman_params'] = state['kalman_params'].copy()
    state2['kalman_params']['observation_covariance'] = [[3.0]]
    result2 = KalmanFilterManager.update_state(state2, weight, timestamp, 'test', {})
    res2 = KalmanFilterManager.create_result(result2, weight, timestamp, 'test', True)
    
    print("\nWith base noise (1.0):")
    print(f"  Confidence: {res1['confidence']:.3f}")
    print(f"  Innovation: {res1['innovation']:.3f}")
    print(f"  Normalized innovation: {res1['normalized_innovation']:.3f}")
    
    print("\nWith 3x noise (3.0):")
    print(f"  Confidence: {res2['confidence']:.3f}")
    print(f"  Innovation: {res2['innovation']:.3f}")
    print(f"  Normalized innovation: {res2['normalized_innovation']:.3f}")
    
    if abs(res1['confidence'] - res2['confidence']) < 0.001:
        print("\n✗ PROBLEM: Confidence is the same despite different noise!")
    else:
        print(f"\n✓ Good: Confidence changed from {res1['confidence']:.3f} to {res2['confidence']:.3f}")

if __name__ == "__main__":
    test_single_measurement()
    check_kalman_update()
