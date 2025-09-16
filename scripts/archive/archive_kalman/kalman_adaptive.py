"""
Adaptive Kalman filter parameters for post-reset period.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import numpy as np

def get_adaptive_kalman_params(
    reset_timestamp: Optional[datetime],
    current_timestamp: datetime,
    base_config: Dict[str, Any],
    adaptive_days: int = 7,
    state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get adaptive Kalman parameters that gradually transition from 
    loose (adaptive) to tight (normal) configuration after a reset.
    
    Uses multipliers from reset parameters to scale base config values.
    """
    
    if reset_timestamp is None:
        return base_config
    
    days_since_reset = (current_timestamp - reset_timestamp).total_seconds() / 86400.0
    
    # Check if we have custom reset parameters in state
    if state and state.get('reset_parameters'):
        reset_params = state['reset_parameters']
        adaptation_days = reset_params.get('adaptation_days', adaptive_days)
        
        # Get multipliers from reset parameters
        initial_var_mult = reset_params.get('initial_variance_multiplier', 5)
        weight_noise_mult = reset_params.get('weight_noise_multiplier', 20)
        trend_noise_mult = reset_params.get('trend_noise_multiplier', 200)
        obs_noise_mult = reset_params.get('observation_noise_multiplier', 0.5)
        
        # Apply multipliers to base config
        base_initial_var = base_config.get('initial_variance', 0.361)
        base_weight_cov = base_config.get('transition_covariance_weight', 0.016)
        base_trend_cov = base_config.get('transition_covariance_trend', 0.0001)
        base_obs_cov = base_config.get('observation_covariance', 3.4)
        
        adaptive_params = {
            'initial_variance': base_initial_var * initial_var_mult,
            'transition_covariance_weight': base_weight_cov * weight_noise_mult,
            'transition_covariance_trend': base_trend_cov * trend_noise_mult,
            'observation_covariance': base_obs_cov * obs_noise_mult,
        }
    else:
        # Use default adaptive parameters (shouldn't happen with proper reset)
        adaptive_params = {
            'initial_variance': 5.0,
            'transition_covariance_weight': 0.5,
            'transition_covariance_trend': 0.01,
            'observation_covariance': 2.0,
        }
    
    # Check if we're still in adaptation period
    if days_since_reset >= adaptation_days:
        return base_config
    
    # Calculate decay factor based on days since reset
    decay_rate = reset_params.get('adaptation_decay_rate', 2.5) if state else 2.5
    measurements_since = state.get('measurements_since_reset', 0) if state else 0
    
    # Use measurement-based decay if available, otherwise time-based
    if measurements_since > 0:
        decay_factor = 1.0 - np.exp(-measurements_since / decay_rate)
    else:
        decay_factor = min(1.0, days_since_reset / adaptation_days)
    
    # Interpolate between adaptive and base parameters
    result = {}
    for key in base_config:
        if key in adaptive_params:
            adaptive_value = adaptive_params[key]
            base_value = base_config[key]
            result[key] = adaptive_value * (1 - decay_factor) + base_value * decay_factor
        else:
            result[key] = base_config[key]
    
    return result

def should_use_adaptive_params(state: Dict[str, Any], adaptive_days: int = 7) -> bool:
    """
    Check if adaptive parameters should be used based on reset history.
    """
    reset_events = state.get('reset_events', [])
    if not reset_events:
        return False
    
    last_reset = reset_events[-1]
    reset_timestamp = last_reset.get('timestamp')
    if not reset_timestamp:
        return False
    
    if isinstance(reset_timestamp, str):
        reset_timestamp = datetime.fromisoformat(reset_timestamp)
    
    current_timestamp = state.get('last_timestamp')
    if not current_timestamp:
        return False
    
    if isinstance(current_timestamp, str):
        current_timestamp = datetime.fromisoformat(current_timestamp)
    
    days_since_reset = (current_timestamp - reset_timestamp).total_seconds() / 86400.0
    
    # Use adaptation_days from reset parameters if available
    reset_params = state.get('reset_parameters', {})
    adaptation_days = reset_params.get('adaptation_days', adaptive_days)
    
    return days_since_reset < adaptation_days

def get_reset_timestamp(state: Dict[str, Any]) -> Optional[datetime]:
    """
    Get the timestamp of the most recent reset event.
    """
    reset_events = state.get('reset_events', [])
    if not reset_events:
        return None
    
    last_reset = reset_events[-1]
    reset_timestamp = last_reset.get('timestamp')
    
    if reset_timestamp and isinstance(reset_timestamp, str):
        reset_timestamp = datetime.fromisoformat(reset_timestamp)
    
    return reset_timestamp
