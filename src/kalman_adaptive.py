"""
Adaptive Kalman filter parameters for post-reset period.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional, Any

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
    
    Args:
        reset_timestamp: When the reset occurred
        current_timestamp: Current measurement timestamp
        base_config: Base Kalman configuration
        adaptive_days: Days over which to transition (default 7)
        state: Optional state dict containing reset parameters
    
    Returns:
        Dictionary with adaptive Kalman parameters
    """
    
    if reset_timestamp is None:
        return base_config
    
    days_since_reset = (current_timestamp - reset_timestamp).total_seconds() / 86400.0
    
    # Check if we have custom reset parameters in state
    if state and state.get('reset_parameters'):
        reset_params = state['reset_parameters']
        adaptive_days = reset_params.get('adaptive_days', adaptive_days)
        
        # Use parameters from reset
        weight_boost = reset_params.get('weight_boost_factor', 10)
        trend_boost = reset_params.get('trend_boost_factor', 100)
        
        # Calculate adaptive values based on reset type
        base_weight_cov = base_config.get('transition_covariance_weight', 0.016)
        base_trend_cov = base_config.get('transition_covariance_trend', 0.0001)
        
        adaptive_params = {
            'initial_variance': 5.0,
            'transition_covariance_weight': base_weight_cov * weight_boost,
            'transition_covariance_trend': base_trend_cov * trend_boost,
            'observation_covariance': 2.0,
        }
    else:
        # Use default adaptive parameters
        adaptive_params = {
            'initial_variance': 5.0,
            'transition_covariance_weight': 0.5,
            'transition_covariance_trend': 0.01,
            'observation_covariance': 2.0,
        }
    
    if days_since_reset >= adaptive_days:
        return base_config
    
    decay_factor = min(1.0, days_since_reset / adaptive_days)
    
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
    
    Args:
        state: Current processor state
        adaptive_days: Days to use adaptive params after reset
    
    Returns:
        True if within adaptive period after reset
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
    
    return days_since_reset < adaptive_days


def get_reset_timestamp(state: Dict[str, Any]) -> Optional[datetime]:
    """
    Get the timestamp of the most recent reset event.
    
    Args:
        state: Current processor state
    
    Returns:
        Timestamp of last reset, or None if no resets
    """
    reset_events = state.get('reset_events', [])
    if not reset_events:
        return None
    
    last_reset = reset_events[-1]
    reset_timestamp = last_reset.get('timestamp')
    
    if reset_timestamp and isinstance(reset_timestamp, str):
        reset_timestamp = datetime.fromisoformat(reset_timestamp)
    
    return reset_timestamp