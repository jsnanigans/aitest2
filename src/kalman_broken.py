"""
Kalman filter logic for weight processing with adaptive parameters and reset management.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from enum import Enum
import numpy as np
from pykalman import KalmanFilter

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()


class ResetType(Enum):
    """Types of resets with different adaptation strategies."""
    HARD = "hard"
    INITIAL = "initial"
    SOFT = "soft"


MANUAL_DATA_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'questionnaire',
    'patient-upload',
    'user-upload',
    'care-team-upload',
    'care-team-entry'
}


class KalmanFilterManager:
    """Manages Kalman filter state and operations."""
    
    @staticmethod
    def initialize_filter(config: Dict[str, Any]) -> KalmanFilter:
        """Initialize a new Kalman filter with given configuration."""
        
        initial_variance = config.get('initial_variance', 0.361)
        transition_covariance_weight = config.get('transition_covariance_weight', 0.016)
        transition_covariance_trend = config.get('transition_covariance_trend', 0.0001)
        observation_covariance = config.get('observation_covariance', 3.4)
        
        transition_matrix = np.array([[1, 1], [0, 1]])
        observation_matrix = np.array([[1, 0]])
        
        transition_covariance = np.array([
            [transition_covariance_weight, 0],
            [0, transition_covariance_trend]
        ])
        
        observation_covariance_matrix = np.array([[observation_covariance]])
        
        initial_state_covariance = np.array([
            [initial_variance, 0],
            [0, initial_variance]
        ])
        
        return KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance_matrix,
            initial_state_covariance=initial_state_covariance
        )
    
    @staticmethod
    def apply_kalman_filter(
        weight: float,
        timestamp: datetime,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply Kalman filter to a weight measurement.
        
        Returns:
            Tuple of (filtered_weight, updated_state)
        """
        
        if config is None:
            config = {
                'initial_variance': 0.361,
                'transition_covariance_weight': 0.016,
                'transition_covariance_trend': 0.0001,
                'observation_covariance': 3.4
            }
        
        if state is None or not state.get('kalman_params'):
            kf = KalmanFilterManager.initialize_filter(config)
            
            initial_state_mean = np.array([weight, 0])
            
            filtered_state_means, filtered_state_covariances = kf.filter(
                np.array([[weight]])
            )
            
            filtered_weight = float(filtered_state_means[-1, 0])
            
            new_state = {
                'kalman_params': {
                    'state_mean': filtered_state_means[-1].tolist(),
                    'state_covariance': filtered_state_covariances[-1].tolist()
                },
                'last_timestamp': timestamp,
                'variance': float(filtered_state_covariances[-1, 0, 0]),
                'trend': float(filtered_state_means[-1, 1])
            }
            
            return filtered_weight, new_state
        
        kalman_params = state['kalman_params']
        prev_state_mean = np.array(kalman_params['state_mean'])
        prev_state_covariance = np.array(kalman_params['state_covariance'])
        
        last_timestamp = state.get('last_timestamp')
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        time_gap = (timestamp - last_timestamp).total_seconds() / 86400.0
        
        if time_gap > 30:
            logger.info(f"Large time gap detected: {time_gap:.1f} days")
            config = KalmanFilterManager.adjust_for_gap(config, time_gap)
        
        kf = KalmanFilterManager.initialize_filter(config)
        
        kf.initial_state_mean = prev_state_mean
        kf.initial_state_covariance = prev_state_covariance
        
        filtered_state_means, filtered_state_covariances = kf.filter(
            np.array([[weight]])
        )
        
        filtered_weight = float(filtered_state_means[-1, 0])
        
        new_state = {
            'kalman_params': {
                'state_mean': filtered_state_means[-1].tolist(),
                'state_covariance': filtered_state_covariances[-1].tolist()
            },
            'last_timestamp': timestamp,
            'variance': float(filtered_state_covariances[-1, 0, 0]),
            'trend': float(filtered_state_means[-1, 1])
        }
        
        return filtered_weight, new_state
    
    @staticmethod
    def adjust_for_gap(config: Dict[str, Any], gap_days: float) -> Dict[str, Any]:
        """
        Adjust Kalman parameters for large time gaps.
        
        Args:
            config: Base configuration
            gap_days: Number of days in the gap
        
        Returns:
            Adjusted configuration
        """
        adjusted_config = config.copy()
        
        if gap_days > 7:
            gap_factor = min(gap_days / 7, 10)
            
            adjusted_config['transition_covariance_weight'] = (
                config.get('transition_covariance_weight', 0.016) * gap_factor
            )
            adjusted_config['transition_covariance_trend'] = (
                config.get('transition_covariance_trend', 0.0001) * gap_factor
            )
            
            logger.debug(f"Adjusted Kalman params for {gap_days:.1f} day gap (factor: {gap_factor:.2f})")
        
        return adjusted_config
    
    @staticmethod
    def reset_kalman_state(
        weight: float,
        timestamp: datetime,
        config: Dict[str, Any],
        reset_type: Optional[ResetType] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reset Kalman filter state with new initial conditions.
        
        Args:
            weight: Current weight measurement
            timestamp: Current timestamp
            config: Kalman configuration
            reset_type: Type of reset (hard, soft, initial)
            reason: Reason for reset
        
        Returns:
            New Kalman state after reset
        """
        
        kf = KalmanFilterManager.initialize_filter(config)
        
        initial_state_mean = np.array([weight, 0])
        
        filtered_state_means, filtered_state_covariances = kf.filter(
            np.array([[weight]])
        )
        
        reset_info = {
            'reset_occurred': True,
            'reset_type': reset_type.value if reset_type else 'unknown',
            'reason': reason or 'Manual reset',
            'timestamp': timestamp,
            'weight_at_reset': weight
        }
        
        new_state = {
            'kalman_params': {
                'state_mean': filtered_state_means[-1].tolist(),
                'state_covariance': filtered_state_covariances[-1].tolist()
            },
            'last_timestamp': timestamp,
            'variance': float(filtered_state_covariances[-1, 0, 0]),
            'trend': float(filtered_state_means[-1, 1]),
            'reset_info': reset_info
        }
        
        return new_state


class ResetManager:
    """Manages different types of resets with configurable parameters."""
    
    @staticmethod
    def should_trigger_reset(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        config: Dict[str, Any]
    ) -> Optional[ResetType]:
        """
        Determine if and what type of reset to trigger.
        
        Priority order:
        1. Initial (no Kalman params)
        2. Hard (30+ day gap)
        3. Soft (manual data with significant change)
        """
        
        if not state or not state.get('kalman_params'):
            return ResetType.INITIAL
        
        last_timestamp = state.get('last_timestamp')
        if last_timestamp:
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            
            gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
            
            hard_reset_threshold = config.get('hard_reset_gap_days', 30)
            if gap_days >= hard_reset_threshold:
                return ResetType.HARD
        
        if source in MANUAL_DATA_SOURCES:
            kalman_params = state.get('kalman_params', {})
            if kalman_params:
                state_mean = kalman_params.get('state_mean', [weight, 0])
                expected_weight = state_mean[0]
                
                soft_reset_threshold = config.get('soft_reset_threshold_kg', 5.0)
                if abs(weight - expected_weight) > soft_reset_threshold:
                    return ResetType.SOFT
        
        return None
    
    @staticmethod
    def get_reset_parameters(reset_type: ResetType, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get reset-specific parameters based on reset type.
        
        Returns dict with:
        - initial_variance_multiplier
        - weight_noise_multiplier
        - trend_noise_multiplier
        - observation_noise_multiplier
        - adaptation_days
        - adaptation_decay_rate
        """
        
        reset_config = config.get('reset_parameters', {})
        type_configs = reset_config.get('by_type', {})
        
        if reset_type == ResetType.INITIAL:
            params = type_configs.get('initial', {})
            defaults = {
                'initial_variance_multiplier': 10,
                'weight_noise_multiplier': 50,
                'trend_noise_multiplier': 500,
                'observation_noise_multiplier': 0.3,
                'adaptation_days': 10,
                'adaptation_decay_rate': 2.0
            }
        elif reset_type == ResetType.HARD:
            params = type_configs.get('hard', {})
            defaults = {
                'initial_variance_multiplier': 5,
                'weight_noise_multiplier': 20,
                'trend_noise_multiplier': 200,
                'observation_noise_multiplier': 0.5,
                'adaptation_days': 7,
                'adaptation_decay_rate': 2.5
            }
        elif reset_type == ResetType.SOFT:
            params = type_configs.get('soft', {})
            defaults = {
                'initial_variance_multiplier': 2,
                'weight_noise_multiplier': 5,
                'trend_noise_multiplier': 10,
                'observation_noise_multiplier': 0.8,
                'adaptation_days': 3,
                'adaptation_decay_rate': 3.0
            }
        else:
            return {}
        
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
        
        return params
    
    @staticmethod
    def execute_reset(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        reset_type: ResetType,
        config: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """
        Execute a reset with appropriate parameters.
        
        Returns:
            Updated state with reset information
        """
        
        reset_params = ResetManager.get_reset_parameters(reset_type, config)
        
        base_config = config.get('kalman', {})
        adaptive_config = ResetManager.apply_reset_multipliers(base_config, reset_params)
        
        new_kalman_state = KalmanFilterManager.reset_kalman_state(
            weight=weight,
            timestamp=timestamp,
            config=adaptive_config,
            reset_type=reset_type,
            reason=reason
        )
        
        reset_event = {
            'timestamp': timestamp,
            'type': reset_type.value,
            'reason': reason,
            'weight_at_reset': weight,
            'parameters': reset_params
        }
        
        reset_events = state.get('reset_events', [])
        reset_events.append(reset_event)
        
        updated_state = {
            **state,
            **new_kalman_state,
            'reset_events': reset_events,
            'reset_parameters': reset_params,
            'reset_timestamp': timestamp,
            'measurements_since_reset': 0
        }
        
        return updated_state
    
    @staticmethod
    def apply_reset_multipliers(
        base_config: Dict[str, Any],
        reset_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply reset multipliers to base Kalman configuration.
        """
        
        return {
            'initial_variance': base_config.get('initial_variance', 0.361) * 
                              reset_params.get('initial_variance_multiplier', 1),
            'transition_covariance_weight': base_config.get('transition_covariance_weight', 0.016) * 
                                           reset_params.get('weight_noise_multiplier', 1),
            'transition_covariance_trend': base_config.get('transition_covariance_trend', 0.0001) * 
                                         reset_params.get('trend_noise_multiplier', 1),
            'observation_covariance': base_config.get('observation_covariance', 3.4) * 
                                    reset_params.get('observation_noise_multiplier', 1)
        }


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
    """
    
    if reset_timestamp is None:
        return base_config
    
    days_since_reset = (current_timestamp - reset_timestamp).total_seconds() / 86400.0
    
    if state and state.get('reset_parameters'):
        reset_params = state['reset_parameters']
        adaptation_days = reset_params.get('adaptation_days', adaptive_days)
        
        initial_var_mult = reset_params.get('initial_variance_multiplier', 5)
        weight_noise_mult = reset_params.get('weight_noise_multiplier', 20)
        trend_noise_mult = reset_params.get('trend_noise_multiplier', 200)
        obs_noise_mult = reset_params.get('observation_noise_multiplier', 0.5)
        
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
        adaptive_params = {
            'initial_variance': 5.0,
            'transition_covariance_weight': 0.5,
            'transition_covariance_trend': 0.01,
            'observation_covariance': 2.0,
        }
    
    if days_since_reset >= adaptation_days:
        return base_config
    
    decay_rate = reset_params.get('adaptation_decay_rate', 2.5) if state else 2.5
    measurements_since = state.get('measurements_since_reset', 0) if state else 0
    
    if measurements_since > 0:
        decay_factor = 1.0 - np.exp(-measurements_since / decay_rate)
    else:
        decay_factor = min(1.0, days_since_reset / adaptation_days)
    
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
    @staticmethod
    def perform_reset(
        state: Dict[str, Any],
        reset_type: ResetType,
        timestamp: datetime,
        weight: float,
        source: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform reset with appropriate parameters (compatibility wrapper).
        
        Returns:
            Tuple of (new_state, reset_event)
        """
        
        # Determine reset reason
        if reset_type == ResetType.INITIAL:
            reason = "Initial measurement"
        elif reset_type == ResetType.HARD:
            last_timestamp = state.get('last_timestamp')
            if last_timestamp:
                if isinstance(last_timestamp, str):
                    last_timestamp = datetime.fromisoformat(last_timestamp)
                gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
                reason = f"Large gap ({gap_days:.0f} days)"
            else:
                reason = "Large gap"
        elif reset_type == ResetType.SOFT:
            reason = f"Manual data from {source}"
        else:
            reason = "Unknown"
        
        # Execute the reset
        new_state = ResetManager.execute_reset(
            state=state,
            weight=weight,
            timestamp=timestamp,
            reset_type=reset_type,
            config=config,
            reason=reason
        )
        
        # Extract reset event for compatibility
        reset_events = new_state.get('reset_events', [])
        if reset_events:
            reset_event = reset_events[-1]
        else:
            reset_event = {
                'timestamp': timestamp,
                'type': reset_type.value,
                'reason': reason,
                'weight_at_reset': weight
            }
        
        return new_state, reset_event
