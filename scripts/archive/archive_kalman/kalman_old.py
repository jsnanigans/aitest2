"""
Kalman filter logic for weight processing.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import numpy as np
from pykalman import KalmanFilter

try:
    from .constants import KALMAN_DEFAULTS
except ImportError:
    from constants import KALMAN_DEFAULTS


class KalmanFilterManager:
    """Manages Kalman filter operations for weight processing."""

    @staticmethod
    def initialize_immediate(
        weight: float,
        timestamp: datetime,
        kalman_config: dict,
        observation_covariance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Initialize Kalman filter immediately with first measurement."""
        initial_variance = kalman_config.get("initial_variance", KALMAN_DEFAULTS['initial_variance'])

        # Use passed observation_covariance if provided, otherwise use config value
        obs_cov = observation_covariance if observation_covariance is not None else kalman_config.get("observation_covariance", KALMAN_DEFAULTS['observation_covariance'])

        kalman_params = {
            'initial_state_mean': [weight, 0],
            'initial_state_covariance': [[initial_variance, 0], [0, 0.001]],
            'transition_covariance': [
                [kalman_config.get("transition_covariance_weight", KALMAN_DEFAULTS['transition_covariance_weight']), 0],
                [0, kalman_config.get("transition_covariance_trend", KALMAN_DEFAULTS['transition_covariance_trend'])]
            ],
            'observation_covariance': [[obs_cov]],
        }

        return {
            'kalman_params': kalman_params,
            'last_state': np.array([[weight, 0]]),
            'last_covariance': np.array([[[initial_variance, 0], [0, 0.001]]]),
            'last_timestamp': timestamp,
            'last_raw_weight': weight,
        }

    @staticmethod
    def update_state(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: dict,
        observation_covariance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update Kalman filter state with new measurement."""
        time_delta_days = 1.0
        if state.get('last_timestamp'):
            last_timestamp = state['last_timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            delta = (timestamp - last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))

        kalman_params = state['kalman_params']

        # Use passed observation_covariance if provided, otherwise use stored value
        obs_cov = observation_covariance if observation_covariance is not None else kalman_params['observation_covariance'][0][0]

        kalman = KalmanFilter(
            transition_matrices=np.array([[1, time_delta_days], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=np.array(kalman_params['initial_state_mean']),
            initial_state_covariance=np.array(kalman_params['initial_state_covariance']),
            transition_covariance=np.array(kalman_params['transition_covariance']),
            observation_covariance=np.array([[obs_cov]]),
        )

        observation = np.array([[weight]])

        if state.get('last_state') is None:
            filtered_state_means, filtered_state_covariances = kalman.filter(observation)
            new_last_state = filtered_state_means
            new_last_covariance = filtered_state_covariances
        else:
            last_state = state['last_state']
            last_covariance = state['last_covariance']

            if len(last_state.shape) > 1:
                current_state = last_state[-1]
                current_covariance = last_covariance[-1]
            else:
                current_state = last_state
                current_covariance = last_covariance[0] if len(last_covariance.shape) > 2 else last_covariance

            filtered_state_mean, filtered_state_covariance = kalman.filter_update(
                current_state,
                current_covariance,
                observation=observation[0],
            )

            if len(last_state.shape) == 1:
                new_last_state = np.array([last_state, filtered_state_mean])
                new_last_covariance = np.array([last_covariance, filtered_state_covariance])
            else:
                new_last_state = np.array([last_state[-1], filtered_state_mean])
                new_last_covariance = np.array([last_covariance[-1], filtered_state_covariance])

        state['last_state'] = new_last_state
        state['last_covariance'] = new_last_covariance
        state['last_timestamp'] = timestamp
        state['last_raw_weight'] = weight

        return state

    @staticmethod
    def calculate_confidence(normalized_innovation: float) -> float:
        """Calculate confidence using smooth exponential decay."""
        alpha = 0.5
        return np.exp(-alpha * normalized_innovation ** 2)

    @staticmethod
    def create_result(
        state: Dict[str, Any],
        weight: float,
        timestamp: datetime,
        source: str,
        accepted: bool,
        observation_covariance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Create result dictionary from current state."""
        if state.get('last_state') is None:
            return None

        last_state = state['last_state']
        if len(last_state.shape) > 1:
            current_state = last_state[-1]
        else:
            current_state = last_state

        filtered_weight = current_state[0]
        trend = current_state[1]

        innovation = weight - filtered_weight

        last_covariance = state.get('last_covariance')
        current_covariance = None
        if last_covariance is not None:
            if len(last_covariance.shape) > 2:
                current_covariance = last_covariance[-1]
            else:
                current_covariance = last_covariance

            # Use passed observation_covariance if provided, otherwise use stored value
            if observation_covariance is not None:
                obs_covariance = observation_covariance
            else:
                obs_covariance = state['kalman_params']['observation_covariance'][0][0]
            innovation_variance = current_covariance[0, 0] + obs_covariance
            normalized_innovation = (
                abs(innovation) / np.sqrt(innovation_variance)
                if innovation_variance > 0
                else 0
            )
        else:
            normalized_innovation = 0

        confidence = KalmanFilterManager.calculate_confidence(normalized_innovation)

        # Calculate confidence intervals (±2σ)
        if current_covariance is not None:
            confidence_interval = 2.0 * np.sqrt(current_covariance[0, 0])
            kalman_upper = filtered_weight + confidence_interval
            kalman_lower = filtered_weight - confidence_interval
            kalman_variance = float(current_covariance[0, 0])
        else:
            kalman_upper = filtered_weight
            kalman_lower = filtered_weight
            kalman_variance = None

        # Calculate prediction error
        prediction_error = innovation if accepted else None

        return {
            "timestamp": timestamp,
            "raw_weight": weight,
            "filtered_weight": float(filtered_weight),
            "trend": float(trend),
            "trend_weekly": float(trend * 7),
            "accepted": accepted,
            "confidence": float(confidence),
            "innovation": float(innovation),
            "normalized_innovation": float(normalized_innovation),
            "source": source,
            "kalman_confidence_upper": float(kalman_upper),
            "kalman_confidence_lower": float(kalman_lower),
            "kalman_variance": kalman_variance,
            "prediction_error": float(prediction_error) if prediction_error is not None else None,
        }

    @staticmethod
    def get_current_state_values(state: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Extract current weight and trend from state."""
        if state.get('last_state') is None:
            return None, None

        last_state = state['last_state']
        if len(last_state.shape) > 1:
            current_state = last_state[-1]
        else:
            current_state = last_state

        return float(current_state[0]), float(current_state[1])

    @staticmethod
    def calculate_time_delta_days(
        current_timestamp: datetime,
        last_timestamp: Optional[datetime]
    ) -> float:
        """Calculate days between measurements."""
        if last_timestamp is None:
            return 1.0

        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)

        delta = (current_timestamp - last_timestamp).total_seconds() / 86400.0
        return max(0.1, delta)
    

    @staticmethod
    def get_adaptive_covariances(measurements_since_reset: int, config: dict) -> Dict[str, float]:
        """
        Get adaptive covariances that start loose after reset and tighten over time.
        This helps the filter adapt quickly to new weight patterns after a gap.
        
        Args:
            measurements_since_reset: Number of measurements since last reset
            config: Kalman configuration dictionary
            
        Returns:
            Dictionary with 'weight' and 'trend' covariance values
        """
        base_weight_cov = config.get('transition_covariance_weight', KALMAN_DEFAULTS['transition_covariance_weight'])
        base_trend_cov = config.get('transition_covariance_trend', KALMAN_DEFAULTS['transition_covariance_trend'])
        
        # Get adaptive settings from config
        adaptive_config = config.get('post_reset_adaptation', {})
        if not adaptive_config.get('enabled', True):
            return {
                'weight': base_weight_cov,
                'trend': base_trend_cov
            }
        
        warmup_measurements = adaptive_config.get('warmup_measurements', 10)
        weight_boost_factor = adaptive_config.get('weight_boost_factor', 10)
        trend_boost_factor = adaptive_config.get('trend_boost_factor', 100)
        decay_rate = adaptive_config.get('decay_rate', 3)
        
        if measurements_since_reset < warmup_measurements:
            # Exponentially decay from boost to 1x over warmup period
            factor = np.exp(-measurements_since_reset / decay_rate)
            
            weight_multiplier = 1 + (weight_boost_factor - 1) * factor
            trend_multiplier = 1 + (trend_boost_factor - 1) * factor
            
            return {
                'weight': base_weight_cov * weight_multiplier,
                'trend': base_trend_cov * trend_multiplier
            }
        else:
            # After warmup, use normal values
            return {
                'weight': base_weight_cov,
                'trend': base_trend_cov
            }
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
"""
Reset Manager for parameterized reset handling.
Supports hard, initial, and soft resets with different adaptation parameters.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import numpy as np

class ResetType(Enum):
    """Types of resets with different adaptation strategies."""
    HARD = "hard"      # 30+ day gaps - aggressive adaptation
    INITIAL = "initial" # First measurement - most aggressive adaptation
    SOFT = "soft"      # Manual data entry - gentle adaptation

MANUAL_DATA_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'questionnaire',
    'patient-upload',
    'user-upload',
    'care-team-upload',
    'care-team-entry'
}

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
        
        # 1. Check for initial reset (no Kalman params yet)
        if not state or not state.get('kalman_params'):
            return ResetType.INITIAL
        
        # 2. Check for hard reset (30+ day gap)
        hard_config = config.get('kalman', {}).get('reset', {}).get('hard', {})
        if hard_config.get('enabled', True):
            last_timestamp = state.get('last_accepted_timestamp') or state.get('last_timestamp')
            if last_timestamp:
                if isinstance(last_timestamp, str):
                    last_timestamp = datetime.fromisoformat(last_timestamp)
                gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
                threshold = hard_config.get('gap_threshold_days', 30)
                if gap_days >= threshold:
                    return ResetType.HARD
        
        # 3. Check for soft reset (manual data with significant change)
        soft_config = config.get('kalman', {}).get('reset', {}).get('soft', {})
        if soft_config.get('enabled', True):
            if source in MANUAL_DATA_SOURCES or source in soft_config.get('trigger_sources', []):
                last_weight = state.get('last_raw_weight') or state.get('last_accepted_weight')
                if last_weight:
                    weight_change = abs(weight - last_weight)
                    min_change = soft_config.get('min_weight_change_kg', 5)
                    
                    if weight_change >= min_change:
                        cooldown_days = soft_config.get('cooldown_days', 3)
                        last_reset = ResetManager.get_last_reset_timestamp(state)
                        if not last_reset or (timestamp - last_reset).total_seconds() / 86400.0 > cooldown_days:
                            return ResetType.SOFT
        
        return None
    
    @staticmethod
    def get_last_reset_timestamp(state: Dict[str, Any]) -> Optional[datetime]:
        """Get timestamp of the most recent reset."""
        reset_events = state.get('reset_events', [])
        if reset_events:
            last_reset = reset_events[-1]
            timestamp = last_reset.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                return timestamp
        
        reset_timestamp = state.get('reset_timestamp')
        if reset_timestamp:
            if isinstance(reset_timestamp, str):
                reset_timestamp = datetime.fromisoformat(reset_timestamp)
            return reset_timestamp
        
        return None
    
    @staticmethod
    def get_reset_parameters(reset_type: ResetType, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get parameters for specific reset type from config.
        
        Returns dict with adaptation parameters using new naming convention.
        """
        
        # Default parameters for each type (using new names)
        defaults = {
            ResetType.INITIAL: {
                # Multipliers for Kalman parameters
                'initial_variance_multiplier': 10,
                'weight_noise_multiplier': 50,
                'trend_noise_multiplier': 500,
                'observation_noise_multiplier': 0.3,
                # Adaptation duration
                'adaptation_measurements': 20,
                'adaptation_days': 21,
                'adaptation_decay_rate': 1.5,
                # Quality scoring
                'quality_acceptance_threshold': 0.25,
                'quality_safety_weight': 0.50,
                'quality_plausibility_weight': 0.05,
                'quality_consistency_weight': 0.05,
                'quality_reliability_weight': 0.40
            },
            ResetType.HARD: {
                # Multipliers for Kalman parameters
                'initial_variance_multiplier': 5,
                'weight_noise_multiplier': 20,
                'trend_noise_multiplier': 200,
                'observation_noise_multiplier': 0.5,
                # Adaptation duration
                'adaptation_measurements': 10,
                'adaptation_days': 7,
                'adaptation_decay_rate': 2.5,
                # Quality scoring
                'quality_acceptance_threshold': 0.35,
                'quality_safety_weight': 0.45,
                'quality_plausibility_weight': 0.10,
                'quality_consistency_weight': 0.10,
                'quality_reliability_weight': 0.35
            },
            ResetType.SOFT: {
                # Multipliers for Kalman parameters
                'initial_variance_multiplier': 2,
                'weight_noise_multiplier': 5,
                'trend_noise_multiplier': 20,
                'observation_noise_multiplier': 0.7,
                # Adaptation duration
                'adaptation_measurements': 15,
                'adaptation_days': 10,
                'adaptation_decay_rate': 4,
                # Quality scoring
                'quality_acceptance_threshold': 0.45,
                'quality_safety_weight': 0.40,
                'quality_plausibility_weight': 0.15,
                'quality_consistency_weight': 0.15,
                'quality_reliability_weight': 0.30
            }
        }
        
        # Get from config or use defaults
        reset_config = config.get('kalman', {}).get('reset', {}).get(reset_type.value, {})
        default_params = defaults[reset_type]
        
        # Merge config with defaults
        params = {}
        for key, default_value in default_params.items():
            params[key] = reset_config.get(key, default_value)
        
        return params
    
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
        Perform reset with appropriate parameters.
        
        Returns:
            Tuple of (new_state, reset_event)
        """
        
        # Get parameters for this reset type
        reset_params = ResetManager.get_reset_parameters(reset_type, config)
        
        # Calculate gap if applicable
        gap_days = None
        last_timestamp = state.get('last_accepted_timestamp') or state.get('last_timestamp')
        if last_timestamp:
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
        
        # Create reset event
        reset_event = {
            'timestamp': timestamp,
            'type': reset_type.value,
            'source': source,
            'weight': weight,
            'last_weight': state.get('last_raw_weight'),
            'gap_days': gap_days,
            'reason': ResetManager.get_reset_reason(reset_type, gap_days, weight, state),
            'parameters': reset_params
        }
        
        # Create new state with reset
        new_state = {
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'measurements_since_reset': 0,
            'reset_type': reset_type.value,
            'reset_parameters': reset_params,
            'reset_timestamp': timestamp,
            'reset_events': state.get('reset_events', []) + [reset_event],
            'last_timestamp': state.get('last_timestamp'),
            'last_source': state.get('last_source'),
            'last_raw_weight': state.get('last_raw_weight'),
            'last_accepted_timestamp': state.get('last_accepted_timestamp'),
            'measurement_history': []
        }
        
        return new_state, reset_event
    
    @staticmethod
    def get_reset_reason(
        reset_type: ResetType,
        gap_days: Optional[float],
        weight: float,
        state: Dict[str, Any]
    ) -> str:
        """Generate human-readable reset reason."""
        
        if reset_type == ResetType.INITIAL:
            return "initial_measurement"
        elif reset_type == ResetType.HARD:
            return f"gap_exceeded_{gap_days:.0f}_days" if gap_days else "gap_exceeded"
        elif reset_type == ResetType.SOFT:
            last_weight = state.get('last_raw_weight')
            if last_weight:
                change = abs(weight - last_weight)
                return f"manual_entry_change_{change:.1f}kg"
            return "manual_entry"
        else:
            return "unknown"
    
    @staticmethod
    def is_in_adaptive_period(
        state: Dict[str, Any],
        timestamp: datetime
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if currently in adaptive period and return parameters if so.
        
        Returns:
            Tuple of (is_adaptive, parameters_dict)
        """
        
        reset_timestamp = state.get('reset_timestamp')
        if not reset_timestamp:
            return False, None
        
        if isinstance(reset_timestamp, str):
            reset_timestamp = datetime.fromisoformat(reset_timestamp)
        
        reset_params = state.get('reset_parameters', {})
        if not reset_params:
            return False, None
        
        # Check measurements-based
        measurements_since = state.get('measurements_since_reset', 0)
        adaptation_measurements = reset_params.get('adaptation_measurements', 10)
        
        # Check time-based
        days_since = (timestamp - reset_timestamp).total_seconds() / 86400.0
        adaptation_days = reset_params.get('adaptation_days', 7)
        
        # In adaptive period if either condition is met
        if measurements_since < adaptation_measurements or days_since < adaptation_days:
            return True, reset_params
        
        return False, None
    
    @staticmethod
    def get_adaptive_factor(
        state: Dict[str, Any],
        timestamp: datetime
    ) -> float:
        """
        Calculate adaptive factor (0-1) based on time/measurements since reset.
        0 = fully adaptive, 1 = normal operation
        """
        
        reset_timestamp = state.get('reset_timestamp')
        if not reset_timestamp:
            return 1.0
        
        if isinstance(reset_timestamp, str):
            reset_timestamp = datetime.fromisoformat(reset_timestamp)
        
        reset_params = state.get('reset_parameters', {})
        decay_rate = reset_params.get('adaptation_decay_rate', 3)
        
        # Use measurements-based decay
        measurements_since = state.get('measurements_since_reset', 0)
        factor = 1.0 - np.exp(-measurements_since / decay_rate)
        
        return min(1.0, max(0.0, factor))
