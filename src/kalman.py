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
    def estimate_trend_from_buffer(measurements: list, pre_gap_trend: Optional[float] = None) -> float:
        """Estimate trend from buffered measurements."""
        if len(measurements) < 2:
            return pre_gap_trend if pre_gap_trend is not None else 0.0
        
        weights = [m['weight'] for m in measurements]
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in measurements]
        
        if len(weights) == 2:
            time_diff_days = (timestamps[1] - timestamps[0]).total_seconds() / 86400.0
            if time_diff_days > 0:
                trend = (weights[1] - weights[0]) / time_diff_days
            else:
                trend = 0.0
        else:
            time_diffs = []
            weight_diffs = []
            for i in range(1, len(weights)):
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400.0
                if time_diff > 0:
                    time_diffs.append(time_diff)
                    weight_diffs.append(weights[i] - weights[i-1])
            
            if time_diffs:
                trends = [wd/td for wd, td in zip(weight_diffs, time_diffs)]
                trend = np.median(trends)
            else:
                trend = 0.0
        
        trend = max(-0.5, min(0.5, trend))
        
        if pre_gap_trend is not None:
            trend = 0.7 * trend + 0.3 * pre_gap_trend
        
        return trend
    
    @staticmethod
    def calculate_adaptive_parameters(gap_days: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate adaptive Kalman parameters based on gap duration."""
        gap_factor = min(gap_days / 30, 3.0)
        
        gap_config = config.get('kalman', {}).get('gap_handling', {})
        gap_variance_multiplier = gap_config.get('gap_variance_multiplier', 2.0)
        trend_variance_multiplier = gap_config.get('trend_variance_multiplier', 20.0)
        
        base_weight_cov = config.get('kalman', {}).get('transition_covariance_weight', KALMAN_DEFAULTS['transition_covariance_weight'])
        base_trend_cov = config.get('kalman', {}).get('transition_covariance_trend', KALMAN_DEFAULTS['transition_covariance_trend'])
        
        adaptive_weight_cov = base_weight_cov * (1 + gap_factor * gap_variance_multiplier)
        adaptive_trend_cov = base_trend_cov * (1 + gap_factor * trend_variance_multiplier)
        
        return {
            'gap_factor': gap_factor,
            'transition_covariance': [
                [adaptive_weight_cov, 0],
                [0, adaptive_trend_cov]
            ]
        }
    
    @staticmethod
    def initialize_from_buffer(buffer_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize Kalman filter from buffered measurements."""
        measurements = buffer_data['measurements']
        gap_days = buffer_data['gap_days']
        
        weights = [m['weight'] for m in measurements]
        initial_weight = np.median(weights)
        
        pre_gap_trend = buffer_data.get('pre_gap_trend')
        initial_trend = KalmanFilterManager.estimate_trend_from_buffer(measurements, pre_gap_trend)
        
        adaptive_params = KalmanFilterManager.calculate_adaptive_parameters(gap_days, config)
        gap_factor = adaptive_params['gap_factor']
        
        kalman_config = config.get('kalman', {})
        initial_variance = kalman_config.get('initial_variance', KALMAN_DEFAULTS['initial_variance'])
        obs_cov = kalman_config.get('observation_covariance', KALMAN_DEFAULTS['observation_covariance'])
        
        adaptive_initial_variance = initial_variance * (1 + gap_factor * 2)
        adaptive_trend_variance = 0.001 * (1 + gap_factor * 10)
        
        kalman_params = {
            'initial_state_mean': [initial_weight, initial_trend],
            'initial_state_covariance': [
                [adaptive_initial_variance, 0],
                [0, adaptive_trend_variance]
            ],
            'transition_covariance': adaptive_params['transition_covariance'],
            'observation_covariance': [[obs_cov]],
        }
        
        state = {
            'kalman_params': kalman_params,
            'last_state': np.array([[initial_weight, initial_trend]]),
            'last_covariance': np.array([[
                [adaptive_initial_variance, 0],
                [0, adaptive_trend_variance]
            ]]),
            'gap_adaptation': {
                'active': True,
                'gap_factor': gap_factor,
                'measurements_since_gap': len(measurements),
                'initial_trend': initial_trend,
                'decay_rate': config.get('kalman', {}).get('gap_handling', {}).get('adaptation_decay_rate', 5.0)
            }
        }
        
        return state
    
    @staticmethod
    def apply_adaptation_decay(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply exponential decay to adaptive parameters."""
        if not state.get('gap_adaptation', {}).get('active'):
            return state
        
        adaptation = state['gap_adaptation']
        measurements_since = adaptation['measurements_since_gap']
        
        decay_factor = np.exp(-measurements_since / adaptation['decay_rate'])
        
        if decay_factor > 0.01:
            gap_factor = adaptation['gap_factor']
            
            kalman_config = config.get('kalman', {})
            base_weight_cov = kalman_config.get('transition_covariance_weight', KALMAN_DEFAULTS['transition_covariance_weight'])
            base_trend_cov = kalman_config.get('transition_covariance_trend', KALMAN_DEFAULTS['transition_covariance_trend'])
            
            gap_config = kalman_config.get('gap_handling', {})
            gap_variance_multiplier = gap_config.get('gap_variance_multiplier', 2.0)
            trend_variance_multiplier = gap_config.get('trend_variance_multiplier', 20.0)
            
            state['kalman_params']['transition_covariance'] = [
                [base_weight_cov * (1 + gap_factor * gap_variance_multiplier * decay_factor), 0],
                [0, base_trend_cov * (1 + gap_factor * trend_variance_multiplier * decay_factor)]
            ]
            
            adaptation['measurements_since_gap'] += 1
        else:
            state['gap_adaptation']['active'] = False
            kalman_config = config.get('kalman', {})
            state['kalman_params']['transition_covariance'] = [
                [kalman_config.get('transition_covariance_weight', KALMAN_DEFAULTS['transition_covariance_weight']), 0],
                [0, kalman_config.get('transition_covariance_trend', KALMAN_DEFAULTS['transition_covariance_trend'])]
            ]
        
        return state
