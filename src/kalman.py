"""
Kalman filter logic for weight processing.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import numpy as np
from pykalman import KalmanFilter


class KalmanFilterManager:
    """Manages Kalman filter operations for weight processing."""
    
    @staticmethod
    def initialize_immediate(
        weight: float,
        timestamp: datetime,
        kalman_config: dict
    ) -> Dict[str, Any]:
        """Initialize Kalman filter immediately with first measurement."""
        initial_variance = kalman_config.get("initial_variance", 1.0)
        
        kalman_params = {
            'initial_state_mean': [weight, 0],
            'initial_state_covariance': [[initial_variance, 0], [0, 0.001]],
            'transition_covariance': [
                [kalman_config.get("transition_covariance_weight", 0.1), 0],
                [0, kalman_config.get("transition_covariance_trend", 0.001)]
            ],
            'observation_covariance': [[kalman_config.get("observation_covariance", 1.0)]],
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
        processing_config: dict
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
        kalman = KalmanFilter(
            transition_matrices=np.array([[1, time_delta_days], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=np.array(kalman_params['initial_state_mean']),
            initial_state_covariance=np.array(kalman_params['initial_state_covariance']),
            transition_covariance=np.array(kalman_params['transition_covariance']),
            observation_covariance=np.array(kalman_params['observation_covariance']),
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
    def reset_state(state: Dict[str, Any], new_weight: float) -> Dict[str, Any]:
        """Reset Kalman state after long gap."""
        state['last_state'] = np.array([[new_weight, 0]])
        state['last_covariance'] = np.array([[[1.0, 0], [0, 0.001]]])
        state['last_raw_weight'] = new_weight
        
        if state.get('kalman_params'):
            state['kalman_params']['initial_state_mean'] = [new_weight, 0]
        
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
        accepted: bool
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
        if last_covariance is not None:
            if len(last_covariance.shape) > 2:
                current_covariance = last_covariance[-1]
            else:
                current_covariance = last_covariance
            
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
        return max(0.1, min(30.0, delta))