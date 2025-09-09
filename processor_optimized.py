"""
Optimized Weight Processor with Adaptive Kalman Filter
Implements council recommendations for improved performance
"""

from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
from pykalman import KalmanFilter


class OptimizedWeightProcessor:
    """
    Enhanced weight processor with adaptive parameter tuning and smooth confidence.
    Key improvements:
    - Adaptive parameter estimation based on user data
    - Smooth exponential confidence function
    - Simplified parameter set (2 key params vs 7)
    - Better outlier handling with user-specific thresholds
    """

    def __init__(self, user_id: str, processing_config: dict, kalman_config: dict):
        self.user_id = user_id
        self.kalman = None
        self.last_state = None
        self.last_covariance = None
        self.last_timestamp = None
        self.baseline = None
        self.measurement_count = 0
        self.init_buffer = []
        
        # Adaptive parameters (will be tuned per-user)
        self.adapted_params = None
        
        # Load from processing config
        self.min_init_readings = processing_config["min_init_readings"]
        self.min_weight = processing_config["min_weight"]
        self.max_weight = processing_config["max_weight"]
        self.max_daily_change = processing_config["max_daily_change"]
        
        # Simplified config - only essential parameters
        self.base_kalman_config = {
            'observation_covariance': kalman_config.get('observation_covariance', 1.0),
            'extreme_threshold': processing_config.get('extreme_threshold', 0.3),
            'reset_gap_days': kalman_config.get('reset_gap_days', 30)
        }
        
        # Advanced params with sensible defaults
        self.advanced_params = {
            'transition_covariance_weight': 0.1,  # From optimization
            'transition_covariance_trend': 0.001,  # From optimization
            'confidence_alpha': 0.5  # For smooth confidence function
        }

    def process_weight(
        self, weight: float, timestamp: datetime, source: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single weight measurement with adaptive filtering."""
        
        if not self._basic_validation(weight):
            return {
                "timestamp": timestamp,
                "raw_weight": weight,
                "accepted": False,
                "reason": "Basic validation failed",
                "source": source,
            }

        if not self.kalman:
            self.init_buffer.append((weight, timestamp))
            
            if len(self.init_buffer) >= self.min_init_readings:
                self._initialize_adaptive_kalman()
            else:
                return None

        time_delta_days = 1.0
        if self.last_timestamp:
            delta = (timestamp - self.last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))
            
            if delta > self.base_kalman_config["reset_gap_days"]:
                self._reset_kalman(weight)

        if self.last_state is not None and self.baseline:
            predicted_weight = (
                self.last_state[0][0] + self.last_state[0][1] * time_delta_days
            )
            deviation = abs(weight - predicted_weight) / predicted_weight
            
            # Use adaptive threshold
            threshold = self.adapted_params.get('extreme_threshold', 
                                                self.base_kalman_config['extreme_threshold'])
            
            if deviation > threshold:
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "filtered_weight": predicted_weight,
                    "trend": self.last_state[0][1],
                    "accepted": False,
                    "reason": f"Extreme deviation: {deviation:.1%}",
                    "confidence": self._calculate_smooth_confidence(deviation / threshold * 5),
                    "source": source,
                }

        observation = np.array([[weight]])
        
        F = np.array([[1, time_delta_days], [0, 1]])
        self.kalman.transition_matrices = F
        
        if self.last_state is None:
            filtered_state_means, filtered_state_covariances = self.kalman.filter(
                observation
            )
            self.last_state = filtered_state_means
            self.last_covariance = filtered_state_covariances
        else:
            filtered_state_mean, filtered_state_covariance = self.kalman.filter_update(
                self.last_state[-1],
                self.last_covariance[-1],
                observation=observation[0],
            )
            
            if self.last_state.shape[0] == 1:
                self.last_state = np.vstack([self.last_state, [filtered_state_mean]])
                self.last_covariance = np.stack(
                    [self.last_covariance[0], filtered_state_covariance]
                )
            else:
                self.last_state = np.array([self.last_state[-1], filtered_state_mean])
                self.last_covariance = np.array(
                    [self.last_covariance[-1], filtered_state_covariance]
                )

        self.last_timestamp = timestamp
        self.measurement_count += 1
        
        current_state = self.last_state[-1]
        filtered_weight = current_state[0]
        trend = current_state[1]
        
        innovation = weight - filtered_weight
        innovation_variance = (
            self.last_covariance[-1][0, 0] + self.kalman.observation_covariance[0, 0]
        )
        normalized_innovation = (
            abs(innovation) / np.sqrt(innovation_variance)
            if innovation_variance > 0
            else 0
        )
        
        # Use smooth confidence function
        confidence = self._calculate_smooth_confidence(normalized_innovation)
        
        return {
            "timestamp": timestamp,
            "raw_weight": weight,
            "filtered_weight": filtered_weight,
            "trend": trend,
            "trend_weekly": trend * 7,
            "accepted": True,
            "confidence": confidence,
            "innovation": innovation,
            "normalized_innovation": normalized_innovation,
            "source": source,
        }

    def _basic_validation(self, weight: float) -> bool:
        """Basic physiological validation."""
        if weight < self.min_weight or weight > self.max_weight:
            return False
        
        if self.last_state is not None:
            last_weight = self.last_state[-1][0]
            change = abs(weight - last_weight) / last_weight
            if change > 0.5:  # 50% change is definitely an error
                return False
        
        return True

    def _analyze_user_characteristics(self) -> Dict:
        """Analyze buffered data to determine user characteristics."""
        weights = [w for w, _ in self.init_buffer]
        
        if len(weights) < 3:
            return {
                'variance': 1.0,
                'noise_level': 'medium',
                'daily_changes': []
            }
        
        # Use robust statistics (median absolute deviation)
        median_weight = np.median(weights)
        mad = np.median([abs(w - median_weight) for w in weights])
        variance = mad * 1.4826  # Convert MAD to std estimate
        
        # Classify noise level
        if variance < 0.5:
            noise_level = 'low'
        elif variance < 1.5:
            noise_level = 'medium'
        else:
            noise_level = 'high'
        
        # Calculate typical daily changes
        daily_changes = []
        for i in range(1, len(weights)):
            change = abs(weights[i] - weights[i-1])
            daily_changes.append(change)
        
        return {
            'variance': variance,
            'noise_level': noise_level,
            'daily_changes': daily_changes
        }

    def _adapt_parameters(self, characteristics: Dict) -> Dict:
        """Adapt Kalman parameters based on user characteristics."""
        
        # Start with base parameters
        adapted = self.base_kalman_config.copy()
        
        # Adapt observation noise to data quality
        variance = characteristics['variance']
        adapted['observation_covariance'] = min(3.0, max(0.5, variance * 1.5))
        
        # Adapt extreme threshold based on noise level
        noise_map = {
            'low': 0.20,     # Tight threshold for clean data
            'medium': 0.25,  # Moderate threshold
            'high': 0.35     # Looser threshold for noisy data
        }
        adapted['extreme_threshold'] = noise_map.get(
            characteristics['noise_level'], 0.30
        )
        
        # Adapt process noise if we have daily change data
        if characteristics['daily_changes']:
            median_change = np.median(characteristics['daily_changes'])
            adapted['transition_covariance_weight'] = min(1.0, max(0.05, median_change * 10))
        else:
            adapted['transition_covariance_weight'] = self.advanced_params['transition_covariance_weight']
        
        adapted['transition_covariance_trend'] = self.advanced_params['transition_covariance_trend']
        
        return adapted

    def _initialize_adaptive_kalman(self):
        """Initialize Kalman filter with adaptive parameters."""
        weights = [w for w, _ in self.init_buffer]
        
        # Analyze user characteristics
        characteristics = self._analyze_user_characteristics()
        
        # Adapt parameters
        self.adapted_params = self._adapt_parameters(characteristics)
        
        # Clean initialization data
        weights_clean = []
        median_weight = np.median(weights)
        for w in weights:
            if abs(w - median_weight) / median_weight < 0.1:
                weights_clean.append(w)
        
        if len(weights_clean) < 3:
            weights_clean = weights
        
        self.baseline = np.median(weights_clean)
        variance = characteristics['variance']
        
        # Initialize Kalman with adapted parameters
        self.kalman = KalmanFilter(
            transition_matrices=np.array([[1, 1], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=np.array([self.baseline, 0]),
            initial_state_covariance=np.array([[variance, 0], [0, 0.001]]),
            transition_covariance=np.array(
                [
                    [self.adapted_params['transition_covariance_weight'], 0],
                    [0, self.adapted_params['transition_covariance_trend']],
                ]
            ),
            observation_covariance=np.array(
                [[self.adapted_params['observation_covariance']]]
            ),
        )
        
        # Process buffered readings
        for weight, timestamp in self.init_buffer:
            self.process_weight(weight, timestamp, "init")
        
        self.init_buffer = []

    def _reset_kalman(self, new_weight: float):
        """Reset Kalman state after long gap."""
        self.baseline = new_weight
        self.last_state = np.array([[new_weight, 0]])
        self.last_covariance = np.array([[[1.0, 0], [0, 0.001]]])
        
        if self.kalman:
            self.kalman.initial_state_mean = np.array([new_weight, 0])

    def _calculate_smooth_confidence(self, normalized_innovation: float) -> float:
        """
        Smooth exponential confidence function (Knuth's recommendation).
        confidence = exp(-alpha * normalized_innovation^2)
        """
        alpha = self.advanced_params.get('confidence_alpha', 0.5)
        return np.exp(-alpha * normalized_innovation ** 2)