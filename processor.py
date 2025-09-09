"""
Simplified Weight Processor with Kalman Filter
Uses pykalman for mathematically correct filtering
"""

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from pykalman import KalmanFilter


class WeightProcessor:
    """
    Stateless recursive weight processor for a single user.
    Combines basic physiological validation with Kalman filtering.
    
    Implements council recommendations:
    - Simplified to 2 key parameters: observation_covariance and extreme_threshold
    - Smooth exponential confidence function (Knuth)
    - Adaptive observation noise estimation (Kleppmann)
    - Maintains mathematical purity with pykalman (Lamport)
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

        # Load from processing config
        self.min_init_readings = processing_config["min_init_readings"]
        self.min_weight = processing_config["min_weight"]
        self.max_weight = processing_config["max_weight"]
        self.max_daily_change = processing_config["max_daily_change"]
        self.extreme_threshold = processing_config["extreme_threshold"]

        # Load from kalman config (will be adapted based on user data)
        self.kalman_config = kalman_config.copy()  # Make a copy for per-user adaptation
        self.base_extreme_threshold = self.extreme_threshold  # Store base value for adaptation

    def process_weight(
        self, weight: float, timestamp: datetime, source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single weight measurement.
        Returns dict with processed result or None if buffering.
        """
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
                self._initialize_kalman()
            else:
                return None

        time_delta_days = 1.0
        if self.last_timestamp:
            delta = (timestamp - self.last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, min(30.0, delta))

            if delta > self.kalman_config["reset_gap_days"]:
                self._reset_kalman(weight)

        if self.last_state is not None and self.baseline:
            predicted_weight = (
                self.last_state[0][0] + self.last_state[0][1] * time_delta_days
            )
            deviation = abs(weight - predicted_weight) / predicted_weight

            if deviation > self.extreme_threshold:
                # Calculate confidence for rejected measurement using smooth function
                # Map deviation to normalized innovation scale (deviation/threshold maps to ~3-5 sigma)
                pseudo_normalized_innovation = (deviation / self.extreme_threshold) * 3.0
                confidence = self._calculate_confidence(pseudo_normalized_innovation)
                
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "filtered_weight": predicted_weight,
                    "trend": self.last_state[0][1],
                    "accepted": False,
                    "reason": f"Extreme deviation: {deviation:.1%}",
                    "confidence": confidence,
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

        confidence = self._calculate_confidence(normalized_innovation)

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

    def _adapt_parameters(self, user_data: list) -> None:
        """
        Adapt Kalman parameters based on user characteristics (Advanced - Step 3).
        This method analyzes the initial data to determine optimal parameters
        for this specific user's weight pattern and measurement noise.
        """
        if len(user_data) < 3:
            return  # Not enough data to adapt
        
        weights = [w for w, _ in user_data] if isinstance(user_data[0], tuple) else user_data
        
        # Calculate variance using MAD (Median Absolute Deviation) - more robust than std
        median_weight = np.median(weights)
        mad = np.median([abs(w - median_weight) for w in weights])
        variance = mad * 1.4826  # Convert MAD to standard deviation estimate
        
        # Adapt observation_covariance based on data quality
        if variance < 0.5:  # Very clean data (high-quality scale)
            self.kalman_config['observation_covariance'] = 0.5
            self.extreme_threshold = 0.20
        elif variance < 1.5:  # Normal variance (typical bathroom scale)
            self.kalman_config['observation_covariance'] = 1.5
            self.extreme_threshold = 0.25
        else:  # High variance (poor scale or hydration swings)
            self.kalman_config['observation_covariance'] = 3.0
            self.extreme_threshold = 0.35
        
        # Adapt process noise based on detected patterns
        if len(weights) > 7:
            # Check for trend (weight loss/gain)
            first_half = np.mean(weights[:len(weights)//2])
            second_half = np.mean(weights[len(weights)//2:])
            trend_magnitude = abs(second_half - first_half) / len(weights)
            
            if trend_magnitude > 0.05:  # Active weight change
                self.kalman_config['transition_covariance_trend'] = 0.001
            else:  # Stable weight
                self.kalman_config['transition_covariance_trend'] = 0.0005
        
        # Log adaptation for debugging (optional)
        if hasattr(self, 'user_id'):
            adaptation_info = {
                'user': self.user_id[:8],
                'variance': round(variance, 2),
                'obs_covar': self.kalman_config['observation_covariance'],
                'threshold': self.extreme_threshold
            }
            # Uncomment for debugging: print(f"Adapted parameters: {adaptation_info}")

    def _basic_validation(self, weight: float) -> bool:
        """Basic physiological validation."""
        if weight < self.min_weight or weight > self.max_weight:
            return False

        if self.last_state is not None:
            last_weight = self.last_state[-1][0]
            change = abs(weight - last_weight) / last_weight
            if change > 0.5:
                return False

        return True

    def _initialize_kalman(self):
        """
        Initialize Kalman filter with adaptive parameters based on user data.
        Implements council recommendations for simplified, adaptive tuning.
        """
        weights = [w for w, _ in self.init_buffer]
        
        # Call the advanced adaptation method (Step 3)
        self._adapt_parameters(self.init_buffer)

        weights_clean = []
        for w in weights:
            if not weights_clean:
                weights_clean.append(w)
            else:
                if abs(w - np.median(weights_clean)) / np.median(weights_clean) < 0.1:
                    weights_clean.append(w)

        if len(weights_clean) < 3:
            weights_clean = weights

        self.baseline = np.median(weights_clean)
        
        # Use the adapted observation_covariance (already set by _adapt_parameters)
        adapted_obs_covariance = self.kalman_config["observation_covariance"]
        
        # Calculate variance for initial state covariance
        mad = np.median([abs(w - self.baseline) for w in weights_clean])
        estimated_std = mad * 1.4826
        variance = estimated_std ** 2 if estimated_std > 0 else self.kalman_config["initial_variance"]

        self.kalman = KalmanFilter(
            transition_matrices=np.array([[1, 1], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=np.array([self.baseline, 0]),
            initial_state_covariance=np.array([[variance, 0], [0, 0.001]]),
            transition_covariance=np.array(
                [
                    [self.kalman_config["transition_covariance_weight"], 0],
                    [0, self.kalman_config["transition_covariance_trend"]],
                ]
            ),
            observation_covariance=np.array([[adapted_obs_covariance]]),
        )

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

    def _calculate_confidence(self, normalized_innovation: float) -> float:
        """
        Calculate confidence score using smooth exponential decay (Knuth's recommendation).
        confidence = exp(-alpha * normalized_innovation²)
        
        This provides continuous confidence scores that better reflect uncertainty,
        replacing the crude step function with a mathematically elegant solution.
        """
        alpha = 0.5  # Tunable parameter, 0.5 gives good decay characteristics
        return np.exp(-alpha * normalized_innovation ** 2)

