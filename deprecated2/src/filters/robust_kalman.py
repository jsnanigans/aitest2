"""
Robust Kalman Filter with Advanced Outlier Handling
Implements a more resilient version that can handle extreme outliers
without corrupting the state estimates
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import numpy as np
from collections import deque

from src.core.types import KalmanState, WeightMeasurement

logger = logging.getLogger(__name__)


class RobustKalmanFilter:
    """
    Enhanced Kalman filter with robust outlier handling mechanisms:
    1. Adaptive outlier rejection with dynamic thresholds
    2. Partial state updates for suspicious measurements
    3. State recovery after extreme outliers
    4. Innovation monitoring with history
    """

    def __init__(self,
                 initial_weight: float,
                 initial_variance: float,
                 process_noise_weight: float = 0.5,
                 process_noise_trend: float = 0.01,
                 measurement_noise: float = 1.0,
                 outlier_threshold: float = 3.0,
                 extreme_outlier_threshold: float = 5.0,
                 innovation_window_size: int = 20,
                 reset_gap_days: int = 30,
                 reset_deviation_threshold: float = 0.5,
                 physiological_min: float = 40.0,
                 physiological_max: float = 300.0):
        """
        Initialize robust Kalman filter with outlier protection.

        Args:
            outlier_threshold: Standard deviation threshold for outlier detection (default 3σ)
            extreme_outlier_threshold: Threshold for extreme outliers that get rejected entirely (default 5σ)
            innovation_window_size: Window size for adaptive threshold calculation
            reset_gap_days: Days of gap that triggers state reset
            reset_deviation_threshold: Fractional deviation that triggers reset (0.5 = 50%)
            physiological_min: Minimum plausible weight in kg
            physiological_max: Maximum plausible weight in kg
        """
        self.state = np.array([initial_weight, 0.0])
        self.covariance = np.array([
            [initial_variance, 0.0],
            [0.0, 0.001]
        ])

        self.Q = np.array([
            [process_noise_weight, 0.0],
            [0.0, process_noise_trend]
        ])

        self.base_R = measurement_noise
        self.R = measurement_noise

        self.H = np.array([1.0, 0.0])

        self.outlier_threshold = outlier_threshold
        self.extreme_outlier_threshold = extreme_outlier_threshold

        self.innovation_history = deque(maxlen=innovation_window_size)
        self.innovation_variance_history = deque(maxlen=innovation_window_size)

        self.measurement_count = 0
        self.outlier_count = 0
        self.extreme_outlier_count = 0
        self.last_timestamp: Optional[datetime] = None

        # State reset parameters
        self.reset_gap_days = reset_gap_days
        self.reset_deviation_threshold = reset_deviation_threshold
        self.physiological_min = physiological_min
        self.physiological_max = physiological_max
        self.reset_count = 0

        # Store initial values for reset
        self.initial_variance = initial_variance

        self.predicted_state = None
        self.predicted_covariance = None

    def predict(self, time_delta_days: float = 1.0) -> Tuple[float, float]:
        """Prediction step with time-aware state transition."""
        F = np.array([
            [1.0, time_delta_days],
            [0.0, 1.0]
        ])

        self.predicted_state = F @ self.state

        Q_scaled = self.Q * time_delta_days

        self.predicted_covariance = F @ self.covariance @ F.T + Q_scaled

        predicted_weight = self.predicted_state[0]
        prediction_variance = self.predicted_covariance[0, 0]

        return predicted_weight, prediction_variance

    def should_reset_state(self, measurement: float, time_gap_days: float) -> Tuple[bool, str]:
        """
        Check if state should be reset based on configurable criteria.

        Returns:
            Tuple of (should_reset, reason)
        """
        # Check for time gap
        if time_gap_days > self.reset_gap_days:
            return True, f"Time gap ({time_gap_days:.0f} days) exceeds threshold ({self.reset_gap_days} days)"

        # Check for physiological bounds on current state
        current_weight = self.state[0]
        if current_weight < self.physiological_min or current_weight > self.physiological_max:
            return True, f"State weight ({current_weight:.1f} kg) outside physiological bounds [{self.physiological_min}, {self.physiological_max}]"

        # Disabled deviation-based reset - large deviations are more likely data errors than valid resets
        # Keeping only time-based and physiological bounds resets
        
        return False, ""

    def reset_state(self, measurement: float, reason: str = ""):
        """
        Reset the Kalman filter state to the current measurement.
        """
        logger.debug(f"Resetting Kalman state to {measurement:.2f} kg. Reason: {reason}")

        self.state = np.array([measurement, 0.0])
        self.covariance = np.array([
            [self.initial_variance, 0.0],
            [0.0, 0.001]
        ])

        # Clear innovation history
        self.innovation_history.clear()
        self.innovation_variance_history.clear()

        self.reset_count += 1

        # Update predicted state to match
        self.predicted_state = self.state.copy()
        self.predicted_covariance = self.covariance.copy()

    def calculate_adaptive_threshold(self) -> float:
        """
        Calculate adaptive outlier threshold based on recent innovation history.
        Uses robust statistics to avoid influence from outliers.
        """
        if len(self.innovation_history) < 5:
            return self.outlier_threshold

        innovations = np.array(self.innovation_history)

        median_abs_innovation = np.median(np.abs(innovations))
        mad = np.median(np.abs(innovations - np.median(innovations)))

        robust_std = 1.4826 * mad

        if robust_std < 0.1:
            return self.outlier_threshold

        adaptive_threshold = max(2.5, min(self.outlier_threshold,
                                         3.0 * robust_std / np.sqrt(np.mean(self.innovation_variance_history))))

        return adaptive_threshold

    def update(self, measurement: float, robust_mode: bool = True) -> Dict[str, float]:
        """
        Enhanced update step with robust outlier handling.

        Args:
            measurement: The weight measurement
            robust_mode: Whether to use robust outlier handling
        """
        innovation = measurement - self.H @ self.predicted_state

        S = self.H @ self.predicted_covariance @ self.H.T + self.R

        normalized_innovation = abs(innovation) / np.sqrt(S) if S > 0 else float('inf')

        self.innovation_history.append(innovation)
        self.innovation_variance_history.append(S)

        adaptive_threshold = self.calculate_adaptive_threshold() if robust_mode else self.outlier_threshold

        if robust_mode and normalized_innovation > self.extreme_outlier_threshold:
            logger.debug(f"Extreme outlier detected: {normalized_innovation:.1f}σ, rejecting measurement")
            self.extreme_outlier_count += 1

            self.state = self.predicted_state.copy()
            self.covariance = self.predicted_covariance.copy()

            return {
                'filtered_weight': float(self.state[0]),
                'trend_kg_per_day': float(self.state[1]),
                'innovation': float(innovation),
                'innovation_variance': float(S),
                'normalized_innovation': float(normalized_innovation),
                'outlier_type': 'extreme',
                'update_applied': False,
                'weight_uncertainty': float(np.sqrt(self.covariance[0, 0])),
                'trend_uncertainty': float(np.sqrt(self.covariance[1, 1]))
            }

        if robust_mode and normalized_innovation > adaptive_threshold:
            logger.info(f"Outlier detected: {normalized_innovation:.1f}σ, applying dampened update")
            self.outlier_count += 1

            damping_factor = adaptive_threshold / normalized_innovation
            damping_factor = max(0.1, min(0.5, damping_factor))

            K_base = (self.predicted_covariance @ self.H.T) / S
            K = K_base * damping_factor

            self.state = self.predicted_state + K * innovation

            I = np.eye(2)
            IKH = I - np.outer(K, self.H)
            self.covariance = IKH @ self.predicted_covariance @ IKH.T + np.outer(K, K) * self.R

            outlier_type = 'moderate'
        else:
            K = (self.predicted_covariance @ self.H.T) / S

            self.state = self.predicted_state + K * innovation

            I = np.eye(2)
            IKH = I - np.outer(K, self.H)
            self.covariance = IKH @ self.predicted_covariance @ IKH.T + np.outer(K, K) * self.R

            outlier_type = 'none'

        self.measurement_count += 1

        return {
            'filtered_weight': float(self.state[0]),
            'trend_kg_per_day': float(self.state[1]),
            'innovation': float(innovation),
            'innovation_variance': float(S),
            'normalized_innovation': float(normalized_innovation),
            'outlier_type': outlier_type,
            'update_applied': True,
            'adaptive_threshold': float(adaptive_threshold),
            'kalman_gain_weight': float(K[0]) if isinstance(K, np.ndarray) else float(K),
            'kalman_gain_trend': float(K[1]) if isinstance(K, np.ndarray) else 0.0,
            'weight_uncertainty': float(np.sqrt(self.covariance[0, 0])),
            'trend_uncertainty': float(np.sqrt(self.covariance[1, 1]))
        }

    def process_measurement(self,
                           measurement: WeightMeasurement,
                           source_trust: Optional[float] = None,
                           robust_mode: bool = True) -> Dict[str, Any]:
        """
        Complete robust Kalman filter cycle with outlier protection and state reset.

        Args:
            measurement: Weight measurement to process
            source_trust: Optional trust factor for adaptive R (0.0 to 1.0)
            robust_mode: Whether to use robust outlier handling (default True)
        """
        time_delta_days = 1.0
        if measurement.timestamp and self.last_timestamp:
            delta = (measurement.timestamp - self.last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, delta)

        # Predict step
        predicted_weight, prediction_variance = self.predict(time_delta_days)

        # Check if state reset is needed
        should_reset, reset_reason = self.should_reset_state(measurement.weight, time_delta_days)
        if should_reset:
            self.reset_state(measurement.weight, reset_reason)
            # After reset, predict again with the new state
            predicted_weight, prediction_variance = self.predict(0.1)  # Small time step after reset

        # Adjust process noise for large gaps (only if not reset)
        gap_factor = 1.0
        if not should_reset and time_delta_days > 30:
            logger.info(f"Large time gap detected ({time_delta_days:.1f} days), increasing process noise")
            gap_factor = min(3.0, 1.0 + (time_delta_days - 30) / 30)
            self.Q = self.Q * gap_factor

        # Adjust measurement noise based on source trust
        if source_trust is not None:
            noise_scale = 2.0 - 1.5 * source_trust
            self.R = self.base_R * noise_scale

        # Update step
        update_results = self.update(measurement.weight, robust_mode=robust_mode)

        # Add reset info to results
        update_results['state_reset'] = should_reset
        if should_reset:
            update_results['reset_reason'] = reset_reason

        # Restore measurement noise
        if source_trust is not None:
            self.R = self.base_R

        # Restore process noise if it was adjusted
        if gap_factor > 1.0:
            self.Q = self.Q / gap_factor

        self.last_timestamp = measurement.timestamp

        return {
            'predicted_weight': predicted_weight,
            'prediction_variance': prediction_variance,
            'time_delta_days': time_delta_days,
            'measurement_count': self.measurement_count,
            'outlier_count': self.outlier_count,
            'extreme_outlier_count': self.extreme_outlier_count,
            'outlier_rate': self.outlier_count / max(1, self.measurement_count),
            **update_results
        }

    def get_state(self) -> KalmanState:
        """Get current filter state."""
        return KalmanState(
            weight=float(self.state[0]),
            trend=float(self.state[1]),
            covariance=self.covariance.tolist(),
            timestamp=self.last_timestamp or datetime.now(),
            measurement_count=self.measurement_count
        )

    def reset_to_baseline(self, weight: float, variance: float):
        """
        Reset filter state to a new baseline.
        Useful after detecting regime changes or initialization issues.
        """
        logger.info(f"Resetting Kalman filter to baseline: {weight:.1f}kg (variance={variance:.3f})")

        self.state = np.array([weight, 0.0])
        self.covariance = np.array([
            [variance, 0.0],
            [0.0, 0.001]
        ])

        self.innovation_history.clear()
        self.innovation_variance_history.clear()

        self.outlier_count = 0
        self.extreme_outlier_count = 0


class AdaptiveValidationGate:
    """
    Enhanced validation gate with adaptive thresholds based on measurement quality.
    """

    def __init__(self,
                 base_gamma: float = 3.0,
                 strict_gamma: float = 2.0,
                 relaxed_gamma: float = 4.0):
        """
        Args:
            base_gamma: Default threshold
            strict_gamma: Strict threshold for high-confidence periods
            relaxed_gamma: Relaxed threshold for noisy periods
        """
        self.base_gamma = base_gamma
        self.strict_gamma = strict_gamma
        self.relaxed_gamma = relaxed_gamma

        self.recent_rejections = deque(maxlen=10)

    def calculate_adaptive_gamma(self) -> float:
        """
        Adjust gamma based on recent rejection rate.
        High rejection rate -> relax threshold slightly
        Low rejection rate -> can be stricter
        """
        if len(self.recent_rejections) < 5:
            return self.base_gamma

        rejection_rate = sum(self.recent_rejections) / len(self.recent_rejections)

        if rejection_rate > 0.3:
            return min(self.relaxed_gamma, self.base_gamma * 1.2)
        elif rejection_rate < 0.05:
            return max(self.strict_gamma, self.base_gamma * 0.8)
        else:
            return self.base_gamma

    def validate(self,
                measurement: float,
                predicted_weight: float,
                innovation_variance: float,
                adaptive: bool = True) -> Tuple[bool, float, float]:
        """
        Enhanced validation with adaptive thresholds.

        Returns:
            (is_valid, normalized_innovation, current_gamma)
        """
        innovation = measurement - predicted_weight

        if innovation_variance <= 0:
            return True, 0.0, self.base_gamma

        normalized_innovation = abs(innovation) / np.sqrt(innovation_variance)

        current_gamma = self.calculate_adaptive_gamma() if adaptive else self.base_gamma

        is_valid = normalized_innovation <= current_gamma

        self.recent_rejections.append(not is_valid)

        if not is_valid:
            logger.debug(f"Adaptive validation gate rejection: {normalized_innovation:.2f}σ > {current_gamma:.1f}σ")

        return is_valid, normalized_innovation, current_gamma
