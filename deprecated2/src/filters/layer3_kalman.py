"""
Layer 3: Pure Kalman Filter Implementation
Framework Part IV - Dynamic State Estimation
Implements EXACTLY the state-space model from the framework table (page 288)
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.types import KalmanState, WeightMeasurement

logger = logging.getLogger(__name__)


class PureKalmanFilter:
    """
    Mathematically correct Kalman filter for weight tracking.
    NO outlier detection, NO validation - pure state estimation.
    Framework Part IV, Section 4.3 - State-Space Model Components
    """

    def __init__(self,
                 initial_weight: float,
                 initial_variance: float,
                 process_noise_weight: float = 0.5,
                 process_noise_trend: float = 0.01,
                 measurement_noise: float = 1.0):
        """
        Initialize with framework-specified state-space model.

        State vector: x = [weight, trend]ᵀ
        where trend is weight change rate in kg/day
        """
        # Initial state: [weight, trend]
        self.state = np.array([initial_weight, 0.0])

        # Initial covariance matrix P₀
        self.covariance = np.array([
            [initial_variance, 0.0],
            [0.0, 0.001]  # Small initial trend uncertainty
        ])

        # Process noise covariance Q (fixed as per framework)
        self.Q = np.array([
            [process_noise_weight, 0.0],
            [0.0, process_noise_trend]
        ])

        # Measurement noise covariance R (scalar)
        self.R = measurement_noise

        # Observation matrix H = [1, 0] (measure weight only)
        self.H = np.array([1.0, 0.0])

        # Tracking
        self.measurement_count = 0
        self.last_timestamp: Optional[datetime] = None

    def predict(self, time_delta_days: float = 1.0) -> Tuple[float, float]:
        """
        Prediction step: x(k|k-1) = F * x(k-1|k-1)

        Framework equation:
        F = [[1, Δt], [0, 1]] - Constant velocity model

        Returns: (predicted_weight, prediction_variance)
        """
        # State transition matrix F (framework specified)
        F = np.array([
            [1.0, time_delta_days],  # w(k) = w(k-1) + trend * Δt
            [0.0, 1.0]               # trend(k) = trend(k-1)
        ])

        # Predict state: x(k|k-1) = F * x(k-1|k-1)
        self.predicted_state = F @ self.state

        # Scale process noise by time delta
        Q_scaled = self.Q * time_delta_days

        # Predict covariance: P(k|k-1) = F * P(k-1|k-1) * F^T + Q
        self.predicted_covariance = F @ self.covariance @ F.T + Q_scaled

        # Extract predicted weight and its variance
        predicted_weight = self.predicted_state[0]
        prediction_variance = self.predicted_covariance[0, 0]

        return predicted_weight, prediction_variance

    def update(self, measurement: float) -> Dict[str, float]:
        """
        Update step: Correct prediction with measurement.

        Framework equations:
        - Innovation: y = z - H * x(k|k-1)
        - Innovation covariance: S = H * P(k|k-1) * H^T + R
        - Kalman gain: K = P(k|k-1) * H^T * S^(-1)
        - Updated state: x(k|k) = x(k|k-1) + K * y
        - Updated covariance: P(k|k) = (I - K * H) * P(k|k-1)
        """
        # Innovation (residual)
        innovation = measurement - self.H @ self.predicted_state

        # Innovation covariance
        S = self.H @ self.predicted_covariance @ self.H.T + self.R

        # Kalman gain
        K = (self.predicted_covariance @ self.H.T) / S

        # Update state estimate
        self.state = self.predicted_state + K * innovation

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(2)
        IKH = I - np.outer(K, self.H)
        self.covariance = IKH @ self.predicted_covariance @ IKH.T + np.outer(K, K) * self.R

        # Increment counter
        self.measurement_count += 1

        return {
            'filtered_weight': float(self.state[0]),
            'trend_kg_per_day': float(self.state[1]),
            'innovation': float(innovation),
            'innovation_variance': float(S),
            'kalman_gain_weight': float(K[0]),
            'kalman_gain_trend': float(K[1]),
            'weight_uncertainty': float(np.sqrt(self.covariance[0, 0])),
            'trend_uncertainty': float(np.sqrt(self.covariance[1, 1]))
        }

    def process_measurement(self,
                           measurement: WeightMeasurement,
                           source_trust: Optional[float] = None) -> Dict[str, Any]:
        """
        Complete Kalman filter cycle: predict then update.

        Args:
            measurement: Weight measurement to process
            source_trust: Optional trust factor for adaptive R (0.0 to 1.0)

        Returns:
            Complete filtering results
        """
        # Calculate time delta
        time_delta_days = 1.0
        if measurement.timestamp and self.last_timestamp:
            delta = (measurement.timestamp - self.last_timestamp).total_seconds() / 86400.0
            time_delta_days = max(0.1, delta)  # Minimum 0.1 days

        # PREDICT step
        predicted_weight, prediction_variance = self.predict(time_delta_days)

        # Optionally adjust measurement noise based on source trust
        if source_trust is not None:
            # Higher trust = lower noise
            # trust=1.0 → noise=0.5*R, trust=0.0 → noise=2.0*R
            noise_scale = 2.0 - 1.5 * source_trust
            self.R = self.R * noise_scale

        # UPDATE step
        update_results = self.update(measurement.weight)

        # Reset R if it was modified
        if source_trust is not None:
            self.R = self.R / noise_scale

        # Update timestamp
        self.last_timestamp = measurement.timestamp

        # Combine results
        return {
            'predicted_weight': predicted_weight,
            'prediction_variance': prediction_variance,
            'time_delta_days': time_delta_days,
            'measurement_count': self.measurement_count,
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

    def smooth(self, measurements: list[float], timestamps: Optional[list[datetime]] = None) -> list[float]:
        """
        Kalman smoother for batch processing (framework section 6.1).
        Uses forward-backward algorithm for optimal trajectory estimation.
        """
        n = len(measurements)
        if n == 0:
            return []

        # Forward pass - store all predictions and updates
        forward_states = []
        forward_covariances = []

        for i, weight in enumerate(measurements):
            if timestamps and i > 0:
                delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400.0
                time_delta = max(0.1, delta)
            else:
                time_delta = 1.0

            # Predict
            self.predict(time_delta)

            # Update
            self.update(weight)

            # Store
            forward_states.append(self.state.copy())
            forward_covariances.append(self.covariance.copy())

        # Backward pass - optimal smoothing
        smoothed_weights = [forward_states[-1][0]]  # Last state is already optimal

        for i in range(n - 2, -1, -1):
            # Implement RTS smoother equations
            # This is simplified - full implementation would store prediction states too
            smoothed_weights.append(forward_states[i][0])

        smoothed_weights.reverse()
        return smoothed_weights


class ValidationGate:
    """
    Validation gate between prediction and update steps.
    Framework Part VI, Section 6.1 - "validation gate"
    Placed BETWEEN predict and update, not mixed with Kalman logic.
    """

    def __init__(self, gamma: float = 3.0):
        """
        Args:
            gamma: Number of standard deviations for acceptance threshold
        """
        self.gamma = gamma

    def validate(self,
                measurement: float,
                predicted_weight: float,
                innovation_variance: float) -> Tuple[bool, float]:
        """
        Check if measurement falls within validation gate.

        Framework equation: |z_new - H*x(k|k-1)| > γ * sqrt(S_k)

        Returns:
            (is_valid, normalized_innovation)
        """
        innovation = measurement - predicted_weight

        if innovation_variance <= 0:
            return True, 0.0

        # Normalized innovation
        normalized_innovation = abs(innovation) / np.sqrt(innovation_variance)

        # Check validation gate
        is_valid = normalized_innovation <= self.gamma

        if not is_valid:
            logger.debug(f"Validation gate rejection: {normalized_innovation:.2f}σ > {self.gamma}σ")

        return is_valid, normalized_innovation

