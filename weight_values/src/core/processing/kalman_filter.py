"""
Custom Kalman Filter implementation.

Implements the standard Kalman filter algorithm as specified at:
https://en.wikipedia.org/wiki/Kalman_filter

This replaces the pykalman dependency with a minimal, dependency-free implementation
that provides the same interface used by our weight processing system.
"""

import numpy as np
from typing import Tuple, Optional


class KalmanFilter:
    """
    Standard Kalman Filter implementation.

    The Kalman filter is an optimal linear estimator that recursively estimates
    the state of a linear dynamic system from noisy measurements.

    State-space model:
        x_k = F_k * x_{k-1} + w_k        (process model)
        z_k = H_k * x_k + v_k            (measurement model)

    where:
        x_k is the state vector at time k
        z_k is the measurement vector at time k
        F_k is the state transition matrix
        H_k is the observation matrix
        w_k ~ N(0, Q_k) is process noise
        v_k ~ N(0, R_k) is measurement noise

    Args:
        transition_matrices: State transition matrix F (n_states x n_states)
        observation_matrices: Observation matrix H (n_observations x n_states)
        initial_state_mean: Initial state estimate x_0 (n_states,)
        initial_state_covariance: Initial covariance P_0 (n_states x n_states)
        transition_covariance: Process noise covariance Q (n_states x n_states)
        observation_covariance: Measurement noise covariance R (n_observations x n_observations)
    """

    def __init__(
        self,
        transition_matrices: np.ndarray,
        observation_matrices: np.ndarray,
        initial_state_mean: np.ndarray,
        initial_state_covariance: np.ndarray,
        transition_covariance: np.ndarray,
        observation_covariance: np.ndarray,
    ):
        # Store parameters as numpy arrays
        self.F = np.asarray(transition_matrices)
        self.H = np.asarray(observation_matrices)
        self.x = np.asarray(initial_state_mean)
        self.P = np.asarray(initial_state_covariance)
        self.Q = np.asarray(transition_covariance)
        self.R = np.asarray(observation_covariance)

        # Validate dimensions
        self.n_states = self.x.shape[0]
        self.n_observations = self.H.shape[0]

        assert self.F.shape == (self.n_states, self.n_states), "F must be n_states x n_states"
        assert self.H.shape == (self.n_observations, self.n_states), "H must be n_obs x n_states"
        assert self.P.shape == (self.n_states, self.n_states), "P must be n_states x n_states"
        assert self.Q.shape == (self.n_states, self.n_states), "Q must be n_states x n_states"
        assert self.R.shape == (self.n_observations, self.n_observations), "R must be n_obs x n_obs"

    def predict(
        self,
        state_mean: np.ndarray,
        state_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter prediction step.

        Predicts the next state and covariance using the process model:
            x̂_{k|k-1} = F * x_{k-1|k-1}
            P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

        Args:
            state_mean: Current state estimate x_{k-1|k-1} (n_states,)
            state_covariance: Current covariance P_{k-1|k-1} (n_states x n_states)

        Returns:
            Tuple of (predicted_state_mean, predicted_state_covariance)
        """
        # x̂_{k|k-1} = F * x_{k-1|k-1}
        predicted_state_mean = self.F @ state_mean

        # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        predicted_state_covariance = self.F @ state_covariance @ self.F.T + self.Q

        return predicted_state_mean, predicted_state_covariance

    def update(
        self,
        predicted_state_mean: np.ndarray,
        predicted_state_covariance: np.ndarray,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter update (correction) step.

        Updates the predicted state with the measurement:
            ỹ_k = z_k - H * x̂_{k|k-1}                    (innovation)
            S_k = H * P_{k|k-1} * H^T + R                 (innovation covariance)
            K_k = P_{k|k-1} * H^T * S_k^{-1}              (Kalman gain)
            x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k              (updated state)
            P_{k|k} = (I - K_k * H) * P_{k|k-1}           (updated covariance)

        Args:
            predicted_state_mean: Predicted state x̂_{k|k-1} (n_states,)
            predicted_state_covariance: Predicted covariance P_{k|k-1} (n_states x n_states)
            observation: Measurement z_k (n_observations,)

        Returns:
            Tuple of (filtered_state_mean, filtered_state_covariance)
        """
        # ỹ_k = z_k - H * x̂_{k|k-1}  (innovation/measurement residual)
        innovation = observation - (self.H @ predicted_state_mean)

        # S_k = H * P_{k|k-1} * H^T + R  (innovation covariance)
        innovation_covariance = self.H @ predicted_state_covariance @ self.H.T + self.R

        # K_k = P_{k|k-1} * H^T * S_k^{-1}  (Kalman gain)
        kalman_gain = predicted_state_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)

        # x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k  (updated state estimate)
        filtered_state_mean = predicted_state_mean + kalman_gain @ innovation

        # P_{k|k} = (I - K_k * H) * P_{k|k-1}  (updated covariance)
        # Using Joseph form for numerical stability:
        # P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(self.n_states) - kalman_gain @ self.H
        filtered_state_covariance = I_KH @ predicted_state_covariance @ I_KH.T + \
                                    kalman_gain @ self.R @ kalman_gain.T

        return filtered_state_mean, filtered_state_covariance

    def filter_update(
        self,
        filtered_state_mean: np.ndarray,
        filtered_state_covariance: np.ndarray,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one complete predict + update cycle.

        This is the interface expected by our existing code. It combines
        the prediction and update steps into a single operation.

        Args:
            filtered_state_mean: Current posterior state x_{k-1|k-1} (n_states,)
            filtered_state_covariance: Current posterior covariance P_{k-1|k-1} (n_states x n_states)
            observation: New measurement z_k (n_observations,)

        Returns:
            Tuple of (new_filtered_state_mean, new_filtered_state_covariance)
        """
        # Predict step
        predicted_mean, predicted_cov = self.predict(filtered_state_mean, filtered_state_covariance)

        # Update step
        filtered_mean, filtered_cov = self.update(predicted_mean, predicted_cov, observation)

        return filtered_mean, filtered_cov

    def filter(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter to a sequence of observations.

        Processes multiple measurements sequentially, starting from the
        initial state (x_0, P_0) provided in the constructor.

        Args:
            observations: Array of measurements, shape (n_timesteps, n_observations)
                         or (n_observations,) for a single measurement

        Returns:
            Tuple of:
                - filtered_state_means: Array of state estimates, shape (n_timesteps, n_states)
                - filtered_state_covariances: Array of covariances, shape (n_timesteps, n_states, n_states)
        """
        # Handle single observation (reshape to 2D)
        observations = np.atleast_2d(observations)
        n_timesteps = observations.shape[0]

        # Initialize output arrays
        filtered_state_means = np.zeros((n_timesteps, self.n_states))
        filtered_state_covariances = np.zeros((n_timesteps, self.n_states, self.n_states))

        # Start with initial state
        current_mean = self.x.copy()
        current_cov = self.P.copy()

        # Process each observation
        for t in range(n_timesteps):
            # Predict
            predicted_mean, predicted_cov = self.predict(current_mean, current_cov)

            # Update
            current_mean, current_cov = self.update(predicted_mean, predicted_cov, observations[t])

            # Store results
            filtered_state_means[t] = current_mean
            filtered_state_covariances[t] = current_cov

        return filtered_state_means, filtered_state_covariances
