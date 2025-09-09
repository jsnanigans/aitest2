"""
Utility functions for Kalman filter parameter learning and analysis.
"""

from pykalman import KalmanFilter
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def learn_parameters_from_data(
    weight_measurements: List[float],
    timestamps: Optional[List] = None
) -> Dict[str, Any]:
    """
    Learn optimal Kalman filter parameters from historical data.

    Args:
        weight_measurements: List of weight measurements
        timestamps: Optional list of timestamps

    Returns:
        Dictionary with learned parameters
    """
    if not weight_measurements or len(weight_measurements) < 2:
        raise ValueError("Need at least 2 measurements for parameter learning")

    measurements_array = np.array(weight_measurements).reshape(-1, 1)

    # Calculate empirical trend
    if timestamps:
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 86400.0
                      for i in range(1, len(timestamps))]
        weight_diffs = [weight_measurements[i] - weight_measurements[i-1]
                        for i in range(1, len(weight_measurements))]
        trends = [wd / td if td > 0 else 0 for wd, td in zip(weight_diffs, time_diffs)]
        initial_trend = np.median(trends) if trends else 0.0
    else:
        trends = [weight_measurements[i] - weight_measurements[i-1]
                  for i in range(1, len(weight_measurements))]
        initial_trend = np.median(trends) if trends else 0.0

    # Initialize Kalman filter for learning
    kf = KalmanFilter(
        initial_state_mean=[np.median(weight_measurements), initial_trend],
        initial_state_covariance=[[10.0, 0.0], [0.0, 0.1]],
        transition_matrices=[[1.0, 1.0], [0.0, 1.0]],
        observation_matrices=[[1.0, 0.0]],
        n_dim_state=2,
        n_dim_obs=1
    )

    # Learn parameters using EM algorithm
    kf = kf.em(measurements_array, n_iter=10)

    if (kf.transition_covariance is not None and
        kf.observation_covariance is not None and
        kf.initial_state_mean is not None and
        kf.initial_state_covariance is not None):

        trans_cov = kf.transition_covariance
        obs_cov = kf.observation_covariance
        init_mean = kf.initial_state_mean
        init_cov = kf.initial_state_covariance

        learned_params = {
            'Q_weight': float(trans_cov[0, 0]),
            'Q_trend': float(trans_cov[1, 1]) if trans_cov.shape[0] > 1 else 0.01,
            'R': float(obs_cov[0, 0]),
            'initial_weight': float(init_mean[0]),
            'initial_trend': float(init_mean[1]) if len(init_mean) > 1 else initial_trend,
            'initial_cov_weight': float(init_cov[0, 0]),
            'initial_cov_trend': float(init_cov[1, 1]) if init_cov.shape[0] > 1 else 0.1
        }

        logger.info(f"Learned parameters - Q_weight: {learned_params['Q_weight']:.4f}, "
                   f"Q_trend: {learned_params['Q_trend']:.6f}, R: {learned_params['R']:.4f}")

        return learned_params

    # Fallback parameters
    return {
        'Q_weight': 0.01,
        'Q_trend': 0.001,
        'R': 2.0,
        'initial_weight': float(np.median(weight_measurements)),
        'initial_trend': initial_trend,
        'initial_cov_weight': 10.0,
        'initial_cov_trend': 0.1
    }