#!/usr/bin/env python3
"""
Direct test of Python KalmanFilter to debug covariance calculation
"""
import sys
from pathlib import Path
import numpy as np

# Add python_lib to path
python_lib_path = Path(__file__).parent / "python_lib" / "src"
sys.path.insert(0, str(python_lib_path))

from weight_processor_lib.core.processing.kalman_filter import KalmanFilter

# Setup from test case - using actual values from config
time_delta_days = 1.0
obs_cov = 5.0
initial_variance = 3.64  # 0.364 * 10 from config

kalman = KalmanFilter(
    transition_matrices=np.array([[1, time_delta_days], [0, 1]]),
    observation_matrices=np.array([[1, 0]]),
    initial_state_mean=np.array([70.0, 0.0]),
    initial_state_covariance=np.array([[initial_variance, 0.0], [0.0, 0.001]]),
    transition_covariance=np.array([[0.04, 0.0], [0.0, 0.005]]),
    observation_covariance=np.array([[obs_cov]]),
)

# First measurement
observation1 = np.array([[70.0]])
filtered_means1, filtered_covs1 = kalman.filter(observation1)

print("=== After first measurement ===")
print("State:", filtered_means1[0])
print("Covariance[0,0]:", filtered_covs1[0][0, 0])
print()

# Second measurement using filter_update
current_state = filtered_means1[0]
current_covariance = filtered_covs1[0]
observation2 = np.array([70.1])

print("=== Before second update ===")
print("Current state:", current_state)
print("Current covariance[0,0]:", current_covariance[0, 0])
print()

# Manual step-by-step to debug
F = kalman.F
Q = kalman.Q
H = kalman.H
R = kalman.R

# Predict
predicted_state = F @ current_state
predicted_cov = F @ current_covariance @ F.T + Q

print("=== After prediction ===")
print("Predicted state:", predicted_state)
print("Predicted cov[0,0]:", predicted_cov[0, 0])
print()

# Update - step by step
innovation = observation2 - (H @ predicted_state)
innovation_cov = H @ predicted_cov @ H.T + R
kalman_gain = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)

print("=== Update calculations ===")
print("Innovation:", innovation)
print("Innovation cov:", innovation_cov)
print("Kalman gain:", kalman_gain.T)
print()

# Joseph form
I_KH = np.eye(2) - kalman_gain @ H
term1 = I_KH @ predicted_cov @ I_KH.T
term2 = kalman_gain @ R @ kalman_gain.T

print("=== Joseph form terms ===")
print("I_KH:")
print(I_KH)
print("Term1 (I_KH * P * I_KH^T)[0,0]:", term1[0, 0])
print("Term2 (K * R * K^T)[0,0]:", term2[0, 0])
print()

filtered_cov_manual = term1 + term2
print("=== Final covariance (manual) ===")
print("Filtered cov[0,0]:", filtered_cov_manual[0, 0])
print()

# Now call filter_update
filtered_state2, filtered_cov2 = kalman.filter_update(current_state, current_covariance, observation2)

print("=== filter_update result ===")
print("Filtered state:", filtered_state2)
print("Filtered cov[0,0]:", filtered_cov2[0, 0])
print()

print("=== Comparison ===")
print(f"Manual: {filtered_cov_manual[0, 0]:.6f}")
print(f"filter_update: {filtered_cov2[0, 0]:.6f}")
print(f"Expected (Python from test): 4.003")
print(f"Expected (TypeScript from test): 2.380")
