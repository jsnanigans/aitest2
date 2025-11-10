"""
Manual step-by-step Kalman calculation in Python to trace the issue
"""

import numpy as np

print('=' * 80)
print('MANUAL KALMAN FILTER CALCULATION - Python')
print('=' * 80)

# Configuration (from debug outputs)
initial_variance_base = 0.364
transition_cov_weight = 0.018
transition_cov_trend = 0.00012
observation_cov = 5.0

# After initial reset multipliers
initial_variance = 3.64  # 0.364 * 10 (initial_variance_multiplier)
obs_cov_with_multiplier = 100.0  # 5 * 20
trans_cov_weight_with_multiplier = 0.9  # 0.018 * 50
trans_cov_trend_with_multiplier = 0.006  # 0.00012 * 50 (also multiplied!)

print('\nConfiguration:')
print(f'  initial_variance: {initial_variance}')
print(f'  transition_cov_weight (with multiplier): {trans_cov_weight_with_multiplier}')
print(f'  transition_cov_trend (with multiplier): {trans_cov_trend_with_multiplier}')
print(f'  observation_cov (with multiplier): {obs_cov_with_multiplier}')

# Measurement 1: Initialize
print('\n' + '=' * 80)
print('MEASUREMENT 1: Initialize')
print('=' * 80)

weight1 = 70.0
x_post1 = np.array([[weight1], [0.0]])  # Column vector
P_post1 = np.array([[initial_variance, 0], [0, 0.001]])

print(f'\nPosterior state x_post1:')
print(x_post1)
print(f'\nPosterior covariance P_post1:')
print(P_post1)
print(f'P_post1[0,0] = {P_post1[0,0]}')

# Measurement 2: Prediction + Update
print('\n' + '=' * 80)
print('MEASUREMENT 2: Predict + Update (1 day later)')
print('=' * 80)

weight2 = 70.1
time_delta = 1.0

# Build matrices
F = np.array([[1, time_delta], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[trans_cov_weight_with_multiplier, 0], [0, trans_cov_trend_with_multiplier]])
R = np.array([[obs_cov_with_multiplier]])

print(f'\nTransition matrix F:')
print(F)
print(f'\nObservation matrix H:')
print(H)
print(f'\nProcess noise Q:')
print(Q)
print(f'\nMeasurement noise R:')
print(R)

# PREDICTION STEP
print('\n' + '--- PREDICTION STEP ---')
x_pred = F @ x_post1
P_pred = F @ P_post1 @ F.T + Q

print(f'\nPredicted state x_pred:')
print(x_pred)
print(f'\nPredicted covariance P_pred:')
print(P_pred)
print(f'P_pred[0,0] = {P_pred[0,0]}')

# Calculate innovation covariance
S = H @ P_pred @ H.T + R
print(f'\nInnovation covariance S:')
print(S)
print(f'S[0,0] = {S[0,0]}')

# UPDATE STEP
print('\n' + '--- UPDATE STEP ---')
z = np.array([[weight2]])
y = z - (H @ x_pred)  # Innovation
print(f'\nMeasurement z:')
print(z)
print(f'\nInnovation y:')
print(y)

# Kalman gain
K = P_pred @ H.T @ np.linalg.inv(S)
print(f'\nKalman gain K:')
print(K)

# Updated state
x_post2 = x_pred + K @ y
print(f'\nUpdated state x_post2:')
print(x_post2)

# Updated covariance (Joseph form)
I = np.eye(2)
I_KH = I - K @ H

# Standard form (less stable)
P_post2_standard = I_KH @ P_pred

# Joseph form (more stable)
P_post2_joseph = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

print(f'\nUpdated covariance P_post2 (STANDARD FORM):')
print(P_post2_standard)
print(f'P_post2_standard[0,0] = {P_post2_standard[0,0]}')

print(f'\nUpdated covariance P_post2 (JOSEPH FORM):')
print(P_post2_joseph)
print(f'P_post2_joseph[0,0] = {P_post2_joseph[0,0]}')

print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'Initial variance (after meas 1): {P_post1[0,0]}')
print(f'Predicted variance (before meas 2 update): {P_pred[0,0]}')
print(f'Final variance (after meas 2 - Standard): {P_post2_standard[0,0]}')
print(f'Final variance (after meas 2 - Joseph): {P_post2_joseph[0,0]}')
print(f'\nPython actual value: 4.00252950373697')
print(f'Manual Joseph form: {P_post2_joseph[0,0]}')
print(f'Manual Standard form: {P_post2_standard[0,0]}')
print(f'TypeScript value: 2.379729588093491')
