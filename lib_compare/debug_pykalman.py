"""
Debug script to understand pykalman's filter_update behavior
"""
import numpy as np
from pykalman import KalmanFilter

# Setup from test case
time_delta_days = 1.0
obs_cov = 5.0  # observation_covariance

kalman = KalmanFilter(
    transition_matrices=np.array([[1, time_delta_days], [0, 1]]),
    observation_matrices=np.array([[1, 0]]),
    initial_state_mean=np.array([70.0, 0.0]),
    initial_state_covariance=np.array([[5.0, 0.0], [0.0, 0.001]]),
    transition_covariance=np.array([[0.04, 0.0], [0.0, 0.005]]),
    observation_covariance=np.array([[obs_cov]]),
)

# First measurement
observation1 = np.array([[70.0]])
filtered_state_means1, filtered_state_covariances1 = kalman.filter(observation1)

print("After first measurement:")
print("State:", filtered_state_means1[0])
print("Covariance:", filtered_state_covariances1[0])
print("Covariance[0,0]:", filtered_state_covariances1[0][0, 0])

# Second measurement using filter_update
current_state = filtered_state_means1[0]
current_covariance = filtered_state_covariances1[0]

observation2 = 70.1

print("\nBefore second update:")
print("Current state:", current_state)
print("Current covariance[0,0]:", current_covariance[0, 0])

# Manual prediction step
F = np.array([[1, time_delta_days], [0, 1]])
Q = np.array([[0.04, 0.0], [0.0, 0.005]])
predicted_state = F @ current_state
predicted_cov = F @ current_covariance @ F.T + Q

print("\nManual prediction:")
print("Predicted state:", predicted_state)
print("Predicted covariance[0,0]:", predicted_cov[0, 0])

# Now call filter_update
filtered_state_mean2, filtered_state_covariance2 = kalman.filter_update(
    current_state,
    current_covariance,
    observation=observation2
)

print("\nAfter filter_update:")
print("Filtered state:", filtered_state_mean2)
print("Filtered covariance:", filtered_state_covariance2)
print("Filtered covariance[0,0]:", filtered_state_covariance2[0, 0])

# Check what pykalman returns
print("\n" + "="*60)
print("KEY QUESTION: Does filter_update return predicted or filtered covariance?")
print(f"Predicted cov[0,0]: {predicted_cov[0, 0]:.6f}")
print(f"filter_update cov[0,0]: {filtered_state_covariance2[0, 0]:.6f}")

if np.isclose(filtered_state_covariance2[0, 0], predicted_cov[0, 0]):
    print("=> filter_update returns PREDICTED covariance!")
else:
    print("=> filter_update returns FILTERED covariance")
