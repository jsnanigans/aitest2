"""Debug script to trace covariance calculations"""
import numpy as np
from datetime import datetime
from weight_values.src.core.processing.kalman import KalmanFilterManager
from weight_values.src.core.constants import KALMAN_DEFAULTS

# Test parameters (matching the logs)
weight = 104.326160
timestamp = datetime.fromisoformat("2025-01-14T00:00:00+00:00")
source = "https://api.iglucose.com"

# Adaptive parameters for INITIAL reset (from config.toml)
initial_variance_multiplier = 10
weight_noise_multiplier = 50
trend_noise_multiplier = 50  # Fixed: was 500 in the code but config says 50
obs_noise_multiplier = 20    # Fixed: was 0.3 in the code but config says 20

# Config overrides defaults
base_initial_var = 0.364
base_weight_cov = 0.018
base_trend_cov = 0.00012  # Config override: not 0.00015
base_obs_cov = 5.0         # Config override: not 3.4

# Apply multipliers
initial_variance = base_initial_var * initial_variance_multiplier
Q_weight = base_weight_cov * weight_noise_multiplier
Q_trend = base_trend_cov * trend_noise_multiplier
# Source noise multiplier = 3.0 for this source
observation_covariance = base_obs_cov * obs_noise_multiplier * 3.0

print(f"Initial variance: {initial_variance}")
print(f"Q_weight: {Q_weight}")
print(f"Q_trend: {Q_trend}")
print(f"Observation covariance: {observation_covariance}")
print()

# Initialize state
kalman_config = {
    "initial_variance": initial_variance,
    "transition_covariance_weight": Q_weight,
    "transition_covariance_trend": Q_trend,
    "observation_covariance": observation_covariance,
}

state = KalmanFilterManager.initialize_immediate(
    weight, timestamp, kalman_config, observation_covariance
)

print("After initialization:")
print(f"last_state shape: {state['last_state'].shape}")
print(f"last_state: {state['last_state']}")
print(f"last_covariance shape: {state['last_covariance'].shape}")
print(f"last_covariance: {state['last_covariance']}")
print()

# Now call predict_next_state (this is what happens in Step 5)
kalman_prediction, innovation_covariance = KalmanFilterManager.predict_next_state(
    state, timestamp
)

print("After predict_next_state:")
print(f"kalman_prediction: {kalman_prediction}")
print(f"innovation_covariance: {innovation_covariance}")
print()

# Let's manually trace through the calculation
last_state = state["last_state"]
last_covariance = state["last_covariance"]

print("Manual calculation:")
print(f"last_state shape: {last_state.shape}")
print(f"last_covariance shape: {last_covariance.shape}")

if len(last_state.shape) > 1:
    posterior_state = last_state[-1]
    posterior_covariance = last_covariance[-1]
else:
    posterior_state = last_state
    posterior_covariance = last_covariance

print(f"posterior_state: {posterior_state}")
print(f"posterior_covariance: {posterior_covariance}")
print()

# Time delta = 0 (same timestamp)
time_delta_days = 0
F = np.array([[1, time_delta_days], [0, 1]])
print(f"F (transition matrix): {F}")

Q = np.array([[Q_weight, 0], [0, Q_trend]])
print(f"Q (process noise): {Q}")

# Predict covariance
predicted_covariance = F @ posterior_covariance @ F.T + Q
print(f"predicted_covariance: {predicted_covariance}")
print(f"predicted_covariance[0,0]: {predicted_covariance[0, 0]}")

# Innovation covariance
R = observation_covariance
manual_innovation_cov = predicted_covariance[0, 0] + R
print(f"R (observation noise): {R}")
print(f"manual_innovation_cov: {manual_innovation_cov}")
