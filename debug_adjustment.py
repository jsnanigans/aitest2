"""Debug the innovation_covariance adjustment logic"""

# From the logs, after predict_next_state we have:
innovation_covariance = 304.54

# From processor.py lines 382-398:
# The code tries to adjust for source-specific noise multiplier

# noise_multiplier for "https://api.iglucose.com" = 3.0
noise_multiplier = 3.0

# Get base_obs_cov from kalman_params
# But kalman_params already has observation_covariance = 300 (after applying obs_noise_multiplier!)
base_obs_cov_from_params = 300.0  # This is WRONG - it's already multiplied

# The adjustment calculation:
predicted_cov_00 = innovation_covariance - base_obs_cov_from_params
print(f"predicted_cov_00 = {innovation_covariance} - {base_obs_cov_from_params} = {predicted_cov_00}")

adjusted_innovation_cov = predicted_cov_00 + (base_obs_cov_from_params * noise_multiplier)
print(f"adjusted_innovation_cov = {predicted_cov_00} + ({base_obs_cov_from_params} * {noise_multiplier}) = {adjusted_innovation_cov}")

print()
print("BUG: base_obs_cov is already 300 (after obs_noise_multiplier), not the base 5.0")
print("This causes double application of the noise multiplier!")
print()

# What it SHOULD be:
base_obs_cov_correct = 5.0
predicted_cov_00_correct = innovation_covariance - 300  # Remove the full R that was added
print(f"Correct calculation:")
print(f"predicted_cov_00 = {innovation_covariance} - 300 = {predicted_cov_00_correct}")

# Then add back with the noise multiplier
adjusted_correct = predicted_cov_00_correct + (base_obs_cov_correct * 20 * noise_multiplier)
print(f"adjusted (correct) = {predicted_cov_00_correct} + ({base_obs_cov_correct} * 20 * {noise_multiplier}) = {adjusted_correct}")
