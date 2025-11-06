/**
 * Debug the innovation_covariance adjustment logic in TypeScript
 */

// From the logs, after predictNextState we should have around 304.54
// But logs show 311.34
const innovation_covariance_from_predict = 304.54;

// noise_multiplier for "https://api.iglucose.com" = 3.0
const noise_multiplier = 3.0;

// Get base_obs_cov from kalman_params
// kalman_params has observation_covariance = [[300]]
const base_obs_cov_from_params = 300.0;

// The adjustment calculation (from processor.ts lines 630-631):
const predicted_cov_00 = innovation_covariance_from_predict - base_obs_cov_from_params;
console.log(`predicted_cov_00 = ${innovation_covariance_from_predict} - ${base_obs_cov_from_params} = ${predicted_cov_00}`);

const adjusted_innovation_cov = predicted_cov_00 + base_obs_cov_from_params * noise_multiplier;
console.log(`adjusted_innovation_cov = ${predicted_cov_00} + (${base_obs_cov_from_params} * ${noise_multiplier}) = ${adjusted_innovation_cov}`);

console.log();
console.log("Expected from Python logic: 904.54");
console.log(`Actual from TS logs: 311.34`);
console.log(`Difference: ${311.34 - 304.54} = 6.8`);
console.log();

// Maybe TS is using a different base_obs_cov?
// If innovation_cov after adjustment is 311.34:
// 311.34 = predicted_cov_00 + (base_obs_cov * 3.0)
// If predicted_cov_00 = innovation_covariance - R_used_in_predict
// Let's work backwards:

// Hypothesis 1: TS is using base R = 100 (not 300)
console.log("Hypothesis: TS might be using a different R value in the adjustment");
const possible_base_R = (311.34 - 4.54) / 3.0;
console.log(`If predicted_cov_00 = 4.54, then base_R = (311.34 - 4.54) / 3.0 = ${possible_base_R}`);

// Hypothesis 2: TS has a different predicted_covariance[0,0]
const ts_predicted_cov_00_needed = 311.34 - (300 * 3.0) + 300;
console.log(`If using same logic as Python, TS predicted_cov[0,0] would be: ${ts_predicted_cov_00_needed}`);
