/**
 * Debug getAdaptiveKalmanParams to see what multipliers are actually being applied
 */

import { getAdaptiveKalmanParams } from '../typescript_lib/src/weight-processor-lib/core/processing/kalman';

async function debugAdaptiveParams() {
  console.log('='.repeat(80));
  console.log('DEBUG: getAdaptiveKalmanParams');
  console.log('='.repeat(80));

  // Load config
  const configPath = `${import.meta.dir}/../typescript_lib/config.json`;
  const config = await Bun.file(configPath).json();
  const kalmanConfig = config.kalman;

  console.log('\nBase Kalman Config:');
  console.log(`  initial_variance: ${kalmanConfig.initial_variance}`);
  console.log(`  transition_covariance_weight: ${kalmanConfig.transition_covariance_weight}`);
  console.log(`  transition_covariance_trend: ${kalmanConfig.transition_covariance_trend}`);
  console.log(`  observation_covariance: ${kalmanConfig.observation_covariance}`);

  // Simulate reset with reset_parameters
  const resetTimestamp = new Date('2025-11-10T10:00:00.000Z');
  const currentTimestamp = new Date('2025-11-11T10:00:00.000Z'); // 1 day later

  // Create a mock state with reset_parameters set (as would be done during reset)
  const mockState = {
    reset_parameters: {
      initial_variance_multiplier: 10,
      weight_noise_multiplier: 50,
      trend_noise_multiplier: 50,
      observation_noise_multiplier: 20,
      adaptation_measurements: 10,
      adaptation_days: 10,
    },
  };

  console.log('\nReset Parameters (from initial reset):');
  console.log(JSON.stringify(mockState.reset_parameters, null, 2));

  // Call getAdaptiveKalmanParams
  console.log('\n' + '='.repeat(80));
  console.log('Calling getAdaptiveKalmanParams with state');
  console.log('='.repeat(80));

  const adaptiveParams = getAdaptiveKalmanParams(
    resetTimestamp,
    currentTimestamp,
    kalmanConfig,
    7,
    mockState as any
  );

  console.log('\nAdaptive Params Returned:');
  console.log(JSON.stringify(adaptiveParams, null, 2));

  console.log('\n' + '='.repeat(80));
  console.log('EXPECTED vs ACTUAL');
  console.log('='.repeat(80));

  const expected = {
    initial_variance: 3.64,  // 0.364 * 10
    transition_covariance_weight: 0.9,  // 0.018 * 50
    transition_covariance_trend: 0.006,  // 0.00012 * 50
    observation_covariance: 100,  // 5 * 20
  };

  console.log('\nExpected (with multipliers):');
  console.log(`  initial_variance: ${expected.initial_variance}`);
  console.log(`  transition_covariance_weight: ${expected.transition_covariance_weight}`);
  console.log(`  transition_covariance_trend: ${expected.transition_covariance_trend}`);
  console.log(`  observation_covariance: ${expected.observation_covariance}`);

  console.log('\nActual (from getAdaptiveKalmanParams):');
  console.log(`  initial_variance: ${adaptiveParams.initial_variance}`);
  console.log(`  transition_covariance_weight: ${adaptiveParams.transition_covariance_weight}`);
  console.log(`  transition_covariance_trend: ${adaptiveParams.transition_covariance_trend}`);
  console.log(`  observation_covariance: ${adaptiveParams.observation_covariance}`);

  console.log('\nDifferences:');
  console.log(`  initial_variance: ${adaptiveParams.initial_variance === expected.initial_variance ? 'MATCH ✓' : `MISMATCH ✗ (${adaptiveParams.initial_variance} vs ${expected.initial_variance})`}`);
  console.log(`  transition_covariance_weight: ${adaptiveParams.transition_covariance_weight === expected.transition_covariance_weight ? 'MATCH ✓' : `MISMATCH ✗ (${adaptiveParams.transition_covariance_weight} vs ${expected.transition_covariance_weight})`}`);
  console.log(`  transition_covariance_trend: ${adaptiveParams.transition_covariance_trend === expected.transition_covariance_trend ? 'MATCH ✓' : `MISMATCH ✗ (${adaptiveParams.transition_covariance_trend} vs ${expected.transition_covariance_trend})`}`);
  console.log(`  observation_covariance: ${adaptiveParams.observation_covariance === expected.observation_covariance ? 'MATCH ✓' : `MISMATCH ✗ (${adaptiveParams.observation_covariance} vs ${expected.observation_covariance})`}`);
}

debugAdaptiveParams().catch(console.error);
