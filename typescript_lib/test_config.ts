/**
 * Quick test to verify config loading matches Python's config.toml values
 */

import { loadConfig } from './src/weight-processor-lib/core/config';
import { SOURCE_PROFILES, KALMAN_DEFAULTS } from './src/weight-processor-lib/core/constants';

console.log('=== Config Loading Test ===\n');

// Test config loading
const config = loadConfig();
console.log('✓ Config loaded successfully\n');

// Test KALMAN_DEFAULTS
console.log('KALMAN_DEFAULTS:');
console.log(`  initial_variance: ${KALMAN_DEFAULTS.initial_variance} (expected: 0.364)`);
console.log(`  transition_covariance_weight: ${KALMAN_DEFAULTS.transition_covariance_weight} (expected: 0.018)`);
console.log(`  transition_covariance_trend: ${KALMAN_DEFAULTS.transition_covariance_trend} (expected: 0.00012)`);
console.log(`  observation_covariance: ${KALMAN_DEFAULTS.observation_covariance} (expected: 5)`);

const kalmanMatch =
  KALMAN_DEFAULTS.initial_variance === 0.364 &&
  KALMAN_DEFAULTS.transition_covariance_weight === 0.018 &&
  KALMAN_DEFAULTS.transition_covariance_trend === 0.00012 &&
  KALMAN_DEFAULTS.observation_covariance === 5;

console.log(`\n${kalmanMatch ? '✓' : '✗'} Kalman defaults ${kalmanMatch ? 'MATCH' : 'MISMATCH'}\n`);

// Test SOURCE_PROFILES
console.log('SOURCE_PROFILES samples:');

const testSources = [
  { name: 'care-team-upload', expectedNoise: 0.5, expectedPriority: 1 },
  { name: 'patient-upload', expectedNoise: 0.7, expectedPriority: 4 },
  { name: 'patient-device', expectedNoise: 1.0, expectedPriority: 3 },
  { name: 'questionnaire', expectedNoise: 0.8, expectedPriority: 1 },
  { name: 'default', expectedNoise: 1.0, expectedPriority: 999 },
];

let sourceMatches = 0;
for (const test of testSources) {
  const profile = SOURCE_PROFILES[test.name];
  const noiseMatch = profile.noise_multiplier === test.expectedNoise;
  const priorityMatch = profile.priority === test.expectedPriority;
  const match = noiseMatch && priorityMatch;

  console.log(`  ${match ? '✓' : '✗'} ${test.name}:`);
  console.log(`      noise_multiplier: ${profile.noise_multiplier} (expected: ${test.expectedNoise}) ${noiseMatch ? '✓' : '✗'}`);
  console.log(`      priority: ${profile.priority} (expected: ${test.expectedPriority}) ${priorityMatch ? '✓' : '✗'}`);

  if (match) sourceMatches++;
}

console.log(`\n${sourceMatches === testSources.length ? '✓' : '✗'} Source profiles: ${sourceMatches}/${testSources.length} match\n`);

// Test reset parameters from config
console.log('Reset Parameters (from config.json):');
console.log('  SOFT reset:');
console.log(`    weight_noise_multiplier: ${config.kalman.reset?.soft?.weight_noise_multiplier} (expected: 5)`);
console.log(`    trend_noise_multiplier: ${config.kalman.reset?.soft?.trend_noise_multiplier} (expected: 20)`);
console.log(`    adaptation_measurements: ${config.kalman.reset?.soft?.adaptation_measurements} (expected: 15)`);
console.log(`    adaptation_days: ${config.kalman.reset?.soft?.adaptation_days} (expected: 10)`);

const softMatch =
  config.kalman.reset?.soft?.weight_noise_multiplier === 5 &&
  config.kalman.reset?.soft?.trend_noise_multiplier === 20 &&
  config.kalman.reset?.soft?.adaptation_measurements === 15 &&
  config.kalman.reset?.soft?.adaptation_days === 10;

console.log(`\n${softMatch ? '✓' : '✗'} SOFT reset parameters ${softMatch ? 'MATCH' : 'MISMATCH'}\n`);

// Summary
const allMatch = kalmanMatch && sourceMatches === testSources.length && softMatch;
console.log('=== Summary ===');
console.log(`${allMatch ? '✓ ALL TESTS PASSED' : '✗ SOME TESTS FAILED'}`);
console.log(`\nConfig file location: ${process.cwd()}/config.json`);

process.exit(allMatch ? 0 : 1);
