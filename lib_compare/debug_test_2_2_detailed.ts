/**
 * Debug script with verbose logging enabled
 */

import { pythonWrapper } from './utils/python_wrapper';
import { typescriptWrapper } from './utils/typescript_wrapper';

async function main() {
  const deviceId = 'test-device-pred';
  const userId = 'test-user-pred';

  const baseTimestamp = Date.now();
  const measurements = [
    {
      weight_kg: 70.0,
      timestamp: baseTimestamp,
      source: 'withings',
    },
    {
      weight_kg: 70.1,
      timestamp: baseTimestamp + 86400000, // 1 day later
      source: 'withings',
    },
  ];

  console.log('=== Running TypeScript with VERBOSE logging ===\n');

  // Enable verbose logging
  process.env.VERBOSE_LOGGING = 'true';

  const tsResult = await typescriptWrapper.processMeasurements({
    deviceId,
    userId,
    measurements,
  });

  console.log('\n=== Results ===');
  console.log('\nMeasurement 1 variance:', tsResult.results[0]?.kalman_variance);
  console.log('Measurement 2 variance:', tsResult.results[1]?.kalman_variance);
  console.log('\nFinal measurements_since_reset:', tsResult.finalState?.measurements_since_reset);
  console.log('Final transition_covariance:', JSON.stringify(tsResult.finalState?.kalman_params?.transition_covariance));
}

main().catch(console.error);
