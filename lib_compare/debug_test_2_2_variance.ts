/**
 * Minimal debug script to isolate the Test 2.2 variance discrepancy
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

  console.log('=== Running Python Implementation ===');
  const pythonResult = await pythonWrapper.processMeasurements({
    deviceId,
    userId,
    measurements,
  });

  console.log('\n=== Running TypeScript Implementation ===');
  const tsResult = await typescriptWrapper.processMeasurements({
    deviceId,
    userId,
    measurements,
  });

  console.log('\n=== Comparison ===');
  console.log('\nMeasurement 1 (initialization):');
  console.log('  Python variance:', pythonResult.results[0]?.kalman_variance);
  console.log('  TypeScript variance:', tsResult.results[0]?.kalman_variance);
  console.log('  Match:', pythonResult.results[0]?.kalman_variance === tsResult.results[0]?.kalman_variance);

  console.log('\nMeasurement 2 (prediction + update):');
  console.log('  Python variance:', pythonResult.results[1]?.kalman_variance);
  console.log('  TypeScript variance:', tsResult.results[1]?.kalman_variance);
  console.log('  Difference:', Math.abs(pythonResult.results[1]?.kalman_variance - tsResult.results[1]?.kalman_variance));
  console.log('  Relative error:', ((Math.abs(pythonResult.results[1]?.kalman_variance - tsResult.results[1]?.kalman_variance) / pythonResult.results[1]?.kalman_variance) * 100).toFixed(2) + '%');

  console.log('\n=== Final State Comparison ===');
  console.log('\nPython final state:');
  console.log('  measurements_since_reset:', pythonResult.finalState?.measurements_since_reset);
  console.log('  reset_type:', pythonResult.finalState?.reset_type);
  console.log('  kalman_params.transition_covariance:', JSON.stringify(pythonResult.finalState?.kalman_params?.transition_covariance));
  console.log('  kalman_params.observation_covariance:', JSON.stringify(pythonResult.finalState?.kalman_params?.observation_covariance));

  console.log('\nTypeScript final state:');
  console.log('  measurements_since_reset:', tsResult.finalState?.measurements_since_reset);
  console.log('  reset_type:', tsResult.finalState?.reset_type);
  console.log('  kalman_params.transition_covariance:', JSON.stringify(tsResult.finalState?.kalman_params?.transition_covariance));
  console.log('  kalman_params.observation_covariance:', JSON.stringify(tsResult.finalState?.kalman_params?.observation_covariance));

  console.log('\n=== Full Results ===');
  console.log('\nPython results[1]:', JSON.stringify(pythonResult.results[1], null, 2));
  console.log('\nTypeScript results[1]:', JSON.stringify(tsResult.results[1], null, 2));
}

main().catch(console.error);
