/**
 * Debug script to investigate the Kalman variance mismatch in Test 2.2
 */

import { pythonWrapper } from './utils/python_wrapper';
import { typescriptWrapper } from './utils/typescript_wrapper';

const input = {
  deviceId: 'test-device-pred',
  userId: 'test-user-pred',
  measurements: [
    {
      weight_kg: 70.0,
      timestamp: Date.now(),
      source: 'withings',
    },
    {
      weight_kg: 70.1,
      timestamp: Date.now() + 86400000, // 1 day later
      source: 'withings',
    },
  ],
};

async function debug() {
  console.log('Running Python...');
  const pyResult = await pythonWrapper.processMeasurements(input);

  console.log('\nPython Results:');
  console.log('Measurement 1:');
  console.log('  kalman_variance:', pyResult.results[0]?.kalman_variance);
  console.log('  kalman_estimate:', pyResult.results[0]?.kalman_estimate);
  console.log('  filtered_weight:', pyResult.results[0]?.filtered_weight);

  console.log('\nMeasurement 2:');
  console.log('  kalman_variance:', pyResult.results[1]?.kalman_variance);
  console.log('  kalman_estimate:', pyResult.results[1]?.kalman_estimate);
  console.log('  kalman_velocity:', pyResult.results[1]?.kalman_velocity);
  console.log('  filtered_weight:', pyResult.results[1]?.filtered_weight);

  console.log('\n' + '='.repeat(80));
  console.log('Running TypeScript...');
  const tsResult = await typescriptWrapper.processMeasurements(input);

  console.log('\nTypeScript Results:');
  console.log('Measurement 1:');
  console.log('  kalman_variance:', tsResult.results[0]?.kalman_variance);
  console.log('  kalman_estimate:', tsResult.results[0]?.kalman_estimate);
  console.log('  filtered_weight:', tsResult.results[0]?.filtered_weight);

  console.log('\nMeasurement 2:');
  console.log('  kalman_variance:', tsResult.results[1]?.kalman_variance);
  console.log('  kalman_estimate:', tsResult.results[1]?.kalman_estimate);
  console.log('  kalman_velocity:', tsResult.results[1]?.kalman_velocity);
  console.log('  filtered_weight:', tsResult.results[1]?.filtered_weight);

  console.log('\n' + '='.repeat(80));
  console.log('\nDifference in kalman_variance (measurement 2):');
  const pyVar = pyResult.results[1]?.kalman_variance;
  const tsVar = tsResult.results[1]?.kalman_variance;
  console.log('Python:', pyVar);
  console.log('TypeScript:', tsVar);
  console.log('Difference:', Math.abs(pyVar - tsVar));
  console.log('Relative difference:', ((Math.abs(pyVar - tsVar) / pyVar) * 100).toFixed(2) + '%');

  // Check the full covariance matrices
  console.log('\n' + '='.repeat(80));
  console.log('\nFinal State Covariances:');
  console.log('Python last_covariance:');
  console.log(JSON.stringify(pyResult.finalState?.last_covariance, null, 2));
  console.log('\nTypeScript last_covariance:');
  console.log(JSON.stringify(tsResult.finalState?.last_covariance, null, 2));
}

debug().catch(console.error);
