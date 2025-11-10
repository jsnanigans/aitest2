/**
 * Debug script to trace kalman_variance calculation step-by-step
 * to find exact divergence between Python and TypeScript
 */

import { processMeasurement } from '../typescript_lib/src/weight-processor-lib/core/processing/processor';
import { InMemoryStore } from '../typescript_lib/src/weight-processor-lib/core/database/memory_store';

async function debugKalmanVariance() {
  console.log('='.repeat(80));
  console.log('DEBUG: Kalman Variance Calculation');
  console.log('='.repeat(80));

  // Load config
  const configPath = `${import.meta.dir}/../typescript_lib/config.json`;
  const config = await Bun.file(configPath).json();

  // Create store
  const store = new InMemoryStore();
  const userId = 'debug-user:debug-device';

  // Fixed timestamps for reproducibility
  const timestamp1 = new Date('2025-11-10T10:00:00.000Z');
  const timestamp2 = new Date(timestamp1.getTime() + 86400000); // 1 day later

  console.log('\n' + '='.repeat(80));
  console.log('MEASUREMENT 1: Initialize Kalman filter');
  console.log('='.repeat(80));
  console.log(`Weight: 70.0 kg`);
  console.log(`Timestamp: ${timestamp1.toISOString()}`);
  console.log(`Source: withings`);

  const result1 = await processMeasurement(
    userId,
    70.0,
    timestamp1,
    'withings',
    config,
    'kg',
    store,
    null
  );

  console.log('\nResult 1:');
  console.log(`  filtered_weight: ${result1.filtered_weight}`);
  console.log(`  trend: ${result1.trend}`);
  console.log(`  kalman_variance: ${result1.kalman_variance}`);
  console.log(`  kalman_confidence_upper: ${result1.kalman_confidence_upper}`);
  console.log(`  kalman_confidence_lower: ${result1.kalman_confidence_lower}`);

  // Get state after first measurement
  const state1 = await store.getState(userId);
  console.log('\nState after measurement 1:');
  console.log(`  last_state shape: ${state1?.last_state?.length} x ${state1?.last_state?.[0]?.rows}x${state1?.last_state?.[0]?.columns}`);
  console.log(`  last_state[0]: [${state1?.last_state?.[0]?.to1DArray()}]`);
  console.log(`  last_state[1]: [${state1?.last_state?.[1]?.to1DArray()}]`);
  console.log(`  last_covariance shape: ${state1?.last_covariance?.length}`);
  console.log(`  last_covariance[0][0,0]: ${state1?.last_covariance?.[0]?.get(0, 0)}`);
  console.log(`  last_covariance[1][0,0]: ${state1?.last_covariance?.[1]?.get(0, 0)}`);

  if (state1?.kalman_params) {
    console.log('\nKalman params:');
    console.log(`  observation_covariance: ${JSON.stringify(state1.kalman_params.observation_covariance)}`);
    console.log(`  transition_covariance: ${JSON.stringify(state1.kalman_params.transition_covariance)}`);
  }

  console.log('\n' + '='.repeat(80));
  console.log('MEASUREMENT 2: Update with prediction step (1 day later)');
  console.log('='.repeat(80));
  console.log(`Weight: 70.1 kg`);
  console.log(`Timestamp: ${timestamp2.toISOString()}`);
  console.log(`Time delta: 1 day`);
  console.log(`Source: withings`);

  // Enable verbose logging for this measurement
  process.env.VERBOSE_LOGGING = 'true';

  const result2 = await processMeasurement(
    userId,
    70.1,
    timestamp2,
    'withings',
    config,
    'kg',
    store,
    null
  );

  process.env.VERBOSE_LOGGING = 'false';

  console.log('\nResult 2:');
  console.log(`  filtered_weight: ${result2.filtered_weight}`);
  console.log(`  trend: ${result2.trend}`);
  console.log(`  kalman_variance: ${result2.kalman_variance}`);
  console.log(`  innovation: ${result2.innovation}`);
  console.log(`  normalized_innovation: ${result2.normalized_innovation}`);
  console.log(`  kalman_confidence_upper: ${result2.kalman_confidence_upper}`);
  console.log(`  kalman_confidence_lower: ${result2.kalman_confidence_lower}`);

  // Get final state
  const state2 = await store.getState(userId);
  console.log('\nState after measurement 2:');
  console.log(`  last_state[0]: [${state2?.last_state?.[0]?.to1DArray()}]`);
  console.log(`  last_state[1]: [${state2?.last_state?.[1]?.to1DArray()}]`);
  console.log(`  last_covariance[0][0,0]: ${state2?.last_covariance?.[0]?.get(0, 0)}`);
  console.log(`  last_covariance[1][0,0]: ${state2?.last_covariance?.[1]?.get(0, 0)}`);

  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`TypeScript kalman_variance (measurement 2): ${result2.kalman_variance}`);
  console.log(`Expected Python value: 4.00252950373697`);
  console.log(`Difference: ${result2.kalman_variance ? (4.00252950373697 - result2.kalman_variance).toFixed(6) : 'N/A'}`);
  console.log(`Relative error: ${result2.kalman_variance ? ((Math.abs(4.00252950373697 - result2.kalman_variance) / 4.00252950373697) * 100).toFixed(2) : 'N/A'}%`);
}

// Run the debug
debugKalmanVariance().catch(console.error);
