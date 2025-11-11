/**
 * Debug script that processes measurements one at a time to inspect state
 */

// Import processor and store directly
import { processMeasurement } from '../typescript_lib/src/weight-processor-lib/core/processing/processor';
import { InMemoryStore } from '../typescript_lib/src/weight-processor-lib/core/database/memory_store';

async function main() {
  // Enable debug logging
  process.env.DEBUG_ADAPTIVE = 'true';

  const config = await Bun.file(`${import.meta.dir}/../typescript_lib/config.json`).json();

  const userId = 'test-device-pred:test-user-pred';
  const stateStore = new InMemoryStore();

  const baseTimestamp = Date.now();

  console.log('=== Processing Measurement 1 (70.0 kg) ===\n');

  const result1 = await processMeasurement(
    userId,
    70.0,
    new Date(baseTimestamp),
    'withings',
    config,
    'kg',
    stateStore,
    1.67  // height
  );

  const state1 = await stateStore.getState(userId);
  console.log('Result 1 kalman_variance:', result1.kalman_variance);
  console.log('State 1 measurements_since_reset:', state1?.measurements_since_reset);
  console.log('State 1 transition_covariance:', JSON.stringify(state1?.kalman_params?.transition_covariance));
  console.log('State 1 reset_type:', state1?.reset_type);
  console.log('State 1 reset_events length:', state1?.reset_events?.length);
  console.log('State 1 reset_timestamp:', state1?.reset_timestamp);

  console.log('\n=== Processing Measurement 2 (70.1 kg, +1 day) ===\n');

  const result2 = await processMeasurement(
    userId,
    70.1,
    new Date(baseTimestamp + 86400000),
    'withings',
    config,
    'kg',
    stateStore,
    1.67
  );

  const state2 = await stateStore.getState(userId);
  console.log('Result 2 kalman_variance:', result2.kalman_variance);
  console.log('State 2 measurements_since_reset:', state2?.measurements_since_reset);
  console.log('State 2 transition_covariance:', JSON.stringify(state2?.kalman_params?.transition_covariance));

  console.log('\n=== Expected vs Actual ===');
  console.log('Expected transition_covariance for measurements_since_reset=1:');
  console.log('  weight: 0.6092222806034339');
  console.log('  trend:  0.004061481870689559');
  console.log('Actual transition_covariance:');
  console.log('  weight:', state2?.kalman_params?.transition_covariance?.[0]?.[0]);
  console.log('  trend: ', state2?.kalman_params?.transition_covariance?.[1]?.[1]);
}

main().catch(console.error);
