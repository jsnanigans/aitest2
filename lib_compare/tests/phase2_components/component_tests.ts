/**
 * Phase 2: Component-Level Tests
 * Tests individual components in isolation to identify exact differences
 */

import { TestRunner, TestCase } from '../../utils/test_runner';
import { pythonWrapper } from '../../utils/python_wrapper';
import { typescriptWrapper } from '../../utils/typescript_wrapper';

/**
 * Test 2.1: Kalman Filter Initialization
 * Tests ONLY the initialization of the Kalman filter state
 * This isolates variance and initial state setup
 */
const test_2_1_kalman_init: TestCase = {
  name: 'Test 2.1: Kalman Filter Initialization',
  description: 'Verify Kalman filter initializes with correct variance and state',
  input: {
    deviceId: 'test-device-init',
    userId: 'test-user-init',
    measurements: [
      {
        weight_kg: 70.0,
        timestamp: Date.now(),
        source: 'withings',
      },
    ],
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    // Extract only Kalman-related fields from first result
    return {
      kalman_variance: result.results[0]?.kalman_variance,
      kalman_estimate: result.results[0]?.kalman_estimate,
      kalman_velocity: result.results[0]?.kalman_velocity,
      kalman_confidence_upper: result.results[0]?.kalman_confidence_upper,
      kalman_confidence_lower: result.results[0]?.kalman_confidence_lower,
      finalState: {
        kalman_state: result.finalState?.kalman_state,
        kalman_covariance: result.finalState?.kalman_covariance,
      },
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    return {
      kalman_variance: result.results[0]?.kalman_variance,
      kalman_estimate: result.results[0]?.kalman_estimate,
      kalman_velocity: result.results[0]?.kalman_velocity,
      kalman_confidence_upper: result.results[0]?.kalman_confidence_upper,
      kalman_confidence_lower: result.results[0]?.kalman_confidence_lower,
      finalState: {
        kalman_state: result.finalState?.kalman_state,
        kalman_covariance: result.finalState?.kalman_covariance,
      },
    };
  },

  comparisonConfig: {
    absoluteTolerance: 1e-6,
    relativeTolerance: 1e-4,
  },
};

/**
 * Test 2.2: Kalman Filter Prediction Step
 * Tests the prediction step of Kalman filter (time update)
 */
const test_2_2_kalman_prediction: TestCase = {
  name: 'Test 2.2: Kalman Filter Prediction',
  description: 'Verify Kalman filter prediction step (state extrapolation)',
  input: {
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
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    // Extract prediction-related fields from second measurement
    return {
      secondMeasurement: {
        kalman_estimate: result.results[1]?.kalman_estimate,
        kalman_variance: result.results[1]?.kalman_variance,
        kalman_velocity: result.results[1]?.kalman_velocity,
      },
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    return {
      secondMeasurement: {
        kalman_estimate: result.results[1]?.kalman_estimate,
        kalman_variance: result.results[1]?.kalman_variance,
        kalman_velocity: result.results[1]?.kalman_velocity,
      },
    };
  },

  comparisonConfig: {
    absoluteTolerance: 1e-6,
    relativeTolerance: 1e-4,
  },
};

/**
 * Test 2.3: Quality Scoring - Overall Score
 * Tests the quality scoring calculation
 */
const test_2_3_quality_score: TestCase = {
  name: 'Test 2.3: Quality Scoring',
  description: 'Verify quality score calculation matches',
  input: {
    deviceId: 'test-device-quality',
    userId: 'test-user-quality',
    measurements: [
      {
        weight_kg: 70.0,
        timestamp: Date.now(),
        source: 'withings',
      },
      {
        weight_kg: 70.2,
        timestamp: Date.now() + 86400000,
        source: 'withings',
      },
    ],
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    return {
      measurement1: {
        quality_score: result.results[0]?.quality_score,
        quality_components: result.results[0]?.quality_components,
      },
      measurement2: {
        quality_score: result.results[1]?.quality_score,
        quality_components: result.results[1]?.quality_components,
      },
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    return {
      measurement1: {
        quality_score: result.results[0]?.quality_score,
        quality_components: result.results[0]?.quality_components,
      },
      measurement2: {
        quality_score: result.results[1]?.quality_score,
        quality_components: result.results[1]?.quality_components,
      },
    };
  },

  comparisonConfig: {
    absoluteTolerance: 1e-6,
    relativeTolerance: 1e-3, // Allow 0.1% difference
  },
};

/**
 * Test 2.4: Output Structure
 * Tests that the output structure matches (field presence)
 */
const test_2_4_output_structure: TestCase = {
  name: 'Test 2.4: Output Structure',
  description: 'Verify both implementations return the same fields',
  input: {
    deviceId: 'test-device-struct',
    userId: 'test-user-struct',
    measurements: [
      {
        weight_kg: 70.0,
        timestamp: Date.now(),
        source: 'withings',
      },
    ],
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    // Return the keys/structure
    return {
      resultKeys: result.results[0] ? Object.keys(result.results[0]).sort() : [],
      stateKeys: result.finalState ? Object.keys(result.finalState).sort() : [],
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    const tsResultKeys = result.results[0] ? Object.keys(result.results[0]).sort() : [];
    const tsStateKeys = result.finalState ? Object.keys(result.finalState).sort() : [];

    // TypeScript can have extra fields (adaptation_state, version) - filter them for comparison
    const tsStateKeysFiltered = tsStateKeys.filter(k => k !== 'adaptation_state' && k !== 'version');

    return {
      resultKeys: tsResultKeys,
      stateKeys: tsStateKeysFiltered, // Use filtered keys for comparison
    };
  },

  comparisonConfig: {
    strictTypes: true,
  },
};

/**
 * Test 2.5: Reset Detection Logic
 * Tests that reset detection works identically
 */
const test_2_5_reset_detection: TestCase = {
  name: 'Test 2.5: Reset Detection',
  description: 'Verify reset detection triggers identically',
  input: {
    deviceId: 'test-device-reset-det',
    userId: 'test-user-reset-det',
    measurements: [
      { weight_kg: 70.0, timestamp: Date.now(), source: 'withings' },
      { weight_kg: 70.1, timestamp: Date.now() + 86400000, source: 'withings' },
      { weight_kg: 60.0, timestamp: Date.now() + 172800000, source: 'withings' }, // Large drop
    ],
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    return {
      measurement3: {
        was_reset: result.results[2]?.was_reset,
        reset_reason: result.results[2]?.reset_reason,
        reset_type: result.results[2]?.reset_type,
        reset_occurred: result.results[2]?.reset_occurred,
      },
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    return {
      measurement3: {
        was_reset: result.results[2]?.was_reset,
        reset_reason: result.results[2]?.reset_reason,
        reset_type: result.results[2]?.reset_type,
        reset_occurred: result.results[2]?.reset_occurred,
      },
    };
  },

  comparisonConfig: {
    strictTypes: true,
  },
};

/**
 * Test 2.6: Acceptance/Rejection Logic
 * Tests that measurements are accepted/rejected identically
 */
const test_2_6_acceptance: TestCase = {
  name: 'Test 2.6: Acceptance/Rejection Logic',
  description: 'Verify measurements are accepted/rejected identically',
  input: {
    deviceId: 'test-device-accept',
    userId: 'test-user-accept',
    measurements: [
      { weight_kg: 70.0, timestamp: Date.now(), source: 'withings' },
      { weight_kg: 30.0, timestamp: Date.now() + 86400000, source: 'withings' }, // Too low
      { weight_kg: 250.0, timestamp: Date.now() + 172800000, source: 'withings' }, // Too high
      { weight_kg: 70.5, timestamp: Date.now() + 259200000, source: 'withings' }, // Normal
    ],
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    return {
      acceptancePattern: result.results.map((r: any) => ({
        accepted: r.accepted,
        rejection_reason: r.rejection_reason,
      })),
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    return {
      acceptancePattern: result.results.map((r: any) => ({
        accepted: r.accepted,
        rejection_reason: r.rejection_reason,
      })),
    };
  },

  comparisonConfig: {
    strictTypes: false,
  },
};

/**
 * Test 2.7: Timestamp Handling
 * Tests that timestamps are converted correctly
 */
const test_2_7_timestamps: TestCase = {
  name: 'Test 2.7: Timestamp Handling',
  description: 'Verify timestamp conversion is consistent',
  input: {
    deviceId: 'test-device-time',
    userId: 'test-user-time',
    measurements: [
      {
        weight_kg: 70.0,
        timestamp: 1699632000000, // Fixed timestamp: 2023-11-10 12:00:00 UTC
        source: 'withings',
      },
    ],
  },

  runPython: async function() {
    const result = await pythonWrapper.processMeasurements(this.input);
    // Convert timestamps to milliseconds for consistent comparison
    return {
      inputMs: this.input.measurements[0].timestamp,
      resultMs: typeof result.results[0]?.timestamp === 'number'
        ? result.results[0].timestamp
        : new Date(result.results[0]?.timestamp).getTime(),
      stateMs: typeof result.finalState?.last_timestamp === 'number'
        ? result.finalState.last_timestamp
        : new Date(result.finalState?.last_timestamp).getTime(),
    };
  },

  runTypeScript: async function() {
    const result = await typescriptWrapper.processMeasurements(this.input);
    // Convert timestamps to milliseconds for consistent comparison
    return {
      inputMs: this.input.measurements[0].timestamp,
      resultMs: typeof result.results[0]?.timestamp === 'string'
        ? new Date(result.results[0].timestamp).getTime()
        : result.results[0]?.timestamp,
      stateMs: typeof result.finalState?.last_timestamp === 'string'
        ? new Date(result.finalState.last_timestamp).getTime()
        : result.finalState?.last_timestamp,
    };
  },

  comparisonConfig: {
    absoluteTolerance: 1000, // Allow 1 second difference for timezone issues
    relativeTolerance: 1e-6,
  },
};

/**
 * Run all Phase 2 tests
 */
async function runPhase2Tests() {
  // Pre-initialize both wrappers to measure only processing time (fair comparison)
  console.log('🔥 Warming up wrappers...');
  console.log('   → Starting Python server...');
  await pythonWrapper.initialize();
  console.log('   ✓ Python server ready');

  console.log('   → Loading TypeScript modules...');
  await typescriptWrapper.initialize();
  console.log('   ✓ TypeScript modules ready\n');

  const runner = new TestRunner('Phase 2: Component Tests');

  const tests = [
    test_2_1_kalman_init,
    test_2_2_kalman_prediction,
    test_2_3_quality_score,
    test_2_4_output_structure,
    test_2_5_reset_detection,
    test_2_6_acceptance,
    test_2_7_timestamps,
  ];

  const results = await runner.runTests(tests);

  // Generate reports
  const reportDir = `${import.meta.dir}/../../reports`;
  await runner.generateReport(`${reportDir}/phase2_report.md`, results);
  await runner.saveResults(`${reportDir}/phase2_results.json`, results);

  // Cleanup
  await pythonWrapper.cleanup();

  // Exit with appropriate code
  process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runPhase2Tests().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
