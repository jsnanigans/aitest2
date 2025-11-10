/**
 * Phase 1: High-Level Integration Tests
 * Tests the complete processing pipeline for identical behavior
 */

import { TestRunner, TestCase } from '../../utils/test_runner';
import { pythonWrapper } from '../../utils/python_wrapper';
import { typescriptWrapper } from '../../utils/typescript_wrapper';
import { readFileSync } from 'fs';
import { resolve } from 'path';

// Load fixtures
const fixturesPath = resolve(import.meta.dir, '../../fixtures/all_fixtures.json');
const fixtures = JSON.parse(readFileSync(fixturesPath, 'utf-8'));

/**
 * Test 1: Basic Processing Flow - Single Measurement
 */
const test1: TestCase = {
  name: 'Test 1: Single Measurement Processing',
  description: 'Process a single measurement and verify initialization',
  input: fixtures.singleMeasurement,

  runPython: async () => {
    return await pythonWrapper.processMeasurements({
      deviceId: fixtures.singleMeasurement.deviceId,
      userId: fixtures.singleMeasurement.userId,
      measurements: fixtures.singleMeasurement.measurements,
    });
  },

  runTypeScript: async () => {
    return await typescriptWrapper.processMeasurements({
      deviceId: fixtures.singleMeasurement.deviceId,
      userId: fixtures.singleMeasurement.userId,
      measurements: fixtures.singleMeasurement.measurements,
    });
  },

  comparisonConfig: {
    absoluteTolerance: 1e-10,
    relativeTolerance: 1e-8,
    ignoreKeys: ['created_at', 'updated_at'], // Timestamps will differ
  },
};

/**
 * Test 2: Multi-Measurement Processing
 */
const test2: TestCase = {
  name: 'Test 2: Multi-Measurement Sequence',
  description: 'Process 10 measurements and verify state evolution',
  input: fixtures.basicSequence,

  runPython: async () => {
    return await pythonWrapper.processMeasurements({
      deviceId: fixtures.basicSequence.deviceId,
      userId: fixtures.basicSequence.userId,
      measurements: fixtures.basicSequence.measurements,
    });
  },

  runTypeScript: async () => {
    return await typescriptWrapper.processMeasurements({
      deviceId: fixtures.basicSequence.deviceId,
      userId: fixtures.basicSequence.userId,
      measurements: fixtures.basicSequence.measurements,
    });
  },

  comparisonConfig: {
    absoluteTolerance: 1e-10,
    relativeTolerance: 1e-8,
    ignoreKeys: ['created_at', 'updated_at'],
  },
};

/**
 * Test 3: Reset Scenario
 */
const test3: TestCase = {
  name: 'Test 3: Reset Scenario',
  description: 'Process measurements with a large change that triggers reset',
  input: fixtures.resetScenario,

  runPython: async () => {
    return await pythonWrapper.processMeasurements({
      deviceId: fixtures.resetScenario.deviceId,
      userId: fixtures.resetScenario.userId,
      measurements: fixtures.resetScenario.measurements,
    });
  },

  runTypeScript: async () => {
    return await typescriptWrapper.processMeasurements({
      deviceId: fixtures.resetScenario.deviceId,
      userId: fixtures.resetScenario.userId,
      measurements: fixtures.resetScenario.measurements,
    });
  },

  comparisonConfig: {
    absoluteTolerance: 1e-10,
    relativeTolerance: 1e-8,
    ignoreKeys: ['created_at', 'updated_at'],
  },
};

/**
 * Test 4: Quality Rejection
 */
const test4: TestCase = {
  name: 'Test 4: Quality Rejection',
  description: 'Mix of good and bad measurements - verify rejection logic',
  input: fixtures.qualityRejection,

  runPython: async () => {
    return await pythonWrapper.processMeasurements({
      deviceId: fixtures.qualityRejection.deviceId,
      userId: fixtures.qualityRejection.userId,
      measurements: fixtures.qualityRejection.measurements,
    });
  },

  runTypeScript: async () => {
    return await typescriptWrapper.processMeasurements({
      deviceId: fixtures.qualityRejection.deviceId,
      userId: fixtures.qualityRejection.userId,
      measurements: fixtures.qualityRejection.measurements,
    });
  },

  comparisonConfig: {
    absoluteTolerance: 1e-10,
    relativeTolerance: 1e-8,
    ignoreKeys: ['created_at', 'updated_at'],
  },
};

/**
 * Test 5: State Persistence
 */
const test5: TestCase = {
  name: 'Test 5: State Persistence',
  description: 'Process in batches - verify state persistence works correctly',
  input: fixtures.statePersistence,

  runPython: async () => {
    // Process first batch
    const firstBatch = fixtures.statePersistence.measurements.filter(
      (m: any) => m.metadata?.batch === 'first'
    );
    const firstResult = await pythonWrapper.processMeasurements({
      deviceId: fixtures.statePersistence.deviceId,
      userId: fixtures.statePersistence.userId,
      measurements: firstBatch,
    });

    // Process second batch (state is preserved in memory)
    const secondBatch = fixtures.statePersistence.measurements.filter(
      (m: any) => m.metadata?.batch === 'second'
    );
    const secondResult = await pythonWrapper.processMeasurements({
      deviceId: fixtures.statePersistence.deviceId,
      userId: fixtures.statePersistence.userId,
      measurements: secondBatch,
    });

    // Combine results
    return {
      results: [...firstResult.results, ...secondResult.results],
      finalState: secondResult.finalState,
    };
  },

  runTypeScript: async () => {
    // Process first batch
    const firstBatch = fixtures.statePersistence.measurements.filter(
      (m: any) => m.metadata?.batch === 'first'
    );
    const firstResult = await typescriptWrapper.processMeasurements({
      deviceId: fixtures.statePersistence.deviceId,
      userId: fixtures.statePersistence.userId,
      measurements: firstBatch,
    });

    // Process second batch (state is preserved in memory)
    const secondBatch = fixtures.statePersistence.measurements.filter(
      (m: any) => m.metadata?.batch === 'second'
    );
    const secondResult = await typescriptWrapper.processMeasurements({
      deviceId: fixtures.statePersistence.deviceId,
      userId: fixtures.statePersistence.userId,
      measurements: secondBatch,
    });

    // Combine results
    return {
      results: [...firstResult.results, ...secondResult.results],
      finalState: secondResult.finalState,
    };
  },

  comparisonConfig: {
    absoluteTolerance: 1e-10,
    relativeTolerance: 1e-8,
    ignoreKeys: ['created_at', 'updated_at'],
  },
};

/**
 * Run all Phase 1 tests
 */
async function runPhase1Tests() {
  const runner = new TestRunner('Phase 1: Integration Tests');

  const tests = [test1, test2, test3, test4, test5];

  const results = await runner.runTests(tests);

  // Generate reports
  const reportDir = resolve(import.meta.dir, '../../reports');
  await runner.generateReport(`${reportDir}/phase1_report.md`, results);
  await runner.saveResults(`${reportDir}/phase1_results.json`, results);

  // Exit with appropriate code
  process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runPhase1Tests().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
