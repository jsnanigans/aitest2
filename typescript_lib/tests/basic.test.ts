/**
 * Basic functionality test for the TypeScript weight processor library
 */

import { test, expect } from 'bun:test';
import {
  InMemoryStore,
  processMeasurement,
  PhysiologicalValidator,
  BMIValidator,
  KalmanFilterManager,
  type ProcessingResult
} from '../src/index';

test('InMemoryStore basic operations', async () => {
  const store = new InMemoryStore();

  // Create and save state
  const state = store.createInitialState();
  expect(state.version).toBe(0);
  expect(state.measurement_history).toEqual([]);

  await store.saveState('user123', state);

  // Retrieve state
  const retrieved = await store.getState('user123');
  expect(retrieved).not.toBeNull();
  expect(retrieved?.version).toBe(0);

  // Delete state
  const deleted = await store.deleteState('user123');
  expect(deleted).toBe(true);

  const afterDelete = await store.getState('user123');
  expect(afterDelete).toBeNull();
});

test('PhysiologicalValidator absolute limits', () => {
  // Valid weight
  const [valid, reason] = PhysiologicalValidator.validateAbsoluteLimits(70.0);
  expect(valid).toBe(true);
  expect(reason).toBeNull();

  // Too low
  const [tooLow, lowReason] = PhysiologicalValidator.validateAbsoluteLimits(20.0);
  expect(tooLow).toBe(false);
  expect(lowReason).toContain('below absolute minimum');

  // Too high
  const [tooHigh, highReason] = PhysiologicalValidator.validateAbsoluteLimits(450.0);
  expect(tooHigh).toBe(false);
  expect(highReason).toContain('above absolute maximum');
});

test('BMIValidator calculate BMI', () => {
  const bmi = BMIValidator.calculateBMI(70, 1.75); // 70kg, 1.75m
  expect(bmi).toBeCloseTo(22.86, 1); // Should be ~22.86
});

test('BMIValidator categorize BMI', () => {
  expect(BMIValidator.categorizeBMI(17.0)).toBe('underweight');
  expect(BMIValidator.categorizeBMI(22.0)).toBe('normal');
  expect(BMIValidator.categorizeBMI(27.0)).toBe('overweight');
  expect(BMIValidator.categorizeBMI(32.0)).toBe('obese');
});

test('BMIValidator unit conversion', () => {
  // Pounds to kg
  const [weightKg, wasConverted, metadata] = BMIValidator.detectAndConvert(
    154.3, // ~70 kg
    'lb',
    1.75,
    'patient-device'
  );

  expect(weightKg).toBeCloseTo(70.0, 1);
  expect(wasConverted).toBe(false); // Regular unit conversion, not BMI conversion
  expect(metadata.conversion).toContain('lb to');
});

test('processMeasurement basic flow', async () => {
  const store = new InMemoryStore();

  const result: ProcessingResult = await processMeasurement(
    'test-user-001',
    70.5,
    new Date(),
    'patient-device',
    {},
    'kg',
    store,
    1.75
  );

  expect(result.accepted).toBeDefined();
  expect(result.filtered_weight).toBeDefined();
  expect(result.quality_score).toBeDefined();
  expect(result.stage).toBeDefined();

  if (result.accepted) {
    expect(result.filtered_weight).toBeGreaterThan(0);
    expect(result.quality_score).toBeGreaterThanOrEqual(0);
    expect(result.quality_score).toBeLessThanOrEqual(1);
  }
});

test('processMeasurement rejects invalid weight', async () => {
  const store = new InMemoryStore();

  // Weight too low (below 30kg absolute minimum)
  const result: ProcessingResult = await processMeasurement(
    'test-user-002',
    15.0,
    new Date(),
    'patient-device',
    {},
    'kg',
    store,
    1.75
  );

  expect(result.accepted).toBe(false);
  if (result.rejection_reason) {
    expect(result.rejection_reason).toContain('absolute minimum');
  }
});

test('processMeasurement multiple measurements', async () => {
  const store = new InMemoryStore();
  const userId = 'test-user-003';
  const baseTime = new Date();

  // First measurement
  const result1 = await processMeasurement(
    userId,
    70.0,
    baseTime,
    'patient-device',
    {},
    'kg',
    store,
    1.75
  );

  expect(result1.accepted).toBe(true);
  expect(result1.filtered_weight).toBeCloseTo(70.0, 1);

  // Second measurement (slightly higher, should be accepted)
  const result2 = await processMeasurement(
    userId,
    70.3,
    new Date(baseTime.getTime() + 60000), // 1 minute later
    'patient-device',
    {},
    'kg',
    store,
    1.75
  );

  expect(result2.accepted).toBe(true);
  expect(result2.filtered_weight).toBeDefined();

  // Third measurement (should have Kalman filtering active)
  const result3 = await processMeasurement(
    userId,
    70.5,
    new Date(baseTime.getTime() + 120000), // 2 minutes later
    'patient-device',
    {},
    'kg',
    store,
    1.75
  );

  expect(result3.accepted).toBe(true);
  expect(result3.filtered_weight).toBeDefined();
});

console.log('\n✅ All basic tests passed!\n');
