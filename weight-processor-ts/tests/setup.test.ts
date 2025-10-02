/**
 * Test setup verification
 *
 * This test file verifies that the test infrastructure is working correctly
 */

import { describe, it, expect } from 'bun:test';
import { expectClose, createMockMeasurement } from './helpers/test-utils';

describe('Test Infrastructure', () => {
  it('should run basic tests', () => {
    expect(true).toBe(true);
  });

  it('should handle numerical comparisons', () => {
    expect(expectClose(1.0, 1.001, 0.01)).toBe(true);
    expect(expectClose(1.0, 1.1, 0.01)).toBe(false);
  });

  it('should create mock measurements', () => {
    const measurement = createMockMeasurement({ weight_kg: 80.0 });
    expect(measurement.weight_kg).toBe(80.0);
    expect(measurement.device_id).toBe('test-device');
  });
});
