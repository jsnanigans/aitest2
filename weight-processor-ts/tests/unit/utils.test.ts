/**
 * Unit tests for utility functions
 */

import { describe, it, expect } from 'bun:test';
import {
  deepCopy,
  parseTimestamp,
  ensureFloat,
  timeDiffSeconds,
  timeDiffDays,
  addSeconds,
  addDays,
  toISOString,
  isPlainObject,
  ensureNumericTypes,
  clamp,
  approxEqual,
  generateId,
} from '../../src/utils';

describe('Utility Functions', () => {
  describe('deepCopy', () => {
    it('should copy primitive values', () => {
      expect(deepCopy(42)).toBe(42);
      expect(deepCopy('hello')).toBe('hello');
      expect(deepCopy(true)).toBe(true);
      expect(deepCopy(null)).toBe(null);
      expect(deepCopy(undefined)).toBe(undefined);
    });

    it('should deep copy arrays', () => {
      const original = [1, 2, [3, 4]];
      const copy = deepCopy(original);
      expect(copy).toEqual(original);
      expect(copy).not.toBe(original);
      expect(copy[2]).not.toBe(original[2]);
    });

    it('should deep copy objects', () => {
      const original = { a: 1, b: { c: 2 } };
      const copy = deepCopy(original);
      expect(copy).toEqual(original);
      expect(copy).not.toBe(original);
      expect(copy.b).not.toBe(original.b);
    });

    it('should copy Date objects', () => {
      const original = new Date('2025-01-01');
      const copy = deepCopy(original);
      expect(copy.getTime()).toBe(original.getTime());
      expect(copy).not.toBe(original);
    });

    it('should handle nested structures', () => {
      const original = {
        array: [1, 2, { nested: 'value' }],
        object: { a: 1, b: [2, 3] },
        date: new Date(),
      };
      const copy = deepCopy(original);
      expect(copy).toEqual(original);
      expect(copy).not.toBe(original);
      expect(copy.array).not.toBe(original.array);
      expect(copy.object).not.toBe(original.object);
      expect(copy.date).not.toBe(original.date);
    });
  });

  describe('parseTimestamp', () => {
    it('should parse ISO 8601 strings', () => {
      const timestamp = '2025-01-15T10:30:00Z';
      const date = parseTimestamp(timestamp);
      expect(date).toBeInstanceOf(Date);
      expect(date.toISOString()).toBe('2025-01-15T10:30:00.000Z');
    });

    it('should handle ISO strings with timezone', () => {
      const timestamp = '2025-01-15T10:30:00+00:00';
      const date = parseTimestamp(timestamp);
      expect(date).toBeInstanceOf(Date);
    });

    it('should handle Date objects', () => {
      const original = new Date('2025-01-15');
      const parsed = parseTimestamp(original);
      expect(parsed).toBe(original);
    });

    it('should handle Unix timestamps', () => {
      const unixTime = 1705318200000; // Some timestamp
      const date = parseTimestamp(unixTime);
      expect(date.getTime()).toBe(unixTime);
    });

    it('should throw on invalid string', () => {
      expect(() => parseTimestamp('not-a-date')).toThrow();
    });
  });

  describe('ensureFloat', () => {
    it('should convert numbers', () => {
      expect(ensureFloat(42)).toBe(42);
      expect(ensureFloat(3.14)).toBe(3.14);
    });

    it('should convert string numbers', () => {
      expect(ensureFloat('42')).toBe(42);
      expect(ensureFloat('3.14')).toBe(3.14);
    });

    it('should handle null and undefined', () => {
      expect(ensureFloat(null)).toBe(0);
      expect(ensureFloat(undefined)).toBe(0);
    });

    it('should return 0 for unconvertible values', () => {
      expect(ensureFloat('not-a-number')).toBe(0);
      expect(ensureFloat({})).toBe(0);
    });
  });

  describe('timeDiffSeconds', () => {
    it('should calculate difference in seconds', () => {
      const date1 = new Date('2025-01-01T00:00:00Z');
      const date2 = new Date('2025-01-01T00:01:00Z');
      expect(timeDiffSeconds(date1, date2)).toBe(60);
    });

    it('should return absolute value', () => {
      const date1 = new Date('2025-01-01T00:01:00Z');
      const date2 = new Date('2025-01-01T00:00:00Z');
      expect(timeDiffSeconds(date1, date2)).toBe(60);
    });

    it('should handle ISO strings', () => {
      const diff = timeDiffSeconds(
        '2025-01-01T00:00:00Z',
        '2025-01-01T00:02:00Z'
      );
      expect(diff).toBe(120);
    });
  });

  describe('timeDiffDays', () => {
    it('should calculate difference in days', () => {
      const date1 = new Date('2025-01-01');
      const date2 = new Date('2025-01-03');
      const diff = timeDiffDays(date1, date2);
      expect(diff).toBeCloseTo(2, 5);
    });
  });

  describe('addSeconds', () => {
    it('should add seconds to timestamp', () => {
      const date = new Date('2025-01-01T00:00:00Z');
      const result = addSeconds(date, 60);
      expect(result.toISOString()).toBe('2025-01-01T00:01:00.000Z');
    });

    it('should subtract seconds with negative value', () => {
      const date = new Date('2025-01-01T00:01:00Z');
      const result = addSeconds(date, -60);
      expect(result.toISOString()).toBe('2025-01-01T00:00:00.000Z');
    });
  });

  describe('addDays', () => {
    it('should add days to timestamp', () => {
      const date = new Date('2025-01-01T00:00:00Z');
      const result = addDays(date, 2);
      expect(result.toISOString()).toBe('2025-01-03T00:00:00.000Z');
    });
  });

  describe('toISOString', () => {
    it('should format Date as ISO string', () => {
      const date = new Date('2025-01-15T10:30:00Z');
      expect(toISOString(date)).toBe('2025-01-15T10:30:00.000Z');
    });
  });

  describe('isPlainObject', () => {
    it('should detect plain objects', () => {
      expect(isPlainObject({})).toBe(true);
      expect(isPlainObject({ a: 1 })).toBe(true);
    });

    it('should reject non-plain objects', () => {
      expect(isPlainObject([])).toBe(false);
      expect(isPlainObject(new Date())).toBe(false);
      expect(isPlainObject(new Map())).toBe(false);
      expect(isPlainObject(new Set())).toBe(false);
      expect(isPlainObject(null)).toBe(false);
      expect(isPlainObject(undefined)).toBe(false);
      expect(isPlainObject(42)).toBe(false);
      expect(isPlainObject('string')).toBe(false);
    });
  });

  describe('ensureNumericTypes', () => {
    it('should convert numeric fields', () => {
      const data = {
        weight: '75.5',
        name: 'John',
      };
      const result = ensureNumericTypes(data);
      expect(result.weight).toBe(75.5);
      expect(result.name).toBe('John');
    });

    it('should handle nested objects', () => {
      const data = {
        measurement: {
          weight: '75.5',
          quality_score: '0.95',
        },
        user_id: 'user123',
      };
      const result = ensureNumericTypes(data);
      expect(result.measurement.weight).toBe(75.5);
      expect(result.measurement.quality_score).toBe(0.95);
      expect(result.user_id).toBe('user123');
    });

    it('should handle arrays', () => {
      const data = [
        { weight: '75.5' },
        { weight: '76.0' },
      ];
      const result = ensureNumericTypes(data);
      expect(result[0].weight).toBe(75.5);
      expect(result[1].weight).toBe(76.0);
    });
  });

  describe('clamp', () => {
    it('should clamp values within range', () => {
      expect(clamp(5, 0, 10)).toBe(5);
      expect(clamp(-5, 0, 10)).toBe(0);
      expect(clamp(15, 0, 10)).toBe(10);
    });

    it('should handle edge cases', () => {
      expect(clamp(0, 0, 10)).toBe(0);
      expect(clamp(10, 0, 10)).toBe(10);
    });
  });

  describe('approxEqual', () => {
    it('should detect approximately equal numbers', () => {
      expect(approxEqual(1.0, 1.0)).toBe(true);
      expect(approxEqual(1.0, 1.0 + 1e-11)).toBe(true);
    });

    it('should detect unequal numbers', () => {
      expect(approxEqual(1.0, 1.1)).toBe(false);
      expect(approxEqual(1.0, 2.0)).toBe(false);
    });

    it('should use custom tolerance', () => {
      expect(approxEqual(1.0, 1.05, 0.1)).toBe(true);
      expect(approxEqual(1.0, 1.15, 0.1)).toBe(false);
    });
  });

  describe('generateId', () => {
    it('should generate unique IDs', () => {
      const id1 = generateId();
      const id2 = generateId();
      expect(id1).not.toBe(id2);
      expect(typeof id1).toBe('string');
      expect(typeof id2).toBe('string');
    });

    it('should generate non-empty strings', () => {
      const id = generateId();
      expect(id.length).toBeGreaterThan(0);
    });
  });

  describe('Integration tests', () => {
    it('should handle measurement data preparation', () => {
      const rawMeasurement = {
        device_id: 'device123',
        user_id: 'user456',
        weight: '75.5',
        timestamp: '2025-01-15T10:30:00Z',
        metadata: {
          quality_score: '0.95',
          source: 'QUESTIONNAIRE_ONBOARDING',
        },
      };

      const processed = ensureNumericTypes(rawMeasurement);
      expect(processed.weight).toBe(75.5);
      expect(processed.metadata.quality_score).toBe(0.95);

      const timestamp = parseTimestamp(processed.timestamp);
      expect(timestamp).toBeInstanceOf(Date);
    });

    it('should handle state deep copying', () => {
      const state = {
        kalman_state: {
          x: [75.0, 0.1],
          P: [[1.0, 0.1], [0.1, 0.5]],
          last_timestamp: new Date('2025-01-15'),
        },
        metadata: {
          measurements_count: 10,
        },
      };

      const copy = deepCopy(state);
      expect(copy).toEqual(state);
      expect(copy).not.toBe(state);
      expect(copy.kalman_state).not.toBe(state.kalman_state);
      expect(copy.kalman_state.last_timestamp).not.toBe(
        state.kalman_state.last_timestamp
      );
    });
  });
});
