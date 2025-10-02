/**
 * Unit tests for statistical functions
 */

import { describe, it, expect } from 'bun:test';
import {
  mean,
  median,
  variance,
  std,
  percentile,
  linearRegression,
  erf,
  normalCdf,
  chi2Cdf,
  mad,
  modifiedZScores,
} from '../../src/core/math/statistics';

describe('Statistical Functions', () => {
  describe('mean', () => {
    it('should calculate mean of array', () => {
      expect(mean([1, 2, 3, 4, 5])).toBe(3);
      expect(mean([10, 20, 30])).toBe(20);
    });

    it('should handle single value', () => {
      expect(mean([42])).toBe(42);
    });

    it('should handle negative values', () => {
      expect(mean([-1, -2, -3])).toBe(-2);
    });

    it('should throw on empty array', () => {
      expect(() => mean([])).toThrow();
    });
  });

  describe('median', () => {
    it('should calculate median of odd-length array', () => {
      expect(median([1, 2, 3, 4, 5])).toBe(3);
      expect(median([5, 1, 3])).toBe(3);
    });

    it('should calculate median of even-length array', () => {
      expect(median([1, 2, 3, 4])).toBe(2.5);
      expect(median([10, 20])).toBe(15);
    });

    it('should handle single value', () => {
      expect(median([42])).toBe(42);
    });

    it('should throw on empty array', () => {
      expect(() => median([])).toThrow();
    });
  });

  describe('variance', () => {
    it('should calculate population variance (ddof=0)', () => {
      const values = [2, 4, 4, 4, 5, 5, 7, 9];
      const result = variance(values, 0);
      expect(result).toBeCloseTo(4.0, 5);
    });

    it('should calculate sample variance (ddof=1)', () => {
      const values = [2, 4, 4, 4, 5, 5, 7, 9];
      const result = variance(values, 1);
      expect(result).toBeCloseTo(4.571, 2);
    });

    it('should handle two values with ddof=1', () => {
      const result = variance([1, 3], 1);
      expect(result).toBe(2);
    });

    it('should throw on empty array', () => {
      expect(() => variance([])).toThrow();
    });
  });

  describe('std', () => {
    it('should calculate standard deviation', () => {
      const values = [2, 4, 4, 4, 5, 5, 7, 9];
      const result = std(values, 0);
      expect(result).toBeCloseTo(2.0, 5);
    });

    it('should calculate sample standard deviation', () => {
      const values = [2, 4, 4, 4, 5, 5, 7, 9];
      const result = std(values, 1);
      expect(result).toBeCloseTo(2.138, 2);
    });
  });

  describe('percentile', () => {
    it('should calculate 25th percentile', () => {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = percentile(values, 25);
      expect(result).toBeCloseTo(3.25, 2);
    });

    it('should calculate 50th percentile (median)', () => {
      const values = [1, 2, 3, 4, 5];
      const result = percentile(values, 50);
      expect(result).toBe(3);
    });

    it('should calculate 75th percentile', () => {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = percentile(values, 75);
      expect(result).toBeCloseTo(7.75, 2);
    });

    it('should handle 0th percentile', () => {
      const values = [1, 2, 3, 4, 5];
      expect(percentile(values, 0)).toBe(1);
    });

    it('should handle 100th percentile', () => {
      const values = [1, 2, 3, 4, 5];
      expect(percentile(values, 100)).toBe(5);
    });

    it('should throw on empty array', () => {
      expect(() => percentile([], 50)).toThrow();
    });

    it('should throw on invalid percentile', () => {
      expect(() => percentile([1, 2, 3], -1)).toThrow();
      expect(() => percentile([1, 2, 3], 101)).toThrow();
    });
  });

  describe('linearRegression', () => {
    it('should fit perfect linear relationship', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [2, 4, 6, 8, 10]; // y = 2x
      const [slope, intercept] = linearRegression(x, y);
      expect(slope).toBeCloseTo(2, 5);
      expect(intercept).toBeCloseTo(0, 5);
    });

    it('should fit line with intercept', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [3, 5, 7, 9, 11]; // y = 2x + 1
      const [slope, intercept] = linearRegression(x, y);
      expect(slope).toBeCloseTo(2, 5);
      expect(intercept).toBeCloseTo(1, 5);
    });

    it('should fit negative slope', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [10, 8, 6, 4, 2]; // y = -2x + 12
      const [slope, intercept] = linearRegression(x, y);
      expect(slope).toBeCloseTo(-2, 5);
      expect(intercept).toBeCloseTo(12, 5);
    });

    it('should throw on mismatched array lengths', () => {
      expect(() => linearRegression([1, 2], [1, 2, 3])).toThrow();
    });

    it('should throw on empty arrays', () => {
      expect(() => linearRegression([], [])).toThrow();
    });

    it('should throw on constant x values', () => {
      const x = [1, 1, 1, 1];
      const y = [1, 2, 3, 4];
      expect(() => linearRegression(x, y)).toThrow();
    });
  });

  describe('erf', () => {
    it('should calculate erf for positive values', () => {
      expect(erf(0)).toBeCloseTo(0, 5);
      expect(erf(1)).toBeCloseTo(0.8427, 3);
      expect(erf(2)).toBeCloseTo(0.9953, 3);
    });

    it('should calculate erf for negative values', () => {
      expect(erf(-1)).toBeCloseTo(-0.8427, 3);
      expect(erf(-2)).toBeCloseTo(-0.9953, 3);
    });

    it('should approach 1 for large positive values', () => {
      expect(erf(3)).toBeGreaterThan(0.999);
      expect(erf(5)).toBeGreaterThan(0.9999);
    });

    it('should be antisymmetric', () => {
      const x = 1.5;
      expect(erf(x)).toBeCloseTo(-erf(-x), 5);
    });
  });

  describe('normalCdf', () => {
    it('should calculate standard normal CDF', () => {
      expect(normalCdf(0)).toBeCloseTo(0.5, 5);
      expect(normalCdf(1)).toBeCloseTo(0.8413, 3);
      expect(normalCdf(-1)).toBeCloseTo(0.1587, 3);
      expect(normalCdf(2)).toBeCloseTo(0.9772, 3);
    });

    it('should handle non-standard normal', () => {
      // N(10, 2)
      expect(normalCdf(10, 10, 2)).toBeCloseTo(0.5, 5);
      expect(normalCdf(12, 10, 2)).toBeCloseTo(0.8413, 3);
    });

    it('should approach 0 and 1 for extreme values', () => {
      expect(normalCdf(-5)).toBeLessThan(0.001);
      expect(normalCdf(5)).toBeGreaterThan(0.999);
    });
  });

  describe('chi2Cdf', () => {
    it('should calculate chi-squared CDF for df=1', () => {
      expect(chi2Cdf(0, 1)).toBe(0);
      expect(chi2Cdf(1, 1)).toBeCloseTo(0.6827, 3);
      expect(chi2Cdf(4, 1)).toBeCloseTo(0.9545, 3);
    });

    it('should handle negative values', () => {
      expect(chi2Cdf(-1, 1)).toBe(0);
    });

    it('should throw for df !== 1', () => {
      expect(() => chi2Cdf(1, 2)).toThrow();
    });
  });

  describe('mad', () => {
    it('should calculate median absolute deviation', () => {
      const values = [1, 2, 3, 4, 5];
      const result = mad(values);
      // median = 3, deviations = [2, 1, 0, 1, 2], median of deviations = 1
      expect(result).toBe(1);
    });

    it('should handle values with outliers', () => {
      const values = [1, 2, 3, 4, 100];
      const result = mad(values);
      // median = 3, deviations = [2, 1, 0, 1, 97], median = 1
      expect(result).toBe(1);
    });
  });

  describe('modifiedZScores', () => {
    it('should calculate modified Z-scores', () => {
      const values = [1, 2, 3, 4, 5];
      const scores = modifiedZScores(values);
      expect(scores.length).toBe(5);
      // Middle value should have Z-score near 0
      expect(Math.abs(scores[2]!)).toBeLessThan(0.1);
    });

    it('should identify outliers', () => {
      const values = [1, 2, 3, 4, 100];
      const scores = modifiedZScores(values);
      // Outlier (100) should have large Z-score
      expect(Math.abs(scores[4]!)).toBeGreaterThan(3);
    });

    it('should throw when MAD is zero', () => {
      const values = [1, 1, 1, 1];
      expect(() => modifiedZScores(values)).toThrow();
    });
  });

  describe('Integration with Kalman filter quality scoring', () => {
    it('should support chi-squared test calculations', () => {
      // Simulate quality score chi-squared test
      const innovation = 0.5; // kg difference from prediction
      const innovationVariance = 1.0; // kg^2

      const chiSquared = (innovation * innovation) / innovationVariance;
      const pValue = 1 - chi2Cdf(chiSquared, 1);

      expect(pValue).toBeGreaterThan(0);
      expect(pValue).toBeLessThan(1);
      // chi-squared = 0.25, df=1 => p-value ≈ 0.617
      expect(pValue).toBeCloseTo(0.617, 2);
    });

    it('should support linear regression for trend analysis', () => {
      // Simulate weight trend over time
      const timestamps = [0, 1, 2, 3, 4]; // days
      const weights = [75.0, 74.8, 74.6, 74.4, 74.2]; // kg

      const [slope, _intercept] = linearRegression(timestamps, weights);

      // Losing 0.2 kg per day
      expect(slope).toBeCloseTo(-0.2, 5);
    });
  });
});
