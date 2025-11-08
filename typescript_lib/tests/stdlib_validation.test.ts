import { describe, it, expect } from 'bun:test';
import {
  isFinite,
  isNaN,
  isNumber,
  isFiniteArray,
  validateNumber,
  validateMatrix,
  validateArray,
} from '../src/weight-processor-lib/core/stdlib-utils';

describe('stdlib validation utilities', () => {
  describe('isFinite', () => {
    it('should return true for finite numbers', () => {
      expect(isFinite(0)).toBe(true);
      expect(isFinite(42)).toBe(true);
      expect(isFinite(-100.5)).toBe(true);
      expect(isFinite(1e-10)).toBe(true);
    });

    it('should return false for Infinity', () => {
      expect(isFinite(Infinity)).toBe(false);
      expect(isFinite(-Infinity)).toBe(false);
    });

    it('should return false for NaN', () => {
      expect(isFinite(NaN)).toBe(false);
    });

    it('should return false for non-numbers', () => {
      expect(isFinite("42" as any)).toBe(false);
      expect(isFinite(undefined as any)).toBe(false);
      expect(isFinite(null as any)).toBe(false);
    });
  });

  describe('isNaN', () => {
    it('should return true for NaN', () => {
      expect(isNaN(NaN)).toBe(true);
    });

    it('should return false for numbers', () => {
      expect(isNaN(0)).toBe(false);
      expect(isNaN(42)).toBe(false);
      expect(isNaN(Infinity)).toBe(false);
    });

    it('should return false for non-numbers', () => {
      expect(isNaN("not a number" as any)).toBe(false);
    });
  });

  describe('isNumber', () => {
    it('should return true for numbers', () => {
      expect(isNumber(0)).toBe(true);
      expect(isNumber(42)).toBe(true);
      expect(isNumber(-100.5)).toBe(true);
      expect(isNumber(NaN)).toBe(true);
      expect(isNumber(Infinity)).toBe(true);
    });

    it('should return false for non-numbers', () => {
      expect(isNumber("42")).toBe(false);
      expect(isNumber(undefined)).toBe(false);
      expect(isNumber(null)).toBe(false);
      expect(isNumber({})).toBe(false);
    });
  });

  describe('isFiniteArray', () => {
    it('should return true for arrays of finite numbers', () => {
      expect(isFiniteArray([1, 2, 3])).toBe(true);
      expect(isFiniteArray([0])).toBe(true);
      expect(isFiniteArray([-100, 200.5, 0.001])).toBe(true);
    });

    it('should return false for arrays containing Infinity', () => {
      expect(isFiniteArray([1, Infinity, 3])).toBe(false);
      expect(isFiniteArray([Infinity])).toBe(false);
    });

    it('should return false for arrays containing NaN', () => {
      expect(isFiniteArray([1, NaN, 3])).toBe(false);
      expect(isFiniteArray([NaN])).toBe(false);
    });

    it('should return false for empty arrays', () => {
      // stdlib returns false for empty arrays (no finite values present)
      expect(isFiniteArray([])).toBe(false);
    });
  });

  describe('validateNumber', () => {
    it('should return true for valid finite numbers', () => {
      expect(validateNumber(0)).toBe(true);
      expect(validateNumber(42)).toBe(true);
      expect(validateNumber(-100.5)).toBe(true);
    });

    it('should return false for undefined', () => {
      expect(validateNumber(undefined)).toBe(false);
    });

    it('should return false for null', () => {
      expect(validateNumber(null as any)).toBe(false);
    });

    it('should return false for NaN', () => {
      expect(validateNumber(NaN)).toBe(false);
    });

    it('should return false for Infinity', () => {
      expect(validateNumber(Infinity)).toBe(false);
      expect(validateNumber(-Infinity)).toBe(false);
    });
  });

  describe('validateArray', () => {
    it('should return true for valid arrays', () => {
      expect(validateArray([1, 2, 3])).toBe(true);
      expect(validateArray([0])).toBe(true);
    });

    it('should return false for empty arrays', () => {
      // stdlib returns false for empty arrays (no finite values present)
      expect(validateArray([])).toBe(false);
    });

    it('should return false for arrays with invalid values', () => {
      expect(validateArray([1, NaN, 3])).toBe(false);
      expect(validateArray([1, Infinity])).toBe(false);
    });
  });

  describe('validateMatrix', () => {
    it('should return true for valid matrices', () => {
      expect(validateMatrix([[1, 2], [3, 4]])).toBe(true);
      expect(validateMatrix([[0]])).toBe(true);
      expect(validateMatrix([[1.5, -2.5], [3.5, 4.5]])).toBe(true);
    });

    it('should return false for matrices with NaN', () => {
      expect(validateMatrix([[1, NaN], [3, 4]])).toBe(false);
      expect(validateMatrix([[NaN]])).toBe(false);
    });

    it('should return false for matrices with Infinity', () => {
      expect(validateMatrix([[1, Infinity], [3, 4]])).toBe(false);
      expect(validateMatrix([[Infinity]])).toBe(false);
    });

    it('should return false for empty matrices', () => {
      // stdlib returns false for empty matrices (no finite values present)
      expect(validateMatrix([])).toBe(false);
      expect(validateMatrix([[]])).toBe(false);
    });
  });

  describe('integration with quality scorer use cases', () => {
    it('should validate Kalman prediction values', () => {
      const kalmanPrediction = 58.5;
      const innovationCovariance = 0.25;

      expect(validateNumber(kalmanPrediction)).toBe(true);
      expect(validateNumber(innovationCovariance)).toBe(true);
      expect(validateNumber(kalmanPrediction) && validateNumber(innovationCovariance)).toBe(true);
    });

    it('should reject invalid Kalman values', () => {
      expect(validateNumber(undefined)).toBe(false);
      expect(validateNumber(NaN)).toBe(false);
      expect(validateNumber(Infinity)).toBe(false);
    });

    it('should validate matrices from Kalman filter', () => {
      const stateMatrix = [[58.5], [0.1]];
      const covarianceMatrix = [[1.0, 0.0], [0.0, 0.5]];

      expect(validateMatrix(stateMatrix)).toBe(true);
      expect(validateMatrix(covarianceMatrix)).toBe(true);
    });

    it('should reject invalid matrices from Kalman filter', () => {
      const invalidState = [[NaN], [0.1]];
      const invalidCovariance = [[1.0, Infinity], [0.0, 0.5]];

      expect(validateMatrix(invalidState)).toBe(false);
      expect(validateMatrix(invalidCovariance)).toBe(false);
    });
  });
});
