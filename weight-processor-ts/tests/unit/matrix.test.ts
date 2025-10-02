/**
 * Unit tests for 2x2 matrix operations
 */

import { describe, it, expect } from 'bun:test';
import {
  eye2,
  multiply2x2,
  multiplyVector2x2,
  transpose2x2,
  add2x2,
  subtract2x2,
  scalarMultiply2x2,
  invert2x2,
  determinant2x2,
  isSymmetric2x2,
  isPositiveDefinite2x2,
  copy2x2,
  copyVector2,
  addVector2,
  subtractVector2,
  scalarMultiplyVector2,
  type Matrix2x2,
  type Vector2,
} from '../../src/core/math/matrix';

describe('Matrix Operations', () => {
  describe('eye2', () => {
    it('should create identity matrix', () => {
      const I = eye2();
      expect(I).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });
  });

  describe('multiply2x2', () => {
    it('should multiply two 2x2 matrices', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const b: Matrix2x2 = [
        [5, 6],
        [7, 8],
      ];
      const result = multiply2x2(a, b);
      // [1*5+2*7, 1*6+2*8] = [19, 22]
      // [3*5+4*7, 3*6+4*8] = [43, 50]
      expect(result).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('should multiply by identity matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const I = eye2();
      const result = multiply2x2(a, I);
      expect(result).toEqual(a);
    });
  });

  describe('multiplyVector2x2', () => {
    it('should multiply matrix by vector', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const x: Vector2 = [5, 6];
      const result = multiplyVector2x2(a, x);
      // [1*5+2*6, 3*5+4*6] = [17, 39]
      expect(result).toEqual([17, 39]);
    });

    it('should multiply identity by vector', () => {
      const I = eye2();
      const x: Vector2 = [5, 6];
      const result = multiplyVector2x2(I, x);
      expect(result).toEqual(x);
    });
  });

  describe('transpose2x2', () => {
    it('should transpose matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const result = transpose2x2(a);
      expect(result).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it('should double transpose to original', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const result = transpose2x2(transpose2x2(a));
      expect(result).toEqual(a);
    });
  });

  describe('add2x2', () => {
    it('should add two matrices', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const b: Matrix2x2 = [
        [5, 6],
        [7, 8],
      ];
      const result = add2x2(a, b);
      expect(result).toEqual([
        [6, 8],
        [10, 12],
      ]);
    });
  });

  describe('subtract2x2', () => {
    it('should subtract two matrices', () => {
      const a: Matrix2x2 = [
        [5, 6],
        [7, 8],
      ];
      const b: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const result = subtract2x2(a, b);
      expect(result).toEqual([
        [4, 4],
        [4, 4],
      ]);
    });
  });

  describe('scalarMultiply2x2', () => {
    it('should multiply matrix by scalar', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const result = scalarMultiply2x2(2, a);
      expect(result).toEqual([
        [2, 4],
        [6, 8],
      ]);
    });

    it('should multiply by zero', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const result = scalarMultiply2x2(0, a);
      expect(result).toEqual([
        [0, 0],
        [0, 0],
      ]);
    });
  });

  describe('determinant2x2', () => {
    it('should calculate determinant', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const det = determinant2x2(a);
      // 1*4 - 2*3 = 4 - 6 = -2
      expect(det).toBe(-2);
    });

    it('should return 1 for identity matrix', () => {
      const I = eye2();
      const det = determinant2x2(I);
      expect(det).toBe(1);
    });

    it('should return 0 for singular matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [2, 4],
      ];
      const det = determinant2x2(a);
      expect(det).toBe(0);
    });
  });

  describe('invert2x2', () => {
    it('should invert matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const inv = invert2x2(a);
      // Verify A * A^{-1} = I
      const product = multiply2x2(a, inv);
      expect(Math.abs(product[0][0] - 1)).toBeLessThan(1e-10);
      expect(Math.abs(product[0][1])).toBeLessThan(1e-10);
      expect(Math.abs(product[1][0])).toBeLessThan(1e-10);
      expect(Math.abs(product[1][1] - 1)).toBeLessThan(1e-10);
    });

    it('should invert identity to identity', () => {
      const I = eye2();
      const inv = invert2x2(I);
      // Check with numerical tolerance due to floating-point arithmetic
      expect(Math.abs(inv[0][0] - 1)).toBeLessThan(1e-10);
      expect(Math.abs(inv[0][1])).toBeLessThan(1e-10);
      expect(Math.abs(inv[1][0])).toBeLessThan(1e-10);
      expect(Math.abs(inv[1][1] - 1)).toBeLessThan(1e-10);
    });

    it('should throw error for singular matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [2, 4],
      ];
      expect(() => invert2x2(a)).toThrow();
    });
  });

  describe('isSymmetric2x2', () => {
    it('should detect symmetric matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [2, 4],
      ];
      expect(isSymmetric2x2(a)).toBe(true);
    });

    it('should detect non-symmetric matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      expect(isSymmetric2x2(a)).toBe(false);
    });

    it('should handle identity matrix', () => {
      const I = eye2();
      expect(isSymmetric2x2(I)).toBe(true);
    });
  });

  describe('isPositiveDefinite2x2', () => {
    it('should detect positive definite matrix', () => {
      const a: Matrix2x2 = [
        [2, 1],
        [1, 2],
      ];
      expect(isPositiveDefinite2x2(a)).toBe(true);
    });

    it('should reject non-positive-definite matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [2, 1],
      ];
      expect(isPositiveDefinite2x2(a)).toBe(false);
    });

    it('should handle identity matrix', () => {
      const I = eye2();
      expect(isPositiveDefinite2x2(I)).toBe(true);
    });

    it('should reject non-symmetric matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      expect(isPositiveDefinite2x2(a)).toBe(false);
    });
  });

  describe('copy2x2', () => {
    it('should deep copy matrix', () => {
      const a: Matrix2x2 = [
        [1, 2],
        [3, 4],
      ];
      const copy = copy2x2(a);
      expect(copy).toEqual(a);
      // Verify it's a deep copy
      copy[0][0] = 99;
      expect(a[0][0]).toBe(1);
    });
  });

  describe('Vector Operations', () => {
    describe('copyVector2', () => {
      it('should deep copy vector', () => {
        const v: Vector2 = [1, 2];
        const copy = copyVector2(v);
        expect(copy).toEqual(v);
        // Verify it's a deep copy
        copy[0] = 99;
        expect(v[0]).toBe(1);
      });
    });

    describe('addVector2', () => {
      it('should add two vectors', () => {
        const a: Vector2 = [1, 2];
        const b: Vector2 = [3, 4];
        const result = addVector2(a, b);
        expect(result).toEqual([4, 6]);
      });
    });

    describe('subtractVector2', () => {
      it('should subtract two vectors', () => {
        const a: Vector2 = [5, 6];
        const b: Vector2 = [1, 2];
        const result = subtractVector2(a, b);
        expect(result).toEqual([4, 4]);
      });
    });

    describe('scalarMultiplyVector2', () => {
      it('should multiply vector by scalar', () => {
        const v: Vector2 = [1, 2];
        const result = scalarMultiplyVector2(3, v);
        expect(result).toEqual([3, 6]);
      });
    });
  });

  describe('Kalman Filter Use Cases', () => {
    it('should support covariance matrix operations', () => {
      // Covariance matrix (must be symmetric and positive definite)
      const P: Matrix2x2 = [
        [1.0, 0.1],
        [0.1, 0.5],
      ];

      expect(isSymmetric2x2(P)).toBe(true);
      expect(isPositiveDefinite2x2(P)).toBe(true);

      // Process noise
      const Q: Matrix2x2 = [
        [0.01, 0],
        [0, 0.01],
      ];

      // Add process noise
      const P_plus_Q = add2x2(P, Q);
      expect(isPositiveDefinite2x2(P_plus_Q)).toBe(true);
    });

    it('should support state transition', () => {
      // State transition matrix (dt = 1.0)
      const F: Matrix2x2 = [
        [1, 1],
        [0, 1],
      ];

      // State vector [weight, velocity]
      const x: Vector2 = [75.0, 0.1];

      // Predict next state
      const x_pred = multiplyVector2x2(F, x);
      expect(x_pred[0]).toBeCloseTo(75.1, 5);
      expect(x_pred[1]).toBeCloseTo(0.1, 5);
    });
  });
});
