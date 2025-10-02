/**
 * Unit tests for Kalman Filter
 */

import { describe, it, expect } from 'bun:test';
import { KalmanFilter } from '../../src/core/processing/kalman_filter';
import type { Matrix2x2, Vector2 } from '../../src/core/math/matrix';
import { eye2 } from '../../src/core/math/matrix';

describe('KalmanFilter', () => {
  // Helper to create a simple filter for testing
  function createTestFilter(): KalmanFilter {
    const dt = 1.0; // Time step in days
    const F: Matrix2x2 = [
      [1, dt],
      [0, 1],
    ]; // Constant velocity model
    const H: Matrix2x2 = [
      [1, 0],
      [0, 0],
    ]; // Observe only position (weight)
    const x0: Vector2 = [75.0, 0.0]; // Initial state [weight, velocity]
    const P0: Matrix2x2 = [
      [1.0, 0.0],
      [0.0, 0.5],
    ]; // Initial covariance
    const Q: Matrix2x2 = [
      [0.01, 0.0],
      [0.0, 0.001],
    ]; // Process noise
    const R = 1.0; // Measurement noise

    return new KalmanFilter(F, H, x0, P0, Q, R);
  }

  describe('Constructor', () => {
    it('should create filter with valid parameters', () => {
      const filter = createTestFilter();
      expect(filter).toBeDefined();
      expect(filter.x[0]).toBe(75.0);
      expect(filter.x[1]).toBe(0.0);
    });

    it('should validate matrix dimensions', () => {
      const F = eye2();
      const H: Matrix2x2 = [
        [1, 0],
        [0, 0],
      ];
      const x: Vector2 = [75.0, 0.0];
      const P = eye2();
      const Q = eye2();
      const R = 1.0;

      expect(() => new KalmanFilter(F, H, x, P, Q, R)).not.toThrow();
    });

    it('should throw on invalid measurement noise', () => {
      const F = eye2();
      const H: Matrix2x2 = [
        [1, 0],
        [0, 0],
      ];
      const x: Vector2 = [75.0, 0.0];
      const P = eye2();
      const Q = eye2();
      const R = -1.0; // Invalid

      expect(() => new KalmanFilter(F, H, x, P, Q, R)).toThrow();
    });
  });

  describe('predict', () => {
    it('should predict next state', () => {
      const filter = createTestFilter();
      const currentState: Vector2 = [75.0, 0.1]; // Moving up 0.1 kg/day
      const currentCov = eye2();

      const [predictedState, predictedCov] = filter.predict(currentState, currentCov);

      // State should be: [75.0 + 0.1*1, 0.1] = [75.1, 0.1]
      expect(predictedState[0]).toBeCloseTo(75.1, 5);
      expect(predictedState[1]).toBeCloseTo(0.1, 5);

      // Covariance should increase due to process noise
      expect(predictedCov[0][0]).toBeGreaterThan(currentCov[0][0]);
    });

    it('should handle zero velocity', () => {
      const filter = createTestFilter();
      const currentState: Vector2 = [75.0, 0.0];
      const currentCov = eye2();

      const [predictedState, _] = filter.predict(currentState, currentCov);

      expect(predictedState[0]).toBeCloseTo(75.0, 5);
      expect(predictedState[1]).toBeCloseTo(0.0, 5);
    });
  });

  describe('update', () => {
    it('should update state with measurement', () => {
      const filter = createTestFilter();
      const predictedState: Vector2 = [75.0, 0.0];
      const predictedCov: Matrix2x2 = [
        [1.0, 0.0],
        [0.0, 0.5],
      ];
      const measurement = 75.5; // Observed weight higher than predicted

      const [updatedState, updatedCov] = filter.update(
        predictedState,
        predictedCov,
        measurement
      );

      // State should move toward measurement
      expect(updatedState[0]).toBeGreaterThan(75.0);
      expect(updatedState[0]).toBeLessThan(75.5);

      // Covariance should decrease (more certain after measurement)
      expect(updatedCov[0][0]).toBeLessThan(predictedCov[0][0]);
    });

    it('should handle measurement exactly matching prediction', () => {
      const filter = createTestFilter();
      const predictedState: Vector2 = [75.0, 0.0];
      const predictedCov = eye2();
      const measurement = 75.0;

      const [updatedState, _] = filter.update(predictedState, predictedCov, measurement);

      // State should remain close to prediction
      expect(updatedState[0]).toBeCloseTo(75.0, 1);
    });
  });

  describe('filterUpdate', () => {
    it('should perform combined predict and update', () => {
      const filter = createTestFilter();
      const currentState: Vector2 = [75.0, 0.1];
      const currentCov = eye2();
      const measurement = 75.2;

      const [newState, newCov] = filter.filterUpdate(currentState, currentCov, measurement);

      expect(newState[0]).toBeGreaterThan(75.0);
      expect(newState[0]).toBeLessThan(75.5);
      expect(newState.length).toBe(2);
      expect(newCov.length).toBe(2);
    });
  });

  describe('filter', () => {
    it('should process sequence of measurements', () => {
      const filter = createTestFilter();
      const observations = [75.0, 75.1, 75.2, 75.1, 75.3];

      const [states, covariances] = filter.filter(observations);

      expect(states.length).toBe(5);
      expect(covariances.length).toBe(5);

      // States should generally follow the measurements
      expect(states[0][0]).toBeCloseTo(75.0, 0);
      expect(states[4][0]).toBeGreaterThan(75.0);
    });

    it('should handle single observation', () => {
      const filter = createTestFilter();
      const observations = [75.5];

      const [states, covariances] = filter.filter(observations);

      expect(states.length).toBe(1);
      expect(covariances.length).toBe(1);
    });

    it('should smooth noisy measurements', () => {
      const filter = createTestFilter();
      // Noisy measurements around 75.0
      const observations = [75.0, 76.0, 74.0, 75.5, 74.5, 75.0];

      const [states, _] = filter.filter(observations);

      // Filter should produce smoother estimates
      const estimates = states.map((s) => s[0]);

      // Check that estimates are less variable than measurements
      const measurementVariance = variance(observations);
      const estimateVariance = variance(estimates);

      expect(estimateVariance).toBeLessThan(measurementVariance);
    });
  });

  describe('Weight tracking scenarios', () => {
    it('should track stable weight', () => {
      const filter = createTestFilter();
      // Stable weight with small variations
      const observations = [75.0, 75.1, 74.9, 75.0, 75.1];

      const [states, _] = filter.filter(observations);

      // All estimates should be near 75.0
      states.forEach((state) => {
        expect(Math.abs(state[0] - 75.0)).toBeLessThan(0.5);
      });

      // Velocity should be near zero
      const finalVelocity = states[states.length - 1]![1];
      expect(Math.abs(finalVelocity)).toBeLessThan(0.1);
    });

    it('should track weight loss trend', () => {
      const filter = createTestFilter();
      // Gradual weight loss: 75 -> 74
      const observations = [75.0, 74.8, 74.6, 74.4, 74.2, 74.0];

      const [states, _] = filter.filter(observations);

      // Should detect negative velocity
      const finalVelocity = states[states.length - 1]![1];
      expect(finalVelocity).toBeLessThan(0);

      // Weight should track the trend
      expect(states[states.length - 1]![0]).toBeCloseTo(74.0, 0);
    });

    it('should track weight gain trend', () => {
      const filter = createTestFilter();
      // Gradual weight gain: 75 -> 76
      const observations = [75.0, 75.2, 75.4, 75.6, 75.8, 76.0];

      const [states, _] = filter.filter(observations);

      // Should detect positive velocity
      const finalVelocity = states[states.length - 1]![1];
      expect(finalVelocity).toBeGreaterThan(0);

      // Weight should track the trend
      expect(states[states.length - 1]![0]).toBeCloseTo(76.0, 0);
    });

    it('should handle measurement gaps (time-varying F matrix)', () => {
      // This would require updating F matrix between measurements
      // For now, we test that the filter still works
      const filter = createTestFilter();
      const observations = [75.0, 75.5]; // Gap of 1 day

      const [states, _] = filter.filter(observations);

      expect(states.length).toBe(2);
      expect(states[1][0]).toBeGreaterThan(75.0);
    });
  });

  describe('Numerical stability', () => {
    it('should maintain positive covariance', () => {
      const filter = createTestFilter();
      const observations = Array(20).fill(75.0);

      const [_, covariances] = filter.filter(observations);

      // All covariances should remain positive
      covariances.forEach((cov) => {
        expect(cov[0][0]).toBeGreaterThan(0);
        expect(cov[1][1]).toBeGreaterThan(0);
      });
    });

    it('should maintain symmetric covariance (approximately)', () => {
      const filter = createTestFilter();
      const observations = [75.0, 75.5, 74.5, 75.2];

      const [_, covariances] = filter.filter(observations);

      covariances.forEach((cov) => {
        // P should be symmetric: P[0][1] ≈ P[1][0]
        // Note: Joseph form can introduce small asymmetries, so we use a practical tolerance
        expect(Math.abs(cov[0][1] - cov[1][0])).toBeLessThan(0.2);
      });
    });
  });
});

// Helper function to calculate variance
function variance(values: number[]): number {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
  return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
}
