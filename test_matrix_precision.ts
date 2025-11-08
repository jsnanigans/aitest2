#!/usr/bin/env bun
/**
 * Test numerical precision differences between ml-matrix and numpy
 */

import { Matrix } from 'ml-matrix';

// Test data: 2x2 innovation covariance matrix (typical values from Kalman filter)
const testMatrices = [
  // Small values (typical for weight measurements)
  [[1.0, 0.1], [0.1, 1.0]],
  [[0.5, 0.05], [0.05, 0.5]],
  [[2.0, 0.3], [0.3, 2.0]],

  // Values from actual Kalman filtering
  [[1.234, 0.123], [0.123, 1.234]],
  [[0.789, 0.056], [0.056, 0.789]],
];

console.log('Testing Matrix Inversion Precision (ml-matrix)\n');
console.log('=' .repeat(60));

for (const matrixData of testMatrices) {
  const m = new Matrix(matrixData);
  console.log('\nOriginal Matrix:');
  console.log(m.to2DArray());

  const inv = m.inverse();
  console.log('Inverse:');
  console.log(inv.to2DArray());

  // Verify: A * A^-1 should equal identity matrix
  const product = m.mmul(inv);
  console.log('Verification (A * A^-1):');
  console.log(product.to2DArray());

  // Check how close to identity
  const identity = Matrix.eye(2);
  const diff = product.sub(identity);
  const maxError = Math.max(...diff.to1DArray().map(Math.abs));
  console.log(`Max error from identity: ${maxError.toExponential(4)}`);
}
