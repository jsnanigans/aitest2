#!/usr/bin/env bun
/**
 * Test matrix operations between ml-matrix and NumPy
 */

import { Matrix, inverse } from "ml-matrix";

// Test case: Kalman innovation covariance
const innovationCov = new Matrix([[5.364]]);

// Test case: Full Kalman update step
const H = new Matrix([[1, 0]]);  // Observation matrix
const R = new Matrix([[5.0]]);    // Observation noise
const P_pred = new Matrix([       // Predicted covariance
  [0.382, 0.0],
  [0.0, 0.00012]
]);

console.log("=== Matrix Inverse Precision Test (TypeScript/ml-matrix) ===\n");

// Test 1: Simple 1x1 inverse
console.log("Test 1: Kalman innovation covariance (1x1)");
console.log("Input:");
console.log(innovationCov.to2DArray());
const inv1 = inverse(innovationCov);
console.log("Inverse:");
console.log(inv1.to2DArray());
console.log(`Raw value: ${inv1.get(0, 0)}`);
console.log();

// Test 2: Full Kalman update step
console.log("Test 2: Full Kalman Update Step");
console.log("H * P * H^T + R (innovation covariance):");
const S = H.mmul(P_pred).mmul(H.transpose()).add(R);
console.log(S.to2DArray());
console.log(`Raw value: ${S.get(0, 0)}`);

console.log("\nInnovation covariance inverse:");
const S_inv = inverse(S);
console.log(S_inv.to2DArray());
console.log(`Raw value: ${S_inv.get(0, 0)}`);

console.log("\nKalman Gain = P * H^T * S^{-1}:");
const K = P_pred.mmul(H.transpose()).mmul(S_inv);
console.log(K.to2DArray());
console.log(`K[0,0] = ${K.get(0, 0)}`);
console.log(`K[1,0] = ${K.get(1, 0)}`);
console.log();

// Test 3: Verify numeric precision with high-precision input
const preciseMatrix = new Matrix([[1.23456789012345]]);
console.log("Test 3: High precision 1x1 matrix");
console.log(`Input: ${preciseMatrix.get(0, 0)}`);
const inv3 = inverse(preciseMatrix);
console.log(`Inverse: ${inv3.get(0, 0)}`);
console.log(`Expected: ${1 / 1.23456789012345}`);
console.log(`Difference: ${Math.abs(inv3.get(0, 0) - (1 / 1.23456789012345))}`);
