#!/usr/bin/env bun
/**
 * Test matrix operations between ml-matrix and NumPy
 * to identify numerical precision differences
 */

import { Matrix } from "ml-matrix";

// Test case 1: Simple 2x2 matrix
const testMatrix1 = new Matrix([
  [4.0, 2.0],
  [2.0, 3.0]
]);

// Test case 2: Typical Kalman innovation covariance (from actual data)
// This is a realistic 1x1 covariance matrix from the Kalman filter
const innovationCov = new Matrix([[5.364]]);

// Test case 3: Typical state covariance matrix (2x2 for weight + trend)
const stateCovariance = new Matrix([
  [0.364, 0.0],
  [0.0, 0.00012]
]);

// Test case 4: A matrix that might have precision issues
const precisionTest = new Matrix([
  [1.23456789012345, 0.98765432109876],
  [0.98765432109876, 2.34567890123456]
]);

console.log("=== Matrix Inverse Precision Test (TypeScript/ml-matrix) ===\n");

// Test 1
console.log("Test 1: Simple 2x2 matrix");
console.log("Input:");
console.log(testMatrix1.to2DArray());
const inv1 = testMatrix1.inverse();
console.log("Inverse:");
console.log(inv1.to2DArray());
console.log("Product (should be identity):");
const product1 = testMatrix1.mmul(inv1);
console.log(product1.to2DArray());
console.log();

// Test 2
console.log("Test 2: Kalman innovation covariance (1x1)");
console.log("Input:");
console.log(innovationCov.to2DArray());
const inv2 = innovationCov.inverse();
console.log("Inverse:");
console.log(inv2.to2DArray());
console.log("Product (should be identity):");
const product2 = innovationCov.mmul(inv2);
console.log(product2.to2DArray());
console.log();

// Test 3
console.log("Test 3: State covariance matrix (2x2)");
console.log("Input:");
console.log(stateCovariance.to2DArray());
const inv3 = stateCovariance.inverse();
console.log("Inverse:");
console.log(inv3.to2DArray());
console.log("Product (should be identity):");
const product3 = stateCovariance.mmul(inv3);
console.log(product3.to2DArray());
console.log();

// Test 4
console.log("Test 4: High precision matrix");
console.log("Input:");
console.log(precisionTest.to2DArray());
const inv4 = precisionTest.inverse();
console.log("Inverse:");
console.log(inv4.to2DArray());
console.log("Product (should be identity):");
const product4 = precisionTest.mmul(inv4);
console.log(product4.to2DArray());
console.log();

// Test 5: Full Kalman update step
console.log("Test 5: Full Kalman Update Step");
const H = new Matrix([[1, 0]]);  // Observation matrix
const R = new Matrix([[5.0]]);    // Observation noise
const P_pred = new Matrix([       // Predicted covariance
  [0.382, 0.0],
  [0.0, 0.00012]
]);

console.log("H * P * H^T + R (innovation covariance):");
const S = H.mmul(P_pred).mmul(H.transpose()).add(R);
console.log(S.to2DArray());

console.log("Innovation covariance inverse:");
const S_inv = S.inverse();
console.log(S_inv.to2DArray());

console.log("Kalman Gain = P * H^T * S^{-1}:");
const K = P_pred.mmul(H.transpose()).mmul(S_inv);
console.log(K.to2DArray());
console.log();

// Output in a format that can be compared with Python
console.log("=== Raw numerical values for comparison ===");
console.log("Test 2 (1x1 innovation covariance):");
console.log(`  Input: ${innovationCov.get(0, 0)}`);
console.log(`  Inverse: ${inv2.get(0, 0)}`);
console.log(`  Product: ${product2.get(0, 0)}`);

console.log("\nTest 5 (Kalman gain):");
console.log(`  K[0,0] = ${K.get(0, 0)}`);
console.log(`  K[1,0] = ${K.get(1, 0)}`);
