#!/usr/bin/env bun
/**
 * Test running Python Kalman filter via Pyodide and compare with ml-matrix
 */

import { loadPyodide } from 'pyodide';
import { Matrix } from 'ml-matrix';

async function testKalmanComparison() {
  console.log('Loading Pyodide...');
  const pyodide = await loadPyodide();
  await pyodide.loadPackage('numpy');
  console.log('✓ Pyodide + numpy loaded\n');

  // Test data: Same innovation covariance matrix from Kalman filter
  const testMatrix = [[1.234, 0.123], [0.123, 1.234]];

  // Test 1: Python (via Pyodide)
  console.log('=== Python (numpy via WASM) ===');
  const pyStart = Date.now();
  const pyResult = pyodide.runPython(`
import numpy as np
import json

m = np.array(${JSON.stringify(testMatrix)})
inv = np.linalg.inv(m)
json.dumps(inv.tolist())
  `);
  const pyTime = Date.now() - pyStart;
  const pyInverse = JSON.parse(pyResult);
  console.log('Inverse:', pyInverse);
  console.log(`Time: ${pyTime}ms\n`);

  // Test 2: TypeScript (ml-matrix)
  console.log('=== TypeScript (ml-matrix) ===');
  const tsStart = Date.now();
  const m = new Matrix(testMatrix);

  // ml-matrix doesn't have .inverse() - need to use the standalone function
  const mlmatrix = await import('ml-matrix');
  const tsInverse = (mlmatrix as any).inverse(m);
  const tsTime = Date.now() - tsStart;
  console.log('Inverse:', tsInverse.to2DArray());
  console.log(`Time: ${tsTime}ms\n`);

  // Test 3: Compare results
  console.log('=== Comparison ===');
  const pyFlat = pyInverse.flat();
  const tsFlat = tsInverse.to1DArray();

  let maxDiff = 0;
  let totalDiff = 0;

  for (let i = 0; i < pyFlat.length; i++) {
    const diff = Math.abs(pyFlat[i] - tsFlat[i]);
    maxDiff = Math.max(maxDiff, diff);
    totalDiff += diff;
  }

  console.log(`Max difference: ${maxDiff.toExponential(4)}`);
  console.log(`Average difference: ${(totalDiff / pyFlat.length).toExponential(4)}`);
  console.log(`Relative error: ${(maxDiff / Math.abs(pyFlat[0]) * 100).toFixed(8)}%`);

  if (maxDiff < 1e-15) {
    console.log('\n✅ Results are IDENTICAL (within floating-point precision)');
  } else if (maxDiff < 1e-10) {
    console.log('\n✅ Results are VERY CLOSE (acceptable tolerance)');
  } else {
    console.log('\n⚠️  Results differ beyond acceptable tolerance');
  }

  // Test 4: Multiple operations (simulating Kalman updates)
  console.log('\n=== Simulating 10 Kalman Filter Updates ===');

  const pyMultiStart = Date.now();
  pyodide.runPython(`
import numpy as np

# Simulate 10 matrix inversions (typical for Kalman filter)
state = np.array([60.0, 0.0])
for i in range(10):
    S = np.array([[1.0 + i*0.01, 0.1], [0.1, 1.0]])
    inv_S = np.linalg.inv(S)
    # Update state (simplified)
    state = state + np.array([0.5, 0.1])
  `);
  const pyMultiTime = Date.now() - pyMultiStart;
  console.log(`Python (Pyodide): ${pyMultiTime}ms`);

  const tsMultiStart = Date.now();
  let state = [60.0, 0.0];
  for (let i = 0; i < 10; i++) {
    const S = new Matrix([[1.0 + i*0.01, 0.1], [0.1, 1.0]]);
    const inv_S = (mlmatrix as any).inverse(S);
    // Update state (simplified)
    state = [state[0] + 0.5, state[1] + 0.1];
  }
  const tsMultiTime = Date.now() - tsMultiStart;
  console.log(`TypeScript (ml-matrix): ${tsMultiTime}ms`);
  console.log(`\nSpeed ratio: ${(pyMultiTime / tsMultiTime).toFixed(2)}x (Pyodide/native)`);
}

testKalmanComparison().catch(console.error);
