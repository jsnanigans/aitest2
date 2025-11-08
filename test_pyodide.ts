#!/usr/bin/env bun
/**
 * Test Pyodide (Python via WASM) for running numpy calculations
 */

import { loadPyodide } from 'pyodide';

async function testPyodide() {
  console.log('Loading Pyodide (this may take a moment)...\n');
  const startLoad = Date.now();

  const pyodide = await loadPyodide();

  const loadTime = Date.now() - startLoad;
  console.log(`✓ Pyodide loaded in ${loadTime}ms\n`);

  // Test basic numpy operation
  console.log('Testing numpy matrix inversion...\n');
  const startCalc = Date.now();

  await pyodide.loadPackage('numpy');

  const result = pyodide.runPython(`
import numpy as np

# Test matrix inversion (same as our Kalman filter uses)
m = np.array([[1.0, 0.1], [0.1, 1.0]])
inv = np.linalg.inv(m)

# Verify
product = m @ inv
max_error = np.max(np.abs(product - np.eye(2)))

print(f"Original Matrix:\\n{m}")
print(f"\\nInverse:\\n{inv}")
print(f"\\nVerification (A * A^-1):\\n{product}")
print(f"\\nMax error from identity: {max_error:.4e}")

# Return the inverse as JSON
import json
json.dumps(inv.tolist())
  `);

  const calcTime = Date.now() - startCalc;
  console.log(`\n✓ Calculation completed in ${calcTime}ms\n`);

  console.log('Inverse matrix (as JS object):');
  console.log(JSON.parse(result));
}

testPyodide().catch(console.error);
