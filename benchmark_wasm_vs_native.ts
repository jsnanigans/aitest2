#!/usr/bin/env bun
/**
 * Comprehensive benchmark: Pyodide vs ml-matrix vs native Python
 */

import { loadPyodide } from 'pyodide';
import { inverse, Matrix } from 'ml-matrix';

async function benchmark() {
  console.log('🔬 Benchmarking WASM vs Native Linear Algebra\n');
  console.log('='.repeat(60));

  // Load Pyodide
  console.log('\n📦 Loading Pyodide + numpy...');
  const loadStart = Date.now();
  const pyodide = await loadPyodide();
  await pyodide.loadPackage('numpy');
  const loadTime = Date.now() - loadStart;
  console.log(`✓ Loaded in ${loadTime}ms`);

  // Test configurations
  const iterations = 100;
  const testMatrix = [[1.234, 0.123], [0.123, 1.234]];

  console.log(`\n📊 Running ${iterations} matrix inversions...\n`);

  // Benchmark 1: Pyodide (Python/numpy via WASM)
  console.log('1️⃣  Python/numpy via Pyodide (WASM)');
  pyodide.runPython(`
import numpy as np
import time

m = np.array(${JSON.stringify(testMatrix)})
  `);

  const pyStart = Date.now();
  pyodide.runPython(`
for i in range(${iterations}):
    inv = np.linalg.inv(m)
  `);
  const pyTime = Date.now() - pyStart;
  console.log(`   Time: ${pyTime}ms`);
  console.log(`   Per operation: ${(pyTime / iterations).toFixed(3)}ms`);

  // Benchmark 2: ml-matrix (TypeScript/JavaScript)
  console.log('\n2️⃣  ml-matrix (native JavaScript)');
  const m = new Matrix(testMatrix);

  const mlStart = Date.now();
  for (let i = 0; i < iterations; i++) {
    const inv = inverse(m);
  }
  const mlTime = Date.now() - mlStart;
  console.log(`   Time: ${mlTime}ms`);
  console.log(`   Per operation: ${(mlTime / iterations).toFixed(3)}ms`);

  // Results
  console.log('\n' + '='.repeat(60));
  console.log('📈 PERFORMANCE SUMMARY\n');
  console.log(`Pyodide overhead: ${(pyTime / mlTime).toFixed(1)}x slower than ml-matrix`);
  console.log(`Pyodide load time: ${loadTime}ms (one-time cost)`);

  // Realistic workload test
  console.log('\n' + '='.repeat(60));
  console.log('🎯 REALISTIC WORKLOAD: Processing 120 measurements\n');

  // Simulate our actual use case: 120 measurements with Kalman updates
  console.log('Simulating weight processing pipeline...');

  const realisticPyStart = Date.now();
  pyodide.runPython(`
import numpy as np

# Simulate 120 measurements with Kalman filter updates
for i in range(120):
    # Innovation covariance (varies slightly each time)
    S = np.array([[1.0 + i*0.001, 0.1], [0.1, 1.0]])
    S_inv = np.linalg.inv(S)
  `);
  const realisticPyTime = Date.now() - realisticPyStart;
  console.log(`   Pyodide: ${realisticPyTime}ms`);

  const realisticMlStart = Date.now();
  for (let i = 0; i < 120; i++) {
    // Innovation covariance
    const S = new Matrix([[1.0 + i*0.001, 0.1], [0.1, 1.0]]);
    const S_inv = inverse(S);
  }
  const realisticMlTime = Date.now() - realisticMlStart;
  console.log(`   ml-matrix: ${realisticMlTime}ms`);

  console.log(`\n   Overhead: ${(realisticPyTime / realisticMlTime).toFixed(1)}x`);

  // Final recommendation
  console.log('\n' + '='.repeat(60));
  console.log('💡 RECOMMENDATION\n');

  const totalTimeWithPyodide = loadTime + realisticPyTime;
  const totalTimeWithMlMatrix = realisticMlTime;

  console.log(`Total time (including load):`);
  console.log(`   Pyodide: ${totalTimeWithPyodide}ms (${loadTime}ms load + ${realisticPyTime}ms compute)`);
  console.log(`   ml-matrix: ${totalTimeWithMlMatrix}ms`);
  console.log(`\nSpeed difference: ${(totalTimeWithPyodide / totalTimeWithMlMatrix).toFixed(1)}x`);

  if (pyTime / mlTime > 20) {
    console.log('\n❌ Pyodide is TOO SLOW for production use');
  } else if (pyTime / mlTime > 5) {
    console.log('\n⚠️  Pyodide has significant overhead but may be acceptable');
  } else {
    console.log('\n✅ Pyodide overhead is acceptable');
  }
}

benchmark().catch(console.error);
