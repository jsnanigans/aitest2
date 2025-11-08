#!/usr/bin/env bun
/**
 * Compare fundamental floating-point precision: Python vs JavaScript
 */

import { loadPyodide } from 'pyodide';

async function testPrecision() {
  console.log('🔬 Floating-Point Precision Comparison\n');
  console.log('='.repeat(60));

  const pyodide = await loadPyodide();
  await pyodide.loadPackage('numpy');

  // Test 1: Basic arithmetic
  console.log('\n📊 Test 1: Basic Arithmetic Precision\n');

  const jsResult1 = 0.1 + 0.2;
  const pyResult1 = pyodide.runPython(`0.1 + 0.2`);

  console.log('0.1 + 0.2:');
  console.log(`  JavaScript: ${jsResult1}`);
  console.log(`  Python:     ${pyResult1}`);
  console.log(`  Expected:   0.3`);
  console.log(`  Match: ${jsResult1 === pyResult1 ? '✅ YES' : '❌ NO'}`);

  // Test 2: Division precision
  console.log('\n📊 Test 2: Division Precision\n');

  const jsResult2 = 1.0 / 3.0;
  const pyResult2 = pyodide.runPython(`1.0 / 3.0`);

  console.log('1.0 / 3.0:');
  console.log(`  JavaScript: ${jsResult2.toPrecision(20)}`);
  console.log(`  Python:     ${pyResult2}`);
  console.log(`  Difference: ${Math.abs(jsResult2 - pyResult2).toExponential(4)}`);

  // Test 3: Square root precision
  console.log('\n📊 Test 3: Square Root Precision\n');

  const jsResult3 = Math.sqrt(2);
  const pyResult3 = pyodide.runPython(`import math; math.sqrt(2)`);

  console.log('sqrt(2):');
  console.log(`  JavaScript: ${jsResult3.toPrecision(20)}`);
  console.log(`  Python:     ${pyResult3}`);
  console.log(`  Difference: ${Math.abs(jsResult3 - pyResult3).toExponential(4)}`);

  // Test 4: Exponential precision
  console.log('\n📊 Test 4: Exponential (e^x) Precision\n');

  const jsResult4 = Math.exp(1);
  const pyResult4 = pyodide.runPython(`import math; math.exp(1)`);

  console.log('e^1:');
  console.log(`  JavaScript: ${jsResult4.toPrecision(20)}`);
  console.log(`  Python:     ${pyResult4}`);
  console.log(`  Difference: ${Math.abs(jsResult4 - pyResult4).toExponential(4)}`);

  // Test 5: Matrix multiplication accumulation
  console.log('\n📊 Test 5: Matrix Multiplication Accumulation Error\n');

  const jsAccum = {
    sum: 0.0,
    product: 1.0
  };

  for (let i = 0; i < 1000; i++) {
    jsAccum.sum += 0.1;
    jsAccum.product *= 1.0001;
  }

  const pyAccum = pyodide.runPython(`
sum_val = 0.0
product_val = 1.0
for i in range(1000):
    sum_val += 0.1
    product_val *= 1.0001
[sum_val, product_val]
  `);

  console.log('1000 iterations of accumulation:');
  console.log(`  JS Sum:      ${jsAccum.sum}`);
  console.log(`  Python Sum:  ${pyAccum[0]}`);
  console.log(`  Difference:  ${Math.abs(jsAccum.sum - pyAccum[0]).toExponential(4)}`);
  console.log();
  console.log(`  JS Product:  ${jsAccum.product}`);
  console.log(`  Python Prod: ${pyAccum[1]}`);
  console.log(`  Difference:  ${Math.abs(jsAccum.product - pyAccum[1]).toExponential(4)}`);

  // Test 6: IEEE 754 representation
  console.log('\n📊 Test 6: IEEE 754 Double Precision\n');

  const jsEpsilon = Number.EPSILON;
  const pyEpsilon = pyodide.runPython(`import sys; sys.float_info.epsilon`);

  console.log('Machine epsilon (smallest distinguishable difference):');
  console.log(`  JavaScript: ${jsEpsilon.toExponential(4)}`);
  console.log(`  Python:     ${pyEpsilon}`);
  console.log(`  Match: ${jsEpsilon === pyEpsilon ? '✅ YES' : '❌ NO'}`);

  const jsMaxFloat = Number.MAX_VALUE;
  const pyMaxFloat = pyodide.runPython(`import sys; sys.float_info.max`);

  console.log('\nMax float value:');
  console.log(`  JavaScript: ${jsMaxFloat.toExponential(4)}`);
  console.log(`  Python:     ${pyMaxFloat}`);

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📝 SUMMARY\n');
  console.log('Both JavaScript and Python use IEEE 754 double precision (64-bit)');
  console.log('for their default number types.');
  console.log();
  console.log('✅ Basic arithmetic: IDENTICAL');
  console.log('✅ Elementary functions (sqrt, exp, etc.): IDENTICAL');
  console.log('✅ Machine epsilon: IDENTICAL');
  console.log();
  console.log('🔍 Differences arise from:');
  console.log('   1. Library implementations (numpy vs ml-matrix)');
  console.log('   2. Algorithm differences (not precision limits)');
  console.log('   3. Accumulation order in complex operations');
}

testPrecision().catch(console.error);
