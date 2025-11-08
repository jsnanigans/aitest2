#!/usr/bin/env bun
/**
 * Test validateNumber to find the exact issue
 */

import * as assert from '@stdlib/assert';
import { validateNumber, isFinite, isNaN } from './typescript_lib/src/weight-processor-lib/core/stdlib-utils';

console.log("\n=== Testing stdlib-utils validation functions ===\n");

// Test values from the actual measurement
const kalmanPrediction = 104.32616;
const innovationCovariance = 1.19736;
const previousWeight = 104.32616;
const timeDiffHours = 480;

console.log("Test 1: kalmanPrediction");
console.log(`  value: ${kalmanPrediction}`);
console.log(`  validateNumber: ${validateNumber(kalmanPrediction)}`);
console.log(`  isFinite: ${isFinite(kalmanPrediction)}`);
console.log(`  isNaN: ${isNaN(kalmanPrediction)}`);
console.log(`  native isFinite: ${Number.isFinite(kalmanPrediction)}`);
console.log(`  native isNaN: ${Number.isNaN(kalmanPrediction)}`);

console.log("\nTest 2: innovationCovariance");
console.log(`  value: ${innovationCovariance}`);
console.log(`  validateNumber: ${validateNumber(innovationCovariance)}`);
console.log(`  isFinite: ${isFinite(innovationCovariance)}`);
console.log(`  isNaN: ${isNaN(innovationCovariance)}`);
console.log(`  native isFinite: ${Number.isFinite(innovationCovariance)}`);
console.log(`  native isNaN: ${Number.isNaN(innovationCovariance)}`);

console.log("\nTest 3: previousWeight");
console.log(`  value: ${previousWeight}`);
console.log(`  validateNumber: ${validateNumber(previousWeight)}`);
console.log(`  isFinite: ${isFinite(previousWeight)}`);
console.log(`  isNaN: ${isNaN(previousWeight)}`);

console.log("\nTest 4: timeDiffHours");
console.log(`  value: ${timeDiffHours}`);
console.log(`  validateNumber: ${validateNumber(timeDiffHours)}`);
console.log(`  isFinite: ${isFinite(timeDiffHours)}`);
console.log(`  isNaN: ${isNaN(timeDiffHours)}`);

// Check what stdlib assert actually exports
console.log("\n=== stdlib/assert exports ===");
console.log("Available functions:", Object.keys(assert).filter(k => k.includes('isnan') || k.includes('isNaN') || k.includes('finite')));

// Test the actual assert functions
console.log("\n=== Direct stdlib/assert tests ===");
console.log("assert.isFinite:", typeof (assert as any).isFinite);
console.log("assert.isnan:", typeof (assert as any).isnan);
console.log("assert.isNaN:", typeof (assert as any).isNaN);
console.log("assert.isNumber:", typeof (assert as any).isNumber);

if ((assert as any).isFinite) {
  console.log("(assert as any).isFinite(104.32616):", (assert as any).isFinite(104.32616));
}

if ((assert as any).isnan) {
  console.log("(assert as any).isnan(104.32616):", (assert as any).isnan(104.32616));
}
