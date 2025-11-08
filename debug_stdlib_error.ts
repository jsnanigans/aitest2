#!/usr/bin/env bun

import { base as statsBase } from '@stdlib/stats';

// Test different scenarios to reproduce the error

console.log('Test 1: Normal array');
try {
  const arr1 = [1, 2, 3, 4, 5];
  const result = (statsBase as any).mean(arr1.length, arr1, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}

console.log('\nTest 2: Empty array');
try {
  const arr2: number[] = [];
  const result = (statsBase as any).mean(arr2.length, arr2, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}

console.log('\nTest 3: Array with NaN');
try {
  const arr3 = [1, NaN, 3];
  const result = (statsBase as any).mean(arr3.length, arr3, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}

console.log('\nTest 4: Undefined as array');
try {
  const arr4 = undefined;
  const result = (statsBase as any).mean((arr4 as any)?.length || 0, arr4, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}

console.log('\nTest 5: NaN as length');
try {
  const arr5 = [1, 2, 3];
  const result = (statsBase as any).mean(NaN, arr5, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}

console.log('\nTest 6: Negative length');
try {
  const arr6 = [1, 2, 3];
  const result = (statsBase as any).mean(-1, arr6, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}

console.log('\nTest 7: Array with undefined values');
try {
  const arr7 = [1, undefined, 3] as any;
  const result = (statsBase as any).mean(arr7.length, arr7, 1);
  console.log('  Result:', result);
} catch (e) {
  console.error('  Error:', (e as Error).message);
}
