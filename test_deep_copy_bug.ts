#!/usr/bin/env bun
/**
 * Test the deepCopy bug with Float64Array
 */

console.log("\n=== Testing deepCopy with Float64Array ===\n");

function deepCopy<T>(obj: T): T {
  if (obj === null || obj === undefined) {
    return obj;
  }

  // Handle Date objects
  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any;
  }

  // Handle arrays
  if (Array.isArray(obj)) {
    return obj.map(item => deepCopy(item)) as any;
  }

  // Handle objects
  if (typeof obj === 'object') {
    const copy: any = {};
    for (const [key, value] of Object.entries(obj)) {
      copy[key] = deepCopy(value);
    }
    return copy;
  }

  // Primitives
  return obj;
}

// Test with Float64Array
const float64 = new Float64Array([104.32616]);
console.log("Original Float64Array:", float64);
console.log("  Type:", float64.constructor.name);
console.log("  Value:", float64[0]);

const copied = deepCopy(float64);
console.log("\nCopied:");
console.log("  Type:", typeof copied);
console.log("  Value:", JSON.stringify(copied));
console.log("  Is Float64Array:", copied instanceof Float64Array);

// Test with object containing Float64Array (like Matrix internals)
const matrixLike = {
  data: [new Float64Array([104.32616]), new Float64Array([0])],
  rows: 2,
  columns: 1
};

console.log("\n\nOriginal matrix-like object:");
console.log(JSON.stringify(matrixLike, null, 2));

const copiedMatrix = deepCopy(matrixLike);
console.log("\n\nCopied matrix-like object:");
console.log(JSON.stringify(copiedMatrix, null, 2));

console.log("\n\nCopied data[0]:");
console.log("  Type:", typeof copiedMatrix.data[0]);
console.log("  Is Float64Array:", copiedMatrix.data[0] instanceof Float64Array);
console.log("  Value:", copiedMatrix.data[0]);
