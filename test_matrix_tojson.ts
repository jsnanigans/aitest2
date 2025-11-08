#!/usr/bin/env bun
/**
 * Test Matrix toJSON method
 */

import { Matrix } from "ml-matrix";

console.log("\n=== Testing Matrix toJSON ===\n");

const mat = new Matrix([[104.32616], [0]]);

console.log("Matrix object:");
console.log(mat);

console.log("\n\nHas toJSON?", typeof (mat as any).toJSON);
if (typeof (mat as any).toJSON === 'function') {
  console.log("toJSON():", (mat as any).toJSON());
}

console.log("\n\nObject.keys(mat):", Object.keys(mat));
console.log("\n\nInspecting internal structure:");
for (const key of Object.keys(mat)) {
  console.log(`  ${key}:`, (mat as any)[key]);
}

// Test what spread operator does
console.log("\n\n{...mat}:");
console.log({...mat});

// Test Object.assign
console.log("\n\nObject.assign({}, mat):");
console.log(Object.assign({}, mat));

// Test if deep copy preserves Matrix
const copy1 = JSON.parse(JSON.stringify(mat));
console.log("\n\nJSON.parse(JSON.stringify(mat)):");
console.log("Type:", typeof copy1);
console.log("instanceof Matrix:", copy1 instanceof Matrix);
console.log("Value:", copy1);

// Test spread in object
const obj = { state: mat };
const objCopy = { ...obj };
console.log("\n\nSpread copy of object containing Matrix:");
console.log("objCopy.state instanceof Matrix:", objCopy.state instanceof Matrix);
console.log("objCopy.state:", objCopy.state);
