#!/usr/bin/env bun
/**
 * Test Matrix clone() and serialization
 */

import { Matrix } from "ml-matrix";

console.log("\n=== Testing Matrix Clone and Serialization ===\n");

// Create a Matrix
const original = new Matrix([[104.32616], [0]]);
console.log("Original Matrix:");
console.log("  Rows:", original.rows);
console.log("  Columns:", original.columns);
console.log("  get(0,0):", original.get(0, 0));
console.log("  get(1,0):", original.get(1, 0));

// Clone it
const cloned = original.clone();
console.log("\nCloned Matrix:");
console.log("  Rows:", cloned.rows);
console.log("  Columns:", cloned.columns);
console.log("  get(0,0):", cloned.get(0, 0));
console.log("  get(1,0):", cloned.get(1, 0));
console.log("  instanceof Matrix:", cloned instanceof Matrix);

// Test JSON serialization
const serialized = JSON.stringify(original);
console.log("\nJSON.stringify(original):");
console.log(serialized);

const clonedSerialized = JSON.stringify(cloned);
console.log("\nJSON.stringify(cloned):");
console.log(clonedSerialized);

// Test what happens when cloned matrix is in an array
const stateArray = [original.clone(), original.clone()];
const stateArraySerialized = JSON.stringify(stateArray);
console.log("\nJSON.stringify([cloned, cloned]):");
console.log(stateArraySerialized);

// Parse it back
const parsed = JSON.parse(stateArraySerialized);
console.log("\nParsed back:");
console.log("  Type:", typeof parsed);
console.log("  Is Array:", Array.isArray(parsed));
console.log("  Length:", parsed.length);
console.log("  First element:", JSON.stringify(parsed[0]));
console.log("  First element type:", typeof parsed[0]);
console.log("  First element instanceof Matrix:", parsed[0] instanceof Matrix);
