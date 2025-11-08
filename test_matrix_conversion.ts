#!/usr/bin/env bun
/**
 * Test Matrix conversion from 2D array
 */

import { Matrix } from "ml-matrix";

console.log("\n=== Testing Matrix Conversion ===\n");

// This is what we get from the state
const currentState = [[104.32616], [0]];

console.log("Input 2D array:", JSON.stringify(currentState));
console.log("Is Array:", Array.isArray(currentState));
console.log("First element is Array:", Array.isArray(currentState[0]));

// Try to create Matrix
const mat = new Matrix(currentState);

console.log("\nCreated Matrix:");
console.log("Rows:", mat.rows);
console.log("Columns:", mat.columns);
console.log("Matrix toString():", mat.toString());

console.log("\nExtracting values:");
console.log("mat.get(0, 0):", mat.get(0, 0));
console.log("mat.get(1, 0):", mat.get(1, 0));

// Test if we need to access it differently
console.log("\nAlternative access:");
console.log("currentState[0][0]:", currentState[0][0]);
console.log("currentState[1][0]:", currentState[1][0]);
