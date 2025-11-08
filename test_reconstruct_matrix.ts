#!/usr/bin/env bun
/**
 * Test reconstructMatrix with corrupted data
 */

import { Matrix } from "ml-matrix";

function reconstructMatrix(data: any): Matrix {
  if (data instanceof Matrix) {
    return data;
  }

  // Handle serialized ml-matrix format
  if (data && typeof data === 'object' && 'rows' in data && 'columns' in data) {
    // Convert the serialized data back to a 2D array
    const rows = data.rows;
    const cols = data.columns;
    const arr: number[][] = [];

    if (data.data && Array.isArray(data.data)) {
      // Data is stored as array of row objects with numeric keys
      for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
          const val = data.data[i]?.[j] ?? data.data[i]?.[j.toString()] ?? 0;
          row.push(val);
        }
        arr.push(row);
      }
    }

    return new Matrix(arr);
  }

  // Fallback: try to construct directly
  return new Matrix(data);
}

console.log("\n=== Testing reconstructMatrix ===\n");

// Test with corrupted data (as seen in logs)
const corrupted = {
  data: [{"0": 104.32616}, {"0": 0}],
  rows: 2,
  columns: 1
};

console.log("Input (corrupted):", JSON.stringify(corrupted));

const mat = reconstructMatrix(corrupted);
console.log("\nReconstructed Matrix:");
console.log("  Rows:", mat.rows);
console.log("  Columns:", mat.columns);
console.log("  get(0,0):", mat.get(0, 0));
console.log("  get(1,0):", mat.get(1, 0));

// Test the actual data access
console.log("\nDebug data access:");
console.log("  corrupted.data[0]:", corrupted.data[0]);
console.log("  corrupted.data[0][0]:", (corrupted.data[0] as any)[0]);
console.log("  corrupted.data[0]['0']:", (corrupted.data[0] as any)["0"]);
