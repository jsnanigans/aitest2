#!/usr/bin/env bun
/**
 * Debug specific measurement IDs that differ between Python and TypeScript
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";

// Parse CSV and find specific measurements
const csvContent = readFileSync("test_user.csv", "utf8");
const records = parse(csvContent, { columns: true });

// IDs that differ
const problematicIds = [
  "726b441f-eb43-47d9-8f3c-845d164e5a5b", // TS accepts, Python rejects
  "1a98b2c3-e023-4757-8d01-d35ef2fb363e", // Python accepts, TS rejects
];

console.log("=== Problematic Measurements ===\n");

for (const id of problematicIds) {
  const measurement = records.find((r: any) => r.id === id);
  if (measurement) {
    console.log(`ID: ${id}`);
    console.log(`  Weight: ${measurement.value_quantity} ${measurement.unit}`);
    console.log(`  Timestamp: ${measurement.timestamp}`);
    console.log(`  Effective: ${measurement.effective_date_time}`);
    console.log(`  Source: ${measurement.source_type}`);

    // Find surrounding measurements for context
    const idx = records.findIndex((r: any) => r.id === id);
    if (idx > 0) {
      const prev = records[idx - 1];
      console.log(`  Previous measurement:`);
      console.log(`    Weight: ${prev.value_quantity} ${prev.unit}`);
      console.log(`    Time: ${prev.effective_date_time}`);
    }
    if (idx < records.length - 1) {
      const next = records[idx + 1];
      console.log(`  Next measurement:`);
      console.log(`    Weight: ${next.value_quantity} ${next.unit}`);
      console.log(`    Time: ${next.effective_date_time}`);
    }
    console.log("");
  }
}