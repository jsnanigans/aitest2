#!/usr/bin/env bun
/**
 * Test script to compare TypeScript processor output with expected output
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { execSync } from "node:child_process";

interface CsvRow {
  id: string;
  user_id: string;
  timestamp: string;
  effective_date_time: string;
  source_type: string;
  value_quantity: string;
  unit: string;
}

interface ComparisonResult {
  passed: boolean;
  expectedCount: number;
  actualCount: number;
  missingInActual: CsvRow[];
  extraInActual: CsvRow[];
  matchedCount: number;
}

/**
 * Parse timestamp to ISO format for comparison
 */
function normalizeTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    return date.toISOString();
  } catch {
    return ts;
  }
}

/**
 * Load and parse CSV file
 */
function loadCsv(filePath: string): CsvRow[] {
  if (!existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }

  const content = readFileSync(filePath, "utf-8");
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];

  return records;
}

/**
 * Create a unique key for a measurement (user_id + timestamp)
 */
function createKey(row: CsvRow): string {
  const timestamp = normalizeTimestamp(row.timestamp || row.effective_date_time);
  return `${row.user_id}|${timestamp}`;
}

/**
 * Compare two CSV files
 */
function compareCsvs(expectedPath: string, actualPath: string): ComparisonResult {
  const expected = loadCsv(expectedPath);
  const actual = loadCsv(actualPath);

  // Create maps keyed by user_id + timestamp
  const expectedMap = new Map<string, CsvRow>();
  for (const row of expected) {
    const key = createKey(row);
    expectedMap.set(key, row);
  }

  const actualMap = new Map<string, CsvRow>();
  for (const row of actual) {
    const key = createKey(row);
    actualMap.set(key, row);
  }

  // Find missing and extra measurements
  const missingInActual: CsvRow[] = [];
  const extraInActual: CsvRow[] = [];
  let matchedCount = 0;

  for (const [key, row] of expectedMap.entries()) {
    if (!actualMap.has(key)) {
      missingInActual.push(row);
    } else {
      matchedCount++;
    }
  }

  for (const [key, row] of actualMap.entries()) {
    if (!expectedMap.has(key)) {
      extraInActual.push(row);
    }
  }

  return {
    passed: missingInActual.length === 0 && extraInActual.length === 0,
    expectedCount: expected.length,
    actualCount: actual.length,
    missingInActual,
    extraInActual,
    matchedCount,
  };
}

/**
 * Format a row for display
 */
function formatRow(row: CsvRow): string {
  const timestamp = row.timestamp || row.effective_date_time;
  const weight = row.value_quantity;
  return `  ID: ${row.id.substring(0, 8)}... | Timestamp: ${timestamp} | Weight: ${weight} ${row.unit}`;
}

/**
 * Main test function
 */
async function main() {
  const USER_ID = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";
  const EXPECTED_CSV = "expected_output_test_user.csv";
  const FILTERED_CSV = "filtered_weights.csv";
  const INPUT_CSV = "test_user.csv";

  console.log("=== TypeScript Weight Processor Output Comparison Test ===\n");

  // Check if input file exists
  if (!existsSync(INPUT_CSV)) {
    console.error(`❌ Input CSV not found: ${INPUT_CSV}`);
    process.exit(1);
  }

  // Check if expected output exists
  if (!existsSync(EXPECTED_CSV)) {
    console.error(`❌ Expected output CSV not found: ${EXPECTED_CSV}`);
    process.exit(1);
  }

  // Run the processor
  console.log(`📊 Running TypeScript processor on ${INPUT_CSV}...`);
  console.log(`   Filtering to user: ${USER_ID}\n`);

  try {
    execSync(
      `bun run local_main.ts --csv-file="${INPUT_CSV}" --user-ids="${USER_ID}" --filtered-csv="${FILTERED_CSV}"`,
      {
        stdio: "inherit",
        cwd: process.cwd(),
      }
    );
  } catch (error) {
    console.error("\n❌ Processor execution failed");
    process.exit(1);
  }

  console.log("\n=== Comparing Output ===\n");

  // Check if output was created
  if (!existsSync(FILTERED_CSV)) {
    console.error(`❌ Output CSV not found: ${FILTERED_CSV}`);
    process.exit(1);
  }

  // Compare the files
  const result = compareCsvs(EXPECTED_CSV, FILTERED_CSV);

  console.log(`Expected count: ${result.expectedCount}`);
  console.log(`Actual count:   ${result.actualCount}`);
  console.log(`Matched count:  ${result.matchedCount}`);
  console.log(`Missing count:  ${result.missingInActual.length}`);
  console.log(`Extra count:    ${result.extraInActual.length}\n`);

  if (result.passed) {
    console.log("✅ PASS: Output matches expected output exactly!\n");
    process.exit(0);
  } else {
    console.log("❌ FAIL: Output does not match expected output\n");

    if (result.missingInActual.length > 0) {
      console.log(`\n📉 Missing measurements (in expected but not in actual): ${result.missingInActual.length}`);
      console.log("   First 10:");
      for (const row of result.missingInActual.slice(0, 10)) {
        console.log(formatRow(row));
      }
    }

    if (result.extraInActual.length > 0) {
      console.log(`\n📈 Extra measurements (in actual but not in expected): ${result.extraInActual.length}`);
      console.log("   First 10:");
      for (const row of result.extraInActual.slice(0, 10)) {
        console.log(formatRow(row));
      }
    }

    // Write detailed comparison report
    const reportPath = "comparison_report.json";
    const report = {
      timestamp: new Date().toISOString(),
      result: {
        passed: result.passed,
        expectedCount: result.expectedCount,
        actualCount: result.actualCount,
        matchedCount: result.matchedCount,
        missingCount: result.missingInActual.length,
        extraCount: result.extraInActual.length,
      },
      missing: result.missingInActual,
      extra: result.extraInActual,
    };

    writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`\n📄 Detailed comparison report saved to: ${reportPath}\n`);

    process.exit(1);
  }
}

// Run the test
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
