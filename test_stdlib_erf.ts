#!/usr/bin/env bun
/**
 * Test stdlib-js erf() function against known values
 * These values will be verified against scipy.special.erf in Python
 */

import erf from "@stdlib/math/base/special/erf";

interface ErfTestCase {
  input: number;
  output: number;
  description: string;
}

function main() {
  console.log("=== Testing stdlib-js erf() function ===\n");

  // Test cases covering common ranges
  const testInputs = [
    0.0,
    0.1,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    -0.1,
    -0.5,
    -1.0,
    -2.0,
    0.02, // Small positive
    -0.02, // Small negative
    5.0, // Large value
    -5.0, // Large negative
    // Values that might appear in chi-squared calculations
    0.6745, // ~50th percentile of normal
    1.96, // ~97.5th percentile of normal
    2.576, // ~99.5th percentile of normal
  ];

  const results: ErfTestCase[] = [];

  console.log("Input\t\tOutput (erf)\t\tPrecision");
  console.log("=".repeat(60));

  for (const x of testInputs) {
    const result = erf(x);
    results.push({
      input: x,
      output: result,
      description: `erf(${x})`,
    });

    console.log(`${x.toFixed(6)}\t${result.toFixed(15)}\t\t15 digits`);
  }

  // Write results to JSON for Python comparison
  Bun.write(
    "ts_erf_results.json",
    JSON.stringify(results, null, 2)
  );

  console.log("\n✅ Results written to ts_erf_results.json");
  console.log("Run Python comparison script to verify against scipy.special.erf");
}

main();
