#!/usr/bin/env bun
/**
 * Test stdlib-js statistical functions
 */

import mean from "@stdlib/stats/base/mean";
import stdev from "@stdlib/stats/base/stdev";
import variance from "@stdlib/stats/base/variance";

interface StatsTestCase {
  data: number[];
  mean: number;
  std: number;
  variance: number;
  description: string;
}

function median(data: number[]): number {
  const sorted = [...data].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function main() {
  console.log("=== Testing stdlib-js Statistical Functions ===\n");

  const testCases = [
    {
      name: "Simple sequence",
      data: [1, 2, 3, 4, 5],
    },
    {
      name: "Real weight data (kg)",
      data: [104.3, 104.3, 113.4, 117.9, 115.4],
    },
    {
      name: "Single value",
      data: [42.0],
    },
    {
      name: "Two values",
      data: [10.0, 20.0],
    },
    {
      name: "With negative values",
      data: [-5, 0, 5, 10],
    },
    {
      name: "Large variance",
      data: [1, 100, 1, 100, 1],
    },
    {
      name: "Disputed measurement context",
      // First few measurements from test data
      data: [104.3, 104.3], // Measurement #1 context
    },
  ];

  const results: StatsTestCase[] = [];

  for (const tc of testCases) {
    const meanVal = mean(tc.data.length, tc.data, 1);
    const stdVal = stdev(tc.data.length, 1, tc.data, 1); // ddof=1 for sample std
    const varVal = variance(tc.data.length, 1, tc.data, 1); // ddof=1 for sample variance
    const medianVal = median(tc.data);

    const result: StatsTestCase = {
      data: tc.data,
      mean: meanVal,
      std: stdVal,
      variance: varVal,
      description: tc.name,
    };

    results.push(result);

    console.log(`\n${tc.name}:`);
    console.log(`  Data: [${tc.data.join(", ")}]`);
    console.log(`  Mean: ${meanVal.toFixed(15)}`);
    console.log(`  Std Dev (sample): ${stdVal.toFixed(15)}`);
    console.log(`  Variance (sample): ${varVal.toFixed(15)}`);
    console.log(`  Median: ${medianVal.toFixed(15)}`);
  }

  // Write results
  Bun.write("ts_stats_results.json", JSON.stringify(results, null, 2));

  console.log("\n✅ Results written to ts_stats_results.json");
}

main();
