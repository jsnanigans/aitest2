/**
 * Test to reproduce TS/PY divergence on measurement 4f07af66.
 *
 * This test processes the full sequence of measurements from test_user.csv
 * up to and including the divergent measurement.
 */

import { describe, test, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { InMemoryStore, processMeasurement } from "../src/index";

interface Measurement {
  id: string;
  user_id: string;
  timestamp: string;
  weight: number;
  unit: string;
  source: string;
}

interface Fixture {
  description: string;
  target_measurement_id: string;
  user_id: string;
  total_measurements: number;
  measurements: Measurement[];
  config: any;
  expected_results: any;
}

function loadFixture(): Fixture {
  const fixturePath = join(__dirname, "../../test_fixtures/divergence_all_120_measurements.json");
  const content = readFileSync(fixturePath, "utf-8");
  return JSON.parse(content);
}

function parseTimestamp(dateStr: string): Date {
  return new Date(dateStr.replace("Z", "+00:00"));
}

describe("TS/PY Divergence", () => {
  test("TypeScript accepts divergent measurement 4f07af66 (requires all 120 measurements)", async () => {
    const fixture = loadFixture();
    const measurements = fixture.measurements;
    const config = fixture.config;
    const userId = fixture.user_id;
    const targetId = fixture.target_measurement_id;

    console.log(`\nProcessing ${measurements.length} measurements up to target...`);

    const store = new InMemoryStore();
    let targetResult: any = null;

    for (let i = 0; i < measurements.length; i++) {
      const m = measurements[i];
      const timestamp = parseTimestamp(m.timestamp);
      const weight = m.weight;

      const result = await processMeasurement(
        userId,
        weight,
        timestamp,
        m.source,
        config,
        m.unit,
        store,
        1.75
      );

      if (m.id === targetId) {
        targetResult = result;
        console.log(`\n[${i + 1}] Target measurement ${targetId.slice(0, 8)}:`);
        console.log(`    Weight: ${weight} kg`);
        console.log(`    Timestamp: ${timestamp.toISOString()}`);
        console.log(`    Accepted: ${result.accepted}`);
        console.log(`    Quality Score: ${result.quality_score ?? "N/A"}`);
      }
    }

    expect(targetResult).not.toBeNull();

    // Document the current behavior
    // TypeScript accepts this measurement
    expect(targetResult.accepted).toBe(true);

    // Quality score should be above threshold
    const quality = targetResult.quality_score;
    expect(quality).toBeDefined();
    expect(quality).toBeGreaterThanOrEqual(0.5);

    console.log(`\n✓ TypeScript accepts measurement ${targetId.slice(0, 8)} as expected`);
    console.log(`  Quality score: ${quality.toFixed(6)} (threshold: 0.5)`);
  });
});
