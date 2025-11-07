/**
 * Test processing a single measurement with pre-configured Kalman state.
 *
 * This test sets up the exact Kalman state from when measurement 4f07af66
 * gets replayed (after 120 measurements), then processes just that one measurement.
 *
 * This isolates the divergence to a single measurement with known state.
 */

import { test, expect } from "bun:test";
import { readFileSync } from "fs";
import { resolve } from "path";
import { InMemoryStore } from "../src/weight-processor-lib/core/database/memory_store";
import { processMeasurement } from "../src/weight-processor-lib/core/processing/processor";
import { floatsEqual, DEFAULT_EPSILON } from "../src/weight-processor-lib/core/utils";

interface KalmanStateFixture {
  description: string;
  target_measurement: {
    id: string;
    timestamp: string;
    weight: number;
    unit: string;
    source: string;
  };
  kalman_state: Record<string, any>;
  config: Record<string, any>;
}

function loadFixture(): KalmanStateFixture {
  const fixturePath = resolve(
    __dirname,
    "../../test_fixtures/kalman_state_replay_divergence.json"
  );
  const data = readFileSync(fixturePath, "utf-8");
  return JSON.parse(data);
}

function setupStoreWithState(fixture: KalmanStateFixture): {
  store: InMemoryStore;
  userId: string;
} {
  const store = new InMemoryStore();
  const userId = "test-user";

  // Set the state directly
  const state = { ...fixture.kalman_state };

  // Convert timestamp strings back to Date objects
  if (state.last_timestamp) {
    state.last_timestamp = new Date(state.last_timestamp);
  }
  if (state.last_accepted_timestamp) {
    state.last_accepted_timestamp = new Date(state.last_accepted_timestamp);
  }
  if (state.reset_timestamp) {
    state.reset_timestamp = new Date(state.reset_timestamp);
  }

  // TypeScript stores arrays as-is (no conversion needed like Python's numpy)

  store.saveState(userId, state);

  return { store, userId };
}

test("Single measurement with Kalman state", async () => {
  const fixture = loadFixture();
  const { store, userId } = setupStoreWithState(fixture);

  const measurement = fixture.target_measurement;
  const config = fixture.config;

  // Process the measurement
  const result = await processMeasurement(
    userId,
    measurement.weight,
    new Date(measurement.timestamp),
    measurement.source,
    config,
    measurement.unit,
    store,
    1.75 // user_height_m
  );

  // Print results for comparison with FULL PRECISION
  console.log("");
  console.log("=".repeat(60));
  console.log("TYPESCRIPT - Single Measurement Test (FULL PRECISION)");
  console.log("=".repeat(60));
  console.log(`Measurement ID: ${measurement.id.substring(0, 16)}...`);
  console.log(`Weight: ${measurement.weight}kg`);
  console.log(`Timestamp: ${measurement.timestamp}`);

  // Print initial state
  console.log("");
  console.log("=".repeat(60));
  console.log("KALMAN STATE BEFORE PROCESSING");
  console.log("=".repeat(60));
  const initialState = store.getState(userId);
  if (initialState) {
    console.log(`Last raw weight: ${initialState.last_raw_weight}`);
    if (initialState.last_state) {
      console.log("Last state (position, velocity):");
      initialState.last_state.forEach((component: any, i: number) => {
        console.log(`  Component ${i}: ${component}`);
      });
    }
    if (initialState.last_covariance) {
      console.log("Last covariance:");
      initialState.last_covariance.forEach((mat: any, i: number) => {
        console.log(`  Covariance ${i}:`);
        mat.forEach((row: any, j: number) => {
          console.log(`    Row ${j}: ${row}`);
        });
      });
    }
    console.log(`Measurements since reset: ${initialState.measurements_since_reset}`);
  }

  console.log("");
  console.log("=".repeat(60));
  console.log("PROCESSING RESULT");
  console.log("=".repeat(60));
  console.log(`  Accepted: ${result.accepted}`);

  // Print quality score with maximum precision
  if (result.quality_score !== null && result.quality_score !== undefined) {
    console.log(`  Quality score: ${result.quality_score.toFixed(18)}`);
  } else {
    console.log(`  Quality score: None`);
  }

  // Print quality components if available
  if (result.quality_components) {
    console.log("\n  Quality Components:");
    for (const [component, value] of Object.entries(result.quality_components)) {
      if (value !== null && value !== undefined) {
        console.log(`    ${component}: ${(value as number).toFixed(18)}`);
      } else {
        console.log(`    ${component}: None`);
      }
    }
  }

  // Print other result fields
  if (result.kalman_estimate !== null && result.kalman_estimate !== undefined) {
    console.log(`  Kalman estimate: ${result.kalman_estimate.toFixed(18)}`);
  } else {
    console.log(`  Kalman estimate: None`);
  }

  console.log(`  Rejection reason: ${result.rejection_reason ?? "N/A"}`);

  // Print final state after processing
  console.log("");
  console.log("=".repeat(60));
  console.log("KALMAN STATE AFTER PROCESSING");
  console.log("=".repeat(60));
  const finalState = store.getState(userId);
  if (finalState) {
    console.log(`Last raw weight: ${finalState.last_raw_weight}`);
    if (finalState.last_state) {
      console.log("Last state (position, velocity):");
      finalState.last_state.forEach((component: any, i: number) => {
        console.log(`  Component ${i}: ${component}`);
      });
    }
    if (finalState.last_covariance) {
      console.log("Last covariance:");
      finalState.last_covariance.forEach((mat: any, i: number) => {
        console.log(`  Covariance ${i}:`);
        mat.forEach((row: any, j: number) => {
          console.log(`    Row ${j}: ${row}`);
        });
      });
    }
  }

  console.log("=".repeat(60));

  // Assertions
  expect(result).toBeDefined();
  expect(result.accepted).toBeDefined();
  expect(result.quality_score).toBeDefined();

  // Epsilon-based comparison to verify we match Python implementation
  // within acceptable floating-point precision tolerance
  const expectedQualityScore = 0.009308080750420552; // From Python
  const qualityScore = result.quality_score ?? 0;

  // Use a larger epsilon for accumulated floating-point errors
  const testEpsilon = 1e-9; // Allow up to 1 billionth difference

  console.log("");
  console.log("Expected (from full 120-measurement run):");
  console.log("  Accepted: False");
  console.log("  Quality score: 0.009308");
  console.log("");
  console.log("Epsilon-based comparison:");
  console.log(`  Expected:  ${expectedQualityScore.toFixed(18)}`);
  console.log(`  Actual:    ${qualityScore.toFixed(18)}`);
  console.log(`  Difference: ${Math.abs(qualityScore - expectedQualityScore).toExponential(3)}`);
  console.log(`  Within epsilon (${testEpsilon}): ${floatsEqual(qualityScore, expectedQualityScore, testEpsilon)}`);

  // Verify quality score matches within epsilon
  expect(floatsEqual(qualityScore, expectedQualityScore, testEpsilon)).toBe(true);

  // Save results for comparison
  const testResult = {
    accepted: result.accepted,
    quality_score: result.quality_score,
    kalman_estimate: result.kalman_estimate,
  };

  console.log("");
  console.log("TypeScript result:");
  console.log(JSON.stringify(testResult, null, 2));
});

// Main function for direct execution
if (import.meta.main) {
  const fixture = loadFixture();
  const { store, userId } = setupStoreWithState(fixture);

  const measurement = fixture.target_measurement;
  const config = fixture.config;

  const result = await processMeasurement(
    userId,
    measurement.weight,
    new Date(measurement.timestamp),
    measurement.source,
    config,
    measurement.unit,
    store,
    1.75
  );

  console.log("");
  console.log("=".repeat(60));
  console.log("TYPESCRIPT - Single Measurement Test");
  console.log("=".repeat(60));
  console.log(`Measurement ID: ${measurement.id.substring(0, 16)}...`);
  console.log(`Weight: ${measurement.weight}kg`);
  console.log("");
  console.log("Result:");
  console.log(`  Accepted: ${result.accepted}`);
  const qualityStr =
    result.quality_score !== null && result.quality_score !== undefined
      ? result.quality_score.toFixed(9)
      : "N/A";
  console.log(`  Quality score: ${qualityStr}`);
  console.log(`  Kalman estimate: ${result.kalman_estimate ?? "N/A"}`);
  console.log(`  Rejection reason: ${result.rejection_reason ?? "N/A"}`);
}
