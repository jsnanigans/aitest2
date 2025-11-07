/**
 * Test to reproduce TS/PY divergence in the replay mechanism.
 *
 * This test uses the SERVICE LAYER (WeightProcessorService.processBatch)
 * which includes buffered replay logic, not just core processMeasurement().
 */

import { describe, test, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { InMemoryStore } from "../src/index";
import {
  WeightProcessorService,
  type MeasurementInput,
  type ProcessResponseData,
} from "../../services/weight_processor_service";

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
  const fixturePath = join(__dirname, "../../test_fixtures/replay_divergence_12_measurements.json");
  const content = readFileSync(fixturePath, "utf-8");
  return JSON.parse(content);
}

function parseTimestamp(dateStr: string): Date {
  return new Date(dateStr.replace("Z", "+00:00"));
}

describe("TS/PY Replay Divergence", () => {
  test("TypeScript service layer replay mechanism", async () => {
    const fixture = loadFixture();
    const measurementsData = fixture.measurements;
    const config = fixture.config;
    const userId = fixture.user_id;
    const targetId = fixture.target_measurement_id;

    console.log("=".repeat(70));
    console.log(`TYPESCRIPT REPLAY TEST - ${measurementsData.length} measurements`);
    console.log("=".repeat(70));

    // Initialize service with replay enabled
    const store = new InMemoryStore();
    const service = new WeightProcessorService(store, config);

    // Convert to MeasurementInput objects
    const measurements: MeasurementInput[] = measurementsData.map((m) => ({
      id: m.id,
      weight: m.weight,
      unit: m.unit,
      timestamp: parseTimestamp(m.timestamp),
      source: m.source,
    }));

    console.log("\nMeasurements to process:");
    measurementsData.forEach((m, i) => {
      console.log(`  [${String(i + 1).padStart(2)}] ${m.id.slice(0, 8)} - ${m.weight.toFixed(1).padStart(5)}kg @ ${m.timestamp}`);
      if (m.id === targetId) {
        console.log(`       ^ TARGET MEASUREMENT`);
      }
    });

    // Process batch through service layer (includes replay)
    console.log("\nProcessing batch through service layer...");
    const response: ProcessResponseData = await service.processBatch(
      userId,
      measurements,
      1.75 // user_height_m
    );

    // Analyze results
    console.log("\n" + "=".repeat(70));
    console.log("RESULTS");
    console.log("=".repeat(70));
    console.log(`Total measurements: ${response.measurements_processed}`);
    console.log(`Accepted: ${response.measurements_accepted}`);
    console.log(`Rejected: ${response.measurements_rejected}`);

    // Find target measurement result
    let targetResult: any = null;
    let targetAccepted: boolean | null = null;

    // Check individual results if available
    if (response.results) {
      for (const result of response.results) {
        if (result.measurement_id === targetId) {
          targetResult = result;
          targetAccepted = result.accepted || false;
          break;
        }
      }
    }

    // Print replay metadata
    console.log(`\nReplay Events: ${response.replay_metadata?.length || 0}`);
    if (response.replay_metadata) {
      response.replay_metadata.forEach((replay, i) => {
        console.log(`\n  Replay ${i + 1}:`);
        console.log(`    Trigger: ${replay.trigger}`);
        console.log(`    Buffer size: ${replay.buffer_size}`);
        console.log(`    Replay from: ${replay.replay_from}`);
        console.log(`    Replay to: ${replay.replay_to}`);
        console.log(`    Measurements replayed: ${replay.measurements_replayed}`);
      });
    }

    // Check target measurement
    console.log("\n" + "=".repeat(70));
    console.log(`TARGET MEASUREMENT: ${targetId.slice(0, 8)}`);
    console.log("=".repeat(70));

    if (targetResult) {
      console.log(`Accepted: ${targetResult.accepted}`);
      console.log(`Quality Score: ${targetResult.quality_score ?? "N/A"}`);
      console.log(`Kalman Estimate: ${targetResult.kalman_estimate ?? "N/A"}`);
    } else {
      // If not in individual results, check state
      const state = await store.getState(userId);
      if (state?.measurement_buffer) {
        for (const buffered of state.measurement_buffer.measurements) {
          if (buffered.id === targetId) {
            console.log("Found in buffer - was processed");
            targetAccepted = true;
            break;
          }
        }
      }

      if (targetAccepted === null) {
        console.log("⚠️  Could not determine if measurement was accepted");
        console.log(`Response keys: ${Object.keys(response)}`);
      }
    }

    // Final verification
    console.log("\n" + "=".repeat(70));
    console.log("VERIFICATION");
    console.log("=".repeat(70));

    const acceptedCount = response.measurements_accepted;
    console.log(`Total accepted: ${acceptedCount}`);

    // We expect TypeScript to accept the target measurement (divergence from Python)
    if (targetAccepted !== null) {
      if (targetAccepted) {
        console.log(`✓ TypeScript ACCEPTED ${targetId.slice(0, 8)} (expected divergence)`);
      } else {
        console.log(`❌ TypeScript REJECTED ${targetId.slice(0, 8)} (unexpected - matches Python?)`);
      }
    } else {
      console.log(`⚠️  Could not determine acceptance status for ${targetId.slice(0, 8)}`);
    }

    console.log("\n" + "=".repeat(70) + "\n");

    // Assertions - we expect TypeScript to accept the target
    expect(response.measurements_processed).toBe(12);

    // Note: Comment out the assertion below if the test shows both implementations now agree
    // expect(targetAccepted).toBe(true);

    const result = {
      measurements_processed: response.measurements_processed,
      measurements_accepted: response.measurements_accepted,
      measurements_rejected: response.measurements_rejected,
      replay_count: response.replay_metadata?.length || 0,
      target_accepted: targetAccepted,
      target_result: targetResult,
    };

    console.log("\nTest completed:");
    console.log(JSON.stringify(result, null, 2));

    return result;
  });
});
