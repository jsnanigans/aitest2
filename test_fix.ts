#!/usr/bin/env bun
/**
 * Quick test to verify the reset_parameters fix
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { InMemoryStore, processMeasurement } from "./typescript_lib/src/index";

const csvFile = "/tmp/debug_user_full.csv";
const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";

const config = {
  database: { backend: "memory" },
  kalman: {
    initial_variance: 0.364,
    transition_covariance_weight: 0.018,
    transition_covariance_trend: 0.00012,
    observation_covariance: 5,
    reset: {
      initial: { enabled: true, observation_noise_multiplier: 20 },
      hard: { enabled: true, gap_threshold_days: 30, observation_noise_multiplier: 0.7 },
      soft: { enabled: true, observation_noise_multiplier: 0.7 },
    },
  },
  quality_scoring: { threshold: 0.46 },
  replay: { buffered_replay_enabled: false },
};

const csvContent = readFileSync(csvFile, "utf-8");
const rows = parse(csvContent, { columns: true, skip_empty_lines: true });

const measurements = rows
  .map((row: any) => ({
    measurementId: row.id,
    weight: parseFloat(row.value_quantity),
    timestamp: new Date(row.effective_date_time || row.timestamp),
    source: row.source_type || "unknown",
  }))
  .sort((a: any, b: any) => a.timestamp.getTime() - b.timestamp.getTime());

const stateStore = new InMemoryStore();
let count = 0;

for (const m of measurements) {
  count++;
  const result = await processMeasurement(userId, m.weight, m.timestamp, m.source, config, "kg", stateStore, null);

  if (count === 48) {
    const state = await stateStore.getState(userId);
    console.log(`\nMeasurement #48: ${m.measurementId}`);
    console.log(`  Weight: ${m.weight} kg`);
    console.log(`  Date: ${m.timestamp.toISOString()}`);
    console.log(`  Accepted: ${result.accepted}`);
    console.log(`  Quality Score: ${result.quality_score?.toFixed(6)}`);
    console.log(`  Kalman Estimate: ${result.kalman_estimate?.toFixed(6)}`);
    console.log(`  Innovation: ${result.innovation?.toFixed(6)}`);
    console.log(`  Kalman Variance: ${result.kalman_variance?.toFixed(6)}`);
    console.log(`  Trend: ${result.trend?.toFixed(8)}`);
    console.log(`\nState after processing:`);
    console.log(`  reset_parameters: ${JSON.stringify(state?.reset_parameters)}`);
    console.log(`  observation_covariance: ${state?.kalman_params?.observation_covariance?.[0]?.[0]}`);
  }
}

console.log(`\nProcessed ${count} measurements`);
