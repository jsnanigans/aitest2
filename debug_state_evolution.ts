#!/usr/bin/env bun
/**
 * Debug Kalman state evolution to find where divergence begins
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";

// Import from typescript_lib
import {
  InMemoryStore,
  processMeasurement,
} from "./typescript_lib/src/index";

// Import service layer
import {
  WeightProcessorService,
  type MeasurementInput,
} from "./services/weight_processor_service";

function getDefaultConfig(): any {
  return {
    database: {
      backend: "memory",
      table_name: "weight-processor-state",
      region: "us-east-1",
    },
    kalman: {
      initial_variance: 0.364,
      transition_covariance_weight: 0.018,
      transition_covariance_trend: 0.00012,
      observation_covariance: 5,
      reset: {
        initial: {
          enabled: true,
          initial_variance_multiplier: 10,
          weight_noise_multiplier: 50,
          trend_noise_multiplier: 50,
          observation_noise_multiplier: 20,
          adaptation_measurements: 10,
          adaptation_days: 10,
          adaptation_decay_rate: 2.5,
        },
        hard: {
          enabled: true,
          gap_threshold_days: 30,
          initial_variance_multiplier: 3,
          weight_noise_multiplier: 5,
          trend_noise_multiplier: 50,
          observation_noise_multiplier: 0.7,
          adaptation_measurements: 10,
          adaptation_days: 7,
          adaptation_decay_rate: 2.5,
        },
        soft: {
          enabled: true,
          min_weight_change_kg: 5,
          cooldown_days: 3,
          trigger_sources: ["questionnaire", "care-team-upload"],
          initial_variance_multiplier: 2,
          weight_noise_multiplier: 5,
          trend_noise_multiplier: 20,
          observation_noise_multiplier: 0.7,
          adaptation_measurements: 15,
          adaptation_days: 10,
          adaptation_decay_rate: 2.5,
        },
      },
    },
    quality_scoring: {
      use_harmonic_mean: true,
      threshold: 0.46,
      components: {
        kalman_fit: { weight: 0.4, enabled: true },
        temporal_consistency: { weight: 0.3, enabled: true },
        anomaly_detection: { weight: 0.2, enabled: true },
        source_reliability: { weight: 0.05, enabled: true },
        trend_alignment: { weight: 0.05, enabled: true },
      },
      component_weights: {
        kalman_fit: 0.4,
        temporal_consistency: 0.3,
        anomaly_detection: 0.2,
        source_reliability: 0.05,
        trend_alignment: 0.05,
      },
    },
    replay: {
      buffered_replay_enabled: false,  // Disable replay for clearer debugging
      buffer_hours: 24,
      max_buffer_measurements: 100,
    },
  };
}

async function main() {
  const csvFile = "/tmp/debug_user_full.csv";
  const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";

  console.log("=== TypeScript State Evolution Debug ===");
  console.log("Disabling replay to see raw processing");
  console.log("");

  // Load CSV
  const csvContent = readFileSync(csvFile, "utf-8");
  const rows = parse(csvContent, {
    columns: true,
    skip_empty_lines: true,
  });

  // Parse measurements
  const measurements: MeasurementInput[] = rows.map((row: any) => ({
    measurementId: row.id,
    weight: parseFloat(row.value_quantity),
    unit: row.unit || "kg",
    timestamp: new Date(row.effective_date_time || row.timestamp),
    source: row.source_type || "unknown",
  }));

  // Sort chronologically
  measurements.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  console.log(`Loaded ${measurements.length} measurements`);
  console.log("");

  // Initialize storage and config
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();

  // Process one measurement at a time to track state
  let measurementCount = 0;

  for (const measurement of measurements) {
    measurementCount++;

    const result = await processMeasurement(
      userId,
      measurement.weight,
      measurement.timestamp,
      measurement.source,
      config,
      measurement.unit,
      stateStore,
      null
    );

    // Print state for measurements around the problematic ones
    const isNearJuly11 = measurement.timestamp >= new Date("2025-07-09") &&
                         measurement.timestamp <= new Date("2025-07-13");
    const isNearJuly26 = measurement.timestamp >= new Date("2025-07-24") &&
                          measurement.timestamp <= new Date("2025-07-28");

    if (isNearJuly11 || isNearJuly26) {
      console.log(`\n[${measurementCount}] ${measurement.measurementId}`);
      console.log(`  Date: ${measurement.timestamp.toISOString()}`);
      console.log(`  Weight: ${measurement.weight} kg`);
      console.log(`  Accepted: ${result.accepted}`);
      console.log(`  Quality: ${result.quality_score?.toFixed(4)}`);

      if (result.quality_components) {
        console.log(`  Components:`);
        console.log(`    kalman_fit: ${result.quality_components.kalman_fit?.toFixed(4)}`);
        console.log(`    temporal_consistency: ${result.quality_components.temporal_consistency?.toFixed(4)}`);
        console.log(`    trend_alignment: ${result.quality_components.trend_alignment?.toFixed(4)}`);
      }

      if (result.kalman_estimate !== undefined) {
        console.log(`  Kalman estimate: ${result.kalman_estimate.toFixed(4)}`);
      }
      if (result.trend !== undefined) {
        console.log(`  Trend (velocity): ${result.trend.toFixed(6)}`);
      }
    }
  }

  console.log(`\nProcessed all ${measurementCount} measurements`);
}

main().catch(console.error);