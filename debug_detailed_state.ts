#!/usr/bin/env bun
/**
 * Detailed Kalman state logging to find divergence point
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
      buffered_replay_enabled: false,  // Disable for clearer debugging
      buffer_hours: 24,
      max_buffer_measurements: 100,
    },
  };
}

async function main() {
  const csvFile = "/tmp/debug_user_full.csv";
  const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";

  console.log("=== TypeScript Detailed State Debug ===\n");

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

  // Initialize storage and config
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();

  // Process measurements and log state for #47-50
  let measurementCount = 0;

  for (const measurement of measurements) {
    measurementCount++;

    // Get state BEFORE processing
    const stateBefore = await stateStore.getState(userId);

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

    // Get state AFTER processing
    const stateAfter = await stateStore.getState(userId);

    // Log detailed state for measurements 47-50
    if (measurementCount >= 47 && measurementCount <= 50) {
      console.log(`\n${"=".repeat(80)}`);
      console.log(`MEASUREMENT #${measurementCount}: ${measurement.measurementId}`);
      console.log(`${"=".repeat(80)}`);
      console.log(`Date: ${measurement.timestamp.toISOString()}`);
      console.log(`Weight: ${measurement.weight} kg`);
      console.log(`Source: ${measurement.source}`);

      if (stateBefore) {
        console.log(`\n--- STATE BEFORE ---`);
        if (stateBefore.kalman_params) {
          const kp = stateBefore.kalman_params;
          console.log(`Kalman State (x):`);
          console.log(`  Weight estimate: ${kp.state_mean?.[0] ?? 'N/A'}`);
          console.log(`  Trend (velocity): ${kp.state_mean?.[1] ?? 'N/A'}`);
          console.log(`Kalman Covariance (P):`);
          console.log(`  P[0,0]: ${kp.state_covariance?.[0]?.[0] ?? 'N/A'}`);
          console.log(`  P[0,1]: ${kp.state_covariance?.[0]?.[1] ?? 'N/A'}`);
          console.log(`  P[1,0]: ${kp.state_covariance?.[1]?.[0] ?? 'N/A'}`);
          console.log(`  P[1,1]: ${kp.state_covariance?.[1]?.[1] ?? 'N/A'}`);
          console.log(`Process Noise (Q):`);
          console.log(`  Q[0,0]: ${kp.process_noise_covariance?.[0]?.[0] ?? 'N/A'}`);
          console.log(`  Q[1,1]: ${kp.process_noise_covariance?.[1]?.[1] ?? 'N/A'}`);
          console.log(`Observation Noise (R): ${kp.observation_covariance ?? 'N/A'}`);
        }
        if (stateBefore.last_raw_weight) {
          console.log(`Last raw weight: ${stateBefore.last_raw_weight}`);
        }
        if (stateBefore.last_timestamp) {
          console.log(`Last timestamp: ${stateBefore.last_timestamp}`);
        }
      }

      console.log(`\n--- PROCESSING RESULT ---`);
      console.log(`Accepted: ${result.accepted}`);
      console.log(`Quality Score: ${result.quality_score?.toFixed(6) ?? 'N/A'}`);
      if (result.quality_components) {
        console.log(`Components:`);
        console.log(`  kalman_fit: ${result.quality_components.kalman_fit?.toFixed(6) ?? 'N/A'}`);
        console.log(`  temporal_consistency: ${result.quality_components.temporal_consistency?.toFixed(6) ?? 'N/A'}`);
        console.log(`  anomaly_detection: ${result.quality_components.anomaly_detection?.toFixed(6) ?? 'N/A'}`);
        console.log(`  source_reliability: ${result.quality_components.source_reliability?.toFixed(6) ?? 'N/A'}`);
        console.log(`  trend_alignment: ${result.quality_components.trend_alignment?.toFixed(6) ?? 'N/A'}`);
      }
      if (result.kalman_estimate !== undefined) {
        console.log(`Kalman estimate: ${result.kalman_estimate.toFixed(6)}`);
      }
      if (result.kalman_variance !== undefined) {
        console.log(`Kalman variance: ${result.kalman_variance.toFixed(6)}`);
      }
      if (result.trend !== undefined) {
        console.log(`Trend (velocity): ${result.trend.toFixed(8)}`);
      }
      if (result.innovation !== undefined) {
        console.log(`Innovation: ${result.innovation.toFixed(6)}`);
      }
      if (result.normalized_innovation !== undefined) {
        console.log(`Normalized innovation: ${result.normalized_innovation.toFixed(6)}`);
      }

      if (stateAfter) {
        console.log(`\n--- STATE AFTER ---`);
        if (stateAfter.kalman_params) {
          const kp = stateAfter.kalman_params;
          console.log(`Kalman State (x):`);
          console.log(`  Weight estimate: ${kp.state_mean?.[0] ?? 'N/A'}`);
          console.log(`  Trend (velocity): ${kp.state_mean?.[1] ?? 'N/A'}`);
          console.log(`Kalman Covariance (P):`);
          console.log(`  P[0,0]: ${kp.state_covariance?.[0]?.[0] ?? 'N/A'}`);
          console.log(`  P[0,1]: ${kp.state_covariance?.[0]?.[1] ?? 'N/A'}`);
          console.log(`  P[1,0]: ${kp.state_covariance?.[1]?.[0] ?? 'N/A'}`);
          console.log(`  P[1,1]: ${kp.state_covariance?.[1]?.[1] ?? 'N/A'}`);
        }
      }
    }
  }

  console.log(`\n\nProcessed all ${measurementCount} measurements`);
}

main().catch(console.error);
