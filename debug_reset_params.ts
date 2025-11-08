#!/usr/bin/env bun
/**
 * Debug script to check reset_parameters and adaptive Kalman params
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

// Import kalman functions
import { getAdaptiveKalmanParams, getResetTimestamp } from "./typescript_lib/src/weight-processor-lib/core/processing/kalman";

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

  console.log("=== TypeScript Reset Parameters Debug ===\n");

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

  // Process measurements and log reset params for #46-48
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

    // Log reset parameters for measurements 46-48
    if (measurementCount >= 46 && measurementCount <= 48) {
      console.log(`\n${"=".repeat(80)}`);
      console.log(`MEASUREMENT #${measurementCount}: ${measurement.measurementId}`);
      console.log(`${"=".repeat(80)}`);
      console.log(`Date: ${measurement.timestamp.toISOString()}`);
      console.log(`Weight: ${measurement.weight} kg`);
      console.log(`Source: ${measurement.source}`);

      console.log(`\n--- STATE BEFORE ---`);
      if (stateBefore?.reset_parameters) {
        console.log(`Reset Parameters:`);
        console.log(`  observation_noise_multiplier: ${stateBefore.reset_parameters.observation_noise_multiplier}`);
        console.log(`  adaptation_days: ${stateBefore.reset_parameters.adaptation_days}`);
        console.log(`  adaptation_decay_rate: ${stateBefore.reset_parameters.adaptation_decay_rate}`);

        const resetTimestamp = getResetTimestamp(stateBefore);
        console.log(`Reset Timestamp: ${resetTimestamp?.toISOString() || 'N/A'}`);

        if (resetTimestamp) {
          const adaptiveParams = getAdaptiveKalmanParams(
            resetTimestamp,
            measurement.timestamp,
            config.kalman,
            7,
            stateBefore
          );
          console.log(`Adaptive Kalman Params:`);
          console.log(`  observation_covariance: ${adaptiveParams.observation_covariance}`);
          console.log(`  transition_covariance_weight: ${adaptiveParams.transition_covariance_weight}`);
        }
      } else {
        console.log(`Reset Parameters: NOT SET`);
      }

      if (stateBefore?.kalman_params) {
        console.log(`Kalman Params (stored):`);
        console.log(`  observation_covariance: ${stateBefore.kalman_params.observation_covariance?.[0]?.[0] || 'N/A'}`);
      } else {
        console.log(`Kalman Params: NOT SET`);
      }

      console.log(`\n--- PROCESSING RESULT ---`);
      console.log(`Accepted: ${result.accepted}`);
      console.log(`Quality Score: ${result.quality_score?.toFixed(6) || 'N/A'}`);

      console.log(`\n--- STATE AFTER ---`);
      if (stateAfter?.reset_parameters) {
        console.log(`Reset Parameters:`);
        console.log(`  observation_noise_multiplier: ${stateAfter.reset_parameters.observation_noise_multiplier}`);
      }

      if (stateAfter?.kalman_params) {
        console.log(`Kalman Params (stored):`);
        console.log(`  observation_covariance: ${stateAfter.kalman_params.observation_covariance?.[0]?.[0] || 'N/A'}`);
      }
    }
  }

  console.log(`\n\nProcessed all ${measurementCount} measurements`);
}

main().catch(console.error);
