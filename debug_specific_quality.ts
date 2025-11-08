#!/usr/bin/env bun
/**
 * Debug specific quality scores for problematic measurements
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";

// Import from typescript_lib
import {
  InMemoryStore,
  processMeasurement,
  type ProcessingResult,
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
      buffered_replay_enabled: true,
      buffer_hours: 24,
      max_buffer_measurements: 100,
    },
  };
}

async function main() {
  const csvFile = "/tmp/debug_user_full.csv";
  const targetIds = [
    "726b441f-eb43-47d9-8f3c-845d164e5a5b", // TS accepts, Py rejects
    "1a98b2c3-e023-4757-8d01-d35ef2fb363e", // Py accepts, TS rejects
  ];

  console.log("=== TypeScript Quality Score Debug ===");
  console.log(`Target IDs: ${targetIds.join(", ")}`);
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

  console.log(`Loaded ${measurements.length} measurements`);

  // Sort chronologically
  measurements.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Initialize storage and service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process batch
  const response = await service.processBatch("ADC64C0B-CB46-41F9-BDA0-CC11A35942D7", measurements);

  console.log(`\nProcessed ${response.measurements_processed} measurements`);
  console.log(`Accepted: ${response.measurements_accepted}, Rejected: ${response.measurements_rejected}`);
  console.log(`Replays triggered: ${response.replay_metadata?.length || 0}`);

  // Find our target measurements in results
  console.log("\n=== Target Measurement Results ===");
  for (const targetId of targetIds) {
    const measurementIndex = measurements.findIndex((m) => m.measurementId === targetId);
    if (measurementIndex >= 0) {
      const result = response.results[measurementIndex];
      const measurement = measurements[measurementIndex];
      console.log(`\nID: ${targetId}`);
      console.log(`  Timestamp: ${measurement.timestamp.toISOString()}`);
      console.log(`  Weight: ${measurement.weight} ${measurement.unit}`);
      console.log(`  Accepted: ${result.accepted}`);
      console.log(`  Quality Score: ${result.quality_score}`);
      console.log(`  Threshold: 0.46`);
      console.log(`  Score vs Threshold: ${result.quality_score - 0.46} (${result.quality_score >= 0.46 ? "PASS" : "FAIL"})`);

      if (result.quality_components) {
        console.log(`  Components:`);
        for (const [key, value] of Object.entries(result.quality_components)) {
          console.log(`    ${key}: ${value}`);
        }
      }

      if (!result.accepted) {
        console.log(`  Rejection reason: ${result.rejection_reason || result.reason || "Unknown"}`);
      }
    } else {
      console.log(`\n⚠️  ${targetId} not found in measurements`);
    }
  }
}

main().catch(console.error);