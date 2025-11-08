#!/usr/bin/env bun
/**
 * Investigate the divergent measurements in detail
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { InMemoryStore } from "./typescript_lib/src/index";
import { WeightProcessorService, type MeasurementInput } from "./services/weight_processor_service";

// IDs that differ between implementations
const PROBLEMATIC_IDS = [
  '726b441f-eb43-47d9-8f3c-845d164e5a5b', // TS only
  '1a98b2c3-e023-4757-8d01-d35ef2fb363e', // Python only
  '510977fa-9d3f-4b50-a667-e676a0cc0791', // Python only
  '86233705-0332-44f1-bc69-bc796220f598', // Python only
];

function parseTimestamp(dateStr: string): Date {
  if (!dateStr) return new Date();
  try {
    if (dateStr.includes("T")) {
      return new Date(dateStr.replace("Z", "+00:00"));
    } else if (dateStr.includes(" ")) {
      return new Date(dateStr.replace(" ", "T") + "Z");
    } else {
      return new Date(dateStr + "T00:00:00Z");
    }
  } catch {
    return new Date();
  }
}

function getDefaultConfig(): any {
  return {
    database: { backend: "memory" },
    kalman: {
      initial_variance: 0.364,
      transition_covariance_weight: 0.018,
      transition_covariance_trend: 0.00015,
      observation_covariance: 3.49,
    },
    quality_scoring: {
      threshold: 0.5,
      components: {
        kalman_fit: { weight: 0.3, enabled: true },
        temporal_consistency: { weight: 0.25, enabled: true },
        anomaly_detection: { weight: 0.25, enabled: true },
        source_reliability: { weight: 0.1, enabled: true },
        trend_alignment: { weight: 0.1, enabled: true },
      },
    },
    processing: { enable_validation: true, enable_quality_scoring: true },
    reset: { time_gap_days: 30, weight_change_threshold_kg: 10 },
    snapshot: { interval_hours: 24, periodic_enabled: true },
    adaptive_noise: { enabled: true },
    replay: { buffered_replay_enabled: false },
  };
}

interface CsvRow {
  id?: string;
  measurement_id?: string;
  user_id: string;
  value_quantity?: string;
  weight?: string;
  unit: string;
  timestamp?: string;
  effective_date_time?: string;
  effectiveDateTime?: string;
  source_type: string;
  [key: string]: any;
}

async function main() {
  const csvPath = "test_user.csv";
  const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";

  // Load CSV
  const content = readFileSync(csvPath, "utf-8");
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];

  // Parse measurements
  const measurements: MeasurementInput[] = [];
  for (const row of records) {
    const measurementId = row.id || row.measurement_id;
    if (!measurementId || row.user_id !== userId) continue;

    const weightStr = (row.value_quantity || row.weight || "").trim();
    if (!weightStr || weightStr.toUpperCase() === "NULL") continue;

    const weight = parseFloat(weightStr);
    if (weight <= 0 || weight > 1000 || isNaN(weight) || !isFinite(weight)) continue;

    const dateStr = row.effective_date_time || row.effectiveDateTime || row.timestamp || "";
    const source = row.source_type || "unknown";
    const unit = (row.unit || "").trim();
    const timestamp = dateStr ? parseTimestamp(dateStr) : new Date();

    measurements.push({ measurementId, weight, unit, timestamp, source });
  }

  // Sort chronologically
  measurements.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Find the divergent measurements
  const divergentIds = [
    "726b441f-eb43-47d9-8f3c-845d164e5a5b", // TS accepts, Python rejects
    "1a98b2c3-e023-4757-8d01-d35ef2fb363e", // Python accepts, TS rejects
    "510977fa-9d3f-4b50-a667-e676a0cc0791", // Python accepts, TS rejects
    "86233705-0332-44f1-bc69-bc796220f598", // Python accepts, TS rejects
  ];

  console.log("\n=== Investigating Divergent Measurements ===\n");

  // Initialize service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process all measurements and track divergent ones
  for (const measurement of measurements) {
    const response = await service.processBatch(userId, [measurement]);
    const result = response.results[0];

    if (divergentIds.includes(measurement.measurementId)) {
      console.log(`\n${"=".repeat(80)}`);
      console.log(`Measurement: ${measurement.measurementId.substring(0, 8)}...`);
      console.log(`  Timestamp: ${measurement.timestamp.toISOString()}`);
      console.log(`  Weight: ${measurement.weight} ${measurement.unit}`);
      console.log(`  Source: ${measurement.source}`);
      console.log(`  Quality Score: ${result.quality_score.toFixed(15)}`);
      console.log(`  Accepted: ${result.accepted}`);
      console.log(`  Threshold: ${config.quality_scoring.threshold}`);

      if (result.quality_components) {
        console.log(`\n  Component Scores:`);
        for (const [name, score] of Object.entries(result.quality_components)) {
          const weight = config.quality_scoring.components[name]?.weight || 0;
          console.log(`    ${name.padEnd(25)} ${(score as number).toFixed(6)} × ${weight.toFixed(2)} = ${((score as number) * weight).toFixed(6)}`);
        }

        // Calculate sum
        const sum = Object.entries(result.quality_components).reduce(
          (acc, [name, score]) => acc + (score as number) * (config.quality_scoring.components[name]?.weight || 0),
          0
        );
        console.log(`    ${"TOTAL".padEnd(25)} ${" ".repeat(9)} ${" ".repeat(6)} ${sum.toFixed(6)}`);
        console.log(`\n  Sum vs Quality Score: ${sum.toFixed(15)} vs ${result.quality_score.toFixed(15)}`);
        console.log(`  Difference: ${Math.abs(sum - result.quality_score).toFixed(15)}`);
      }

      if (result.rejection_reason) {
        console.log(`  Rejection Reason: ${result.rejection_reason}`);
      }
    }
  }
}

main().catch(console.error);
