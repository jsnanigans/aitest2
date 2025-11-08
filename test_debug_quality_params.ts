#!/usr/bin/env bun
/**
 * Debug quality scoring parameters to see what's being passed
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { InMemoryStore } from "./typescript_lib/src/index";
import { WeightProcessorService, type MeasurementInput } from "./services/weight_processor_service";

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
  // Enable verbose logging
  process.env.VERBOSE_LOGGING = "true";

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

  console.log("\n=== Debug Quality Scoring Parameters ===\n");

  // Initialize service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process first 2 measurements
  for (let i = 0; i < 2; i++) {
    const measurement = measurements[i];

    console.log(`\n\n${"=".repeat(80)}`);
    console.log(`[${i}] Processing ${measurement.measurementId.substring(0, 8)}...`);
    console.log(`    Timestamp: ${measurement.timestamp.toISOString()}`);
    console.log(`    Weight: ${measurement.weight} ${measurement.unit}`);
    console.log("=".repeat(80));

    const response = await service.processBatch(userId, [measurement]);
    const result = response.results[0];

    console.log(`\n[Result] Quality Score: ${result.quality_score.toFixed(15)}`);
    console.log(`[Result] Accepted: ${result.accepted}`);

    if (result.quality_components) {
      console.log(`\n[Result] Component Scores:`);
      for (const [name, score] of Object.entries(result.quality_components)) {
        console.log(`  ${name.padEnd(25)} ${(score as number).toFixed(15)}`);
      }
    }
  }
}

main().catch(console.error);
