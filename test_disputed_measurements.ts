#!/usr/bin/env bun
/**
 * Test disputed measurements to see exact quality scores
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";
import {
  InMemoryStore,
} from "./typescript_lib/src/index";
import {
  WeightProcessorService,
  type MeasurementInput,
} from "./services/weight_processor_service";

// IDs that differ between TS and Python
const DISPUTED_IDS = [
  "5b022b9c-509e-4a9f-bd5c-7857733bf2f8", // TS accepts, Python rejects
  "726b441f-eb43-47d9-8f3c-845d164e5a5b", // TS accepts, Python rejects
  "d957b0de-58fc-4e96-b351-a81cfc10e54c", // TS accepts, Python rejects
  "df8e3da2-5f5d-4177-b535-4e4fc8e59bd0", // TS accepts, Python rejects
  "86233705-0332-44f1-bc69-bc796220f598", // Python accepts, TS rejects
];

function parseTimestamp(dateStr: string): Date {
  if (!dateStr) return new Date();
  try {
    if (dateStr.includes("T")) {
      const normalized = dateStr.replace("Z", "+00:00");
      return new Date(normalized);
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
    processing: {
      enable_validation: true,
      enable_quality_scoring: true,
    },
    reset: {
      time_gap_days: 30,
      weight_change_threshold_kg: 10,
    },
    snapshot: {
      interval_hours: 24,
      periodic_enabled: true,
    },
    adaptive_noise: { enabled: true },
    replay: {
      buffered_replay_enabled: true,
      buffer_hours: 24,
      max_buffer_measurements: 100,
    },
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
  const content = readFileSync(csvPath, "utf-8");
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];

  // Load all measurements for the user
  const measurements: MeasurementInput[] = [];
  const idToRow = new Map<string, CsvRow>();

  for (const row of records) {
    const measurementId = row.id || row.measurement_id;
    if (!measurementId) continue;

    idToRow.set(measurementId, row);

    const weightStr = (row.value_quantity || row.weight || "").trim();
    if (!weightStr || weightStr.toUpperCase() === "NULL") continue;

    const weight = parseFloat(weightStr);
    if (weight <= 0 || weight > 1000 || isNaN(weight) || !isFinite(weight)) continue;

    const dateStr = row.effective_date_time || row.effectiveDateTime || row.timestamp || "";
    const source = row.source_type || "unknown";
    const unit = (row.unit || "").trim();

    const timestamp = dateStr ? parseTimestamp(dateStr) : new Date();

    measurements.push({
      measurementId,
      weight,
      unit,
      timestamp,
      source,
    });
  }

  // Sort chronologically
  measurements.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Initialize service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process all measurements
  const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";
  const response = await service.processBatch(userId, measurements);

  // Find disputed measurements and their quality scores
  console.log("\n=== Disputed Measurements Quality Scores ===\n");

  for (const disputedId of DISPUTED_IDS) {
    const idx = measurements.findIndex((m) => m.measurementId === disputedId);
    if (idx === -1) {
      console.log(`${disputedId}: NOT FOUND`);
      continue;
    }

    const result = response.results[idx];
    const measurement = measurements[idx];
    const row = idToRow.get(disputedId);

    console.log(`ID: ${disputedId}`);
    console.log(`  Timestamp: ${measurement.timestamp.toISOString()}`);
    console.log(`  Weight: ${measurement.weight} ${measurement.unit}`);
    console.log(`  Quality Score: ${result.quality_score.toFixed(15)}`);
    console.log(`  Accepted: ${result.accepted}`);
    console.log(`  Threshold: 0.5`);
    console.log(`  Margin: ${(result.quality_score - 0.5).toFixed(15)}`);
    console.log();
  }

  console.log("\n=== Summary ===");
  console.log(`Total measurements processed: ${response.measurements_processed}`);
  console.log(`Total accepted: ${response.measurements_accepted}`);
  console.log(`Total rejected: ${response.measurements_rejected}`);
}

main().catch(console.error);
