#!/usr/bin/env bun
/**
 * Inspect the state structure after first measurement
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

  console.log("\n=== State Inspection ===\n");

  // Initialize service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process first measurement
  console.log("[0] Processing first measurement...");
  await service.processBatch(userId, [measurements[0]]);

  // Get state after first measurement
  const state = await stateStore.getState(userId);

  console.log("\n=== State after first measurement ===");
  console.log("State keys:", Object.keys(state || {}));
  console.log("\nlast_state type:", typeof state?.last_state);
  console.log("last_state is Array:", Array.isArray(state?.last_state));
  console.log("last_state length:", (state?.last_state as any)?.length);

  if (state?.last_state) {
    console.log("\nlast_state contents:");
    console.log(JSON.stringify(state.last_state, null, 2));

    // Check if it's an array
    if (Array.isArray(state.last_state)) {
      console.log("\nlast_state is an array with", state.last_state.length, "elements");
      const lastElement = state.last_state[state.last_state.length - 1];
      console.log("Last element type:", typeof lastElement);
      console.log("Last element:", JSON.stringify(lastElement, null, 2));
    }
  }

  console.log("\nlast_covariance type:", typeof state?.last_covariance);
  console.log("last_covariance is Array:", Array.isArray(state?.last_covariance));

  if (state?.last_covariance) {
    console.log("\nlast_covariance contents:");
    console.log(JSON.stringify(state.last_covariance, null, 2));
  }

  console.log("\nkalman_params:");
  console.log(JSON.stringify(state?.kalman_params, null, 2));
}

main().catch(console.error);
