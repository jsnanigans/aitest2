#!/usr/bin/env bun
/**
 * Phase 1: Sequential Comparison - Find First Divergence Point
 *
 * Processes measurements one-by-one and outputs Kalman state after each.
 * This helps identify EXACTLY where TS and Python first diverge.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { InMemoryStore } from "./typescript_lib/src/index";
import { WeightProcessorService } from "./services/weight_processor_service";
import type { MeasurementInput } from "./services/weight_processor_service";

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
    replay: { buffered_replay_enabled: true, buffer_hours: 24, max_buffer_measurements: 100 },
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

interface StateSnapshot {
  measurementIndex: number;
  measurementId: string;
  timestamp: string;
  weight: number;
  accepted: boolean;
  qualityScore: number;
  kalmanState: {
    weight: number;
    velocity: number;
  } | null;
  kalmanCovariance: number[][] | null;
  processNoise: number[][] | null;
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

  console.log(`Loaded ${measurements.length} measurements for user ${userId.substring(0, 12)}...`);

  // Initialize service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process ONE measurement at a time and capture state
  const snapshots: StateSnapshot[] = [];

  for (let i = 0; i < measurements.length; i++) {
    const measurement = measurements[i];

    // Process single measurement
    const response = await service.processBatch(userId, [measurement]);
    const result = response.results[0];

    // Get current Kalman state from store
    const state = await stateStore.getState(userId);

    const snapshot: StateSnapshot = {
      measurementIndex: i,
      measurementId: measurement.measurementId,
      timestamp: measurement.timestamp.toISOString(),
      weight: measurement.weight,
      accepted: result.accepted,
      qualityScore: result.quality_score,
      kalmanState: state?.kalman_filter ? {
        weight: state.kalman_filter.state[0],
        velocity: state.kalman_filter.state[1],
      } : null,
      kalmanCovariance: state?.kalman_filter?.covariance || null,
      processNoise: state?.kalman_filter?.process_noise || null,
    };

    snapshots.push(snapshot);

    // Log progress every 10 measurements
    if ((i + 1) % 10 === 0) {
      console.log(`Processed ${i + 1}/${measurements.length} measurements...`);
    }
  }

  // Write results to JSON
  const outputPath = "ts_sequential_states.json";
  writeFileSync(outputPath, JSON.stringify(snapshots, null, 2), "utf-8");

  console.log(`\n✅ Sequential state snapshots written to: ${outputPath}`);
  console.log(`Total measurements processed: ${snapshots.length}`);
  console.log(`Accepted: ${snapshots.filter(s => s.accepted).length}`);
  console.log(`Rejected: ${snapshots.filter(s => !s.accepted).length}`);
}

main().catch(console.error);
