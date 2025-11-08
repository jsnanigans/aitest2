#!/usr/bin/env bun
/**
 * Quality Component Breakdown for Divergent Measurement
 *
 * Processes the first divergent measurement (#1) and extracts
 * detailed breakdown of each quality component
 */

import { readFileSync } from "node:fs";
import { parse } from "csv-parse/sync";
import { InMemoryStore } from "./typescript_lib/src/index";
import { processMeasurement } from "./typescript_lib/src/weight-processor-lib/core/processing/processor";
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
    replay: { buffered_replay_enabled: false, buffer_hours: 24, max_buffer_measurements: 100 },
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
  const targetMeasurementId = "0bb4ca6c-d123-4461-8cae-a40297230843"; // Divergent measurement #1

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

  console.log(`\n=== Quality Component Breakdown ===`);
  console.log(`Target measurement: ${targetMeasurementId}`);
  console.log(`Processing first ${measurements.findIndex(m => m.measurementId === targetMeasurementId) + 1} measurements...\n`);

  // Initialize
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();

  // Process measurements up to and including target
  const targetIndex = measurements.findIndex(m => m.measurementId === targetMeasurementId);

  for (let i = 0; i <= targetIndex; i++) {
    const measurement = measurements[i];

    console.log(`\n[${i}] Processing ${measurement.measurementId.substring(0, 8)}...`);
    console.log(`   Timestamp: ${measurement.timestamp.toISOString()}`);
    console.log(`   Weight: ${measurement.weight} ${measurement.unit}`);

    const result = await processMeasurement(
      userId,
      measurement.weight,
      measurement.timestamp,
      measurement.source,
      config,
      measurement.unit,
      stateStore,
      null // user height
    );

    console.log(`   Accepted: ${result.accepted}`);
    console.log(`   Quality Score: ${result.quality_score.toFixed(15)}`);

    // For the target measurement, get detailed breakdown
    if (measurement.measurementId === targetMeasurementId) {
      console.log(`\n🎯 DETAILED BREAKDOWN FOR DIVERGENT MEASUREMENT:`);

      if (result.quality_details) {
        console.log(`\n   Quality Components:`);

        const components = result.quality_details.component_scores || {};
        const weights = config.quality_scoring.components;

        for (const [name, score] of Object.entries(components)) {
          const weight = weights[name]?.weight || 0;
          const contribution = (score as number) * weight;
          console.log(`     ${name}:`);
          console.log(`       Raw score: ${(score as number).toFixed(15)}`);
          console.log(`       Weight: ${weight}`);
          console.log(`       Contribution: ${contribution.toFixed(15)}`);
        }

        console.log(`\n   Final Quality Score: ${result.quality_score.toFixed(15)}`);
        console.log(`   Threshold: 0.5`);
        console.log(`   Decision: ${result.accepted ? "ACCEPT" : "REJECT"}`);

        // Check if detailed metrics available
        if (result.quality_details.metrics) {
          console.log(`\n   Additional Metrics:`);
          console.log(`   ${JSON.stringify(result.quality_details.metrics, null, 2)}`);
        }
      } else {
        console.log(`   ⚠️  No quality_details available in result`);
      }

      // Get Kalman state
      const state = await stateStore.getState(userId);
      if (state?.kalman_filter) {
        console.log(`\n   Kalman State:`);
        console.log(`     Weight: ${state.kalman_filter.state[0].toFixed(15)}`);
        console.log(`     Velocity: ${state.kalman_filter.state[1].toFixed(15)}`);
        console.log(`     Covariance: [[${state.kalman_filter.covariance[0].map(v => v.toFixed(6)).join(", ")}],`);
        console.log(`                  [${state.kalman_filter.covariance[1].map(v => v.toFixed(6)).join(", ")}]]`);
      }

      // Write detailed results
      const breakdown = {
        measurementId: measurement.measurementId,
        measurementIndex: i,
        timestamp: measurement.timestamp.toISOString(),
        weight: measurement.weight,
        unit: measurement.unit,
        source: measurement.source,
        qualityScore: result.quality_score,
        accepted: result.accepted,
        qualityDetails: result.quality_details,
        kalmanState: state?.kalman_filter ? {
          weight: state.kalman_filter.state[0],
          velocity: state.kalman_filter.state[1],
          covariance: state.kalman_filter.covariance,
        } : null,
      };

      Bun.write("ts_quality_breakdown.json", JSON.stringify(breakdown, null, 2));
      console.log(`\n✅ Detailed breakdown written to ts_quality_breakdown.json`);
    }
  }
}

main().catch(console.error);
