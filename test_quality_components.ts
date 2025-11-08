#!/usr/bin/env bun
/**
 * Extract and compare quality component scores for divergent measurement
 * Uses service layer which returns quality_details
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
    replay: { buffered_replay_enabled: false }, // Disable replay for clean test
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

  console.log("\n=== Quality Component Comparison ===\n");

  // Initialize service
  const stateStore = new InMemoryStore();
  const config = getDefaultConfig();
  const service = new WeightProcessorService(stateStore, config);

  // Process first 2 measurements one at a time
  const results: any[] = [];

  for (let i = 0; i < 2; i++) {
    const measurement = measurements[i];

    console.log(`\n[${i}] Processing ${measurement.measurementId.substring(0, 8)}...`);
    console.log(`    Timestamp: ${measurement.timestamp.toISOString()}`);
    console.log(`    Weight: ${measurement.weight} ${measurement.unit}`);

    const response = await service.processBatch(userId, [measurement]);
    const result = response.results[0];

    console.log(`    Quality Score: ${result.quality_score.toFixed(15)}`);
    console.log(`    Accepted: ${result.accepted}`);

    // Extract component scores
    if (result.quality_components) {
      const components = result.quality_components;
      const weights = config.quality_scoring.components;

      console.log(`\n    Component Breakdown:`);
      for (const [name, score] of Object.entries(components)) {
        const weight = weights[name]?.weight || 0;
        const contribution = (score as number) * weight;
        console.log(`      ${name.padEnd(25)} ${(score as number).toFixed(15)} × ${weight.toFixed(2)} = ${contribution.toFixed(15)}`);
      }

      results.push({
        index: i,
        measurementId: measurement.measurementId,
        timestamp: measurement.timestamp.toISOString(),
        weight: measurement.weight,
        qualityScore: result.quality_score,
        accepted: result.accepted,
        componentScores: components,
        componentWeights: Object.fromEntries(
          Object.keys(components).map(name => [name, weights[name]?.weight || 0])
        ),
        componentContributions: Object.fromEntries(
          Object.entries(components).map(([name, score]) => [
            name,
            (score as number) * (weights[name]?.weight || 0)
          ])
        ),
      });
    } else {
      console.log(`    ⚠️  No quality_components in result`);
      results.push({
        index: i,
        measurementId: measurement.measurementId,
        timestamp: measurement.timestamp.toISOString(),
        weight: measurement.weight,
        qualityScore: result.quality_score,
        accepted: result.accepted,
        error: "No quality_details available",
      });
    }
  }

  // Focus on measurement #1 (the divergent one)
  console.log("\n\n" + "=".repeat(80));
  console.log("🎯 DIVERGENT MEASUREMENT #1 BREAKDOWN");
  console.log("=".repeat(80));

  const divergent = results[1];
  console.log(`\nMeasurement ID: ${divergent.measurementId}`);
  console.log(`Timestamp: ${divergent.timestamp}`);
  console.log(`Weight: ${divergent.weight} kg`);
  console.log(`\nTypeScript Quality Score: ${divergent.qualityScore.toFixed(15)}`);
  console.log(`Expected Python Score:     0.977765... (31% higher!)`);

  if (divergent.componentScores) {
    console.log(`\nComponent Scores (TypeScript):`);
    console.log(`${"Component".padEnd(25)} ${"Raw Score".padEnd(20)} Weight  Contribution`);
    console.log("=".repeat(80));

    for (const [name, score] of Object.entries(divergent.componentScores)) {
      const weight = divergent.componentWeights[name];
      const contribution = divergent.componentContributions[name];
      console.log(
        `${name.padEnd(25)} ${(score as number).toFixed(15).padEnd(20)} ${weight.toFixed(2).padEnd(6)} ${contribution.toFixed(15)}`
      );
    }

    // Calculate sum
    const sum = Object.values(divergent.componentContributions).reduce((a: number, b) => a + (b as number), 0);
    console.log("=".repeat(80));
    console.log(`${"TOTAL".padEnd(25)} ${" ".repeat(20)} ${" ".repeat(6)} ${sum.toFixed(15)}`);
    console.log(`\n✅ Sum matches quality_score: ${Math.abs(sum - divergent.qualityScore) < 1e-10}`);
  }

  // Write results
  Bun.write("ts_quality_components.json", JSON.stringify(results, null, 2));
  console.log(`\n✅ Results written to ts_quality_components.json`);
  console.log(`\nNext: Run Python equivalent to compare component-by-component!`);
}

main().catch(console.error);
