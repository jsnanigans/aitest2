#!/usr/bin/env bun
/**
 * Debug temporal consistency inputs for divergent measurement.
 * Focus on measurement ID: 726b441f-eb43-47d9-8f3c-845d164e5a5b
 */

import { readFileSync } from "fs";
import { parse } from "csv-parse/sync";
import { loadConfig } from "./typescript_lib/src/weight-processor-lib/core/config";
import { InMemoryStore } from "./typescript_lib/src/index";
import { WeightProcessorService } from "./services/weight_processor_service";

const TARGET_ID = "726b441f-eb43-47d9-8f3c-845d164e5a5b";

interface CsvRow {
  id?: string;
  user_id: string;
  value_quantity?: string;
  unit: string;
  timestamp?: string;
  effective_date_time?: string;
  effectiveDateTime?: string;
  source_type: string;
  [key: string]: any;
}

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

async function main() {
  console.log(`=== Debugging Temporal Consistency Inputs for ${TARGET_ID} ===\n`);

  // Load CSV
  const csvPath = "test_user.csv";
  const content = readFileSync(csvPath, "utf-8");
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];

  // Extract all measurements
  const userMeasurements: any[] = [];
  let targetIndex = -1;

  for (const row of records) {
    const measurementId = row.id || row.measurement_id;
    const userId = row.user_id;
    const weightStr = (row.value_quantity || row.weight || "").trim();

    if (!userId || !measurementId || !weightStr) continue;

    const weight = parseFloat(weightStr);
    if (isNaN(weight) || weight <= 0) continue;

    const dateStr = row.effective_date_time || row.effectiveDateTime || row.timestamp || "";
    const source = row.source_type || "unknown";
    const unit = (row.unit || "").trim();

    if (!unit) continue;

    const timestamp = parseTimestamp(dateStr);

    const measurement = {
      measurementId,
      weight,
      unit,
      timestamp,
      source,
    };

    if (measurementId === TARGET_ID) {
      targetIndex = userMeasurements.length;
      console.log(`Found target measurement at index ${targetIndex}:`);
      console.log(`  ID: ${measurementId}`);
      console.log(`  Timestamp: ${timestamp.toISOString()}`);
      console.log(`  Weight: ${weight} ${unit}`);
      console.log(`  Source: ${source}\n`);
    }

    userMeasurements.push(measurement);
  }

  if (targetIndex === -1) {
    console.error("Target measurement not found!");
    return;
  }

  // Sort by timestamp
  userMeasurements.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Find new index after sorting
  targetIndex = userMeasurements.findIndex(m => m.measurementId === TARGET_ID);
  console.log(`After sorting, target is at index ${targetIndex}\n`);

  // Show previous measurements
  console.log("Previous 3 measurements:");
  for (let i = Math.max(0, targetIndex - 3); i < targetIndex; i++) {
    const m = userMeasurements[i];
    console.log(`  [${i}] ${m.timestamp.toISOString()} - ${m.weight}kg - ${m.source}`);
  }
  console.log();

  // Calculate what previousWeight and timeDiffHours should be
  if (targetIndex > 0) {
    const targetMeasurement = userMeasurements[targetIndex];
    const previousMeasurement = userMeasurements[targetIndex - 1];

    const timeDiffMs = targetMeasurement.timestamp.getTime() - previousMeasurement.timestamp.getTime();
    const timeDiffHours = timeDiffMs / (1000 * 60 * 60);

    console.log("Expected inputs to temporal_consistency:");
    console.log(`  previousWeight: ${previousMeasurement.weight}`);
    console.log(`  timeDiffHours: ${timeDiffHours.toFixed(2)}`);
    console.log(`  weight: ${targetMeasurement.weight}`);
    console.log(`  weightChange: ${Math.abs(targetMeasurement.weight - previousMeasurement.weight).toFixed(2)}\n`);
  }

  // Now run full processing to see what actually happens
  console.log("Running full processing with VERBOSE_LOGGING...\n");

  const config = loadConfig();
  const store = new InMemoryStore();
  const service = new WeightProcessorService(store, config);
  const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";

  // Set verbose logging
  process.env.VERBOSE_LOGGING = "true";

  const response = await service.processBatch(userId, userMeasurements);

  // Find result for target measurement
  const result = response.results[targetIndex];

  console.log("\n\n=== Result for Target Measurement ===");
  console.log(`Accepted: ${result.accepted}`);
  console.log(`Quality Score: ${result.quality_score?.toFixed(6)}`);
  if (result.quality_components) {
    console.log("Quality Components:");
    for (const [component, score] of Object.entries(result.quality_components)) {
      console.log(`  ${component}: ${(score as number).toFixed(6)}`);
    }
  }
}

main().catch(console.error);
