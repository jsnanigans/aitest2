#!/usr/bin/env bun
/**
 * Debug script to trace processing of divergent measurements.
 * Compares TypeScript and Python behavior for specific measurement IDs.
 */

import { readFileSync } from "fs";
import { parse } from "csv-parse/sync";

// Target measurement IDs that differ between implementations
const DIVERGENT_IDS = [
  "726b441f-eb43-47d9-8f3c-845d164e5a5b", // TS accepts, PY rejects
  "1a98b2c3-e023-4757-8d01-d35ef2fb363e", // PY accepts, TS rejects
  "510977fa-9d3f-4b50-a667-e676a0cc0791", // PY accepts, TS rejects
  "70d7918e-87d9-4968-84a4-b2bfec488e76", // PY accepts, TS rejects
  "86233705-0332-44f1-bc69-bc796220f598", // PY accepts, TS rejects
];

// Import TypeScript implementation
import { loadConfig } from "./typescript_lib/src/weight-processor-lib/core/config";
import { InMemoryStore } from "./typescript_lib/src/index";
import { WeightProcessorService } from "./services/weight_processor_service";

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

async function main() {
  console.log("=== Debugging Divergent Measurements ===\n");
  console.log("Target IDs:");
  for (const id of DIVERGENT_IDS) {
    console.log(`  ${id}`);
  }
  console.log();

  // Load CSV
  const csvPath = "test_user.csv";
  const content = readFileSync(csvPath, "utf-8");
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];

  // Extract all measurements for the user
  const userMeasurements: any[] = [];
  let divergentMeasurements: any[] = [];

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

    userMeasurements.push(measurement);

    if (DIVERGENT_IDS.includes(measurementId)) {
      divergentMeasurements.push({ ...measurement, originalRow: row });
    }
  }

  console.log(`Loaded ${userMeasurements.length} measurements for user`);
  console.log(`Found ${divergentMeasurements.length} divergent measurements\n`);

  // Sort by timestamp
  userMeasurements.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Initialize TypeScript implementation
  const config = loadConfig();
  const store = new InMemoryStore();
  const service = new WeightProcessorService(store, config);

  const userId = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7";

  // Process all measurements
  console.log("Processing all measurements...\n");
  const response = await service.processBatch(userId, userMeasurements);

  console.log(`Total processed: ${response.measurements_processed}`);
  console.log(`Total accepted: ${response.measurements_accepted}`);
  console.log(`Total rejected: ${response.measurements_rejected}\n`);

  // Analyze divergent measurements
  console.log("=== Divergent Measurement Analysis ===\n");

  for (const divergent of divergentMeasurements) {
    // Find the result for this measurement
    const measurementIndex = userMeasurements.findIndex(
      m => m.measurementId === divergent.measurementId
    );

    if (measurementIndex >= 0 && measurementIndex < response.results.length) {
      const result = response.results[measurementIndex];

      console.log(`ID: ${divergent.measurementId}`);
      console.log(`  Timestamp: ${divergent.timestamp.toISOString()}`);
      console.log(`  Source: ${divergent.source}`);
      console.log(`  Weight: ${divergent.weight} ${divergent.unit}`);
      console.log(`  Accepted: ${result.accepted}`);
      console.log(`  Quality Score: ${result.quality_score?.toFixed(6)}`);

      if (result.quality_components) {
        console.log(`  Quality Components:`);
        for (const [component, score] of Object.entries(result.quality_components)) {
          console.log(`    ${component}: ${(score as number).toFixed(6)}`);
        }
      }

      if (!result.accepted) {
        console.log(`  Rejection Reason: ${result.rejection_reason || 'Unknown'}`);
      }

      console.log(`  Kalman Estimate: ${result.kalman_estimate?.toFixed(3)}`);
      console.log(`  Innovation: ${result.innovation?.toFixed(3)}`);
      console.log(`  Normalized Innovation: ${result.normalized_innovation?.toFixed(3)}`);
      console.log();
    }
  }
}

main().catch(console.error);
