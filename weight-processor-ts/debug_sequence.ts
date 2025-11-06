#!/usr/bin/env bun
/**
 * Debug script to trace quality scoring through a sequence of measurements
 */

import { readFileSync } from 'node:fs';
import { parse } from 'csv-parse/sync';
import { WeightProcessorService } from './src/services/weight_processor_service';
import { ConfigManager } from './src/config/config_manager';

interface CsvRow {
  id: string;
  user_id: string;
  timestamp: string;
  effective_date_time: string;
  source_type: string;
  value_quantity: string;
  unit: string;
}

const config = ConfigManager.loadConfig();
config.database.backend = 'memory';

const service = new WeightProcessorService(config);

const USER_ID = 'ADC64C0B-CB46-41F9-BDA0-CC11A35942D7';
const DEVICE_ID = 'test-device';

// Load test data
const content = readFileSync('test_user.csv', 'utf-8');
const records = parse(content, {
  columns: true,
  skip_empty_lines: true,
  trim: true,
}) as CsvRow[];

console.log('=== Processing First 10 Measurements ===\n');

async function processSequence() {
for (let i = 0; i < Math.min(10, records.length); i++) {
  const row = records[i];

  const measurement = {
    userId: row.user_id,
    deviceId: DEVICE_ID,
    weight: parseFloat(row.value_quantity),
    unit: row.unit as any,
    timestamp: new Date(row.timestamp || row.effective_date_time),
    source: row.source_type,
  };

  const result = await service.process(measurement);

  console.log(`\n[${i + 1}] ID: ${row.id.substring(0, 8)}...`);
  console.log(`    Weight: ${measurement.weight} kg`);
  console.log(`    Timestamp: ${measurement.timestamp.toISOString()}`);
  console.log(`    Accepted: ${result.accepted}`);

  if (result.qualityScore) {
    console.log(`    Quality Score: ${result.qualityScore.overall.toFixed(4)}`);
    console.log(`    Components:`);
    for (const [key, value] of Object.entries(result.qualityScore.components)) {
      console.log(`      ${key}: ${value.toFixed(4)}`);
    }
    if (!result.accepted) {
      console.log(`    Rejection Reason: ${result.qualityScore.rejectionReason}`);
    }
  }
}
}

processSequence().catch(console.error);
