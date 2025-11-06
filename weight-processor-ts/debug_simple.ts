#!/usr/bin/env bun
/**
 * Simple debug to check why 43.2 kg is accepted in TS but rejected in Python
 */

import { readFileSync } from 'node:fs';
import { parse } from 'csv-parse/sync';
import { processMeasurement } from './src/core/processing/processor';
import { ProcessorStateDB } from './src/core/database/database';
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

const stateStore = new ProcessorStateDB();
const USER_ID = 'ADC64C0B-CB46-41F9-BDA0-CC11A35942D7';

// Load test data
const content = readFileSync('test_user.csv', 'utf-8');
const records = parse(content, {
  columns: true,
  skip_empty_lines: true,
  trim: true,
}) as CsvRow[];

console.log('=== Processing Measurements to Find Pattern ===\n');

async function processAll() {
  // Process first 10 measurements
  for (let i = 0; i < Math.min(10, records.length); i++) {
    const row = records[i];
    const weight = parseFloat(row.value_quantity);
    const timestamp = new Date(row.timestamp || row.effective_date_time);
    const source = row.source_type;

    const result = await processMeasurement(
      USER_ID,
      weight,
      timestamp,
      source,
      config,
      stateStore,
      'kg',
      null
    );

    const accepted = result.accepted ? '✅' : '❌';
    console.log(`[${i + 1}] ${accepted} ${weight.toFixed(2)} kg - ${row.id.substring(0, 8)}...`);

    if (result.qualityScore) {
      console.log(`   Quality: ${result.qualityScore.overall.toFixed(4)} (threshold: ${result.qualityScore.threshold})`);
      console.log(`   Components: ${JSON.stringify(result.qualityScore.components, null, 2).replace(/\n/g, '\n   ')}`);

      if (!result.accepted && result.qualityScore.rejectionReason) {
        console.log(`   Reason: ${result.qualityScore.rejectionReason}`);
      }
    }
    console.log('');
  }
}

processAll().catch(console.error);
