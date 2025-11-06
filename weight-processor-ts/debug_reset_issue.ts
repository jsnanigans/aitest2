#!/usr/bin/env bun
/**
 * Debug why hard resets happen on every measurement
 */

import { readFileSync } from 'node:fs';
import { parse } from 'csv-parse/sync';
import { processMeasurement } from './src/core/processing/processor';
import { ProcessorStateDB } from './src/core/database/database';
import { ConfigManager } from './src/config/config_manager';
import { ResetManager } from './src/core/processing/reset_manager';

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

console.log('=== Debugging Reset Logic ===\n');

async function processAndDebug() {
  // Process first 3 measurements
  for (let i = 0; i < Math.min(3, records.length); i++) {
    const row = records[i];
    const weight = parseFloat(row.value_quantity);
    const timestamp = new Date(row.timestamp || row.effective_date_time);
    const source = row.source_type;

    console.log(`\n[${ i + 1}] Processing: ${weight.toFixed(2)} kg at ${timestamp.toISOString()}`);

    // Check state BEFORE processing
    const stateBefore = stateStore.get_state(USER_ID);
    console.log(`   Before - lastAcceptedTimestamp: ${stateBefore?.lastAcceptedTimestamp || 'NONE'}`);
    console.log(`   Before - lastTimestamp: ${stateBefore?.lastTimestamp || 'NONE'}`);

    // Check if reset will trigger
    const resetType = ResetManager.shouldTriggerReset(
      stateBefore || {},
      weight,
      timestamp,
      source,
      config
    );
    console.log(`   Reset check: ${resetType || 'NO RESET'}`);

    // Process the measurement
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

    console.log(`   Result: ${result.accepted ? 'ACCEPTED ✅' : 'REJECTED ❌'}`);

    // Check state AFTER processing
    const stateAfter = stateStore.get_state(USER_ID);
    console.log(`   After  - lastAcceptedTimestamp: ${stateAfter?.lastAcceptedTimestamp || 'NONE'}`);
    console.log(`   After  - lastTimestamp: ${stateAfter?.lastTimestamp || 'NONE'}`);
    console.log(`   After  - resetTimestamp: ${stateAfter?.resetTimestamp || 'NONE'}`);

    // Calculate gap to next measurement
    if (i < records.length - 1) {
      const nextRow = records[i + 1];
      const nextTimestamp = new Date(nextRow.timestamp || nextRow.effective_date_time);
      const lastAccepted = stateAfter?.lastAcceptedTimestamp;

      if (lastAccepted) {
        const last = typeof lastAccepted === 'string' ? new Date(lastAccepted) : lastAccepted;
        const gapDays = (nextTimestamp.getTime() - last.getTime()) / (86400.0 * 1000);
        console.log(`   Gap to next measurement: ${gapDays.toFixed(1)} days (threshold: 30)`);
        console.log(`   Should trigger hard reset next? ${gapDays >= 30}`);
      }
    }
  }
}

processAndDebug().catch(console.error);
