#!/usr/bin/env bun
/**
 * Debug script to trace Kalman filter state measurement-by-measurement.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { WeightProcessorService, type MeasurementInput } from './src/services/weight_processor_service';
import { ProcessorStateDB } from './src/core/database/database';
import { ConfigManager } from './src/config/config_manager';

function parseTimestamp(dateStr: string): Date {
  if (!dateStr) {
    return new Date();
  }
  try {
    if (dateStr.includes('T')) {
      return new Date(dateStr);
    } else {
      return new Date(dateStr + 'T00:00:00.000Z');
    }
  } catch {
    return new Date();
  }
}

async function main() {
  // Load CSV data
  const csvPath = '../test_small.csv';
  const userId = 'ADC64C0B-CB46-41F9-BDA0-CC11A35942D7';

  const csvContent = readFileSync(csvPath, 'utf-8');
  const lines = csvContent.trim().split('\n');
  const headers = lines[0]!.split(',');

  const measurements: MeasurementInput[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i]!.split(',');
    const row: Record<string, string> = {};

    headers.forEach((header, idx) => {
      row[header] = values[idx] || '';
    });

    if (row['user_id'] !== userId) {
      continue;
    }

    const measurement: MeasurementInput = {
      measurement_id: row['id']!,
      weight: parseFloat(row['value_quantity']!),
      unit: row['unit']!,
      timestamp: parseTimestamp(row['effective_date_time']!),
      source: row['source_type'] || 'unknown',
    };

    measurements.push(measurement);
  }

  // Sort by timestamp
  measurements.sort((a, b) => {
    const tsA = typeof a.timestamp === 'string' ? new Date(a.timestamp) : a.timestamp;
    const tsB = typeof b.timestamp === 'string' ? new Date(b.timestamp) : b.timestamp;
    return tsA.getTime() - tsB.getTime();
  });

  // Initialize service
  const stateStore = new ProcessorStateDB();
  const configPath = join(process.cwd(), '../weight_values/config.toml');
  const manager = new ConfigManager();
  const config = manager.loadConfig(configPath);
  config.database.backend = 'memory';
  const service = new WeightProcessorService(stateStore, config);

  console.log('=== TYPESCRIPT KALMAN STATE TRACE ===');
  console.log(`Processing ${measurements.length} measurements\n`);

  // Process one by one with detailed logging
  for (let i = 0; i < measurements.length; i++) {
    const measurement = measurements[i]!;

    // Get state BEFORE processing
    const stateBefore = stateStore.get_state(userId);

    // Process single measurement
    const response = await service.processBatch(userId, [measurement]);
    const result = response.results[0];

    // Get state AFTER processing
    const stateAfter = stateStore.get_state(userId);

    if (result) {
      const ts = typeof measurement.timestamp === 'string' ? new Date(measurement.timestamp) : measurement.timestamp;
      const dateStr = ts.toISOString().slice(0, 10);
      const kalmanEst = result.kalman_estimate ?? measurement.weight;
      const innovation = measurement.weight - kalmanEst;
      console.log(
        `[${i.toString().padStart(2)}] ${dateStr} | ` +
        `raw=${measurement.weight.toFixed(3).padStart(7)} | ` +
        `kalman=${kalmanEst.toFixed(3).padStart(7)} | ` +
        `innovation=${innovation >= 0 ? '+' : ''}${innovation.toFixed(3).padStart(7)} | ` +
        `accepted=${result.accepted}`
      );

      // Show detailed state after processing
      if (stateAfter?.last_state) {
        const state = stateAfter.last_state;
        const covariance = stateAfter.last_covariance;

        console.log(
          `      state=[${state[0].toFixed(3).padStart(7)}, ${state[1] >= 0 ? '+' : ''}${state[1].toFixed(5)}] ` +
          `P[0,0]=${covariance[0][0]!.toFixed(4).padStart(7)}`
        );
      }
    }
  }
}

main().catch(console.error);
