/**
 * Helper script for find_divergence.py
 * Processes measurements with TypeScript and outputs results as JSON
 */

import { readFileSync } from 'fs';
import { InMemoryStore } from './typescript_lib/src/index';
import { processMeasurement } from './typescript_lib/src/index';

interface Measurement {
  id: string;
  userId: string;
  timestamp: string;
  weight: number;
  unit: string;
  source: string;
}

interface InputData {
  measurements: Measurement[];
  config: any;
}

async function main() {
  const inputFile = process.argv[2];
  if (!inputFile) {
    console.error('Usage: bun run find_divergence_helper.ts <input_file.json>');
    process.exit(1);
  }

  // Read input
  const input: InputData = JSON.parse(readFileSync(inputFile, 'utf-8'));

  // Suppress console output during processing
  const originalLog = console.log;
  const originalError = console.error;
  console.log = () => {};
  console.error = () => {};

  // Initialize store
  const store = new InMemoryStore();

  // Process measurements
  const results: any[] = [];

  for (const measurement of input.measurements) {
    const timestamp = new Date(measurement.timestamp);
    const result = await processMeasurement(
      measurement.userId,
      measurement.weight,
      timestamp,
      measurement.source,
      input.config,
      measurement.unit,
      store,
      1.75 // user_height_m
    );

    results.push({
      id: measurement.id,
      accepted: result.accepted || false,
      quality_score: result.quality_score,
      timestamp: measurement.timestamp,
    });
  }

  // Restore console and output results as JSON
  console.log = originalLog;
  console.error = originalError;

  console.log(JSON.stringify(results));
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
