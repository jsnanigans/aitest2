/**
 * TypeScript helper for extracting divergence sequence
 * Processes measurements using the full service (including replay)
 */

import { readFileSync } from 'fs';
import { InMemoryStore } from './typescript_lib/src/index';
import { WeightProcessorService } from './services/weight_processor_service';

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
    console.error('Usage: bun run extract_divergence_helper.ts <input_file.json>');
    process.exit(1);
  }

  // Suppress console output during processing
  const originalLog = console.log;
  const originalError = console.error;
  console.log = () => {};
  console.error = () => {};

  try {
    const input: InputData = JSON.parse(readFileSync(inputFile, 'utf-8'));
    const store = new InMemoryStore();
    const service = new WeightProcessorService(store, input.config);

    if (input.measurements.length === 0) {
      throw new Error('No measurements provided');
    }

    const userId = input.measurements[0].userId;

    // Convert to service format
    const measurements = input.measurements.map((m) => ({
      id: m.id,
      weight: m.weight,
      timestamp: new Date(m.timestamp),
      source: m.source,
      unit: m.unit,
    }));

    // Process batch (includes automatic replay)
    const response = await service.processBatch(userId, measurements);

    // Extract results - match by index since results might not have IDs
    const results = response.results.map((r: any, index: number) => ({
      id: measurements[index].id,
      accepted: r.accepted || false,
      quality_score: r.quality_score,
    }));

    // Restore console and output
    console.log = originalLog;
    console.error = originalError;

    console.log(JSON.stringify(results));
  } catch (error) {
    console.log = originalLog;
    console.error = originalError;
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
