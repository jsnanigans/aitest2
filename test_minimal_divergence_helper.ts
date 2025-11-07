/**
 * TypeScript helper for minimal divergence test
 */

import { readFileSync } from 'fs';
import { InMemoryStore, processMeasurement } from './typescript_lib/src/index';

interface SetupMeasurement {
  weight: number;
  ts: string;
}

interface ReplayMeasurement {
  id: string;
  weight: number;
  ts: string;
}

interface InputData {
  setup_measurements: SetupMeasurement[];
  replay_measurements: ReplayMeasurement[];
  config: any;
}

async function main() {
  const inputFile = process.argv[2];
  if (!inputFile) {
    console.error('Usage: bun run test_minimal_divergence_helper.ts <input_file.json>');
    process.exit(1);
  }

  // Suppress console output
  const originalLog = console.log;
  const originalError = console.error;
  console.log = () => {};
  console.error = () => {};

  try {
    const input: InputData = JSON.parse(readFileSync(inputFile, 'utf-8'));
    const store = new InMemoryStore();
    const userId = 'test-user';

    // Process setup measurements
    for (const m of input.setup_measurements) {
      await processMeasurement(
        userId,
        m.weight,
        new Date(m.ts),
        'test',
        input.config,
        'kg',
        store,
        1.75
      );
    }

    // Create snapshot
    const snapshotTs = new Date(input.replay_measurements[0].ts);
    await store.saveStateSnapshot(userId, snapshotTs);

    // Restore snapshot (simulating replay mechanism)
    const restored = await store.checkAndRestoreSnapshot(userId, snapshotTs);
    if (!restored.snapshot_found) {
      throw new Error('Snapshot not found');
    }

    // Process replay measurements
    const results: any[] = [];
    for (const m of input.replay_measurements) {
      const result = await processMeasurement(
        userId,
        m.weight,
        new Date(m.ts),
        'test',
        input.config,
        'kg',
        store,
        1.75
      );

      results.push({
        id: m.id,
        accepted: result.accepted || false,
        quality_score: result.quality_score,
      });
    }

    // Restore console and output
    console.log = originalLog;
    console.error = originalError;

    console.log(JSON.stringify({ results }));
  } catch (error) {
    console.log = originalLog;
    console.error = originalError;
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
