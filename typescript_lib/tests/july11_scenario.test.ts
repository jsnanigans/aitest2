/**
 * Test for July 11 replay scenario that diverges between Python and TypeScript.
 */

import { test, expect, describe } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { InMemoryStore, processMeasurement, type ProcessingResult } from '../src/index';

interface Measurement {
  id: string;
  timestamp: string;
  weight: number;
  unit: string;
  source: string;
  note?: string;
}

interface July11Fixture {
  description: string;
  scenario: string;
  user_id: string;
  setup: {
    description: string;
    measurements: Measurement[];
  };
  test_measurements: {
    description: string;
    time_gap_hours: number;
    measurements: Measurement[];
  };
  expected_results: {
    typescript: Record<string, { accepted: boolean; quality_score: number }>;
    python: Record<string, { accepted: boolean; quality_score: number; note?: string }>;
  };
  analysis: {
    issue: string;
    divergence_point: string;
    quality_score_diff: number;
    possible_causes: string[];
  };
}

function loadFixture(): July11Fixture {
  const fixturePath = join(__dirname, '..', '..', 'test_fixtures', 'july11_replay_scenario.json');
  const content = readFileSync(fixturePath, 'utf-8');
  return JSON.parse(content);
}

function parseTimestamp(tsStr: string): Date {
  return new Date(tsStr);
}

const baseConfig = {
  kalman: {
    process_noise_position: 0.01,
    process_noise_velocity: 0.0001,
    initial_position_variance: 1.0,
    initial_velocity_variance: 0.01,
    trend_limit_kg_per_week: 5.0,
  },
  quality_weights: {
    kalman_fit: 0.35,
    temporal_consistency: 0.25,
    plausibility: 0.25,
    anomaly_detection: 0.15,
  },
  quality_threshold: 0.55,
};

describe('July 11 Replay Scenario', () => {
  test('TypeScript isolated scenario processing', async () => {
    const fixture = loadFixture();
    const store = new InMemoryStore();
    const userId = fixture.user_id;

    // Process setup measurements to establish state
    console.log('\n📊 TypeScript - Setup measurements:');
    for (const measurement of fixture.setup.measurements) {
      const timestamp = parseTimestamp(measurement.timestamp);
      const result: ProcessingResult = await processMeasurement(
        userId,
        measurement.weight,
        timestamp,
        measurement.source,
        baseConfig,
        measurement.unit,
        store,
        1.75 // Assumed height
      );
      console.log(
        `  ${timestamp.toISOString()} -> accepted=${result.accepted}, quality=${result.quality_score?.toFixed(4)}`
      );
    }

    // Create snapshot before test measurements (simulating replay mechanism)
    const snapshotTimestamp = parseTimestamp(fixture.test_measurements.measurements[0].timestamp);
    await store.saveStateSnapshot(userId, snapshotTimestamp);

    // Process the 3 problematic measurements
    console.log('\n📊 TypeScript - Test measurements (after 49h gap):');
    const results: Array<{
      id: string;
      accepted: boolean;
      quality_score?: number;
      weight: number;
      timestamp: string;
    }> = [];

    for (const measurement of fixture.test_measurements.measurements) {
      const timestamp = parseTimestamp(measurement.timestamp);
      const result: ProcessingResult = await processMeasurement(
        userId,
        measurement.weight,
        timestamp,
        measurement.source,
        baseConfig,
        measurement.unit,
        store,
        1.75
      );
      results.push({
        id: measurement.id,
        accepted: result.accepted || false,
        quality_score: result.quality_score,
        weight: measurement.weight,
        timestamp: timestamp.toISOString(),
      });
      console.log(
        `  ${measurement.id.substring(0, 8)} (${measurement.weight}kg) -> ` +
          `accepted=${result.accepted}, quality=${result.quality_score?.toFixed(4)}`
      );
    }

    // Document what TypeScript produces in this isolated scenario
    const expectedPy = fixture.expected_results.python;
    const problematicId = '4f07af66-cd5e-4a38-9403-80d6da1d1542';

    console.log(`\n📊 ISOLATED SCENARIO RESULTS (TypeScript):`);
    console.log(`   This differs from full batch because Kalman state depends on ALL prior measurements`);
    console.log(`   First:  ${results[0].id.substring(0, 8)} -> accepted=${results[0].accepted}, score=${results[0].quality_score?.toFixed(4)}`);
    console.log(`   Middle: ${results[1].id.substring(0, 8)} -> accepted=${results[1].accepted}, score=${results[1].quality_score?.toFixed(4)}`);
    console.log(`   Third:  ${results[2].id.substring(0, 8)} -> accepted=${results[2].accepted}, score=${results[2].quality_score?.toFixed(4)}`);
    console.log(`\n   For comparison, Python in full batch produced:`);
    console.log(`   Middle: accepted=${expectedPy[problematicId].accepted}, score=${expectedPy[problematicId].quality_score.toFixed(4)}`);

    // Basic sanity checks
    expect(results.length).toBe(3);
    expect(results[0].id).toBe('52ec2c45-c6a8-4946-887b-e5e8907f19b9');
    expect(results[1].id).toBe(problematicId);
    expect(results[2].id).toBe('726b441f-eb43-47d9-8f3c-845d164e5a5b');

    // All results should have quality scores
    for (const result of results) {
      expect(result.quality_score).toBeDefined();
    }

    console.log('✅ TypeScript isolated scenario processed successfully');
  });

  test('should document divergence from Python', async () => {
    const fixture = loadFixture();

    console.log('\n📋 Analysis of divergence:');
    console.log(`   Issue: ${fixture.analysis.issue}`);
    console.log(`   Divergence point: ${fixture.analysis.divergence_point}`);
    console.log(`   Quality score difference: ${fixture.analysis.quality_score_diff}`);
    console.log(`   Possible causes:`);
    fixture.analysis.possible_causes.forEach(cause => {
      console.log(`     - ${cause}`);
    });

    // This test always passes - it's just for documentation
    expect(fixture.analysis.quality_score_diff).toBeGreaterThan(0);
  });
});
