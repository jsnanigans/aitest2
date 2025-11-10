/**
 * Data generator for creating test fixtures
 * Generates measurements, edge cases, and stress test data
 */

export interface Measurement {
  weight_kg: number;
  timestamp: number;
  source: string;
  metadata?: Record<string, any>;
}

export interface MeasurementFixture {
  deviceId: string;
  userId: string;
  measurements: Measurement[];
  description: string;
  expectedBehavior?: string;
}

export interface EdgeCaseConfig {
  type: 'very_small' | 'very_large' | 'rapid' | 'long_gap' | 'out_of_order' | 'corrupted';
  count?: number;
}

export class DataGenerator {
  private baseTimestamp: number;
  private currentTimestamp: number;

  constructor(baseTimestamp?: number) {
    this.baseTimestamp = baseTimestamp ?? Date.now();
    this.currentTimestamp = this.baseTimestamp;
  }

  /**
   * Generate a basic sequence of measurements
   */
  generateBasicSequence(count: number = 10, avgWeight: number = 70.0): MeasurementFixture {
    const measurements: Measurement[] = [];

    for (let i = 0; i < count; i++) {
      // Add small random variation (±0.5 kg)
      const weight = avgWeight + (Math.random() - 0.5) * 1.0;

      measurements.push({
        weight_kg: parseFloat(weight.toFixed(2)),
        timestamp: this.currentTimestamp,
        source: 'withings',
      });

      // Increment by 1 day (86400 seconds)
      this.currentTimestamp += 86400;
    }

    return {
      deviceId: 'test-device-001',
      userId: 'test-user-001',
      measurements,
      description: `Basic sequence of ${count} measurements around ${avgWeight} kg`,
      expectedBehavior: 'All measurements should be accepted and processed',
    };
  }

  /**
   * Generate a single measurement
   */
  generateSingleMeasurement(weight: number = 70.0): MeasurementFixture {
    return {
      deviceId: 'test-device-single',
      userId: 'test-user-single',
      measurements: [
        {
          weight_kg: weight,
          timestamp: this.currentTimestamp,
          source: 'withings',
        },
      ],
      description: `Single measurement at ${weight} kg`,
      expectedBehavior: 'First measurement should initialize Kalman filter',
    };
  }

  /**
   * Generate a reset scenario
   */
  generateResetScenario(): MeasurementFixture {
    const measurements: Measurement[] = [];

    // First sequence: stable weight around 70 kg
    for (let i = 0; i < 5; i++) {
      measurements.push({
        weight_kg: 70.0 + (Math.random() - 0.5) * 0.5,
        timestamp: this.currentTimestamp,
        source: 'withings',
      });
      this.currentTimestamp += 86400; // 1 day
    }

    // Sudden large change (trigger reset) - 10 kg drop
    measurements.push({
      weight_kg: 60.0,
      timestamp: this.currentTimestamp,
      source: 'withings',
      metadata: { expected_reset: true },
    });
    this.currentTimestamp += 86400;

    // Continue with new stable weight
    for (let i = 0; i < 5; i++) {
      measurements.push({
        weight_kg: 60.0 + (Math.random() - 0.5) * 0.5,
        timestamp: this.currentTimestamp,
        source: 'withings',
      });
      this.currentTimestamp += 86400;
    }

    return {
      deviceId: 'test-device-reset',
      userId: 'test-user-reset',
      measurements,
      description: 'Reset scenario: 5 measurements @ 70kg → sudden drop to 60kg → 5 measurements @ 60kg',
      expectedBehavior: 'Should trigger reset after large weight change',
    };
  }

  /**
   * Generate quality rejection scenario
   */
  generateQualityRejection(): MeasurementFixture {
    const measurements: Measurement[] = [];

    // Good measurements
    measurements.push({
      weight_kg: 70.0,
      timestamp: this.currentTimestamp,
      source: 'withings',
    });
    this.currentTimestamp += 86400;

    measurements.push({
      weight_kg: 70.2,
      timestamp: this.currentTimestamp,
      source: 'withings',
    });
    this.currentTimestamp += 86400;

    // Bad measurement - physiologically implausible (too low)
    measurements.push({
      weight_kg: 30.0,
      timestamp: this.currentTimestamp,
      source: 'withings',
      metadata: { expected_rejection: true },
    });
    this.currentTimestamp += 86400;

    // Bad measurement - physiologically implausible (too high)
    measurements.push({
      weight_kg: 250.0,
      timestamp: this.currentTimestamp,
      source: 'withings',
      metadata: { expected_rejection: true },
    });
    this.currentTimestamp += 86400;

    // Good measurement - resume normal
    measurements.push({
      weight_kg: 70.1,
      timestamp: this.currentTimestamp,
      source: 'withings',
    });
    this.currentTimestamp += 86400;

    return {
      deviceId: 'test-device-quality',
      userId: 'test-user-quality',
      measurements,
      description: 'Quality rejection: good measurements interspersed with bad ones',
      expectedBehavior: 'Bad measurements should be rejected based on quality score',
    };
  }

  /**
   * Generate state persistence scenario
   */
  generateStatePersistence(): MeasurementFixture {
    const measurements: Measurement[] = [];

    // First batch
    for (let i = 0; i < 5; i++) {
      measurements.push({
        weight_kg: 70.0 + i * 0.1,
        timestamp: this.currentTimestamp,
        source: 'withings',
        metadata: { batch: 'first' },
      });
      this.currentTimestamp += 86400;
    }

    // Second batch (to be processed after state save/restore)
    for (let i = 0; i < 5; i++) {
      measurements.push({
        weight_kg: 70.5 + i * 0.1,
        timestamp: this.currentTimestamp,
        source: 'withings',
        metadata: { batch: 'second' },
      });
      this.currentTimestamp += 86400;
    }

    return {
      deviceId: 'test-device-persist',
      userId: 'test-user-persist',
      measurements,
      description: 'State persistence: process first batch, save state, restore, process second batch',
      expectedBehavior: 'Results should be identical whether processed together or in two batches',
    };
  }

  /**
   * Generate edge case data
   */
  generateEdgeCase(config: EdgeCaseConfig): MeasurementFixture {
    const measurements: Measurement[] = [];
    const count = config.count ?? 10;

    switch (config.type) {
      case 'very_small':
        // Very small weight values (near minimum)
        for (let i = 0; i < count; i++) {
          measurements.push({
            weight_kg: 40.0 + Math.random() * 0.1,
            timestamp: this.currentTimestamp,
            source: 'withings',
          });
          this.currentTimestamp += 86400;
        }
        return {
          deviceId: 'test-device-edge',
          userId: 'test-user-edge',
          measurements,
          description: 'Edge case: very small weight values (40 kg)',
          expectedBehavior: 'Should handle near-minimum physiological weights',
        };

      case 'very_large':
        // Very large weight values (near maximum)
        for (let i = 0; i < count; i++) {
          measurements.push({
            weight_kg: 200.0 + Math.random() * 1.0,
            timestamp: this.currentTimestamp,
            source: 'withings',
          });
          this.currentTimestamp += 86400;
        }
        return {
          deviceId: 'test-device-edge',
          userId: 'test-user-edge',
          measurements,
          description: 'Edge case: very large weight values (200 kg)',
          expectedBehavior: 'Should handle near-maximum physiological weights',
        };

      case 'rapid':
        // Rapid measurements (seconds apart)
        for (let i = 0; i < count; i++) {
          measurements.push({
            weight_kg: 70.0 + (Math.random() - 0.5) * 0.1,
            timestamp: this.currentTimestamp,
            source: 'withings',
          });
          this.currentTimestamp += 60; // 1 minute
        }
        return {
          deviceId: 'test-device-edge',
          userId: 'test-user-edge',
          measurements,
          description: 'Edge case: rapid measurements (1 minute apart)',
          expectedBehavior: 'Should handle very small time deltas',
        };

      case 'long_gap':
        // Long gaps between measurements (months)
        for (let i = 0; i < count; i++) {
          measurements.push({
            weight_kg: 70.0 + (Math.random() - 0.5) * 2.0,
            timestamp: this.currentTimestamp,
            source: 'withings',
          });
          this.currentTimestamp += 30 * 86400; // 30 days
        }
        return {
          deviceId: 'test-device-edge',
          userId: 'test-user-edge',
          measurements,
          description: 'Edge case: long gaps between measurements (30 days)',
          expectedBehavior: 'Should handle large time gaps with increased uncertainty',
        };

      case 'out_of_order':
        // Out-of-order timestamps
        const baseWeights = Array.from({ length: count }, (_, i) => ({
          weight_kg: 70.0 + i * 0.1,
          timestamp: this.currentTimestamp + i * 86400,
          source: 'withings',
        }));
        // Shuffle
        for (let i = baseWeights.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [baseWeights[i], baseWeights[j]] = [baseWeights[j], baseWeights[i]];
        }
        return {
          deviceId: 'test-device-edge',
          userId: 'test-user-edge',
          measurements: baseWeights,
          description: 'Edge case: out-of-order measurements',
          expectedBehavior: 'Should handle or reject out-of-order measurements',
        };

      default:
        throw new Error(`Unknown edge case type: ${config.type}`);
    }
  }

  /**
   * Generate stress test data
   */
  generateStressTest(count: number = 1000): MeasurementFixture {
    const measurements: Measurement[] = [];
    let weight = 70.0;

    for (let i = 0; i < count; i++) {
      // Random walk with small steps
      weight += (Math.random() - 0.5) * 0.2;
      weight = Math.max(40, Math.min(200, weight)); // Keep in reasonable range

      measurements.push({
        weight_kg: parseFloat(weight.toFixed(2)),
        timestamp: this.currentTimestamp,
        source: i % 3 === 0 ? 'fitbit' : 'withings', // Mix sources
      });

      this.currentTimestamp += 86400; // 1 day
    }

    return {
      deviceId: 'test-device-stress',
      userId: 'test-user-stress',
      measurements,
      description: `Stress test: ${count} measurements with random walk`,
      expectedBehavior: 'Should process large sequences efficiently',
    };
  }

  /**
   * Reset timestamp to base
   */
  reset(): void {
    this.currentTimestamp = this.baseTimestamp;
  }

  /**
   * Set custom timestamp
   */
  setTimestamp(timestamp: number): void {
    this.currentTimestamp = timestamp;
  }
}

/**
 * Export all fixtures to JSON
 */
export async function generateAllFixtures(outputDir: string): Promise<void> {
  const generator = new DataGenerator();
  const fixtures: Record<string, MeasurementFixture> = {};

  // Generate all fixtures
  fixtures.singleMeasurement = generator.generateSingleMeasurement();

  generator.reset();
  fixtures.basicSequence = generator.generateBasicSequence();

  generator.reset();
  fixtures.resetScenario = generator.generateResetScenario();

  generator.reset();
  fixtures.qualityRejection = generator.generateQualityRejection();

  generator.reset();
  fixtures.statePersistence = generator.generateStatePersistence();

  // Edge cases
  generator.reset();
  fixtures.edgeVerySmall = generator.generateEdgeCase({ type: 'very_small' });

  generator.reset();
  fixtures.edgeVeryLarge = generator.generateEdgeCase({ type: 'very_large' });

  generator.reset();
  fixtures.edgeRapid = generator.generateEdgeCase({ type: 'rapid' });

  generator.reset();
  fixtures.edgeLongGap = generator.generateEdgeCase({ type: 'long_gap' });

  // Stress test
  generator.reset();
  fixtures.stressTest = generator.generateStressTest(100); // Start with 100, can increase

  // Write to files
  await Bun.write(
    `${outputDir}/all_fixtures.json`,
    JSON.stringify(fixtures, null, 2)
  );

  console.log(`Generated ${Object.keys(fixtures).length} fixtures in ${outputDir}/all_fixtures.json`);
}
