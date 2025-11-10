/**
 * Wrapper for calling TypeScript weight processor implementation
 */

import type { Measurement } from './data_generator';

// We'll import from the actual TypeScript library
// For now, let's create a dynamic import to avoid build issues

export interface TypeScriptProcessorInput {
  deviceId: string;
  userId: string;
  measurements: Measurement[];
}

export interface TypeScriptProcessorResult {
  results: any[];
  finalState: any;
}

export class TypeScriptWrapper {
  private processMeasurement: any;
  private InMemoryStore: any;
  private config: any;

  async initialize() {
    // Dynamic import of the TypeScript library
    const processorModule = await import(
      '../../typescript_lib/src/weight-processor-lib/core/processing/processor'
    );
    const storeModule = await import(
      '../../typescript_lib/src/weight-processor-lib/core/database/memory_store'
    );

    this.processMeasurement = processorModule.processMeasurement;
    this.InMemoryStore = storeModule.InMemoryStore;

    // Load config from config.json
    const configPath = `${import.meta.dir}/../../typescript_lib/config.json`;
    this.config = await Bun.file(configPath).json();
  }

  /**
   * Process measurements using TypeScript implementation
   */
  async processMeasurements(input: TypeScriptProcessorInput): Promise<TypeScriptProcessorResult> {
    if (!this.processMeasurement) {
      await this.initialize();
    }

    // Create a fresh state store for this test
    const stateStore = new this.InMemoryStore();

    // Combine device_id and user_id for consistency with Python
    const combinedUserId = `${input.deviceId}:${input.userId}`;

    const results: any[] = [];

    // Process each measurement using the function-based API
    for (const measurement of input.measurements) {
      const result = await this.processMeasurement(
        combinedUserId,
        measurement.weight_kg,
        new Date(measurement.timestamp),
        measurement.source,
        this.config,  // config
        'kg', // unit
        stateStore,
        null  // user_height_m
      );

      results.push(this.serializeResult(result));
    }

    // Get final state
    const finalState = await stateStore.getState(combinedUserId);

    return {
      results,
      finalState: this.serializeState(finalState),
    };
  }

  /**
   * Serialize result for comparison
   */
  private serializeResult(result: any): any {
    if (!result) return null;

    // Convert to plain object, handling any special types
    return JSON.parse(JSON.stringify(result));
  }

  /**
   * Serialize state for comparison
   */
  private serializeState(state: any): any {
    if (!state) return null;

    // Convert to plain object
    return JSON.parse(JSON.stringify(state));
  }

  /**
   * Test single measurement processing
   */
  async processSingle(
    deviceId: string,
    userId: string,
    measurement: Measurement
  ): Promise<any> {
    const result = await this.processMeasurements({
      deviceId,
      userId,
      measurements: [measurement],
    });

    return result.results[0];
  }

  /**
   * Process measurements one by one and return all results
   */
  async processSequence(
    deviceId: string,
    userId: string,
    measurements: Measurement[]
  ): Promise<any[]> {
    const result = await this.processMeasurements({
      deviceId,
      userId,
      measurements,
    });

    return result.results;
  }
}

/**
 * Default TypeScript wrapper instance
 */
export const typescriptWrapper = new TypeScriptWrapper();
