/**
 * Service layer for weight processing operations.
 *
 * Provides a simplified interface for:
 * - Processing single measurements
 * - Processing batches of measurements
 * - Managing state (get/reset)
 * - Replay operations (future)
 *
 * This is a simplified version suitable for both CLI and library use.
 * For full AWS API functionality, see the Python version.
 */

import type { StateStore, ProcessorState, ProcessResult, Config } from '../models';
import { processMeasurement } from '../core/processing/processor';

/**
 * Interface for measurement input (simplified from API models)
 */
export interface MeasurementInput {
  measurement_id?: string;
  weight: number;
  unit?: string;
  timestamp: Date | string;
  source: string;
  user_height_m?: number;
}

/**
 * Interface for batch processing results
 */
export interface BatchProcessResult {
  user_id: string;
  measurements_processed: number;
  measurements_accepted: number;
  measurements_rejected: number;
  results: ProcessResult[];
  processing_time_ms: number;
}

/**
 * Service layer for weight processing operations.
 */
export class WeightProcessorService {
  private state_store: StateStore;
  private config: Config;

  /**
   * Initialize service.
   *
   * @param state_store - State storage backend
   * @param config - Configuration dictionary
   */
  constructor(state_store: StateStore, config: Config) {
    this.state_store = state_store;
    this.config = config;
  }

  /**
   * Process a single measurement for a user.
   *
   * @param user_id - User identifier
   * @param measurement - Measurement to process
   * @returns ProcessResult with acceptance status and details
   */
  async process_single(user_id: string, measurement: MeasurementInput): Promise<ProcessResult> {
    // Parse timestamp if string
    let timestamp: Date;
    if (typeof measurement.timestamp === 'string') {
      timestamp = new Date(measurement.timestamp);
    } else {
      timestamp = measurement.timestamp;
    }

    // Call the processor
    const result = await processMeasurement(
      user_id,
      measurement.weight,
      timestamp,
      measurement.source,
      this.config,
      this.state_store,
      measurement.unit || 'kg',
      measurement.user_height_m
    );

    return result;
  }

  /**
   * Process a batch of measurements for a user.
   *
   * Measurements are processed in chronological order.
   * Each measurement is processed independently through the normal pipeline.
   *
   * @param user_id - User identifier
   * @param measurements - List of measurements to process
   * @returns BatchProcessResult with results for all measurements
   */
  async process_batch(user_id: string, measurements: MeasurementInput[]): Promise<BatchProcessResult> {
    const start_time = Date.now();

    // Sort measurements chronologically
    const sorted_measurements = [...measurements].sort((a, b) => {
      const ts_a = typeof a.timestamp === 'string' ? new Date(a.timestamp) : a.timestamp;
      const ts_b = typeof b.timestamp === 'string' ? new Date(b.timestamp) : b.timestamp;
      return ts_a.getTime() - ts_b.getTime();
    });

    // Process each measurement
    const results: ProcessResult[] = [];
    let accepted_count = 0;
    let rejected_count = 0;

    for (const measurement of sorted_measurements) {
      try {
        const result = await this.process_single(user_id, measurement);
        results.push(result);

        if (result.accepted) {
          accepted_count++;
        } else {
          rejected_count++;
        }
      } catch (e) {
        const error = e as Error;
        console.error(`Error processing measurement for ${user_id}: ${error.message}`);

        // Create error result
        const error_result: ProcessResult = {
          accepted: false,
          rejected: true,
          timestamp: typeof measurement.timestamp === 'string'
            ? new Date(measurement.timestamp)
            : measurement.timestamp,
          source: measurement.source,
          raw_weight: measurement.weight,
          reason: `Processing error: ${error.message}`,
          stage: 'processing'
        };
        results.push(error_result);
        rejected_count++;
      }
    }

    const processing_time_ms = Date.now() - start_time;

    return {
      user_id,
      measurements_processed: results.length,
      measurements_accepted: accepted_count,
      measurements_rejected: rejected_count,
      results,
      processing_time_ms
    };
  }

  /**
   * Get the current processing state for a user.
   *
   * @param user_id - User identifier
   * @returns ProcessorState or null if no state exists
   */
  get_state(user_id: string): ProcessorState | null {
    return this.state_store.get_state(user_id);
  }

  /**
   * Reset (delete) the processing state for a user.
   *
   * This clears all Kalman filter state, history, and snapshots for the user.
   * The next measurement for this user will initialize a new state.
   *
   * @param user_id - User identifier
   * @returns True if state was deleted, false if no state existed
   */
  reset_state(user_id: string): boolean {
    try {
      // Check if state exists
      const state = this.state_store.get_state(user_id);
      if (!state) {
        console.warn(`No state found for user ${user_id}`);
        return false;
      }

      // Delete the state
      this.state_store.delete_state(user_id);
      console.info(`Reset state for user ${user_id}`);
      return true;
    } catch (e) {
      const error = e as Error;
      console.error(`Error resetting state for user ${user_id}: ${error.message}`);
      return false;
    }
  }

  /**
   * Get statistics about the state store.
   *
   * @returns Statistics object
   */
  get_stats(): { total_users: number; config: Config } {
    // This is a simple implementation - can be extended
    return {
      total_users: 0, // Would need to track this in state store
      config: this.config
    };
  }

  /**
   * Process measurements for multiple users in batch.
   *
   * This is a convenience method for processing measurements for many users.
   *
   * @param measurements_by_user - Map of user_id to list of measurements
   * @param progress_callback - Optional callback for progress updates
   * @returns Map of user_id to batch results
   */
  process_multi_user(
    measurements_by_user: Map<string, MeasurementInput[]>,
    progress_callback?: (user_id: string, progress: number, total: number) => void
  ): Map<string, BatchProcessResult> {
    const results = new Map<string, BatchProcessResult>();
    const total_users = measurements_by_user.size;
    let processed_users = 0;

    for (const [user_id, measurements] of measurements_by_user.entries()) {
      try {
        const result = this.process_batch(user_id, measurements);
        results.set(user_id, result);

        processed_users++;
        if (progress_callback) {
          progress_callback(user_id, processed_users, total_users);
        }
      } catch (e) {
        const error = e as Error;
        console.error(`Error processing user ${user_id}: ${error.message}`);

        // Create error result
        const error_result: BatchProcessResult = {
          user_id,
          measurements_processed: 0,
          measurements_accepted: 0,
          measurements_rejected: measurements.length,
          results: [],
          processing_time_ms: 0
        };
        results.set(user_id, error_result);
      }
    }

    return results;
  }
}
