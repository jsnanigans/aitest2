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
 * Interface for replay metadata
 */
export interface ReplayMetadata {
  trigger: string;
  buffer_size: number;
  replay_from: string;
  replay_to: string;
  measurements_replayed: number;
  duration_seconds: number;
  timestamp: string;
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
  replay_metadata?: ReplayMetadata[];
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
   * Process a batch of measurements for a user with automatic buffered replay.
   *
   * Measurements are processed in chronological order.
   * Accepted measurements are buffered and replayed automatically when:
   * - Last measurement in batch (AND buffer >= 2)
   * - Time window exceeded (AND buffer >= 2)
   * - Buffer size limit reached (AND buffer >= 2)
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

    // Check if buffered replay is enabled
    const buffered_replay_enabled = this.config.replay?.buffered_replay_enabled ?? true;

    // Initialize buffer for replay processing (only if enabled)
    const buffer: MeasurementInput[] = [];
    let buffer_start_time: Date | null = null;
    const replay_metadata: ReplayMetadata[] = [];

    // Process each measurement
    const results: ProcessResult[] = [];
    let accepted_count = 0;
    let rejected_count = 0;

    for (let i = 0; i < sorted_measurements.length; i++) {
      const measurement = sorted_measurements[i];

      // Check if replay should be triggered BEFORE processing current measurement
      if (buffered_replay_enabled && buffer.length > 0) {
        const buffer_hours = this.config.replay?.buffer_hours ?? 24;

        // Check if current measurement is outside the time window from the last buffered measurement
        const last_buffered = buffer[buffer.length - 1];
        const last_buffered_time = typeof last_buffered.timestamp === 'string'
          ? new Date(last_buffered.timestamp)
          : last_buffered.timestamp;
        const current_timestamp = typeof measurement.timestamp === 'string'
          ? new Date(measurement.timestamp)
          : measurement.timestamp;
        const time_gap_hours = (current_timestamp.getTime() - last_buffered_time.getTime()) / (1000 * 3600);

        // If time gap exceeds buffer window
        if (time_gap_hours >= buffer_hours) {
          // Trigger replay if we have enough measurements
          if (buffer.length >= 2) {
            const buffer_first_ts = typeof buffer[0].timestamp === 'string'
              ? new Date(buffer[0].timestamp)
              : buffer[0].timestamp;
            const buffer_last_ts = typeof buffer[buffer.length - 1].timestamp === 'string'
              ? new Date(buffer[buffer.length - 1].timestamp)
              : buffer[buffer.length - 1].timestamp;

            console.log(
              `Triggering replay for user ${user_id}: trigger=time_gap, ` +
              `buffer_size=${buffer.length}, time_gap=${time_gap_hours.toFixed(1)}h, ` +
              `buffer_range=${buffer_first_ts.toISOString()} to ${buffer_last_ts.toISOString()}`
            );

            // Execute replay
            const replay_output = await this._execute_buffered_replay(
              user_id, buffer, buffer_start_time!
            );

            // Merge replay results into original results
            this._merge_replay_results(results, replay_output, buffer);

            // Track replay metadata
            replay_metadata.push({
              trigger: "time_gap",
              buffer_size: buffer.length,
              replay_from: buffer_start_time!.toISOString(),
              replay_to: buffer_last_ts.toISOString(),
              measurements_replayed: buffer.length,
              duration_seconds: replay_output.duration_seconds || 0,
              timestamp: new Date().toISOString(),
            });
          } else {
            console.log(
              `Time gap ${time_gap_hours.toFixed(1)}h exceeds buffer window but only ${buffer.length} measurement(s) in buffer - no replay`
            );
          }

          // Clear buffer for next window (regardless of whether replay triggered)
          buffer.length = 0;
          buffer_start_time = null;
        }
      }

      try {
        const result = await this.process_single(user_id, measurement);
        results.push(result);

        if (result.accepted) {
          accepted_count++;
        } else {
          rejected_count++;
        }

        // Buffer management: Add ALL measurements to buffer (accepted or rejected)
        // This allows replays to reconsider rejected measurements with better context
        if (buffered_replay_enabled) {
          // Create snapshot before first buffered measurement in the window
          if (buffer.length === 0) {
            const ts = typeof measurement.timestamp === 'string'
              ? new Date(measurement.timestamp)
              : measurement.timestamp;
            buffer_start_time = ts;
            this.state_store.save_state_snapshot(user_id, buffer_start_time);
            console.log(`Created snapshot for user ${user_id} at ${buffer_start_time.toISOString()}`);
          }

          // Add measurement to buffer (both accepted and rejected)
          buffer.push(measurement);
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

      // Check if replay should be triggered at batch end
      if (buffered_replay_enabled) {
        const is_last = (i === sorted_measurements.length - 1);
        const current_timestamp = typeof measurement.timestamp === 'string'
          ? new Date(measurement.timestamp)
          : measurement.timestamp;
        const should_replay = this._should_trigger_replay(buffer, current_timestamp, is_last);

        if (should_replay && buffer.length > 0) {
          // Determine trigger reason
          let trigger_reason: string;
          if (is_last) {
            trigger_reason = "batch_end";
          } else if (buffer.length >= (this.config.replay?.max_buffer_measurements ?? 100)) {
            trigger_reason = "buffer_overflow";
          } else {
            trigger_reason = "time_window";
          }

          const buffer_first_ts = typeof buffer[0].timestamp === 'string'
            ? new Date(buffer[0].timestamp)
            : buffer[0].timestamp;
          const buffer_last_ts = typeof buffer[buffer.length - 1].timestamp === 'string'
            ? new Date(buffer[buffer.length - 1].timestamp)
            : buffer[buffer.length - 1].timestamp;

          console.log(
            `Triggering replay for user ${user_id}: trigger=${trigger_reason}, ` +
            `buffer_size=${buffer.length}, time_range=${buffer_first_ts.toISOString()} to ${buffer_last_ts.toISOString()}`
          );

          // Execute replay
          const replay_output = await this._execute_buffered_replay(
            user_id, buffer, buffer_start_time!
          );

          // Merge replay results into original results
          this._merge_replay_results(results, replay_output, buffer);

          // Track replay metadata
          replay_metadata.push({
            trigger: trigger_reason,
            buffer_size: buffer.length,
            replay_from: buffer_start_time!.toISOString(),
            replay_to: buffer_last_ts.toISOString(),
            measurements_replayed: buffer.length,
            duration_seconds: replay_output.duration_seconds || 0,
            timestamp: new Date().toISOString(),
          });

          // Clear buffer for next window
          buffer.length = 0;
          buffer_start_time = null;
        }
      }
    }

    const processing_time_ms = Date.now() - start_time;

    return {
      user_id,
      measurements_processed: results.length,
      measurements_accepted: accepted_count,
      measurements_rejected: rejected_count,
      results,
      processing_time_ms,
      replay_metadata: replay_metadata.length > 0 ? replay_metadata : undefined
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
  async process_multi_user(
    measurements_by_user: Map<string, MeasurementInput[]>,
    progress_callback?: (user_id: string, progress: number, total: number) => void
  ): Promise<Map<string, BatchProcessResult>> {
    const results = new Map<string, BatchProcessResult>();
    const total_users = measurements_by_user.size;
    let processed_users = 0;

    for (const [user_id, measurements] of measurements_by_user.entries()) {
      try {
        const result = await this.process_batch(user_id, measurements);
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

  /**
   * CamelCase wrapper for process_batch (for compatibility)
   */
  async processBatch(user_id: string, measurements: MeasurementInput[]): Promise<BatchProcessResult> {
    return this.process_batch(user_id, measurements);
  }

  /**
   * CamelCase wrapper for process_single (for compatibility)
   */
  async processSingle(user_id: string, measurement: MeasurementInput): Promise<ProcessResult> {
    return this.process_single(user_id, measurement);
  }

  /**
   * Determine if replay should be triggered for the current buffer.
   *
   * Replay is triggered when:
   * 1. Last measurement in batch (is_last=true) AND buffer has >= 2 measurements
   * 2. Time window exceeded (buffer_hours) AND buffer has >= 2 measurements
   * 3. Buffer size limit reached AND buffer has >= 2 measurements
   *
   * @param buffer - List of buffered measurements
   * @param current_timestamp - Timestamp of current measurement being processed
   * @param is_last - Whether this is the last measurement in the batch
   * @returns True if replay should be triggered, False otherwise
   */
  private _should_trigger_replay(
    buffer: MeasurementInput[],
    current_timestamp: Date,
    is_last: boolean
  ): boolean {
    // Minimum buffer size: need at least 2 measurements to replay
    if (buffer.length < 2) {
      return false;
    }

    // Trigger 1: Last measurement in batch
    if (is_last) {
      return true;
    }

    // Trigger 2: Time window exceeded
    const buffer_hours = this.config.replay?.buffer_hours ?? 24;
    const first_timestamp = typeof buffer[0].timestamp === 'string'
      ? new Date(buffer[0].timestamp)
      : buffer[0].timestamp;
    const hours_elapsed = (current_timestamp.getTime() - first_timestamp.getTime()) / (1000 * 3600);

    if (hours_elapsed >= buffer_hours) {
      return true;
    }

    // Trigger 3: Buffer size limit (safety)
    const max_buffer = this.config.replay?.max_buffer_measurements ?? 100;
    if (buffer.length >= max_buffer) {
      return true;
    }

    return false;
  }

  /**
   * Execute replay for buffered measurements.
   *
   * @param user_id - User identifier
   * @param buffer - List of buffered measurements to replay
   * @param buffer_start_time - Timestamp to replay from (snapshot timestamp)
   * @returns Replay result with processing results (includes 'duration_seconds')
   */
  private async _execute_buffered_replay(
    user_id: string,
    buffer: MeasurementInput[],
    buffer_start_time: Date
  ): Promise<any> {
    try {
      console.log(
        `Executing buffered replay for user ${user_id}: ` +
        `buffer_size=${buffer.length}, replay_from=${buffer_start_time.toISOString()}`
      );

      // Track replay performance
      const replay_start = Date.now();

      // Import and call replay service
      const { replayMeasurements } = await import('./replay_service');

      const replay_output = await replayMeasurements(
        user_id,
        buffer,
        buffer_start_time,
        this.state_store,
        this.config
      );

      const replay_duration = (Date.now() - replay_start) / 1000;

      if (!replay_output.success) {
        const error_msg = replay_output.error || 'Unknown error';
        console.error(`Replay failed for user ${user_id}: ${error_msg}`);
        throw new Error(`Replay failed: ${error_msg}`);
      }

      console.log(
        `Replay completed for user ${user_id}: ` +
        `processed=${replay_output.processed_count || 0}, ` +
        `accepted=${replay_output.accepted_count || 0}, ` +
        `duration=${replay_duration.toFixed(2)}s`
      );

      // Add duration to output
      replay_output.duration_seconds = Math.round(replay_duration * 100) / 100;

      return replay_output;

    } catch (e) {
      const error = e as Error;
      console.error(`Replay execution failed for user ${user_id}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Merge replay results back into original results list.
   *
   * @param original_results - Original processing results
   * @param replay_output - Replay service output
   * @param buffer - List of buffered measurements that were replayed
   */
  private _merge_replay_results(
    original_results: ProcessResult[],
    replay_output: any,
    buffer: MeasurementInput[]
  ): void {
    // Create lookup map: measurement_id -> replay result
    const replay_map = new Map<string, any>();
    for (const r of replay_output.results || []) {
      replay_map.set(r.uuid, r);
    }

    // Create set of buffered measurement IDs for quick lookup
    const buffered_ids = new Set<string>();
    for (const m of buffer) {
      const id = m.measurement_id || `${m.timestamp}`;
      buffered_ids.add(id);
    }

    // Update original results with replay data
    for (let i = 0; i < original_results.length; i++) {
      const original = original_results[i];
      const measurement_id = original.uuid || `${original.timestamp}`;

      // Check if this measurement was in the buffer and has replay data
      if (buffered_ids.has(measurement_id) && replay_map.has(measurement_id)) {
        const replay_data = replay_map.get(measurement_id);

        // Update result with replay data - use replay data for all processing fields
        original_results[i] = {
          ...original,
          accepted: replay_data.accepted ?? original.accepted,
          quality_score: replay_data.quality_score ?? original.quality_score,
          kalman_estimate: replay_data.kalman_estimate ?? original.kalman_estimate,
          reason: replay_data.rejection_reason ?? original.reason,
          stage: replay_data.processing_stage ?? original.stage,
        };

        console.log(
          `Updated result for measurement ${measurement_id}: ` +
          `accepted=${original_results[i].accepted}, quality_score=${original_results[i].quality_score}`
        );
      }
    }
  }
}
