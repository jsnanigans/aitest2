/**
 * Service layer for weight processing operations.
 *
 * Minimal implementation using only typescript_lib core library.
 * Mirrors the Python be_implementation_service WeightProcessorService.
 */

import {
  processMeasurement,
  type ProcessingResult,
  type StateStore,
} from "../typescript_lib/src/index";

/**
 * Interface for measurement input
 */
export interface MeasurementInput {
  measurementId?: string;
  weight: number;
  unit?: string;
  timestamp: Date;
  source: string;
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
 * Interface for batch processing response
 */
export interface ProcessResponseData {
  userId: string;
  measurements_processed: number;
  measurements_accepted: number;
  measurements_rejected: number;
  results: ProcessingResult[];
  replay_metadata?: ReplayMetadata[];
}

/**
 * Service layer for weight processing operations.
 */
export class WeightProcessorService {
  private stateStore: StateStore;
  private config: any;

  /**
   * Initialize service.
   *
   * @param stateStore - State storage backend
   * @param config - Configuration dictionary
   */
  constructor(stateStore: StateStore, config: any) {
    this.stateStore = stateStore;
    this.config = config;
  }

  /**
   * Process a batch of measurements for a user with automatic buffered replay.
   *
   * Measurements are processed in chronological order.
   * Accepted measurements are buffered and replayed automatically when:
   * - Last measurement in batch (AND buffer >= 2)
   * - Time gap exceeded (AND buffer >= 2)
   * - Buffer size limit reached (AND buffer >= 2)
   *
   * @param userId - User identifier
   * @param measurements - List of measurements to process
   * @returns ProcessResponseData with results for all measurements
   */
  async processBatch(
    userId: string,
    measurements: MeasurementInput[]
  ): Promise<ProcessResponseData> {
    const startTime = Date.now();

    // Sort measurements chronologically
    const sortedMeasurements = [...measurements].sort((a, b) => {
      return a.timestamp.getTime() - b.timestamp.getTime();
    });

    // Check if buffered replay is enabled
    const bufferedReplayEnabled =
      this.config.replay?.buffered_replay_enabled ?? true;

    // Initialize buffer for replay processing (only if enabled)
    const buffer: MeasurementInput[] = [];
    let bufferStartTime: Date | null = null;
    const replayMetadata: ReplayMetadata[] = [];

    // Process each measurement
    const results: ProcessingResult[] = [];
    let acceptedCount = 0;
    let rejectedCount = 0;

    for (let i = 0; i < sortedMeasurements.length; i++) {
      const measurement = sortedMeasurements[i];
      const isLast = i === sortedMeasurements.length - 1;

      // Check if replay should be triggered BEFORE processing current measurement (time_gap trigger)
      if (bufferedReplayEnabled && buffer.length > 0) {
        const bufferHours = this.config.replay?.buffer_hours ?? 24;

        // Check if current measurement is outside the time window from the last buffered measurement
        const lastBuffered = buffer[buffer.length - 1];
        const lastBufferedTime = lastBuffered.timestamp;
        const currentTimestamp = measurement.timestamp;
        const timeGapHours =
          (currentTimestamp.getTime() - lastBufferedTime.getTime()) /
          (1000 * 3600);

        // If time gap exceeds buffer window
        if (timeGapHours >= bufferHours) {
          // Trigger replay if we have enough measurements
          if (buffer.length >= 2) {
            const bufferFirstTs = buffer[0].timestamp;
            const bufferLastTs = buffer[buffer.length - 1].timestamp;

            console.log(
              `Triggering replay for user ${userId}: trigger=time_gap, ` +
                `buffer_size=${buffer.length}, time_gap=${timeGapHours.toFixed(1)}h, ` +
                `buffer_range=${bufferFirstTs.toISOString()} to ${bufferLastTs.toISOString()}`
            );

            // Execute replay
            const replayOutput = await this._executeBufferedReplay(
              userId,
              buffer,
              bufferStartTime!
            );

            // Merge replay results into original results
            this._mergeReplayResults(results, replayOutput, buffer);

            // Track replay metadata
            replayMetadata.push({
              trigger: "time_gap",
              buffer_size: buffer.length,
              replay_from: bufferStartTime!.toISOString(),
              replay_to: bufferLastTs.toISOString(),
              measurements_replayed: buffer.length,
              duration_seconds: replayOutput.duration_seconds || 0,
              timestamp: new Date().toISOString(),
            });
          } else {
            console.log(
              `Time gap ${timeGapHours.toFixed(1)}h exceeds buffer window but only ${buffer.length} measurement(s) in buffer - no replay`
            );
          }

          // Clear buffer for next window (regardless of whether replay triggered)
          buffer.length = 0;
          bufferStartTime = null;
        }
      }

      // Process the measurement
      try {
        const result = await processMeasurement(
          userId,
          measurement.weight,
          measurement.timestamp,
          measurement.source,
          this.config,
          measurement.unit || "kg",
          this.stateStore,
          null // user_height_m
        );

        results.push(result);

        if (result.accepted) {
          acceptedCount++;
        } else {
          rejectedCount++;
        }

        // Buffer management: Add ALL measurements to buffer (accepted or rejected)
        // This allows replays to reconsider rejected measurements with better context
        if (bufferedReplayEnabled) {
          // Create snapshot before first buffered measurement in the window
          if (buffer.length === 0) {
            bufferStartTime = measurement.timestamp;
            await this.stateStore.saveStateSnapshot(userId, bufferStartTime);
            console.log(
              `Created snapshot for user ${userId} at ${bufferStartTime.toISOString()}`
            );
          }

          // Add measurement to buffer (both accepted and rejected)
          buffer.push(measurement);
        }
      } catch (e) {
        const error = e as Error;
        console.error(
          `Error processing measurement for ${userId}: ${error.message}`
        );

        // Create error result
        const errorResult: ProcessingResult = {
          accepted: false,
          rejected: true,
          timestamp: measurement.timestamp,
          source: measurement.source,
          raw_weight: measurement.weight,
          reason: `Processing error: ${error.message}`,
          stage: "processing",
        };
        results.push(errorResult);
        rejectedCount++;
      }

      // Check if replay should be triggered at batch end or buffer overflow
      if (bufferedReplayEnabled && buffer.length > 0) {
        const shouldReplay = this._shouldTriggerReplay(
          buffer,
          measurement.timestamp,
          isLast
        );

        if (shouldReplay) {
          // Determine trigger reason
          let triggerReason: string;
          if (isLast) {
            triggerReason = "batch_end";
          } else if (
            buffer.length >=
            (this.config.replay?.max_buffer_measurements ?? 100)
          ) {
            triggerReason = "buffer_overflow";
          } else {
            triggerReason = "time_window";
          }

          const bufferFirstTs = buffer[0].timestamp;
          const bufferLastTs = buffer[buffer.length - 1].timestamp;

          console.log(
            `Triggering replay for user ${userId}: trigger=${triggerReason}, ` +
              `buffer_size=${buffer.length}, time_range=${bufferFirstTs.toISOString()} to ${bufferLastTs.toISOString()}`
          );

          // Execute replay
          const replayOutput = await this._executeBufferedReplay(
            userId,
            buffer,
            bufferStartTime!
          );

          // Merge replay results into original results
          this._mergeReplayResults(results, replayOutput, buffer);

          // Track replay metadata
          replayMetadata.push({
            trigger: triggerReason,
            buffer_size: buffer.length,
            replay_from: bufferStartTime!.toISOString(),
            replay_to: bufferLastTs.toISOString(),
            measurements_replayed: buffer.length,
            duration_seconds: replayOutput.duration_seconds || 0,
            timestamp: new Date().toISOString(),
          });

          // Clear buffer for next window
          buffer.length = 0;
          bufferStartTime = null;
        }
      }
    }

    const processingTimeMs = Date.now() - startTime;

    return {
      userId,
      measurements_processed: results.length,
      measurements_accepted: acceptedCount,
      measurements_rejected: rejectedCount,
      results,
      replay_metadata: replayMetadata.length > 0 ? replayMetadata : undefined,
    };
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
   * @param currentTimestamp - Timestamp of current measurement being processed
   * @param isLast - Whether this is the last measurement in the batch
   * @returns True if replay should be triggered, False otherwise
   */
  private _shouldTriggerReplay(
    buffer: MeasurementInput[],
    currentTimestamp: Date,
    isLast: boolean
  ): boolean {
    // Minimum buffer size: need at least 2 measurements to replay
    if (buffer.length < 2) {
      return false;
    }

    // Trigger 1: Last measurement in batch
    if (isLast) {
      return true;
    }

    // Trigger 2: Time window exceeded
    const bufferHours = this.config.replay?.buffer_hours ?? 24;
    const firstTimestamp = buffer[0].timestamp;
    const hoursElapsed =
      (currentTimestamp.getTime() - firstTimestamp.getTime()) / (1000 * 3600);

    if (hoursElapsed >= bufferHours) {
      return true;
    }

    // Trigger 3: Buffer size limit (safety)
    const maxBuffer = this.config.replay?.max_buffer_measurements ?? 100;
    if (buffer.length >= maxBuffer) {
      return true;
    }

    return false;
  }

  /**
   * Execute replay for buffered measurements.
   *
   * @param userId - User identifier
   * @param buffer - List of buffered measurements to replay
   * @param bufferStartTime - Timestamp to replay from (snapshot timestamp)
   * @returns Replay result with processing results (includes 'duration_seconds')
   */
  private async _executeBufferedReplay(
    userId: string,
    buffer: MeasurementInput[],
    bufferStartTime: Date
  ): Promise<any> {
    try {
      console.log(
        `Executing buffered replay for user ${userId}: ` +
          `buffer_size=${buffer.length}, replay_from=${bufferStartTime.toISOString()}`
      );

      // Track replay performance
      const replayStart = Date.now();

      // Step 1: Restore state snapshot
      const snapshotResult = await this.stateStore.checkAndRestoreSnapshot(
        userId,
        bufferStartTime
      );

      if (!snapshotResult.snapshot_found) {
        throw new Error(
          `No snapshot found for user ${userId} at ${bufferStartTime.toISOString()}`
        );
      }

      console.log(
        `Restored snapshot for user ${userId} from ${snapshotResult.snapshot_timestamp?.toISOString()}`
      );

      // Step 2: Reprocess buffered measurements
      const replayResults: ProcessingResult[] = [];
      let processedCount = 0;
      let acceptedCount = 0;

      for (const measurement of buffer) {
        try {
          const result = await processMeasurement(
            userId,
            measurement.weight,
            measurement.timestamp,
            measurement.source,
            this.config,
            measurement.unit || "kg",
            this.stateStore,
            null // user_height_m
          );

          replayResults.push(result);
          processedCount++;

          if (result.accepted) {
            acceptedCount++;
          }
        } catch (e) {
          const error = e as Error;
          console.error(
            `Error during replay for measurement: ${error.message}`
          );

          // Create error result
          const errorResult: ProcessingResult = {
            accepted: false,
            rejected: true,
            timestamp: measurement.timestamp,
            source: measurement.source,
            raw_weight: measurement.weight,
            reason: `Replay error: ${error.message}`,
            stage: "replay",
          };
          replayResults.push(errorResult);
          processedCount++;
        }
      }

      const replayDuration = (Date.now() - replayStart) / 1000;

      console.log(
        `Replay completed for user ${userId}: ` +
          `processed=${processedCount}, ` +
          `accepted=${acceptedCount}, ` +
          `duration=${replayDuration.toFixed(2)}s`
      );

      return {
        success: true,
        processed_count: processedCount,
        accepted_count: acceptedCount,
        results: replayResults,
        duration_seconds: Math.round(replayDuration * 100) / 100,
      };
    } catch (e) {
      const error = e as Error;
      console.error(
        `Replay execution failed for user ${userId}: ${error.message}`
      );
      return {
        success: false,
        error: error.message,
        results: [],
        duration_seconds: 0,
      };
    }
  }

  /**
   * Merge replay results back into original results list.
   *
   * @param originalResults - Original processing results
   * @param replayOutput - Replay service output
   * @param buffer - List of buffered measurements that were replayed
   */
  private _mergeReplayResults(
    originalResults: ProcessingResult[],
    replayOutput: any,
    buffer: MeasurementInput[]
  ): void {
    // Create lookup map: timestamp -> replay result
    const replayMap = new Map<string, ProcessingResult>();
    for (const r of replayOutput.results || []) {
      const key = r.timestamp.toISOString();
      replayMap.set(key, r);
    }

    // Create set of buffered measurement timestamps for quick lookup
    const bufferedTimestamps = new Set<string>();
    for (const m of buffer) {
      bufferedTimestamps.add(m.timestamp.toISOString());
    }

    // Update original results with replay data
    for (let i = 0; i < originalResults.length; i++) {
      const original = originalResults[i];
      const timestampKey = original.timestamp.toISOString();

      // Check if this measurement was in the buffer and has replay data
      if (
        bufferedTimestamps.has(timestampKey) &&
        replayMap.has(timestampKey)
      ) {
        const replayData = replayMap.get(timestampKey)!;

        // Update result with replay data - use replay data for all processing fields
        originalResults[i] = {
          ...original,
          accepted: replayData.accepted ?? original.accepted,
          quality_score: replayData.quality_score ?? original.quality_score,
          filtered_weight: replayData.filtered_weight ?? original.filtered_weight,
          reason: replayData.reason ?? original.reason,
          stage: replayData.stage ?? original.stage,
        };

        console.log(
          `Updated result for measurement ${timestampKey}: ` +
            `accepted=${originalResults[i].accepted}, quality_score=${originalResults[i].quality_score}`
        );
      }
    }
  }

  /**
   * Get the current processing state for a user.
   *
   * @param userId - User identifier
   * @returns ProcessorState or null if no state exists
   */
  async getState(userId: string): Promise<any | null> {
    return this.stateStore.getState(userId);
  }

  /**
   * Reset (delete) the processing state for a user.
   *
   * This clears all Kalman filter state, history, and snapshots for the user.
   * The next measurement for this user will initialize a new state.
   *
   * @param userId - User identifier
   * @returns True if state was deleted, false if no state existed
   */
  async resetState(userId: string): Promise<boolean> {
    try {
      // Check if state exists
      const state = await this.stateStore.getState(userId);
      if (!state) {
        console.warn(`No state found for user ${userId}`);
        return false;
      }

      // Delete the state
      await this.stateStore.deleteState(userId);
      return true;
    } catch (e) {
      const error = e as Error;
      console.error(
        `Error resetting state for user ${userId}: ${error.message}`
      );
      return false;
    }
  }
}
