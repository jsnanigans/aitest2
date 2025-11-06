/**
 * Simple replay service for MVP.
 *
 * Provides functionality to replay measurements from a specific timestamp,
 * restoring state from snapshots and reprocessing measurements.
 */

import type { StateStore, Config, ProcessResult } from '../models';
import { processMeasurement } from '../core/processing/processor';
import type { MeasurementInput } from './weight_processor_service';

/**
 * Result interface for replay operations
 */
export interface ReplayResult {
  uuid: string;
  accepted: boolean;
  quality_score?: number;
  kalman_estimate?: number;
}

/**
 * Output interface for replay_measurements function
 */
export interface ReplayOutput {
  success: boolean;
  processed_count?: number;
  accepted_count?: number;
  rejected_count?: number;
  results?: ReplayResult[];
  error?: string;
}

/**
 * Simple replay: restore state and reprocess measurements.
 *
 * @param user_id - User identifier
 * @param measurements - All measurements to replay
 * @param replay_from - Timestamp to replay from
 * @param state_store - Database instance
 * @param config - Configuration
 * @param user_height_m - User height in meters (optional)
 * @returns Result object with processing results
 */
export async function replayMeasurements(
  user_id: string,
  measurements: MeasurementInput[],
  replay_from: Date,
  state_store: StateStore,
  config: Config,
  user_height_m?: number
): Promise<ReplayOutput> {
  try {
    // Step 1: Get snapshot before replay_from
    const snapshot = state_store.get_snapshot(user_id, replay_from);

    if (snapshot) {
      // Restore from snapshot
      state_store.save_state(user_id, snapshot);
      console.log(`Restored state from snapshot at ${replay_from.toISOString()}`);
    } else {
      // No snapshot - reset state
      state_store.delete_state(user_id);
      console.log('No snapshot found, starting fresh');
    }

    // Step 2: Filter and sort measurements
    const replay_measurements = measurements
      .filter(m => {
        const ts = typeof m.timestamp === 'string' ? new Date(m.timestamp) : m.timestamp;
        return ts >= replay_from;
      })
      .sort((a, b) => {
        const ts_a = typeof a.timestamp === 'string' ? new Date(a.timestamp) : a.timestamp;
        const ts_b = typeof b.timestamp === 'string' ? new Date(b.timestamp) : b.timestamp;
        return ts_a.getTime() - ts_b.getTime();
      });

    // Step 3: Process measurements
    const results: ReplayResult[] = [];
    let accepted_count = 0;
    let rejected_count = 0;

    for (const measurement of replay_measurements) {
      const timestamp = typeof measurement.timestamp === 'string'
        ? new Date(measurement.timestamp)
        : measurement.timestamp;

      const result = await processMeasurement(
        user_id,
        measurement.weight,
        timestamp,
        measurement.source,
        config,
        state_store,
        measurement.unit || 'kg',
        user_height_m
      );

      results.push({
        uuid: measurement.measurement_id || `${user_id}_${timestamp.getTime()}`,
        accepted: result.accepted || false,
        quality_score: result.quality_score,
        kalman_estimate: result.kalman_estimate,
      });

      if (result.accepted) {
        accepted_count++;
      } else {
        rejected_count++;
      }
    }

    // Step 4: Create snapshot after replay
    state_store.save_state_snapshot(user_id, new Date());

    return {
      success: true,
      processed_count: replay_measurements.length,
      accepted_count,
      rejected_count,
      results,
    };

  } catch (e) {
    const error = e as Error;
    console.error(`Replay failed for user ${user_id}:`, error);
    return {
      success: false,
      error: error.message
    };
  }
}
