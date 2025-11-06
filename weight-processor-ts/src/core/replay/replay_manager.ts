/**
 * Replay Manager for Replay Data Quality Processing
 *
 * Handles state restoration and chronological reprocessing of clean measurements.
 * Provides atomic operations with rollback capability for safe replay processing.
 *
 * Key responsibilities:
 * - Restore Kalman state from historical snapshots
 * - Chronologically replay clean measurements
 * - Provide rollback capability on failures
 * - Ensure atomic state transitions
 *
 * Ported from Python: weight_values/src/core/replay/replay_manager.py
 */

import type { StateStore, ProcessorState, Config } from '../../models';
import { process_measurement } from '../processing/processor';
import { prepareMeasurementForProcessing } from '../processing/type_conversion';
import { KalmanFilterManager } from '../processing/kalman';

/**
 * Custom exception for replay operation errors.
 */
export class ReplayError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ReplayError';
  }
}

/**
 * Exception for replay operation timeouts.
 */
export class ReplayTimeoutError extends ReplayError {
  constructor(message: string) {
    super(message);
    this.name = 'ReplayTimeoutError';
  }
}

/**
 * Exception for state-related replay errors.
 */
export class ReplayStateError extends ReplayError {
  constructor(message: string) {
    super(message);
    this.name = 'ReplayStateError';
  }
}

/**
 * Result interface for replay operations.
 */
interface ReplayResult {
  success: boolean;
  error?: string;
  user_id: string;
  reason?: string;
  measurements_replayed?: number;
  processing_time_seconds?: number;
  final_state?: ProcessorState | null;
  restore_point?: any;
  processed_count?: number;
  failed_measurement?: Record<string, any>;
  attempts?: number;
  buffer_start_time?: Date;
}

/**
 * Result interface for state restoration.
 */
interface RestoreResult {
  success: boolean;
  error?: string;
  user_id: string;
  restore_snapshot?: any;
  restored_to_time?: Date;
  attempts?: number;
  buffer_start_time?: Date;
  snapshot_timestamp?: Date;
}

/**
 * Result interface for chronological replay.
 */
interface ChronologicalReplayResult {
  success: boolean;
  error?: string;
  user_id: string;
  processed_count?: number;
  last_result?: any;
  failed_measurement?: Record<string, any>;
}

/**
 * Statistics interface for replay operations.
 */
interface ReplayStats {
  active_backups: number;
  max_processing_time: number;
  preserve_immediate_results: boolean;
  require_rollback_confirmation: boolean;
}

/**
 * Manages state restoration and measurement replay for replay processing.
 * All operations are designed to be atomic with rollback capability.
 */
export class ReplayManager {
  private db: StateStore;
  private config: Config;
  private max_processing_time: number;
  private require_rollback_confirmation: boolean;
  private preserve_immediate_results: boolean;

  // Backup storage for rollback
  private _backup_states: Map<string, ProcessorState>;

  /**
   * Initialize replay manager.
   *
   * @param db - State database instance
   * @param config - Configuration dictionary
   */
  constructor(db: StateStore, config: Config = {} as Config) {
    this.db = db;
    this.config = config;

    // Safety configuration
    this.max_processing_time = (config as any).max_processing_time_seconds || 60;
    this.require_rollback_confirmation = (config as any).require_rollback_confirmation || false;
    this.preserve_immediate_results = (config as any).preserve_immediate_results !== false;

    // Backup storage for rollback
    this._backup_states = new Map();
  }

  /**
   * Replay clean measurements for a user with full rollback capability.
   *
   * @param user_id - User identifier
   * @param clean_measurements - List of measurements without outliers
   * @param buffer_start_time - Start time of the buffer window (for state restoration)
   * @returns Result dictionary with success status and details
   */
  replay_clean_measurements(
    user_id: string,
    clean_measurements: Array<Record<string, any>>,
    buffer_start_time: Date
  ): ReplayResult {
    const start_time = Date.now();

    try {
      // Step 0: Check if replay already in progress (prevent concurrent replay)
      const current_state = this.db.get_state(user_id);
      if (current_state && (current_state as any).replay_in_progress) {
        const replay_start = (current_state as any).replay_started_at;
        return {
          success: false,
          error: `Replay already in progress (started: ${replay_start})`,
          user_id,
          reason: 'concurrent_replay_prevented'
        };
      }

      // Step 1: Create backup of current state
      if (!this._create_state_backup(user_id)) {
        return {
          success: false,
          error: 'Failed to create state backup',
          user_id
        };
      }

      // Step 1.5: Set replay_in_progress flag
      if (!this._set_replay_in_progress(user_id, true)) {
        this._restore_state_from_backup(user_id);
        return {
          success: false,
          error: 'Failed to set replay_in_progress flag',
          user_id
        };
      }

      // Step 2: Restore state to before buffer start
      const restore_result = this._restore_state_to_buffer_start(user_id, buffer_start_time);
      if (!restore_result.success) {
        this._restore_state_from_backup(user_id);
        return restore_result as ReplayResult;
      }

      // Step 2.5: Check trajectory continuity to prevent impossible jumps
      const current_backup_state = this._backup_states.get(user_id);
      const restored_state = this.db.get_state(user_id);

      if (
        current_backup_state &&
        (current_backup_state as any).lastState !== null &&
        restored_state &&
        (restored_state as any).lastState !== null
      ) {
        try {
          const [backup_weight] = KalmanFilterManager.getCurrentStateValues(current_backup_state);
          const [restored_weight] = KalmanFilterManager.getCurrentStateValues(restored_state);

          if (backup_weight !== null && restored_weight !== null) {
            // Ensure both are floats to avoid type errors
            const weight_jump = Math.abs(backup_weight - restored_weight);
            // If restoration would cause >15kg jump, skip replay processing
            if (weight_jump > 15.0) {
              console.warn(
                `Skipping replay processing for ${user_id}: would cause ${weight_jump.toFixed(1)}kg trajectory jump`
              );
              this._restore_state_from_backup(user_id);
              return {
                success: false,
                error: `Trajectory continuity check failed: ${weight_jump.toFixed(1)}kg jump exceeds 15kg limit`,
                user_id
              };
            }
          }
        } catch (e) {
          const error = e as Error;
          // Continue with replay if check fails
        }
      }

      // Step 3: Replay measurements chronologically
      const replay_result = this._replay_measurements_chronologically(
        user_id,
        clean_measurements,
        start_time
      );
      if (!replay_result.success) {
        this._restore_state_from_backup(user_id);
        return replay_result as ReplayResult;
      }

      // Step 4: Verify state was saved before clearing backup
      // (Transactional safety - don't clear backup until we confirm save)
      const final_state = this.db.get_state(user_id);
      if (!final_state) {
        console.error(`Failed to retrieve state after replay for ${user_id}`);
        this._restore_state_from_backup(user_id);
        this._set_replay_in_progress(user_id, false);
        return {
          success: false,
          error: 'State verification failed after replay',
          user_id
        };
      }

      // Step 5: Clear replay flag and backup (only after verification)
      this._set_replay_in_progress(user_id, false);
      this._clear_state_backup(user_id);

      return {
        success: true,
        user_id,
        measurements_replayed: clean_measurements.length,
        processing_time_seconds: (Date.now() - start_time) / 1000,
        final_state,
        restore_point: restore_result.restore_snapshot
      };

    } catch (e) {
      const error = e as Error;
      console.error(`Replay failed for user ${user_id}: ${error.message}`);
      // Emergency rollback
      this._restore_state_from_backup(user_id);
      this._set_replay_in_progress(user_id, false);
      return {
        success: false,
        error: `Replay exception: ${error.message}`,
        user_id,
        processing_time_seconds: (Date.now() - start_time) / 1000
      };
    }
  }

  /**
   * Create a backup of the current state for rollback capability.
   *
   * @param user_id - User identifier
   * @returns True if backup created successfully
   */
  private _create_state_backup(user_id: string): boolean {
    try {
      const current_state = this.db.get_state(user_id);
      if (current_state) {
        // Create deep copy for backup
        this._backup_states.set(user_id, JSON.parse(JSON.stringify(current_state)));
        return true;
      } else {
        console.warn(`No current state found for user ${user_id}`);
        return false;
      }
    } catch (e) {
      const error = e as Error;
      console.error(`Failed to create backup for user ${user_id}: ${error.message}`);
      return false;
    }
  }

  /**
   * Restore state from backup.
   *
   * @param user_id - User identifier
   * @returns True if restoration successful
   */
  private _restore_state_from_backup(user_id: string): boolean {
    try {
      const backup_state = this._backup_states.get(user_id);
      if (backup_state) {
        this.db.save_state(user_id, backup_state);
        return true;
      } else {
        console.error(`No backup found for user ${user_id}`);
        return false;
      }
    } catch (e) {
      const error = e as Error;
      console.error(`Failed to restore from backup for user ${user_id}: ${error.message}`);
      return false;
    }
  }

  /**
   * Clear backup state after successful commit.
   *
   * @param user_id - User identifier
   */
  private _clear_state_backup(user_id: string): void {
    if (this._backup_states.has(user_id)) {
      this._backup_states.delete(user_id);
    }
  }

  /**
   * Set or clear replay_in_progress flag in user state.
   *
   * @param user_id - User identifier
   * @param in_progress - True to set flag, False to clear
   * @returns True if successfully set/cleared
   */
  private _set_replay_in_progress(user_id: string, in_progress: boolean): boolean {
    try {
      const state = this.db.get_state(user_id);
      if (!state) {
        console.error(`No state found for user ${user_id} when setting replay flag`);
        return false;
      }

      const mutable_state = state as any;
      if (in_progress) {
        mutable_state.replay_in_progress = true;
        mutable_state.replay_started_at = new Date().toISOString();
      } else {
        mutable_state.replay_in_progress = false;
        delete mutable_state.replay_started_at;
      }

      this.db.save_state(user_id, state);
      return true;

    } catch (e) {
      const error = e as Error;
      console.error(`Failed to set replay_in_progress flag for ${user_id}: ${error.message}`);
      return false;
    }
  }

  /**
   * Validate snapshot has required fields and reasonable values.
   *
   * @param snapshot - Snapshot to validate
   * @param user_id - User identifier (for logging)
   * @returns True if snapshot is valid
   */
  private _validate_snapshot(snapshot: any, user_id: string): boolean {
    if (!snapshot) {
      console.warn(`Snapshot is null for ${user_id}`);
      return false;
    }

    // Check required fields
    const required_fields = ['last_state', 'last_timestamp'];
    for (const field of required_fields) {
      if (!(field in snapshot)) {
        console.warn(`Snapshot missing required field '${field}' for ${user_id}`);
        return false;
      }
    }

    // Validate last_state structure
    const last_state = snapshot.lastState;
    if (last_state === null) {
      console.warn(`Snapshot has null last_state for ${user_id}`);
      return false;
    }

    // Check if it's a valid array/list with weight component
    try {
      if (Array.isArray(last_state)) {
        if (last_state.length === 0) {
          console.warn(`Snapshot has empty last_state for ${user_id}`);
          return false;
        }
        // Try to access weight (first element)
        const weight = parseFloat(last_state[0]);
        if (weight <= 0 || weight > 500) {  // Sanity check
          console.warn(`Snapshot has invalid weight ${weight}kg for ${user_id}`);
          return false;
        }
      }
    } catch (e) {
      const error = e as Error;
      console.warn(`Snapshot has invalid last_state structure for ${user_id}: ${error.message}`);
      return false;
    }

    // Validate timestamp
    const timestamp = snapshot.lastTimestamp;
    try {
      if (typeof timestamp === 'string') {
        new Date(timestamp.replace('Z', '+00:00'));
      } else if (!(timestamp instanceof Date)) {
        console.warn(`Snapshot has invalid timestamp type for ${user_id}`);
        return false;
      }
    } catch (e) {
      const error = e as Error;
      console.warn(`Snapshot has invalid timestamp for ${user_id}: ${error.message}`);
      return false;
    }

    return true;
  }

  /**
   * Restore user state to just before the buffer start time using atomic operations.
   * Includes retry logic for transient failures and snapshot validation.
   *
   * @param user_id - User identifier
   * @param buffer_start_time - Time to restore state before
   * @returns Result dictionary with success status
   */
  private _restore_state_to_buffer_start(
    user_id: string,
    buffer_start_time: Date
  ): RestoreResult {
    const max_retries = 3;
    let last_error: string | null = null;

    for (let attempt = 0; attempt < max_retries; attempt++) {
      try {
        // Log retry attempt if not first try
        if (attempt > 0) {
          // Exponential backoff: 0.1s, 0.2s, 0.4s
          const delay = 100 * Math.pow(2, attempt);
          // Sleep equivalent in JavaScript
          Bun.sleep(delay);
        }

        // Use atomic check-and-restore method to prevent race condition
        const result = this.db.check_and_restore_snapshot?.(user_id, buffer_start_time);

        // If check_and_restore_snapshot doesn't exist, use fallback
        if (!result) {
          console.error(`check_and_restore_snapshot not implemented on StateStore`);
          return {
            success: false,
            error: 'check_and_restore_snapshot not implemented',
            user_id,
            attempts: attempt + 1
          };
        }

        // Validate snapshot before using it
        if (result.success) {
          const snapshot = result.snapshot;
          if (!this._validate_snapshot(snapshot, user_id)) {
            console.error(`Snapshot validation failed for ${user_id}`);
            return {
              success: false,
              error: 'Snapshot validation failed',
              user_id,
              attempts: attempt + 1
            };
          }
        }

        if (result.success) {
          return {
            success: true,
            user_id,
            restore_snapshot: result.snapshot,
            restored_to_time: result.snapshot_timestamp,
            attempts: attempt + 1
          };
        }

        // Log the specific error
        const error_msg = result.error || 'Unknown error';
        console.warn(`Attempt ${attempt + 1}/${max_retries} failed for ${user_id}: ${error_msg}`);
        last_error = error_msg;

        // Don't retry if snapshot doesn't exist (not a transient failure)
        if (error_msg.includes('No snapshot found')) {
          console.error(
            `No snapshot exists for ${user_id} before ${buffer_start_time}, aborting retries`
          );
          return {
            success: false,
            error: error_msg,
            user_id,
            buffer_start_time,
            attempts: attempt + 1
          };
        }

      } catch (e) {
        const error = e as Error;
        last_error = `State restoration exception: ${error.message}`;
        console.error(
          `Exception during state restoration for ${user_id} ` +
          `(attempt ${attempt + 1}/${max_retries}): ${error.message}`
        );

        // Don't retry on programming errors
        if (error.name === 'TypeError' || error.name === 'ReferenceError') {
          return {
            success: false,
            error: last_error,
            user_id,
            attempts: attempt + 1
          };
        }
      }
    }

    // All retries exhausted
    console.error(
      `Failed to restore state for ${user_id} after ${max_retries} attempts. ` +
      `Last error: ${last_error}`
    );
    return {
      success: false,
      error: `All ${max_retries} restore attempts failed. Last error: ${last_error}`,
      user_id,
      buffer_start_time,
      attempts: max_retries
    };
  }

  /**
   * Replay measurements in chronological order.
   *
   * @param user_id - User identifier
   * @param clean_measurements - List of clean measurements
   * @param start_time - Processing start time for timeout check (milliseconds)
   * @returns Result dictionary with success status
   */
  private _replay_measurements_chronologically(
    user_id: string,
    clean_measurements: Array<Record<string, any>>,
    start_time: number
  ): ChronologicalReplayResult {
    try {
      // Ensure all measurements have proper types and sort by timestamp
      const prepared_measurements = clean_measurements.map(m => prepareMeasurementForProcessing(m));
      const sorted_measurements = prepared_measurements.sort((a, b) => {
        const ts_a = a.timestamp instanceof Date ? a.timestamp : new Date(a.timestamp);
        const ts_b = b.timestamp instanceof Date ? b.timestamp : new Date(b.timestamp);
        return ts_a.getTime() - ts_b.getTime();
      });

      let processed_count = 0;
      let last_result: any = null;

      for (const measurement of sorted_measurements) {
        // Check timeout
        if (Date.now() - start_time > this.max_processing_time * 1000) {
          return {
            success: false,
            error: `Processing timeout after ${this.max_processing_time}s`,
            user_id,
            processed_count
          };
        }

        // Process measurement through normal pipeline
        try {
          // Prepare measurement with proper types
          const clean_measurement = prepareMeasurementForProcessing(measurement);

          // Extract measurement data
          const weight = clean_measurement.weight;  // Now guaranteed to be float
          let timestamp = clean_measurement.timestamp;
          const source = clean_measurement.source || 'replay-replay';
          const unit = clean_measurement.unit || 'kg';

          // Ensure timestamp is a Date
          if (typeof timestamp === 'string') {
            timestamp = new Date(timestamp);
          }

          // Process through existing pipeline with SAME config as real-time
          // No relaxed thresholds - temporal filtering already happened
          const result = process_measurement(
            user_id,
            weight,
            timestamp,
            source,
            this.config,  // Use same config as real-time
            this.db,
            unit
          );
          last_result = result;

          if (!result.accepted) {
            // Continue processing - rejection is normal and expected
          }

          processed_count++;

        } catch (e) {
          const error = e as Error;
          console.error(`Failed to process measurement during replay: ${error.message}`);
          console.error(`Stack trace: ${error.stack}`);
          return {
            success: false,
            error: `Measurement processing failed: ${error.message}`,
            user_id,
            processed_count,
            failed_measurement: measurement
          };
        }
      }

      return {
        success: true,
        user_id,
        processed_count,
        last_result
      };

    } catch (e) {
      const error = e as Error;
      console.error(`Chronological replay failed for user ${user_id}: ${error.message}`);
      return {
        success: false,
        error: `Chronological replay exception: ${error.message}`,
        user_id
      };
    }
  }

  /**
   * Manually rollback user state to backup.
   *
   * @param user_id - User identifier
   * @returns True if rollback successful
   */
  rollback_user_state(user_id: string): boolean {
    if (this.require_rollback_confirmation) {
      console.warn(`Rollback requires confirmation for user ${user_id}`);
      return false;
    }

    return this._restore_state_from_backup(user_id);
  }

  /**
   * Check if a backup exists for the user.
   *
   * @param user_id - User identifier
   * @returns True if backup exists
   */
  has_backup(user_id: string): boolean {
    return this._backup_states.has(user_id);
  }

  /**
   * Get statistics about replay operations.
   *
   * @returns Statistics dictionary
   */
  get_replay_stats(): ReplayStats {
    return {
      active_backups: this._backup_states.size,
      max_processing_time: this.max_processing_time,
      preserve_immediate_results: this.preserve_immediate_results,
      require_rollback_confirmation: this.require_rollback_confirmation
    };
  }

  /**
   * Clean up old backups to prevent memory leaks.
   *
   * @returns Number of backups cleaned up
   */
  cleanup_old_backups(): number {
    const cleaned_count = this._backup_states.size;
    this._backup_states.clear();
    return cleaned_count;
  }
}
