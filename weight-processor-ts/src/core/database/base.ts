/**
 * Base interface for state storage
 *
 * Ported from Python: weight_values/src/core/database/base.py
 */

import type { ProcessorState } from '../../models';

/**
 * Abstract interface for state storage.
 * Implementations can be in-memory, file-based, or database-backed.
 */
export interface StateStore {
  /**
   * Retrieve state for a user
   */
  get_state(user_id: string): ProcessorState | null;

  /**
   * Save state for a user
   */
  save_state(user_id: string, state: ProcessorState): void;

  /**
   * Delete state for a user
   * Returns true if deleted, false if user not found
   */
  delete_state(user_id: string): boolean;

  /**
   * Create an empty initial state with required fields
   */
  create_initial_state(): ProcessorState;

  /**
   * Save a snapshot of current state (for replay functionality)
   * Returns true if snapshot saved successfully
   */
  save_state_snapshot(user_id: string, timestamp: Date): boolean;

  /**
   * Get the most recent snapshot for a user
   * Used by periodic snapshot logic to determine when to create next snapshot
   */
  get_latest_snapshot?(user_id: string): ProcessorState | null;

  /**
   * Get the nearest snapshot before the given timestamp
   */
  get_snapshot(user_id: string, timestamp: Date): ProcessorState | null;

  /**
   * Restore state from the latest snapshot
   * Returns true if restored, false if no snapshot found
   */
  restore_state_snapshot?(user_id: string): boolean;

  /**
   * Get measurements for a user within a time window
   * Used by replay trigger logic to find measurements in the 72-hour window
   */
  get_measurements_in_window(
    user_id: string,
    start_time: Date,
    end_time: Date
  ): Array<{
    timestamp: Date;
    weight: number;
    source: string;
    unit: string;
    metadata: Record<string, any>;
  }>;

  /**
   * Check if a snapshot exists before buffer_start_time and restore it atomically
   */
  check_and_restore_snapshot(
    user_id: string,
    buffer_start_time: Date
  ): {
    success: boolean;
    snapshot?: ProcessorState;
    snapshot_timestamp?: Date;
    user_id?: string;
    error?: string;
  };

  /**
   * Export all states to CSV (simplified version)
   * Returns number of users exported
   */
  export_to_csv?(filepath: string): number;
}
