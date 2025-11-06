/**
 * Simple in-memory state database for weight processor.
 * Stores Kalman filter states without persistence.
 *
 * Ported from Python: weight_values/src/core/database/database.py
 */

import type { ProcessorState } from '../../models';
import type { StateStore } from './base';

/**
 * Deep copy helper function (only used for snapshots)
 */
function deepCopy<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

interface Snapshot {
  timestamp: Date;
  snapshotTime: string;
  state: ProcessorState;
}

/**
 * In-memory state storage for weight processor.
 * Stores and retrieves Kalman state for each user.
 */
export class ProcessorStateDB implements StateStore {
  private states: Map<string, ProcessorState>;
  private _snapshots: Map<string, Snapshot[]>;

  /**
   * Initialize in-memory state database
   */
  constructor(storage_path?: string) {
    this.states = new Map();
    this._snapshots = new Map();
  }

  /**
   * Retrieve state for a user.
   * Returns direct reference for in-memory processing.
   */
  get_state(user_id: string): ProcessorState | null {
    const state = this.states.get(user_id);
    return state || null;
  }

  /**
   * Save state for a user.
   * Stores direct reference for in-memory processing.
   */
  save_state(user_id: string, state: ProcessorState): void {
    this.states.set(user_id, state);
  }

  /**
   * Delete state for a user.
   * Returns true if deleted, false if user not found.
   */
  delete_state(user_id: string): boolean {
    const hadState = this.states.has(user_id);
    if (hadState) {
      this.states.delete(user_id);
      this._snapshots.delete(user_id);
      return true;
    }
    return false;
  }

  /**
   * Create an empty initial state with required fields.
   */
  create_initial_state(): ProcessorState {
    return {
      userId: '', // Will be set by caller
      kalmanParams: null,
      lastState: null,
      lastCovariance: null,
      lastTimestamp: null,
      lastAcceptedTimestamp: null,
      lastSource: null,
      lastRawWeight: null,
      measurementHistory: [],
      resetEvents: [],
      measurementsSinceReset: 0
    };
  }

  /**
   * Save a snapshot of current state (for replay functionality).
   * Returns true if snapshot saved successfully.
   */
  save_state_snapshot(user_id: string, timestamp: Date): boolean {
    const state = this.states.get(user_id);
    if (state) {
      if (!this._snapshots.has(user_id)) {
        this._snapshots.set(user_id, []);
      }

      const snapshot: Snapshot = {
        timestamp: timestamp,
        snapshotTime: timestamp.toISOString(),
        state: deepCopy(state)
      };

      const snapshots = this._snapshots.get(user_id)!;
      snapshots.push(snapshot);

      // Keep only last 10 snapshots (10 days with 24-hour intervals)
      snapshots.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
      if (snapshots.length > 10) {
        this._snapshots.set(user_id, snapshots.slice(-10));
      }

      return true;
    }
    return false;
  }

  /**
   * Get the most recent snapshot for a user.
   * Used by periodic snapshot logic to determine when to create next snapshot.
   */
  get_latest_snapshot(user_id: string): ProcessorState | null {
    const snapshots = this._snapshots.get(user_id);
    if (!snapshots || snapshots.length === 0) {
      return null;
    }

    // Get the most recent snapshot (list is kept sorted)
    const latest = snapshots[snapshots.length - 1];
    return deepCopy(latest.state);
  }

  /**
   * Get the nearest snapshot before the given timestamp.
   */
  get_snapshot(user_id: string, timestamp: Date): ProcessorState | null {
    const snapshots = this._snapshots.get(user_id);
    if (!snapshots || snapshots.length === 0) {
      return null;
    }

    // Find the most recent snapshot before the timestamp
    const suitable_snapshots = snapshots.filter((s) => s.timestamp < timestamp);

    if (suitable_snapshots.length === 0) {
      return null;
    }

    // Return the most recent one
    const latest = suitable_snapshots.reduce((prev, curr) =>
      curr.timestamp > prev.timestamp ? curr : prev
    );

    return deepCopy(latest.state);
  }

  /**
   * Restore state from the latest snapshot.
   * Returns true if restored, false if no snapshot found.
   */
  restore_state_snapshot(user_id: string): boolean {
    const latest_snapshot_state = this.get_latest_snapshot(user_id);
    if (latest_snapshot_state) {
      this.states.set(user_id, deepCopy(latest_snapshot_state));
      return true;
    }
    return false;
  }

  /**
   * Get measurements for a user within a time window.
   * Used by replay trigger logic to find measurements in the 72-hour window.
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
  }> {
    const state = this.get_state(user_id);
    if (!state || !state.measurementHistory) {
      return [];
    }

    const measurements: Array<{
      timestamp: Date;
      weight: number;
      source: string;
      unit: string;
      metadata: Record<string, any>;
    }> = [];

    for (const m of state.measurementHistory) {
      let timestamp = m.timestamp;
      if (timestamp === undefined) {
        continue;
      }

      // Ensure timestamp is Date
      if (typeof timestamp === 'string') {
        timestamp = new Date(timestamp.replace('Z', '+00:00'));
      }

      // Check if in window
      if (timestamp >= start_time && timestamp < end_time) {
        measurements.push({
          timestamp: timestamp,
          weight: m.weight!,
          source: m.source || 'unknown',
          unit: m.unit || 'kg',
          metadata: m.metadata || {}
        });
      }
    }

    return measurements;
  }

  /**
   * Check if a snapshot exists before buffer_start_time and restore it atomically.
   */
  check_and_restore_snapshot(
    user_id: string,
    buffer_start_time: Date
  ): {
    success: boolean;
    snapshot?: ProcessorState;
    snapshot_timestamp?: Date | string | null;
    user_id?: string;
    error?: string;
  } {
    const snapshot_state = this.get_snapshot(user_id, buffer_start_time);
    if (snapshot_state) {
      // Restore the state
      this.states.set(user_id, deepCopy(snapshot_state));
      return {
        success: true,
        snapshot: snapshot_state,
        snapshot_timestamp: snapshot_state.lastTimestamp || buffer_start_time,
        user_id: user_id
      };
    } else {
      return {
        success: false,
        error: `No snapshot found for user ${user_id} before ${buffer_start_time}`,
        user_id: user_id
      };
    }
  }

  /**
   * Export all states to CSV (simplified version).
   * Returns number of users exported.
   */
  export_to_csv(filepath: string): number {
    // This would require Bun's file system API or similar
    // For now, just return the count
    console.warn('export_to_csv not yet implemented for TypeScript');
    return this.states.size;
  }
}

// Global instance - kept for backward compatibility
let _db_instance: ProcessorStateDB | null = null;

/**
 * Get the global state database instance.
 * This is kept for backward compatibility - new code should use
 * the factory or create instances directly.
 */
export function get_state_db(): ProcessorStateDB {
  if (_db_instance === null) {
    _db_instance = new ProcessorStateDB();
  }
  return _db_instance;
}

/**
 * Reset the global database instance (useful for testing).
 */
export function reset_db(): void {
  _db_instance = null;
}
