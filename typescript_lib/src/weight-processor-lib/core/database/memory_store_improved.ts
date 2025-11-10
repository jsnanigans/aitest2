/**
 * Improved in-memory storage using immutability patterns.
 *
 * Browser-optimized design:
 * - States are frozen (immutable) after creation
 * - Snapshots are just references to frozen states (zero-cost)
 * - No deep copying needed (structural sharing)
 * - Mutation attempts throw errors (fail-fast debugging)
 */

import type { KalmanState } from '../types';
import { StateStore, SnapshotResult } from './base';
import { Matrix } from 'ml-matrix';

interface Snapshot {
  timestamp: Date;
  state: Readonly<KalmanState>; // Reference to frozen state
}

/**
 * Deep freeze helper - recursively freezes objects but skips Matrix instances.
 * Matrix objects need to remain mutable for ml-matrix library operations.
 */
function deepFreeze<T>(obj: T): Readonly<T> {
  // Freeze the object itself
  Object.freeze(obj);

  // Recursively freeze nested objects, but skip Matrix instances
  Object.getOwnPropertyNames(obj).forEach(prop => {
    const value = (obj as any)[prop];
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
      // Skip freezing Matrix objects (they need to be mutable for ml-matrix)
      if (value instanceof Matrix) {
        return;
      }
      // Skip freezing arrays of Matrix objects
      if (Array.isArray(value) && value.length > 0 && value[0] instanceof Matrix) {
        return;
      }
      deepFreeze(value);
    }
  });

  return obj as Readonly<T>;
}

/**
 * Clone Matrix arrays for new state.
 * Creates new Matrix instances so state updates don't affect snapshots.
 */
function cloneMatrixArrays(state: KalmanState): KalmanState {
  return {
    ...state,
    last_state: state.last_state?.map(m => m.clone()),
    last_covariance: state.last_covariance?.map(m => m.clone()),
    measurement_history: [...state.measurement_history],
    reset_events: [...state.reset_events],
  };
}

/**
 * Improved in-memory state store using immutability patterns.
 *
 * Design:
 * - Current states are frozen after saving (immutable)
 * - Snapshots are just references to frozen states (zero-cost!)
 * - No deep copying needed (faster than Python version)
 * - Mutation attempts throw errors in strict mode (safer)
 */
export class InMemoryStoreImproved extends StateStore {
  private states: Map<string, Readonly<KalmanState>>;
  private snapshots: Map<string, Snapshot[]>;

  constructor() {
    super();
    this.states = new Map();
    this.snapshots = new Map();
    console.log('Initialized InMemoryStoreImproved (immutable frozen states)');
  }

  /**
   * Retrieve state for a user.
   * Returns frozen (immutable) state - caller must clone before modifying!
   */
  async getState(userId: string): Promise<KalmanState | null> {
    const state = this.states.get(userId);
    if (!state) return null;

    // Return a clone with new Matrix instances so caller can modify
    return cloneMatrixArrays(state as KalmanState);
  }

  /**
   * Save state for a user.
   * State is cloned and frozen to prevent external modifications.
   */
  async saveState(userId: string, state: KalmanState): Promise<boolean> {
    // Clone Matrix arrays to avoid shared references
    const cloned = cloneMatrixArrays(state);

    // Freeze to make immutable (prevents accidental mutations)
    const frozen = deepFreeze(cloned);

    this.states.set(userId, frozen);
    return true;
  }

  /**
   * Delete state for a user.
   */
  async deleteState(userId: string): Promise<boolean> {
    const existed = this.states.has(userId);
    if (existed) {
      this.states.delete(userId);
      this.snapshots.delete(userId);
    }
    return existed;
  }

  /**
   * Create an empty initial state.
   */
  createInitialState(): KalmanState {
    return {
      kalman_params: null,
      last_state: undefined,
      last_covariance: undefined,
      last_timestamp: null,
      last_accepted_timestamp: null,
      last_source: null,
      last_raw_weight: null,
      measurement_history: [],
      reset_events: [],
      measurements_since_reset: 0,
      adaptation_state: null,
      version: 1,
    };
  }

  /**
   * Save a snapshot of the current state.
   * Zero-cost operation: just stores a reference to the frozen state!
   */
  async saveStateSnapshot(userId: string, timestamp: Date): Promise<boolean> {
    const currentState = this.states.get(userId);
    if (!currentState) {
      console.warn(`Cannot save snapshot for ${userId}: no state found`);
      return false;
    }

    // Get or create snapshot array
    let userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots) {
      userSnapshots = [];
      this.snapshots.set(userId, userSnapshots);
    }

    // Just store a reference! State is already frozen, so it's safe.
    // This is MUCH faster than deep copying.
    userSnapshots.push({
      timestamp,
      state: currentState,
    });

    // Sort snapshots by timestamp
    userSnapshots.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    console.debug(
      `Saved snapshot for ${userId} at ${timestamp.toISOString()} ` +
        `(total snapshots: ${userSnapshots.length}) [zero-cost reference]`
    );
    return true;
  }

  /**
   * Restore state from the latest snapshot.
   */
  async restoreLatestSnapshot(userId: string): Promise<boolean> {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      console.warn(`No snapshots found for ${userId}`);
      return false;
    }

    // Get the latest snapshot (just a reference to frozen state)
    const latestSnapshot = userSnapshots[userSnapshots.length - 1];
    this.states.set(userId, latestSnapshot.state);

    console.debug(
      `Restored latest snapshot for ${userId} ` +
        `(from ${userSnapshots.length} available) [zero-cost reference]`
    );
    return true;
  }

  /**
   * Get the nearest snapshot before the given timestamp.
   */
  async getSnapshot(userId: string, timestamp: Date): Promise<KalmanState | null> {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      return null;
    }

    // Find the most recent snapshot before the given timestamp
    const matchingSnapshot = userSnapshots
      .filter((s) => s.timestamp <= timestamp)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    if (!matchingSnapshot) return null;

    // Return a clone so caller can modify
    return cloneMatrixArrays(matchingSnapshot.state as KalmanState);
  }

  /**
   * Get the most recent snapshot for a user.
   */
  async getLatestSnapshot(userId: string): Promise<KalmanState | null> {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      return null;
    }

    const latestSnapshot = userSnapshots[userSnapshots.length - 1];

    // Return a clone so caller can modify
    return cloneMatrixArrays(latestSnapshot.state as KalmanState);
  }

  /**
   * Check if a snapshot exists and restore it atomically.
   */
  async checkAndRestoreSnapshot(
    userId: string,
    bufferStartTime: Date
  ): Promise<SnapshotResult> {
    const result: SnapshotResult = {
      snapshot_found: false,
      snapshot_restored: false,
      snapshot_timestamp: null,
    };

    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      return result;
    }

    // Find snapshot at or before buffer start time
    const matchingSnapshot = userSnapshots
      .filter((s) => s.timestamp <= bufferStartTime)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    if (!matchingSnapshot) {
      return result;
    }

    result.snapshot_found = true;
    result.snapshot_timestamp = matchingSnapshot.timestamp;

    // Restore the snapshot (just update reference to frozen state)
    this.states.set(userId, matchingSnapshot.state);
    result.snapshot_restored = true;

    console.info(
      `Restored snapshot for ${userId} from ${matchingSnapshot.timestamp.toISOString()} ` +
        `for buffer starting at ${bufferStartTime.toISOString()} [zero-cost reference]`
    );

    return result;
  }

  /**
   * Export all states to CSV.
   */
  async exportToCsv(filepath: string): Promise<number> {
    if (this.states.size === 0) {
      console.warn('No states to export');
      return 0;
    }

    const rows: string[] = [
      'user_id,last_timestamp,last_raw_weight,measurements_since_reset,version',
    ];

    for (const [userId, state] of this.states.entries()) {
      const row = [
        userId,
        state.last_timestamp?.toISOString() ?? '',
        state.last_raw_weight ?? '',
        state.measurements_since_reset,
        state.version,
      ].join(',');
      rows.push(row);
    }

    await Bun.write(filepath, rows.join('\n'));
    console.info(`Exported ${this.states.size} states to ${filepath}`);
    return this.states.size;
  }

  // Helper methods

  clearAll(): void {
    this.states.clear();
    this.snapshots.clear();
    console.info('Cleared all states and snapshots');
  }

  getUserCount(): number {
    return this.states.size;
  }

  getUserIds(): string[] {
    return Array.from(this.states.keys());
  }

  getSnapshotCount(userId: string): number {
    const userSnapshots = this.snapshots.get(userId);
    return userSnapshots ? userSnapshots.length : 0;
  }

  clearSnapshots(userId: string): void {
    this.snapshots.delete(userId);
    console.debug(`Cleared snapshots for ${userId}`);
  }

  clearAllSnapshots(): void {
    this.snapshots.clear();
    console.debug('Cleared all snapshots');
  }
}
