/**
 * In-memory implementation of StateStore for testing and development.
 */

import { StateStore, KalmanState, SnapshotResult } from './base.js';

interface Snapshot {
  timestamp: Date;
  state: KalmanState;
}

/**
 * In-memory state storage for testing and development.
 *
 * This implementation stores all data in memory using Maps.
 * Data is NOT persisted and will be lost when the process ends.
 *
 * Features:
 * - Thread-safe operations (in single-threaded JS, no locks needed)
 * - Fast for testing and development
 * - No external dependencies
 * - Supports snapshots for replay functionality
 *
 * Usage:
 *     const db = new InMemoryStore();
 *     const state = db.createInitialState();
 *     await db.saveState("user123", state);
 *     const retrieved = await db.getState("user123");
 */
export class InMemoryStore extends StateStore {
  private states: Map<string, KalmanState>;
  private snapshots: Map<string, Snapshot[]>;

  constructor() {
    super();
    this.states = new Map();
    this.snapshots = new Map();
    console.log('Initialized InMemoryStore');
  }

  /**
   * Retrieve state for a user.
   */
  async getState(userId: string): Promise<KalmanState | null> {
    const state = this.states.get(userId);
    // Return a deep copy to prevent external modifications
    return state ? this.deepCopy(state) : null;
  }

  /**
   * Save state for a user.
   */
  async saveState(userId: string, state: KalmanState): Promise<boolean> {
    // Store a deep copy to prevent external modifications
    this.states.set(userId, this.deepCopy(state));
    return true;
  }

  /**
   * Delete state for a user.
   */
  async deleteState(userId: string): Promise<boolean> {
    const existed = this.states.has(userId);
    if (existed) {
      this.states.delete(userId);
      // Also delete snapshots
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
      adaptation_state: {},
      version: 0,
    };
  }

  /**
   * Save a snapshot of current state.
   */
  async saveStateSnapshot(userId: string, timestamp: Date): Promise<boolean> {
    const currentState = this.states.get(userId);
    if (!currentState) {
      console.warn(`Cannot save snapshot for ${userId}: no current state`);
      return false;
    }

    // Initialize snapshots array if needed
    if (!this.snapshots.has(userId)) {
      this.snapshots.set(userId, []);
    }

    // Add snapshot
    const userSnapshots = this.snapshots.get(userId)!;
    userSnapshots.push({
      timestamp,
      state: this.deepCopy(currentState),
    });

    // Sort snapshots by timestamp
    userSnapshots.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    console.debug(
      `Saved snapshot for ${userId} at ${timestamp.toISOString()} ` +
        `(total snapshots: ${userSnapshots.length})`
    );
    return true;
  }

  /**
   * Restore state from the latest snapshot.
   */
  async restoreStateSnapshot(userId: string): Promise<boolean> {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      console.warn(`Cannot restore snapshot for ${userId}: no snapshots exist`);
      return false;
    }

    // Get the latest snapshot
    const latestSnapshot = userSnapshots[userSnapshots.length - 1];

    // Restore state (deep copy to prevent modifications)
    this.states.set(userId, this.deepCopy(latestSnapshot.state));

    console.debug(
      `Restored latest snapshot for ${userId} ` +
        `(from ${userSnapshots.length} available)`
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

    // Find the latest snapshot before or at the timestamp
    let matchingSnapshot: KalmanState | null = null;
    for (const snapshot of userSnapshots) {
      if (snapshot.timestamp <= timestamp) {
        matchingSnapshot = snapshot.state;
      } else {
        break; // List is sorted, so we can stop here
      }
    }

    // Return deep copy to prevent modifications
    return matchingSnapshot ? this.deepCopy(matchingSnapshot) : null;
  }

  /**
   * Get the most recent snapshot for a user.
   */
  async getLatestSnapshot(userId: string): Promise<KalmanState | null> {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      return null;
    }

    // Get the latest snapshot
    const latestSnapshot = userSnapshots[userSnapshots.length - 1];

    // Return deep copy to prevent modifications
    return this.deepCopy(latestSnapshot.state);
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

    // Check if we have snapshots for this user
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      return result;
    }

    // Find the nearest snapshot before buffer_start_time
    let matchingSnapshot: Snapshot | null = null;

    for (const snapshot of userSnapshots) {
      if (snapshot.timestamp <= bufferStartTime) {
        matchingSnapshot = snapshot;
      } else {
        break; // List is sorted
      }
    }

    if (!matchingSnapshot) {
      return result;
    }

    // Found a snapshot
    result.snapshot_found = true;
    result.snapshot_timestamp = matchingSnapshot.timestamp;

    // Restore it
    this.states.set(userId, this.deepCopy(matchingSnapshot.state));
    result.snapshot_restored = true;

    console.info(
      `Restored snapshot for ${userId} from ${matchingSnapshot.timestamp.toISOString()} ` +
        `for buffer starting at ${bufferStartTime.toISOString()}`
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

    // Prepare CSV data
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

    // Write to file (using Bun's file API)
    await Bun.write(filepath, rows.join('\n'));

    console.info(`Exported ${this.states.size} states to ${filepath}`);
    return this.states.size;
  }

  // Additional helper methods

  /**
   * Clear all states and snapshots.
   *
   * Useful for testing to reset to a clean state.
   */
  clearAll(): void {
    this.states.clear();
    this.snapshots.clear();
    console.info('Cleared all states and snapshots');
  }

  /**
   * Get list of all user IDs with stored states.
   */
  listUsers(): string[] {
    return Array.from(this.states.keys());
  }

  /**
   * Get number of snapshots stored for a user.
   */
  getSnapshotCount(userId: string): number {
    const userSnapshots = this.snapshots.get(userId);
    return userSnapshots ? userSnapshots.length : 0;
  }

  /**
   * Clear all snapshots for a user.
   */
  clearSnapshots(userId: string): number {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots) {
      return 0;
    }
    const count = userSnapshots.length;
    this.snapshots.delete(userId);
    console.debug(`Cleared ${count} snapshots for ${userId}`);
    return count;
  }

  toString(): string {
    return `InMemoryStore(states=${this.states.size}, users_with_snapshots=${this.snapshots.size})`;
  }

  /**
   * Deep copy helper to prevent external modifications.
   */
  private deepCopy<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
  }
}
