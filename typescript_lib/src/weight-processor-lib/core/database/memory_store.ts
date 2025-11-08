/**
 * Simple in-memory storage for Kalman filter states.
 *
 * Stores direct references - NO deepCopy, NO serialization/deserialization.
 * Matrix objects stay as Matrix objects throughout.
 */

import type { KalmanState } from '../types';
import { StateStore } from './base';

interface Snapshot {
  timestamp: Date;
  state: KalmanState;
}

export interface SnapshotResult {
  snapshot_found: boolean;
  snapshot_restored: boolean;
  snapshot_timestamp: Date | null;
}

/**
 * Simple in-memory state store using direct references.
 * No deepCopy, no serialization - just Map storage.
 */
export class InMemoryStore extends StateStore {
  private states: Map<string, KalmanState>;
  private snapshots: Map<string, Snapshot[]>;

  constructor() {
    super();
    this.states = new Map();
    this.snapshots = new Map();
    console.log('Initialized InMemoryStore (direct references)');
  }

  /**
   * Retrieve state for a user.
   * Returns direct reference - caller should not mutate!
   */
  async getState(userId: string): Promise<KalmanState | null> {
    return this.states.get(userId) ?? null;
  }

  /**
   * Save state for a user.
   * Stores direct reference - no copying.
   */
  async saveState(userId: string, state: KalmanState): Promise<boolean> {
    this.states.set(userId, state);
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
      adaptation_state: null,
      version: 1,
    };
  }

  /**
   * Save a snapshot of the current state at a specific timestamp.
   * Stores direct reference to state - no copying.
   */
  async saveStateSnapshot(userId: string, timestamp: Date): Promise<boolean> {
    const currentState = this.states.get(userId);
    if (!currentState) {
      console.warn(
        `Cannot save snapshot for ${userId}: no state found`
      );
      return false;
    }

    // Get or create snapshot array for this user
    let userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots) {
      userSnapshots = [];
      this.snapshots.set(userId, userSnapshots);
    }

    // Add snapshot (direct reference to state)
    userSnapshots.push({
      timestamp,
      state: currentState,
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
  async restoreLatestSnapshot(userId: string): Promise<boolean> {
    const userSnapshots = this.snapshots.get(userId);
    if (!userSnapshots || userSnapshots.length === 0) {
      console.warn(`No snapshots found for ${userId}`);
      return false;
    }

    // Get the latest snapshot
    const latestSnapshot = userSnapshots[userSnapshots.length - 1];

    // Restore the state (direct reference)
    this.states.set(userId, latestSnapshot.state);

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

    // Find the most recent snapshot before the given timestamp
    const matchingSnapshot = userSnapshots
      .filter((s) => s.timestamp <= timestamp)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    return matchingSnapshot ? matchingSnapshot.state : null;
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

    // Return direct reference
    return latestSnapshot.state;
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

    // Restore the snapshot state (direct reference)
    this.states.set(userId, matchingSnapshot.state);
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
   * Get the number of users with stored states.
   */
  getUserCount(): number {
    return this.states.size;
  }

  /**
   * Get all user IDs with stored states.
   */
  getUserIds(): string[] {
    return Array.from(this.states.keys());
  }

  /**
   * Get snapshot count for a user.
   */
  getSnapshotCount(userId: string): number {
    const userSnapshots = this.snapshots.get(userId);
    return userSnapshots ? userSnapshots.length : 0;
  }

  /**
   * Clear snapshots for a user.
   */
  clearSnapshots(userId: string): void {
    this.snapshots.delete(userId);
    console.debug(`Cleared snapshots for ${userId}`);
  }

  /**
   * Clear all snapshots for all users.
   */
  clearAllSnapshots(): void {
    this.snapshots.clear();
    console.debug('Cleared all snapshots');
  }
}
