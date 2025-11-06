/**
 * Abstract base interface for state storage.
 */

export interface KalmanState {
  // Core Kalman filter state
  kalman_params: any;
  last_state: number[][] | undefined; // (2, 2) array for weight and velocity state
  last_covariance: number[][][] | undefined; // (2, 2, 2) array for covariance matrices
  last_timestamp: Date | null;
  last_accepted_timestamp: Date | null;
  last_source: string | null;
  last_raw_weight: number | null;

  // History and events
  measurement_history: any[];
  reset_events: any[];

  // Counters and metadata
  measurements_since_reset: number;
  adaptation_state: Record<string, any>;
  version: number;

  // Additional processing fields (optional, for algorithm compatibility)
  last_accepted_weight?: number;
  reset_type?: string;
  reset_parameters?: any;
  reset_timestamp?: Date;
}

export interface SnapshotResult {
  snapshot_found: boolean;
  snapshot_restored: boolean;
  snapshot_timestamp: Date | null;
}

/**
 * Abstract interface for state storage backends.
 */
export abstract class StateStore {
  /**
   * Retrieve state for a user.
   */
  abstract getState(userId: string): Promise<KalmanState | null>;

  /**
   * Save state for a user.
   */
  abstract saveState(userId: string, state: KalmanState): Promise<boolean>;

  /**
   * Delete state for a user.
   */
  abstract deleteState(userId: string): Promise<boolean>;

  /**
   * Create an empty initial state.
   */
  abstract createInitialState(): KalmanState;

  /**
   * Save a snapshot of current state.
   */
  abstract saveStateSnapshot(userId: string, timestamp: Date): Promise<boolean>;

  /**
   * Restore state from snapshot.
   */
  abstract restoreStateSnapshot(userId: string): Promise<boolean>;

  /**
   * Get the nearest snapshot before the given timestamp.
   */
  abstract getSnapshot(userId: string, timestamp: Date): Promise<KalmanState | null>;

  /**
   * Get the most recent snapshot for a user.
   */
  abstract getLatestSnapshot(userId: string): Promise<KalmanState | null>;

  /**
   * Check if a snapshot exists and restore it atomically.
   */
  abstract checkAndRestoreSnapshot(
    userId: string,
    bufferStartTime: Date
  ): Promise<SnapshotResult>;

  /**
   * Export all states to CSV.
   */
  abstract exportToCsv(filepath: string): Promise<number>;
}
