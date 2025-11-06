/**
 * Database module exports.
 * Storage abstraction layer for state persistence.
 */

export { StateStore } from './base.js';
export type { KalmanState, SnapshotResult } from './base.js';

export { InMemoryStore } from './memory_store.js';
