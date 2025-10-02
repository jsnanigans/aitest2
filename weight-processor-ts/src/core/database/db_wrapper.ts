/**
 * State storage utilities
 * Helper functions for state management (no actual database calls in this implementation)
 *
 * Ported from Python: weight_values/src/core/database/db_wrapper.py
 */

import type { ProcessorState } from '../../models';
import type { StateStore } from './base';

/**
 * Get user state or create initial state if not found
 */
export function get_or_create_state(db: StateStore, user_id: string): ProcessorState {
  let state = db.get_state(user_id);
  if (state === null) {
    state = db.create_initial_state();
    db.save_state(user_id, state);
  }
  return state;
}

/**
 * Check if user has any state
 */
export function has_state(db: StateStore, user_id: string): boolean {
  return db.get_state(user_id) !== null;
}

/**
 * Get the number of users with state
 */
export function get_user_count(db: StateStore): number {
  // This is implementation-specific
  // For ProcessorStateDB, we'd need to expose a method
  // For now, return -1 to indicate not implemented
  return -1;
}

/**
 * Clear all state (useful for testing)
 */
export function clear_all_state(db: StateStore): void {
  // This is implementation-specific
  // Not all StateStore implementations may support this
  console.warn('clear_all_state not implemented for generic StateStore');
}
