/**
 * Validates state before persistence and makes persistence decisions.
 */

import type { ProcessorState } from '../../models';

export class PersistenceValidator {
  /**
   * Validate state structure before persistence.
   */
  static validateState(
    state: ProcessorState,
    userId: string,
    reason: string
  ): [boolean, string | null] {
    // Basic validation
    if (!state || typeof state !== 'object') {
      return [false, 'State is null or not an object'];
    }

    // All validations passed
    return [true, null];
  }

  /**
   * Decide whether state should be persisted.
   */
  static shouldPersist(
    state: ProcessorState,
    previousState: ProcessorState | null,
    userId: string,
    reason: string
  ): [boolean, string] {
    // Always persist after successful processing
    if (reason === 'successful_processing') {
      return [true, 'Successful processing'];
    }

    // Persist if state has changed significantly
    if (!previousState) {
      return [true, 'Initial state'];
    }

    // Check if Kalman state has been updated
    if (state.lastTimestamp !== previousState.lastTimestamp) {
      return [true, 'State updated'];
    }

    // No significant changes
    return [false, 'No significant state changes'];
  }

  /**
   * Create audit log entry (simplified - just console log for now).
   */
  static createAuditLog(
    userId: string,
    action: string,
    state: ProcessorState | null,
    success: boolean,
    reason: string,
    error: string | null
  ): void {
    const logEntry = {
      timestamp: new Date().toISOString(),
      userId,
      action,
      success,
      reason,
      error,
    };

    if (!success) {
      console.warn('Persistence audit log:', logEntry);
    } else {
    }
  }
}
