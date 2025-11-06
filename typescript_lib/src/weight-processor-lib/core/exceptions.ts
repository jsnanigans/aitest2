/**
 * Custom exceptions for the weight processing system.
 *
 * These exceptions provide clear error signaling for data corruption
 * and validation failures in the Kalman state management system.
 */

/**
 * Raised when data corruption is detected in stored state.
 *
 * This indicates that the stored Kalman state has been corrupted
 * beyond automatic recovery and requires manual intervention.
 */
export class DataCorruptionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DataCorruptionError';
    Object.setPrototypeOf(this, DataCorruptionError.prototype);
  }
}

/**
 * Raised when state validation fails.
 *
 * This indicates that the Kalman state does not meet expected
 * validation criteria (shape, values, completeness).
 */
export class StateValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StateValidationError';
    Object.setPrototypeOf(this, StateValidationError.prototype);
  }
}

/**
 * Raised when automatic recovery attempts fail.
 *
 * This indicates that the system attempted to recover from
 * corrupted state but was unable to produce a valid result.
 */
export class RecoveryFailedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RecoveryFailedError';
    Object.setPrototypeOf(this, RecoveryFailedError.prototype);
  }
}
