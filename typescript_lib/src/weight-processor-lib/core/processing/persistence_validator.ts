/**
 * Persistence Validator Module
 *
 * Validates state before persistence to prevent invalid or corrupted data from being saved.
 * Provides audit trail and ensures state integrity.
 */

import type { KalmanState } from '../database/base.js';
import { KalmanParams } from './kalman.js';

/**
 * Validation result tuple: [isValid, errorMessage?]
 */
export type ValidationResult = [boolean, string | null];

/**
 * Persistence decision tuple: [shouldPersist, auditMessage]
 */
export type PersistenceDecision = [boolean, string];

/**
 * Audit log entry interface
 */
export interface AuditLogEntry {
  timestamp: string;
  user_id: string;
  action: string;
  success: boolean;
  reason: string;
  error?: string;
  state_summary?: {
    has_kalman_state: boolean;
    has_timestamp: boolean;
    measurements_count: number;
    current_weight?: number;
  };
}

/**
 * Validates state before persistence operations.
 */
export class PersistenceValidator {
  // Required fields that must be present in every state
  private static readonly REQUIRED_FIELDS = new Set([
    'last_state',
    'kalman_params',
    'last_timestamp',
  ]);

  // Fields that should be numeric if present
  private static readonly NUMERIC_FIELDS = new Set([
    'measurements_since_reset',
    'last_raw_weight',
  ]);

  // Maximum reasonable weight in kg
  private static readonly MAX_WEIGHT_KG = 500;
  private static readonly MIN_WEIGHT_KG = 10;

  /**
   * Validate state before persistence.
   *
   * @param state - The state dictionary to validate
   * @param userId - User identifier for logging
   * @param reason - Reason for persistence (for audit trail)
   * @returns Tuple of [isValid, errorMessage]
   */
  static validateState(
    state: KalmanState,
    userId: string,
    reason: string = 'unknown'
  ): ValidationResult {
    if (!state || Object.keys(state).length === 0) {
      return [false, 'State is None or empty'];
    }

    // Check required fields
    const stateKeys = new Set(Object.keys(state));
    const missingFields = [...this.REQUIRED_FIELDS].filter(
      (field) => !stateKeys.has(field)
    );

    if (missingFields.length > 0) {
      return [false, `Missing required fields: ${missingFields.join(', ')}`];
    }

    // Validate Kalman state structure
    const kalmanState = state.last_state;
    if (!this._validateKalmanState(kalmanState)) {
      return [false, 'Invalid Kalman state structure'];
    }

    // Validate Kalman parameters
    const kalmanParams = state.kalman_params;
    if (!this._validateKalmanParams(kalmanParams)) {
      return [false, 'Invalid Kalman parameters'];
    }

    // Validate timestamp
    const lastTimestamp = state.last_timestamp;
    if (!this._validateTimestamp(lastTimestamp)) {
      return [false, 'Invalid last_timestamp'];
    }

    // Validate numeric fields
    for (const field of this.NUMERIC_FIELDS) {
      if (field in state) {
        const value = (state as any)[field];
        if (typeof value !== 'number') {
          return [
            false,
            `Field '${field}' must be numeric, got ${typeof value}`,
          ];
        }

        // Special validation for weight
        if (field === 'last_raw_weight') {
          if (value < this.MIN_WEIGHT_KG || value > this.MAX_WEIGHT_KG) {
            return [
              false,
              `Weight ${value} outside valid range [${this.MIN_WEIGHT_KG}, ${this.MAX_WEIGHT_KG}]`,
            ];
          }
        }
      }
    }

    // Check for consistency between related fields
    if (state.last_accepted_timestamp && state.last_timestamp) {
      const lastTs = state.last_timestamp;
      const acceptedTs = state.last_accepted_timestamp;
      if (acceptedTs instanceof Date && lastTs instanceof Date) {
        if (acceptedTs > lastTs) {
          return [
            false,
            'last_accepted_timestamp cannot be after last_timestamp',
          ];
        }
      }
    }

    // Log successful validation with audit trail
    console.debug(
      `State validation passed for user ${userId}, reason: ${reason}`
    );

    return [true, null];
  }

  /**
   * Validate Kalman state structure.
   */
  private static _validateKalmanState(kalmanState: any): boolean {
    // Allow undefined/null for initial states
    if (kalmanState === undefined || kalmanState === null) {
      return true;
    }

    try {
      // Handle arrays
      if (Array.isArray(kalmanState)) {
        if (kalmanState.length === 0) {
          return true; // Empty array is OK for initial state
        }

        let weight: number;

        // Check if it's a 2D array
        if (Array.isArray(kalmanState[0])) {
          // 2D array [[weight], [trend]] or multiple states
          if (kalmanState.length === 0) {
            return true; // Empty is OK
          }
          // Get the last state
          const lastState = kalmanState[kalmanState.length - 1];
          if (Array.isArray(lastState)) {
            if (lastState.length >= 1) {
              weight = Number(lastState[0]);
            } else {
              weight = Number(lastState);
            }
          } else {
            weight = Number(lastState);
          }
        } else {
          // 1D array [weight, trend]
          if (kalmanState.length < 1) {
            return false;
          }
          weight = Number(kalmanState[0]);
        }

        // Weight should be reasonable
        if (weight < this.MIN_WEIGHT_KG || weight > this.MAX_WEIGHT_KG) {
          return false;
        }
      } else {
        return false;
      }
    } catch (error) {
      return false;
    }

    return true;
  }

  /**
   * Validate Kalman parameters structure.
   */
  private static _validateKalmanParams(kalmanParams: any): boolean {
    if (!kalmanParams || typeof kalmanParams !== 'object') {
      return false;
    }

    // Handle two possible formats for transition covariance
    const hasMatrixFormat = 'transition_covariance' in kalmanParams;
    const hasIndividualFormat =
      'transition_covariance_weight' in kalmanParams &&
      'transition_covariance_trend' in kalmanParams;

    // Must have at least one format
    if (!hasMatrixFormat && !hasIndividualFormat) {
      return false;
    }

    // Validate transition covariance based on format
    if (hasMatrixFormat) {
      // Validate as 2x2 matrix
      const transCov = kalmanParams.transition_covariance;
      if (!Array.isArray(transCov) || transCov.length !== 2) {
        return false;
      }
      for (const row of transCov) {
        if (!Array.isArray(row) || row.length !== 2) {
          return false;
        }
        for (const val of row) {
          try {
            const floatVal = Number(val);
            if (isNaN(floatVal) || floatVal < 0) {
              return false;
            }
          } catch (error) {
            return false;
          }
        }
      }
    }

    if (hasIndividualFormat) {
      // Validate individual fields
      for (const field of [
        'transition_covariance_weight',
        'transition_covariance_trend',
      ]) {
        const val = (kalmanParams as any)[field];
        try {
          const floatVal = Number(val);
          if (isNaN(floatVal) || floatVal < 0) {
            return false;
          }
        } catch (error) {
          return false;
        }
      }
    }

    // Validate observation covariance
    if (!('observation_covariance' in kalmanParams)) {
      return false;
    }

    const obsCov = kalmanParams.observation_covariance;
    if (Array.isArray(obsCov)) {
      if (
        obsCov.length !== 1 ||
        !Array.isArray(obsCov[0]) ||
        obsCov[0].length !== 1
      ) {
        return false;
      }
      try {
        const floatVal = Number(obsCov[0][0]);
        if (isNaN(floatVal) || floatVal < 0) {
          return false;
        }
      } catch (error) {
        return false;
      }
    } else {
      try {
        const floatVal = Number(obsCov);
        if (isNaN(floatVal) || floatVal < 0) {
          return false;
        }
      } catch (error) {
        return false;
      }
    }

    return true;
  }

  /**
   * Validate timestamp field.
   */
  private static _validateTimestamp(timestamp: any): boolean {
    if (timestamp === null || timestamp === undefined) {
      return false;
    }

    // Accept either Date or ISO string
    if (timestamp instanceof Date) {
      return !isNaN(timestamp.getTime());
    }

    if (typeof timestamp === 'string') {
      try {
        const date = new Date(timestamp.replace('Z', '+00:00'));
        return !isNaN(date.getTime());
      } catch (error) {
        return false;
      }
    }

    return false;
  }

  /**
   * Determine if state should be persisted based on changes and validity.
   *
   * @param state - Current state
   * @param previousState - Previous state (if any)
   * @param userId - User identifier
   * @param reason - Reason for potential persistence
   * @returns Tuple of [shouldPersist, auditMessage]
   */
  static shouldPersist(
    state: KalmanState,
    previousState: KalmanState | null | undefined,
    userId: string,
    reason: string = 'processing'
  ): PersistenceDecision {
    // First validate the current state
    const [isValid, error] = this.validateState(state, userId, reason);
    if (!isValid) {
      const auditMsg = `State validation failed for user ${userId}: ${error}`;
      console.warn(auditMsg);
      return [false, auditMsg];
    }

    // If no previous state, we should persist
    if (!previousState) {
      const auditMsg = `Initial state persistence for user ${userId}, reason: ${reason}`;
      console.info(auditMsg);
      return [true, auditMsg];
    }

    // Check if there are meaningful changes
    const hasChanges = this._hasMeaningfulChanges(state, previousState);

    if (hasChanges) {
      const auditMsg = `State has meaningful changes for user ${userId}, reason: ${reason}`;
      console.debug(auditMsg);
      return [true, auditMsg];
    } else {
      const auditMsg = `No meaningful state changes for user ${userId}, skipping persistence`;
      console.debug(auditMsg);
      return [false, auditMsg];
    }
  }

  /**
   * Check if there are meaningful changes between states.
   *
   * Ignores minor floating point differences and focuses on significant changes.
   */
  private static _hasMeaningfulChanges(
    state: KalmanState,
    previousState: KalmanState
  ): boolean {
    // Fields that indicate meaningful changes
    const significantFields = new Set([
      'last_state',
      'last_accepted_timestamp',
      'last_raw_weight',
      'measurements_since_reset',
      'reset_type',
      'last_source',
    ]);

    for (const field of significantFields) {
      const fieldInState = field in state;
      const fieldInPrevState = field in previousState;

      if (!fieldInState && !fieldInPrevState) {
        continue;
      }

      // Field added or removed
      if (fieldInState !== fieldInPrevState) {
        return true;
      }

      const currentVal = (state as any)[field];
      const prevVal = (previousState as any)[field];

      // Special handling for Kalman state (check for significant weight change)
      if (
        field === 'last_state' &&
        currentVal !== null &&
        currentVal !== undefined &&
        prevVal !== null &&
        prevVal !== undefined
      ) {
        try {
          let currentWeight: number;
          let prevWeight: number;

          // Handle arrays
          if (Array.isArray(currentVal)) {
            if (Array.isArray(currentVal[0])) {
              // 2D array
              const lastState = currentVal[currentVal.length - 1];
              currentWeight = Number(
                Array.isArray(lastState) ? lastState[0] : lastState
              );
            } else {
              // 1D array
              currentWeight = Number(currentVal[0]);
            }
          } else {
            currentWeight = Number(currentVal);
          }

          if (Array.isArray(prevVal)) {
            if (Array.isArray(prevVal[0])) {
              // 2D array
              const lastState = prevVal[prevVal.length - 1];
              prevWeight = Number(
                Array.isArray(lastState) ? lastState[0] : lastState
              );
            } else {
              // 1D array
              prevWeight = Number(prevVal[0]);
            }
          } else {
            prevWeight = Number(prevVal);
          }

          // Consider > 0.01 kg change as significant
          if (Math.abs(currentWeight - prevWeight) > 0.01) {
            return true;
          }
        } catch (error) {
          return true; // Structure changed
        }
      }
      // For other fields, check for any change
      else if (!this._deepEqual(currentVal, prevVal)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Deep equality check for values
   */
  private static _deepEqual(a: any, b: any): boolean {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (typeof a !== typeof b) return false;

    // Handle dates
    if (a instanceof Date && b instanceof Date) {
      return a.getTime() === b.getTime();
    }

    // Handle arrays
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (!this._deepEqual(a[i], b[i])) return false;
      }
      return true;
    }

    // Handle objects
    if (typeof a === 'object' && typeof b === 'object') {
      const keysA = Object.keys(a);
      const keysB = Object.keys(b);
      if (keysA.length !== keysB.length) return false;
      for (const key of keysA) {
        if (!this._deepEqual(a[key], b[key])) return false;
      }
      return true;
    }

    return false;
  }

  /**
   * Create an audit log entry for persistence operations.
   *
   * @param userId - User identifier
   * @param action - Action being performed (e.g., "persist", "skip", "validate")
   * @param state - State being processed
   * @param success - Whether the operation succeeded
   * @param reason - Reason for the operation
   * @param error - Error message if operation failed
   * @returns Audit log entry dictionary
   */
  static createAuditLog(
    userId: string,
    action: string,
    state: KalmanState,
    success: boolean,
    reason: string,
    error?: string
  ): AuditLogEntry {
    const auditEntry: AuditLogEntry = {
      timestamp: new Date().toISOString(),
      user_id: userId,
      action,
      success,
      reason,
    };

    if (error) {
      auditEntry.error = error;
    }

    // Add state summary (avoid logging full state for privacy)
    if (state) {
      auditEntry.state_summary = {
        has_kalman_state: 'last_state' in state,
        has_timestamp: 'last_timestamp' in state,
        measurements_count: state.measurements_since_reset || 0,
      };

      // Add weight if present (useful for debugging)
      if (state.last_state) {
        try {
          const lastState = state.last_state;
          let weight: number;

          if (Array.isArray(lastState)) {
            if (Array.isArray(lastState[0])) {
              // 2D array
              const lastRow = lastState[lastState.length - 1];
              weight = Number(Array.isArray(lastRow) ? lastRow[0] : lastRow);
            } else {
              // 1D array
              weight = Number(lastState[0]);
            }
          } else {
            weight = Number(lastState);
          }

          auditEntry.state_summary.current_weight = Math.round(weight * 100) / 100;
        } catch (error) {
          // Ignore extraction errors
        }
      }
    }

    console.info(`Persistence audit: ${JSON.stringify(auditEntry)}`);
    return auditEntry;
  }
}
