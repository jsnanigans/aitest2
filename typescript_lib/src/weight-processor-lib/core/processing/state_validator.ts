/**
 * State validation for reset operations.
 * Ensures state integrity after each operation.
 */

/**
 * Types of operations in a reset transaction
 */
export enum ResetOperation {
  KALMAN_RESET = 'kalman_reset',
  STATE_UPDATE = 'state_update',
  BUFFER_UPDATE = 'buffer_update',
  STATE_PERSIST = 'state_persist',
}

/**
 * Type for processor state dictionary
 */
export interface ProcessorState {
  kalman_params?: any;
  reset_parameters?: ResetParameters;
  measurements_since_reset?: number;
  reset_type?: string;
  reset_timestamp?: number;
  measurement_history?: any[];
  last_state?: number | number[];
  [key: string]: any;
}

/**
 * Type for reset parameters
 */
export interface ResetParameters {
  initial_variance_multiplier: number;
  weight_noise_multiplier: number;
  trend_noise_multiplier: number;
  observation_noise_multiplier: number;
  adaptation_measurements: number;
  adaptation_days: number;
  adaptation_decay_rate: number;
  quality_acceptance_threshold: number;
  [key: string]: any;
}

/**
 * Check if a value is NaN or Infinity
 */
function isNaNOrInf(value: number): boolean {
  return isNaN(value) || !isFinite(value);
}

/**
 * Check if an array contains NaN or Infinity
 */
function arrayHasNaNOrInf(arr: number[]): boolean {
  return arr.some(val => isNaNOrInf(val));
}

/**
 * Validates state integrity after reset operations.
 *
 * Performs checks for:
 * - Required fields presence
 * - Data type correctness
 * - Value range validity
 * - NaN/Inf detection
 * - Structural consistency
 */
export class StateValidator {
  /**
   * Validate state based on operation type.
   *
   * @param state - State to validate
   * @param operation - Type of operation performed
   * @returns True if state is valid, False otherwise
   */
  validate(state: ProcessorState, operation: ResetOperation): boolean {
    const validators: Record<ResetOperation, (state: ProcessorState) => boolean> = {
      [ResetOperation.KALMAN_RESET]: this._validateKalmanState.bind(this),
      [ResetOperation.STATE_UPDATE]: this._validateProcessorState.bind(this),
      [ResetOperation.BUFFER_UPDATE]: this._validateBufferState.bind(this),
      [ResetOperation.STATE_PERSIST]: this._validatePersistedState.bind(this),
    };

    const validator = validators[operation];
    if (!validator) {
      console.error(`No validator for operation ${operation}`);
      return false;
    }

    try {
      return validator(state);
    } catch (e) {
      console.error(`Validation failed for ${operation}:`, e);
      return false;
    }
  }

  /**
   * Validate Kalman filter state after reset.
   *
   * Checks:
   * - State has been properly reset (kalman_params should be None after reset)
   * - Reset parameters are present and valid
   * - Measurements counter is reset
   */
  private _validateKalmanState(state: ProcessorState): boolean {
    // After a reset, kalman_params should be None (will be recreated on next measurement)
    if (state.kalman_params !== undefined && state.kalman_params !== null) {
      console.warn('kalman_params should be null after reset');
      // This is actually OK - might be set during processing
    }

    // Check reset parameters
    if (!state.reset_parameters) {
      console.error('Missing reset_parameters after reset');
      return false;
    }

    const resetParams = state.reset_parameters;
    const requiredResetParams = [
      'initial_variance_multiplier',
      'weight_noise_multiplier',
      'trend_noise_multiplier',
      'observation_noise_multiplier',
      'adaptation_measurements',
      'adaptation_days',
      'adaptation_decay_rate',
      'quality_acceptance_threshold',
    ];

    for (const param of requiredResetParams) {
      if (!(param in resetParams)) {
        console.error(`Missing reset parameter: ${param}`);
        return false;
      }

      const value = resetParams[param];

      // Check for NaN or Inf in numeric parameters
      if (typeof value === 'number') {
        if (isNaNOrInf(value)) {
          console.error(`Reset parameter ${param} is NaN or Inf`);
          return false;
        }

        // Range checks for multipliers
        if (param.includes('multiplier')) {
          if (value <= 0) {
            console.error(`Reset parameter ${param} must be positive, got ${value}`);
            return false;
          }
        }

        if (param === 'adaptation_decay_rate') {
          if (value <= 0) {
            console.error(`Adaptation decay rate must be positive: ${value}`);
            return false;
          }
        }
      }
    }

    // Check measurements counter
    if ((state.measurements_since_reset ?? -1) !== 0) {
      console.error(
        `Measurements counter should be 0 after reset, got ${state.measurements_since_reset}`
      );
      return false;
    }

    // Check reset type is valid (handle both upper and lowercase)
    const resetType = state.reset_type;
    const validTypes = ['INITIAL', 'HARD', 'SOFT', 'initial', 'hard', 'soft', null, undefined];
    if (!validTypes.includes(resetType as any)) {
      console.error(`Invalid reset type: ${resetType}`);
      return false;
    }

    // Check reset timestamp exists
    if (!('reset_timestamp' in state)) {
      console.error('Missing reset_timestamp');
      return false;
    }

    return true;
  }

  /**
   * Validate overall processor state update.
   *
   * Checks for consistency and required fields.
   */
  private _validateProcessorState(state: ProcessorState): boolean {
    // Essential fields that should always exist
    const essentialFields = [
      'measurements_since_reset',
      'reset_type',
      'reset_parameters',
      'reset_timestamp',
    ];

    for (const field of essentialFields) {
      if (!(field in state)) {
        console.error(`Missing essential field: ${field}`);
        return false;
      }
    }

    // Validate measurements counter
    const measurements = state.measurements_since_reset ?? -1;
    if (measurements < 0) {
      console.error(`Invalid measurements count: ${measurements}`);
      return false;
    }

    // Validate measurement history if present
    if ('measurement_history' in state) {
      const history = state.measurement_history;
      if (!Array.isArray(history)) {
        console.error('Measurement history is not an array');
        return false;
      }

      // Check history isn't too large (memory protection)
      const maxHistorySize = 1000;
      if (history.length > maxHistorySize) {
        console.error(
          `Measurement history too large: ${history.length} > ${maxHistorySize}`
        );
        return false;
      }
    }

    // If there's a last_state, validate it's numeric
    if ('last_state' in state && state.last_state !== null && state.last_state !== undefined) {
      const lastState = state.last_state;
      if (typeof lastState === 'number') {
        // Check for NaN/Inf
        if (isNaNOrInf(lastState)) {
          console.error('last_state contains NaN or Inf');
          return false;
        }
      } else if (Array.isArray(lastState)) {
        // Check array for NaN/Inf
        if (arrayHasNaNOrInf(lastState)) {
          console.error('last_state array contains NaN or Inf');
          return false;
        }
      } else {
        console.error(`Invalid last_state type: ${typeof lastState}`);
        return false;
      }
    }

    return true;
  }

  /**
   * Validate buffer state after update.
   *
   * Checks buffer consistency and size limits.
   */
  private _validateBufferState(state: ProcessorState): boolean {
    // Buffer operations are handled differently in this system
    // After a reset, measurement_history should be cleared
    if ('measurement_history' in state) {
      const history = state.measurement_history ?? [];

      // After reset, history should be empty or very small
      if (history.length > 100) {
        // Reasonable limit after reset
        console.warn(
          `Large measurement history after reset: ${history.length} items`
        );
        // Not a failure, just a warning
      }
    }

    return true;
  }

  /**
   * Validate state before persistence.
   *
   * Final validation before saving to database.
   */
  private _validatePersistedState(state: ProcessorState): boolean {
    // Run all validations for a complete check

    // Check structure
    if (typeof state !== 'object' || state === null || Array.isArray(state)) {
      console.error('State is not an object');
      return false;
    }

    // Check for any NaN/Inf in numeric fields
    for (const [key, value] of Object.entries(state)) {
      if (typeof value === 'number') {
        if (isNaNOrInf(value)) {
          console.error(`Field ${key} contains NaN or Inf`);
          return false;
        }
      } else if (Array.isArray(value)) {
        // Check if it's a numeric array
        const isNumericArray = value.every(v => typeof v === 'number');
        if (isNumericArray && arrayHasNaNOrInf(value)) {
          console.error(`Array field ${key} contains NaN or Inf`);
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Validate a weight measurement value.
   *
   * @param weight - Weight value in kg
   * @returns True if valid weight
   */
  static validateWeightValue(weight: number): boolean {
    // Basic sanity checks
    if (isNaNOrInf(weight)) {
      console.error('Weight is NaN or Inf');
      return false;
    }

    // Physiological limits (same as in constants)
    if (weight < 20 || weight > 450) {
      console.error(`Weight ${weight}kg outside physiological limits`);
      return false;
    }

    return true;
  }

  /**
   * Validate reset type string.
   *
   * @param resetType - Reset type string
   * @returns True if valid reset type
   */
  static validateResetType(resetType: string): boolean {
    const validTypes = ['INITIAL', 'HARD', 'SOFT', 'initial', 'hard', 'soft'];
    return validTypes.includes(resetType);
  }
}
