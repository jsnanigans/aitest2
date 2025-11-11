/**
 * Kalman filter logic for weight processing using ml-matrix.
 */

import { Matrix } from 'ml-matrix';
import { KALMAN_DEFAULTS } from '../constants.js';
import { KalmanFilter } from './kalman_filter.js';
import type { KalmanState } from '../database/base.js';
import { ResetType } from './reset_manager.js';

/**
 * Reconstruct a Matrix object from serialized JSON data.
 * Handles both Matrix instances and serialized plain objects.
 */
function reconstructMatrix(data: any): Matrix {
  if (data instanceof Matrix) {
    return data;
  }

  // Handle serialized ml-matrix format
  if (data && typeof data === 'object' && 'rows' in data && 'columns' in data) {
    // Convert the serialized data back to a 2D array
    const rows = data.rows;
    const cols = data.columns;
    const arr: number[][] = [];

    if (data.data && Array.isArray(data.data)) {
      // Data is stored as array of row objects with numeric keys
      for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
          const val = data.data[i]?.[j] ?? data.data[i]?.[j.toString()] ?? 0;
          row.push(val);
        }
        arr.push(row);
      }
    }

    return new Matrix(arr);
  }

  // Fallback: try to construct directly
  return new Matrix(data);
}

/**
 * Convert value to number, handling nested arrays and objects.
 */
function ensureFloat(value: any): any {
  if (typeof value === 'number') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(v => ensureFloat(v));
  }
  if (typeof value === 'object' && value !== null) {
    const result: any = {};
    for (const [k, v] of Object.entries(value)) {
      result[k] = ensureFloat(v);
    }
    return result;
  }
  return value;
}


/**
 * Kalman parameters interface
 */
export interface KalmanParams {
  initial_state_mean: number[];
  initial_state_covariance: number[][];
  transition_covariance: number[][];
  observation_covariance: number[][];
}

/**
 * Kalman result interface
 */
export interface KalmanResult {
  timestamp: Date;
  raw_weight: number;
  filtered_weight: number;
  trend: number;
  trend_weekly: number;
  accepted: boolean;
  confidence: number;
  innovation: number;
  normalized_innovation: number;
  source: string;
  kalman_confidence_upper: number;
  kalman_confidence_lower: number;
  kalman_variance: number | null;
  prediction_error: number | null;
}

/**
 * Reset parameters interface
 */
export interface ResetParameters {
  initial_variance_multiplier?: number;
  weight_noise_multiplier?: number;
  trend_noise_multiplier?: number;
  observation_noise_multiplier?: number;
  adaptation_measurements?: number;
  adaptation_days?: number;
  adaptation_decay_rate?: number;
  quality_acceptance_threshold?: number;
  quality_safety_weight?: number;
  quality_plausibility_weight?: number;
  quality_consistency_weight?: number;
  quality_reliability_weight?: number;
}

/**
 * Reset event interface
 */
export interface ResetEvent {
  timestamp: Date;
  type: string;
  source: string;
  weight: number;
  last_weight?: number;
  gap_days?: number;
  reason: string;
  parameters: ResetParameters;
}

/**
 * Adaptive covariances result
 */
export interface AdaptiveCovariances {
  weight: number;
  trend: number;
}

/**
 * Reset types
 */

/**
 * Manual data sources
 */
export const MANUAL_DATA_SOURCES = new Set([
  'internal-questionnaire',
  'initial-questionnaire',
  'questionnaire',
  'user-upload',
  'care-team-upload',
  'care-team-entry',
]);

/**
 * Manages Kalman filter operations for weight processing.
 */
export class KalmanFilterManager {
  /**
   * Initialize Kalman filter immediately with first measurement.
   */
  static initializeImmediate(
    weight: number,
    timestamp: Date,
    kalmanConfig: Record<string, any>,
    observationCovariance?: number
  ): KalmanState {
    const initialVariance =
      kalmanConfig.initial_variance ?? KALMAN_DEFAULTS.initial_variance;

    // Use passed observation_covariance if provided, otherwise use config value
    const obsCov =
      observationCovariance !== undefined
        ? observationCovariance
        : (kalmanConfig.observation_covariance ?? KALMAN_DEFAULTS.observation_covariance);

    const kalmanParams: KalmanParams = {
      initial_state_mean: [weight, 0],
      initial_state_covariance: [
        [initialVariance, 0],
        [0, 0.001],
      ],
      transition_covariance: [
        [
          kalmanConfig.transition_covariance_weight ??
            KALMAN_DEFAULTS.transition_covariance_weight,
          0,
        ],
        [
          0,
          kalmanConfig.transition_covariance_trend ??
            KALMAN_DEFAULTS.transition_covariance_trend,
        ],
      ],
      observation_covariance: [[obsCov]],
    };

    // Create Matrix objects for initial state
    const initialStateMean = Matrix.columnVector([weight, 0]);
    const initialStateCovariance = new Matrix([
      [initialVariance, 0],
      [0, 0.001],
    ]);

    // Return state with 2 copies (history of 2)
    return {
      kalman_params: kalmanParams,
      last_state: [initialStateMean.clone(), initialStateMean.clone()],
      last_covariance: [initialStateCovariance.clone(), initialStateCovariance.clone()],
      last_timestamp: timestamp,
      last_raw_weight: weight,
      last_accepted_timestamp: null,
      last_source: null,
      measurement_history: [],
      reset_events: [],
      measurements_since_reset: 0,
    };
  }

  /**
   * Update Kalman filter state with new measurement.
   */
  static updateState(
    state: KalmanState,
    weight: number,
    timestamp: Date,
    source: string,
    processingConfig: Record<string, any>,
    observationCovariance?: number
  ): KalmanState {
    let timeDeltaDays = 1.0;
    if (state.last_timestamp) {
      const lastTimestamp = state.last_timestamp;
      const delta = (timestamp.getTime() - lastTimestamp.getTime()) / 86400000.0;
      timeDeltaDays = Math.max(0.1, Math.min(30.0, delta));
    }

    const kalmanParams = state.kalman_params!;

    // Use passed observation_covariance if provided, otherwise use stored value
    const obsCov =
      observationCovariance !== undefined
        ? observationCovariance
        : kalmanParams.observation_covariance[0][0];

    const kalman = new KalmanFilter(
      [
        [1, timeDeltaDays],
        [0, 1],
      ],
      [[1, 0]],
      ensureFloat(kalmanParams.initial_state_mean),
      kalmanParams.initial_state_covariance,
      ensureFloat(kalmanParams.transition_covariance),
      [[obsCov]]
    );

    const observation = [weight];

    let newLastState: Matrix[];
    let newLastCovariance: Matrix[];

    if (!state.last_state) {
      const [filteredStateMeans, filteredStateCovariances] = kalman.filter([[weight]]);
      newLastState = filteredStateMeans;
      newLastCovariance = filteredStateCovariances;

      if (process.env.VERBOSE_LOGGING) {
        const firstState = newLastState[0];
        if (firstState instanceof Matrix) {
          console.log(`[KalmanInit] First state: weight=${firstState.get(0, 0)}, velocity=${firstState.get(1, 0)}`);
        }
      }
    } else {
      const lastState = state.last_state;
      const lastCovariance = state.last_covariance!;

      // Reconstruct Matrix objects from serialized state if needed
      // (State may have been serialized/deserialized from storage)
      const reconstructedState = lastState.map(s => reconstructMatrix(s));
      const reconstructedCovariance = lastCovariance.map(c => reconstructMatrix(c));

      // Get the most recent state and covariance
      const currentState = reconstructedState[reconstructedState.length - 1];
      const currentCovariance = reconstructedCovariance[reconstructedCovariance.length - 1];

      if (process.env.VERBOSE_LOGGING && currentState instanceof Matrix) {
        const w = currentState.get(0, 0);
        const v = currentState.get(1, 0);
        if (isNaN(w) || isNaN(v)) {
          console.log(`[KalmanBeforeUpdate] READING NaN from state! weight=${w}, velocity=${v}`);
        }
      }

      const [filteredStateMean, filteredStateCovariance] = kalman.filterUpdate(
        currentState,
        currentCovariance,
        observation
      );

      if (process.env.VERBOSE_LOGGING && filteredStateMean instanceof Matrix) {
        const w = filteredStateMean.get(0, 0);
        const v = filteredStateMean.get(1, 0);
        if (isNaN(w) || isNaN(v)) {
          console.log(`[KalmanAfterUpdate] filterUpdate RETURNED NaN! weight=${w}, velocity=${v}, input=${weight}`);
        } else {
          console.log(`[KalmanAfterUpdate] filterUpdate returned: weight=${w}, velocity=${v}`);
        }
      }

      // Keep last 2 states: previous and current
      newLastState = [reconstructedState[reconstructedState.length - 1], filteredStateMean];
      newLastCovariance = [
        reconstructedCovariance[reconstructedCovariance.length - 1],
        filteredStateCovariance,
      ];
    }

    state.last_state = newLastState;
    state.last_covariance = newLastCovariance;
    state.last_timestamp = timestamp;
    state.last_raw_weight = weight;

    // Debug: Check if we're storing NaN
    if (process.env.VERBOSE_LOGGING) {
      const lastStateVec = newLastState[newLastState.length - 1];
      if (lastStateVec instanceof Matrix) {
        const weight = lastStateVec.get(0, 0);
        const velocity = lastStateVec.get(1, 0);
        if (isNaN(weight) || isNaN(velocity)) {
          console.log(`[KalmanUpdateState] STORING NaN! weight=${weight}, velocity=${velocity}`);
          console.log(`[KalmanUpdateState] Input weight was: ${weight}`);
        }
      }
    }

    return state;
  }

  /**
   * Calculate confidence using smooth exponential decay.
   */
  static calculateConfidence(normalizedInnovation: number): number {
    const alpha = 0.5;
    return Math.exp(-alpha * normalizedInnovation ** 2);
  }

  /**
   * Create result dictionary from current state.
   */
  static createResult(
    state: KalmanState,
    weight: number,
    timestamp: Date,
    source: string,
    accepted: boolean,
    observationCovariance?: number
  ): KalmanResult | null {
    if (!state.last_state) {
      return null;
    }

    const lastState = state.last_state;

    // Reconstruct Matrix objects from serialized state if needed
    const reconstructedState = lastState.map(s => reconstructMatrix(s));

    // Get the most recent state
    const currentState = reconstructedState[reconstructedState.length - 1];

    const filteredWeight = currentState.get(0, 0);
    const trend = currentState.get(1, 0);

    const innovation = weight - filteredWeight;

    const lastCovariance = state.last_covariance;
    let currentCovariance: Matrix | null = null;
    let normalizedInnovation = 0;
    let kalmanUpper = filteredWeight;
    let kalmanLower = filteredWeight;
    let kalmanVariance: number | null = null;

    if (lastCovariance) {
      if (lastCovariance.length > 0) {
        // Reconstruct Matrix objects from serialized covariance if needed
        const reconstructedCovariance = lastCovariance.map(c => reconstructMatrix(c));
        currentCovariance = reconstructedCovariance[reconstructedCovariance.length - 1];
      }

      if (currentCovariance) {
        // Use passed observation_covariance if provided, otherwise use stored value
        const obsCovariance =
          observationCovariance !== undefined
            ? observationCovariance
            : state.kalman_params!.observation_covariance[0][0];
        const innovationVariance = currentCovariance.get(0, 0) + obsCovariance;
        normalizedInnovation =
          innovationVariance > 0
            ? Math.abs(innovation) / Math.sqrt(innovationVariance)
            : 0;
      }
    }

    const confidence = KalmanFilterManager.calculateConfidence(normalizedInnovation);

    // Calculate confidence intervals (±2σ)
    if (currentCovariance) {
      const confidenceInterval = 2.0 * Math.sqrt(currentCovariance.get(0, 0));
      kalmanUpper = filteredWeight + confidenceInterval;
      kalmanLower = filteredWeight - confidenceInterval;
      kalmanVariance = currentCovariance.get(0, 0);
    }

    // Calculate prediction error
    const predictionError = accepted ? innovation : null;

    return {
      timestamp,
      raw_weight: weight,
      filtered_weight: filteredWeight,
      trend,
      trend_weekly: trend * 7,
      accepted,
      confidence,
      innovation,
      normalized_innovation: normalizedInnovation,
      source,
      kalman_confidence_upper: kalmanUpper,
      kalman_confidence_lower: kalmanLower,
      kalman_variance: kalmanVariance,
      prediction_error: predictionError,
    };
  }

  /**
   * Extract current weight and trend from state.
   */
  static getCurrentStateValues(state: KalmanState): [number | null, number | null] {
    if (!state.last_state) {
      return [null, null];
    }

    const lastState = state.last_state;

    if (process.env.VERBOSE_LOGGING) {
      console.log(`[KalmanGetState] lastState type: ${typeof lastState}, is Array: ${Array.isArray(lastState)}, length: ${(lastState as any)?.length}`);
    }

    const currentState = lastState[lastState.length - 1];

    if (process.env.VERBOSE_LOGGING) {
      console.log(`[KalmanGetState] currentState type: ${typeof currentState}, is Array: ${Array.isArray(currentState)}, is Matrix: ${currentState instanceof Matrix}`);
      if (Array.isArray(currentState)) {
        console.log(`[KalmanGetState] currentState (array): ${JSON.stringify(currentState)}`);
      } else if (currentState && typeof currentState === 'object') {
        console.log(`[KalmanGetState] currentState (object) keys: ${Object.keys(currentState).join(', ')}`);
        console.log(`[KalmanGetState] currentState (object): ${JSON.stringify(currentState).substring(0, 200)}`);
      }
    }

    // Handle non-Matrix objects
    if (!(currentState instanceof Matrix)) {
      // If it's a serialized Matrix object (has rows, columns, data properties)
      if (currentState && typeof currentState === 'object' && 'rows' in currentState && 'columns' in currentState && 'data' in currentState) {
        const { rows, columns, data } = currentState as any;
        if (process.env.VERBOSE_LOGGING) {
          console.log(`[KalmanGetState] Serialized matrix: rows=${rows}, columns=${columns}, data=${JSON.stringify(data)}`);
        }

        // Use reconstructMatrix logic - handle both array and object data formats
        const arr: number[][] = [];
        if (data && Array.isArray(data)) {
          for (let i = 0; i < rows; i++) {
            const row: number[] = [];
            for (let j = 0; j < columns; j++) {
              // Handle both data[i][j] (array) and data[i]["j"] (object with string keys)
              const val = data[i]?.[j] ?? data[i]?.[j.toString()] ?? 0;
              row.push(val);
            }
            arr.push(row);
          }
        }

        const mat = new Matrix(arr);
        const weight = mat.get(0, 0);
        const velocity = mat.get(1, 0);
        if (process.env.VERBOSE_LOGGING) {
          console.log(`[KalmanGetState] Converted from serialized: weight=${weight}, velocity=${velocity}`);
        }
        return [weight, velocity];
      }

      // Try to convert if it's an array
      if (Array.isArray(currentState)) {
        if (Array.isArray(currentState[0])) {
          // 2D array
          if (process.env.VERBOSE_LOGGING) {
            console.log(`[KalmanGetState] Converting 2D array to Matrix`);
          }
          const mat = new Matrix(currentState);
          const weight = mat.get(0, 0);
          const velocity = mat.get(1, 0);
          if (process.env.VERBOSE_LOGGING) {
            console.log(`[KalmanGetState] Converted from 2D array: weight=${weight}, velocity=${velocity}`);
          }
          return [weight, velocity];
        } else {
          // 1D array
          if (process.env.VERBOSE_LOGGING) {
            console.log(`[KalmanGetState] Converting 1D array to column vector`);
          }
          const mat = Matrix.columnVector(currentState);
          const weight = mat.get(0, 0);
          const velocity = mat.get(1, 0);
          if (process.env.VERBOSE_LOGGING) {
            console.log(`[KalmanGetState] Converted from 1D array: weight=${weight}, velocity=${velocity}`);
          }
          return [weight, velocity];
        }
      }

      throw new Error('getCurrentStateValues: currentState is not a Matrix and could not be converted');
    }

    const weight = currentState.get(0, 0);
    const velocity = currentState.get(1, 0);

    if (process.env.VERBOSE_LOGGING) {
      console.log(`[KalmanGetState] Extracted from Matrix: weight=${weight}, velocity=${velocity}`);
      if (isNaN(weight) || isNaN(velocity)) {
        console.log(`[KalmanGetState] NaN detected! weight=${weight}, velocity=${velocity}`);
        console.log(`[KalmanGetState] currentState:`, currentState);
      }
    }

    return [weight, velocity];
  }

  /**
   * Calculate days between measurements.
   */
  static calculateTimeDeltaDays(
    currentTimestamp: Date,
    lastTimestamp: Date | null
  ): number {
    if (!lastTimestamp) {
      return 1.0;
    }

    const delta = (currentTimestamp.getTime() - lastTimestamp.getTime()) / 86400000.0;
    return Math.max(0.1, delta);
  }

  /**
   * Get prediction for the next timestamp WITHOUT updating state.
   * This is the true Kalman prediction step, used for quality scoring.
   *
   * @param state - Current Kalman state
   * @param timestamp - Timestamp to predict for
   * @param config - Optional config with Kalman parameters
   * @returns Tuple of [predicted_weight, innovation_covariance]
   */
  static predictNextState(
    state: KalmanState,
    timestamp: Date,
    config?: Record<string, any>
  ): [number | null, number | null] {
    // Check if we have a valid state
    if (!state || !state.last_state || !state.kalman_params) {
      return [null, null];
    }

    // Get last timestamp
    const lastTimestamp = state.last_timestamp;
    if (!lastTimestamp) {
      return [null, null];
    }

    // Calculate time delta
    const timeDeltaDays = KalmanFilterManager.calculateTimeDeltaDays(timestamp, lastTimestamp);

    // Get last posterior state and covariance
    const lastState = state.last_state;
    const lastCovariance = state.last_covariance!;

    let posteriorState = lastState[lastState.length - 1];
    let posteriorCovariance = lastCovariance[lastCovariance.length - 1];

    // Ensure they're actual Matrix objects (handle deserialized states)
    if (!(posteriorState instanceof Matrix)) {
      posteriorState = reconstructMatrix(posteriorState);
    }

    if (!(posteriorCovariance instanceof Matrix)) {
      posteriorCovariance = reconstructMatrix(posteriorCovariance);
    }

    // Build transition matrix F
    const F = new Matrix([
      [1, timeDeltaDays],
      [0, 1],
    ]);

    // Get process noise Q from kalman_params
    const kalmanParams = state.kalman_params;
    const Q = new Matrix(ensureFloat(kalmanParams.transition_covariance));

    // Predict state: x_pred = F * x_posterior
    const predictedState = F.mmul(posteriorState as Matrix);

    // Predict covariance: P_pred = F * P_posterior * F' + Q
    const predictedCovariance = F
      .mmul(posteriorCovariance as Matrix)
      .mmul(F.transpose())
      .add(Q);

    // Extract predicted weight (first element of state vector)
    const predictedWeight = predictedState.get(0, 0);

    // Calculate innovation covariance for the measurement
    // S = H * P_pred * H' + R, where H = [1, 0] for weight observation
    // Since H = [1, 0], this simplifies to P_pred[0,0] + R
    const R = kalmanParams.observation_covariance[0][0];
    const innovationCovariance = predictedCovariance.get(0, 0) + R;

    if (process.env.VERBOSE_LOGGING) {
      if (isNaN(predictedWeight) || isNaN(innovationCovariance)) {
        console.log(`[KalmanPredict] NaN detected! predictedWeight=${predictedWeight}, innovationCovariance=${innovationCovariance}`);
        console.log(`[KalmanPredict] posteriorState[0,0]=${(posteriorState as Matrix).get(0, 0)}, posteriorState[1,0]=${(posteriorState as Matrix).get(1, 0)}`);
        console.log(`[KalmanPredict] posteriorCov[0,0]=${(posteriorCovariance as Matrix).get(0, 0)}, R=${R}`);
        console.log(`[KalmanPredict] predictedCov[0,0]=${predictedCovariance.get(0, 0)}`);
        console.log(`[KalmanPredict] timeDeltaDays=${timeDeltaDays}`);
      }
    }

    return [predictedWeight, innovationCovariance];
  }

  /**
   * Get adaptive covariances that start loose after reset and tighten over time.
   * This helps the filter adapt quickly to new weight patterns after a gap.
   *
   * @param measurementsSinceReset - Number of measurements since last reset
   * @param config - Kalman configuration dictionary
   * @returns Dictionary with 'weight' and 'trend' covariance values
   */
  static getAdaptiveCovariances(
    measurementsSinceReset: number,
    config: Record<string, any>
  ): AdaptiveCovariances {
    const baseWeightCov =
      config.transition_covariance_weight ?? KALMAN_DEFAULTS.transition_covariance_weight;
    const baseTrendCov =
      config.transition_covariance_trend ?? KALMAN_DEFAULTS.transition_covariance_trend;

    // Get adaptive settings from config
    const adaptiveConfig = config.post_reset_adaptation ?? {};

    const warmupMeasurements = adaptiveConfig.warmup_measurements ?? 10;
    const weightBoostFactor = adaptiveConfig.weight_boost_factor ?? 10;
    const trendBoostFactor = adaptiveConfig.trend_boost_factor ?? 100;
    const decayRate = adaptiveConfig.decay_rate ?? 3;

    if (measurementsSinceReset < warmupMeasurements) {
      // Exponentially decay from boost to 1x over warmup period
      const factor = Math.exp(-measurementsSinceReset / decayRate);

      const weightMultiplier = 1 + (weightBoostFactor - 1) * factor;
      const trendMultiplier = 1 + (trendBoostFactor - 1) * factor;

      return {
        weight: baseWeightCov * weightMultiplier,
        trend: baseTrendCov * trendMultiplier,
      };
    } else {
      // After warmup, use normal values
      return { weight: baseWeightCov, trend: baseTrendCov };
    }
  }
}

/**
 * Get adaptive Kalman parameters that gradually transition from
 * loose (adaptive) to tight (normal) configuration after a reset.
 *
 * Uses multipliers from reset parameters to scale base config values.
 */
export function getAdaptiveKalmanParams(
  resetTimestamp: Date | null,
  currentTimestamp: Date,
  baseConfig: Record<string, any>,
  adaptiveDays: number = 7,
  state?: KalmanState
): Record<string, any> {
  if (!resetTimestamp) {
    return baseConfig;
  }

  const daysSinceReset =
    (currentTimestamp.getTime() - resetTimestamp.getTime()) / 86400000.0;

  // Initialize adaptation_days (may be overridden by reset_parameters)
  let adaptationDaysValue = adaptiveDays;
  let resetParams: ResetParameters = {}; // Default empty reset params

  // Check if we have custom reset parameters in state
  if (state?.reset_parameters) {
    resetParams = state.reset_parameters;
    adaptationDaysValue = resetParams.adaptation_days ?? adaptiveDays;

    // Get multipliers from reset parameters (NO hardcoded defaults)
    if (!resetParams || !resetParams.initial_variance_multiplier || !resetParams.weight_noise_multiplier ||
        !resetParams.trend_noise_multiplier || !resetParams.observation_noise_multiplier) {
      throw new Error(
        'Reset parameters missing required multipliers. Must be provided or loaded from config.json.'
      );
    }

    const initialVarMult = resetParams.initial_variance_multiplier;
    const weightNoiseMult = resetParams.weight_noise_multiplier;
    const trendNoiseMult = resetParams.trend_noise_multiplier;
    const obsNoiseMult = resetParams.observation_noise_multiplier;

    // Apply multipliers to base config (NO hardcoded defaults - must come from config.json)
    if (!baseConfig.initial_variance || !baseConfig.transition_covariance_weight ||
        !baseConfig.transition_covariance_trend || !baseConfig.observation_covariance) {
      throw new Error(
        'Base Kalman config missing required values. Config must be loaded from config.json.'
      );
    }

    const baseInitialVar = baseConfig.initial_variance;
    const baseWeightCov = baseConfig.transition_covariance_weight;
    const baseTrendCov = baseConfig.transition_covariance_trend;
    const baseObsCov = baseConfig.observation_covariance;

    const adaptiveParams = {
      initial_variance: baseInitialVar * initialVarMult,
      transition_covariance_weight: baseWeightCov * weightNoiseMult,
      transition_covariance_trend: baseTrendCov * trendNoiseMult,
      observation_covariance: baseObsCov * obsNoiseMult,
    };

    // Check if we're still in adaptation period
    if (daysSinceReset >= adaptationDaysValue) {
      return baseConfig;
    }

    // Calculate decay factor based on days since reset
    const decayRate = resetParams.adaptation_decay_rate ?? 2.5;
    const measurementsSince = state?.measurements_since_reset ?? 0;

    // DEBUG logging
    if (process.env.DEBUG_ADAPTIVE) {
      console.log(`[getAdaptiveKalmanParams] measurements_since_reset=${measurementsSince}, decay_rate=${decayRate}`);
    }

    // Use measurement-based decay if available, otherwise time-based
    let decayFactor: number;
    if (measurementsSince > 0) {
      decayFactor = 1.0 - Math.exp(-measurementsSince / decayRate);
    } else {
      decayFactor = Math.min(1.0, daysSinceReset / adaptationDaysValue);
    }

    // DEBUG logging
    if (process.env.DEBUG_ADAPTIVE) {
      console.log(`[getAdaptiveKalmanParams] decay_factor=${decayFactor}`);
    }

    // Interpolate between adaptive and base parameters
    const result: Record<string, any> = {};
    for (const key of Object.keys(baseConfig)) {
      if (key in adaptiveParams) {
        const adaptiveValue = (adaptiveParams as any)[key];
        const baseValue = baseConfig[key];
        result[key] = adaptiveValue * (1 - decayFactor) + baseValue * decayFactor;
      } else {
        result[key] = baseConfig[key];
      }
    }

    return result;
  } else {
    // Use default adaptive parameters (shouldn't happen with proper reset)
    const adaptiveParams = {
      initial_variance: 5.0,
      transition_covariance_weight: 0.5,
      transition_covariance_trend: 0.01,
      observation_covariance: 2.0,
    };

    // Check if we're still in adaptation period
    if (daysSinceReset >= adaptationDaysValue) {
      return baseConfig;
    }

    const decayFactor = Math.min(1.0, daysSinceReset / adaptationDaysValue);

    // Interpolate between adaptive and base parameters
    const result: Record<string, any> = {};
    for (const key of Object.keys(baseConfig)) {
      if (key in adaptiveParams) {
        const adaptiveValue = (adaptiveParams as any)[key];
        const baseValue = baseConfig[key];
        result[key] = adaptiveValue * (1 - decayFactor) + baseValue * decayFactor;
      } else {
        result[key] = baseConfig[key];
      }
    }

    return result;
  }
}

/**
 * Check if adaptive parameters should be used based on reset history.
 */
export function shouldUseAdaptiveParams(
  state: KalmanState,
  adaptiveDays: number = 7
): boolean {
  const resetEvents = state.reset_events ?? [];
  if (resetEvents.length === 0) {
    return false;
  }

  const lastReset = resetEvents[resetEvents.length - 1];
  const resetTimestamp = lastReset.timestamp;
  if (!resetTimestamp) {
    return false;
  }

  const currentTimestamp = state.last_timestamp;
  if (!currentTimestamp) {
    return false;
  }

  const daysSinceReset =
    (currentTimestamp.getTime() - resetTimestamp.getTime()) / 86400000.0;

  // Use adaptation_days from reset parameters if available
  const resetParams = state.reset_parameters ?? {};
  const adaptationDays = resetParams.adaptation_days ?? adaptiveDays;

  return daysSinceReset < adaptationDays;
}

/**
 * Get the timestamp of the most recent reset event.
 */
export function getResetTimestamp(state: KalmanState): Date | null {
  const resetEvents = state.reset_events ?? [];
  if (resetEvents.length === 0) {
    return null;
  }

  const lastReset = resetEvents[resetEvents.length - 1];
  return lastReset.timestamp ?? null;
}

/**
 * Reset Manager for parameterized reset handling.
 * Supports hard, initial, and soft resets with different adaptation parameters.
 */
export class ResetManager {
  /**
   * Determine if and what type of reset to trigger.
   *
   * Priority order:
   * 1. Initial (no Kalman params)
   * 2. Hard (30+ day gap)
   * 3. Soft (manual data with significant change)
   */
  static shouldTriggerReset(
    state: KalmanState,
    weight: number,
    timestamp: Date,
    source: string,
    config: Record<string, any>
  ): ResetType | null {
    // 1. Check for initial reset (no Kalman params yet)
    if (!state || !state.kalman_params) {
      return ResetType.INITIAL;
    }

    // 2. Check for hard reset gap
    const hardConfig = config.kalman?.reset?.hard ?? {};
    const hardEnabled = hardConfig.enabled ?? true;

    if (hardEnabled) {
      const lastTimestamp = state.last_accepted_timestamp ?? state.last_timestamp;
      if (lastTimestamp) {
        const gapDays = (timestamp.getTime() - lastTimestamp.getTime()) / 86400000.0;
        const threshold = hardConfig.gap_threshold_days ?? 30;
        if (gapDays >= threshold) {
          return ResetType.HARD;
        }
      }
    }

    // 3. Check for soft reset (manual data with significant change)
    const softConfig = config.kalman?.reset?.soft ?? {};
    const softEnabled = softConfig.enabled ?? true;

    if (softEnabled) {
      const triggerSources = softConfig.trigger_sources ?? [];
      if (MANUAL_DATA_SOURCES.has(source) || triggerSources.includes(source)) {
        const lastWeight = state.last_raw_weight ?? state.last_accepted_weight;
        if (lastWeight !== undefined) {
          const weightChange = Math.abs(weight - lastWeight);
          const minChange = softConfig.min_weight_change_kg ?? 5;

          if (weightChange >= minChange) {
            const cooldownDays = softConfig.cooldown_days ?? 3;
            const lastReset = ResetManager.getLastResetTimestamp(state);
            if (
              !lastReset ||
              (timestamp.getTime() - lastReset.getTime()) / 86400000.0 > cooldownDays
            ) {
              return ResetType.SOFT;
            }
          }
        }
      }
    }

    return null;
  }

  /**
   * Get timestamp of the most recent reset.
   */
  static getLastResetTimestamp(state: KalmanState): Date | null {
    const resetEvents = state.reset_events ?? [];
    if (resetEvents.length > 0) {
      const lastReset = resetEvents[resetEvents.length - 1];
      return lastReset.timestamp ?? null;
    }

    return state.reset_timestamp ?? null;
  }

  /**
   * Get parameters for specific reset type from config.
   *
   * Returns dict with adaptation parameters using new naming convention.
   */
  static getResetParameters(
    resetType: ResetType,
    config: Record<string, any>
  ): ResetParameters {
    // Default parameters for each type (using new names)
    const defaults: Record<ResetType, ResetParameters> = {
      [ResetType.INITIAL]: {
        // Multipliers for Kalman parameters
        initial_variance_multiplier: 10,
        weight_noise_multiplier: 50,
        trend_noise_multiplier: 500,
        observation_noise_multiplier: 0.3,
        // Adaptation duration
        adaptation_measurements: 20,
        adaptation_days: 21,
        adaptation_decay_rate: 1.5,
        // Quality scoring
        quality_acceptance_threshold: 0.25,
        quality_safety_weight: 0.5,
        quality_plausibility_weight: 0.05,
        quality_consistency_weight: 0.05,
        quality_reliability_weight: 0.4,
      },
      [ResetType.HARD]: {
        // Multipliers for Kalman parameters
        initial_variance_multiplier: 5,
        weight_noise_multiplier: 20,
        trend_noise_multiplier: 200,
        observation_noise_multiplier: 0.5,
        // Adaptation duration
        adaptation_measurements: 10,
        adaptation_days: 7,
        adaptation_decay_rate: 2.5,
        // Quality scoring
        quality_acceptance_threshold: 0.35,
        quality_safety_weight: 0.45,
        quality_plausibility_weight: 0.1,
        quality_consistency_weight: 0.1,
        quality_reliability_weight: 0.35,
      },
      [ResetType.SOFT]: {
        // Multipliers for Kalman parameters
        initial_variance_multiplier: 2,
        weight_noise_multiplier: 20,
        trend_noise_multiplier: 200,
        observation_noise_multiplier: 0.7,
        // Adaptation duration
        adaptation_measurements: 10,
        adaptation_days: 7,
        adaptation_decay_rate: 2.5,
        // Quality scoring
        quality_acceptance_threshold: 0.35,
        quality_safety_weight: 0.45,
        quality_plausibility_weight: 0.1,
        quality_consistency_weight: 0.1,
        quality_reliability_weight: 0.35,
      },
    };

    // Get from config or use defaults
    const resetConfig = config.kalman?.reset?.[resetType] ?? {};
    const defaultParams = defaults[resetType];

    // Merge config with defaults
    const params: ResetParameters = {};
    for (const [key, defaultValue] of Object.entries(defaultParams)) {
      (params as any)[key] = resetConfig[key] ?? defaultValue;
    }

    return params;
  }

  /**
   * Perform reset with appropriate parameters.
   *
   * @returns Tuple of [new_state, reset_event]
   */
  static performReset(
    state: KalmanState,
    resetType: ResetType,
    timestamp: Date,
    weight: number,
    source: string,
    config: Record<string, any>
  ): [KalmanState, ResetEvent] {
    // Get parameters for this reset type
    const resetParams = ResetManager.getResetParameters(resetType, config);

    // Calculate gap if applicable
    let gapDays: number | undefined;
    const lastTimestamp = state.last_accepted_timestamp ?? state.last_timestamp;
    if (lastTimestamp) {
      gapDays = (timestamp.getTime() - lastTimestamp.getTime()) / 86400000.0;
    }

    // Create reset event
    const resetEvent: ResetEvent = {
      timestamp,
      type: resetType,
      source,
      weight,
      last_weight: state.last_raw_weight ?? undefined,
      gap_days: gapDays,
      reason: ResetManager.getResetReason(resetType, gapDays, weight, state),
      parameters: resetParams,
    };

    // Create new state with reset
    const newState: KalmanState = {
      kalman_params: null,
      last_state: undefined,
      last_covariance: undefined,
      measurements_since_reset: 0,
      reset_type: resetType,
      reset_parameters: resetParams,
      reset_timestamp: timestamp,
      reset_events: [...(state.reset_events ?? []), resetEvent],
      last_timestamp: state.last_timestamp,
      last_source: state.last_source,
      last_raw_weight: state.last_raw_weight,
      last_accepted_timestamp: state.last_accepted_timestamp,
      measurement_history: [],
    };

    if (process.env.DEBUG_ADAPTIVE) {
      console.log(`[performReset] Created newState with reset_events length=${newState.reset_events.length}`);
    }

    return [newState, resetEvent];
  }

  /**
   * Generate human-readable reset reason.
   */
  static getResetReason(
    resetType: ResetType,
    gapDays: number | undefined,
    weight: number,
    state: KalmanState
  ): string {
    if (resetType === ResetType.INITIAL) {
      return 'initial_measurement';
    } else if (resetType === ResetType.HARD) {
      return gapDays ? `gap_exceeded_${Math.floor(gapDays)}_days` : 'gap_exceeded';
    } else if (resetType === ResetType.SOFT) {
      const lastWeight = state.last_raw_weight;
      if (lastWeight !== undefined && lastWeight !== null) {
        const change = Math.abs(weight - lastWeight);
        return `manual_entry_change_${change.toFixed(1)}kg`;
      }
      return 'manual_entry';
    } else {
      return 'unknown';
    }
  }

  /**
   * Check if currently in adaptive period and return parameters if so.
   *
   * @returns Tuple of [is_adaptive, parameters_dict]
   */
  static isInAdaptivePeriod(
    state: KalmanState,
    timestamp: Date
  ): [boolean, ResetParameters | null] {
    const resetTimestamp = state.reset_timestamp;
    if (!resetTimestamp) {
      return [false, null];
    }

    const resetParams = state.reset_parameters;
    if (!resetParams) {
      return [false, null];
    }

    // Check measurements-based
    const measurementsSince = state.measurements_since_reset ?? 0;
    const adaptationMeasurements = resetParams.adaptation_measurements ?? 10;

    // Check time-based
    const daysSince = (timestamp.getTime() - resetTimestamp.getTime()) / 86400000.0;
    const adaptationDays = resetParams.adaptation_days ?? 7;

    // In adaptive period if either condition is met
    if (measurementsSince < adaptationMeasurements || daysSince < adaptationDays) {
      return [true, resetParams];
    }

    return [false, null];
  }

  /**
   * Calculate adaptive factor (0-1) based on time/measurements since reset.
   * 0 = fully adaptive, 1 = normal operation
   */
  static getAdaptiveFactor(state: KalmanState, timestamp: Date): number {
    const resetTimestamp = state.reset_timestamp;
    if (!resetTimestamp) {
      return 1.0;
    }

    const resetParams = state.reset_parameters ?? {};
    const decayRate = resetParams.adaptation_decay_rate ?? 3;

    // Use measurements-based decay
    const measurementsSince = state.measurements_since_reset ?? 0;
    const factor = 1.0 - Math.exp(-measurementsSince / decayRate);

    return Math.min(1.0, Math.max(0.0, factor));
  }
}
