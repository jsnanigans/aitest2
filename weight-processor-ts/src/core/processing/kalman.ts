/**
 * Kalman filter logic for weight processing.
 * Manages Kalman filter operations including initialization, updates, predictions,
 * and adaptive parameters for post-reset periods.
 */

import { KalmanFilter } from './kalman_filter';
import type { Vector2, Matrix2x2 } from '../math/matrix';
import { KALMAN_DEFAULTS } from '../../constants';
import type {
  ProcessorState,
  KalmanParams,
  ProcessResult,
} from '../../models';

/**
 * Convert value to float, handling nested structures
 */
function ensureFloatFromDecimal(value: any): any {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value === 'number') {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? value : parsed;
  }
  if (Array.isArray(value)) {
    return value.map(v => ensureFloatFromDecimal(v));
  }
  if (typeof value === 'object') {
    const result: any = {};
    for (const [k, v] of Object.entries(value)) {
      result[k] = ensureFloatFromDecimal(v);
    }
    return result;
  }
  return value;
}

export class KalmanFilterManager {
  /**
   * Initialize Kalman filter immediately with first measurement.
   */
  static initializeImmediate(
    weight: number,
    timestamp: Date,
    kalmanConfig: any,
    observationCovariance?: number
  ): Partial<ProcessorState> {
    const initialVariance = kalmanConfig.initial_variance ?? KALMAN_DEFAULTS.initial_variance;

    // Use passed observation_covariance if provided, otherwise use config value
    const obsCov = observationCovariance !== undefined
      ? observationCovariance
      : (kalmanConfig.observation_covariance ?? KALMAN_DEFAULTS.observation_covariance);

    const kalmanParams: KalmanParams = {
      initialStateMean: [weight, 0],
      initialStateCovariance: [[initialVariance, 0], [0, 0.001]],
      transitionCovariance: [
        [
          kalmanConfig.transition_covariance_weight ?? KALMAN_DEFAULTS.transition_covariance_weight,
          0,
        ],
        [
          0,
          kalmanConfig.transition_covariance_trend ?? KALMAN_DEFAULTS.transition_covariance_trend,
        ],
      ],
      observationCovariance: [[obsCov]],
    };

    return {
      kalmanParams,
      lastState: [weight, 0],  // Store as Vector2, not sequence
      lastCovariance: [[initialVariance, 0], [0, 0.001]],  // Store as Matrix2x2, not sequence
      lastTimestamp: timestamp,
      lastRawWeight: weight,
    };
  }

  /**
   * Update Kalman filter state with new measurement.
   */
  static updateState(
    state: ProcessorState,
    weight: number,
    timestamp: Date,
    _source: string,  // Prefix unused params with _
    _processingConfig: any,
    observationCovariance?: number
  ): ProcessorState {
    let timeDeltaDays = 1.0;
    if (state.lastTimestamp) {
      const lastTimestamp = typeof state.lastTimestamp === 'string'
        ? new Date(state.lastTimestamp)
        : state.lastTimestamp;
      const delta = (timestamp.getTime() - lastTimestamp.getTime()) / (86400.0 * 1000);
      timeDeltaDays = Math.max(0.1, Math.min(30.0, delta));
    }

    const kalmanParams = state.kalmanParams!;

    // Use passed observation_covariance if provided, otherwise use stored value
    const obsCov = observationCovariance !== undefined
      ? observationCovariance
      : parseFloat(String(kalmanParams.observationCovariance?.[0]?.[0] ?? 5.0));

    const kalman = new KalmanFilter(
      [[1, timeDeltaDays], [0, 1]],  // transitionMatrix
      [[1, 0], [0, 0]],  // observationMatrix (2x2 with second row zeros)
      ensureFloatFromDecimal(kalmanParams.initialStateMean) as Vector2,
      kalmanParams.initialStateCovariance as Matrix2x2,
      ensureFloatFromDecimal(kalmanParams.transitionCovariance) as Matrix2x2,
      obsCov  // observationCovariance (scalar)
    );

    const observation: number = weight;  // Single observation value

    // Use current state or initialize
    const currentState = state.lastState || kalmanParams.initialStateMean;
    const currentCovariance = state.lastCovariance || kalmanParams.initialStateCovariance;

    const [filteredStateMean, filteredStateCovariance] = kalman.filterUpdate(
      currentState,
      currentCovariance,
      observation
    );

    return {
      ...state,
      lastState: filteredStateMean,
      lastCovariance: filteredStateCovariance,
      lastTimestamp: timestamp,
      lastRawWeight: weight,
    };
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
    state: ProcessorState,
    weight: number,
    timestamp: Date,
    source: string,
    accepted: boolean,
    observationCovariance?: number
  ): Partial<ProcessResult> | null {
    if (!state.lastState) {
      return null;
    }

    const currentState = state.lastState;  // Vector2
    const filteredWeight = currentState[0];
    const trend = currentState[1];
    const innovation = weight - filteredWeight;

    const currentCovariance = state.lastCovariance;  // Matrix2x2 or null
    let normalizedInnovation = 0;

    if (currentCovariance) {
      // Use passed observation_covariance if provided, otherwise use stored value
      const obsCovariance = observationCovariance !== undefined
        ? observationCovariance
        : parseFloat(String(state.kalmanParams!.observationCovariance[0]![0]));

      const innovationVariance = currentCovariance[0][0] + obsCovariance;
      normalizedInnovation = innovationVariance > 0
        ? Math.abs(innovation) / Math.sqrt(innovationVariance)
        : 0;
    }

    const confidence = KalmanFilterManager.calculateConfidence(normalizedInnovation);

    // Calculate confidence intervals (±2σ)
    let kalmanUpper: number;
    let kalmanLower: number;
    let kalmanVariance: number | null;

    if (currentCovariance) {
      const confidenceInterval = 2.0 * Math.sqrt(currentCovariance[0][0]);
      kalmanUpper = filteredWeight + confidenceInterval;
      kalmanLower = filteredWeight - confidenceInterval;
      kalmanVariance = currentCovariance[0][0];
    } else {
      kalmanUpper = filteredWeight;
      kalmanLower = filteredWeight;
      kalmanVariance = null;
    }

    // Calculate prediction error
    const predictionError = accepted ? innovation : undefined;

    return {
      timestamp: timestamp.toISOString(),  // Convert Date to string
      rawWeight: weight,
      filteredWeight,
      trend,
      trendWeekly: trend * 7,
      accepted,
      confidence,
      innovation,
      normalizedInnovation,
      source,
      kalmanConfidenceUpper: kalmanUpper,
      kalmanConfidenceLower: kalmanLower,
      kalmanVariance: kalmanVariance ?? undefined,  // Convert null to undefined for optional field
      predictionError,
    };
  }

  /**
   * Extract current weight and trend from state.
   */
  static getCurrentStateValues(state: ProcessorState): [number | null, number | null] {
    if (!state.lastState) {
      return [null, null];
    }

    const currentState = state.lastState;  // Vector2
    return [currentState[0], currentState[1]];
  }

  /**
   * Calculate days between measurements.
   */
  static calculateTimeDeltaDays(
    currentTimestamp: Date,
    lastTimestamp: Date | null | undefined
  ): number {
    if (!lastTimestamp) {
      return 1.0;
    }

    const last = typeof lastTimestamp === 'string'
      ? new Date(lastTimestamp)
      : lastTimestamp;

    const delta = (currentTimestamp.getTime() - last.getTime()) / (86400.0 * 1000);
    return Math.max(0.1, delta);
  }

  /**
   * Get prediction for the next timestamp WITHOUT updating state.
   * This is the true Kalman prediction step, used for quality scoring.
   *
   * Returns [predicted_weight, innovation_covariance] or [null, null]
   */
  static predictNextState(
    state: ProcessorState,
    timestamp: Date,
    _config?: any  // Prefix unused param with _
  ): [number | null, number | null] {
    // Check if we have a valid state
    if (
      !state ||
      !state.lastState ||
      !state.kalmanParams
    ) {
      return [null, null];
    }

    // Get last timestamp
    const lastTimestamp = state.lastTimestamp;
    if (!lastTimestamp) {
      return [null, null];
    }

    // Convert to Date if string
    const lastTimestampDate = typeof lastTimestamp === 'string'
      ? new Date(lastTimestamp)
      : lastTimestamp;

    // Calculate time delta
    const timeDeltaDays = KalmanFilterManager.calculateTimeDeltaDays(
      timestamp,
      lastTimestampDate
    );

    // Get last posterior state and covariance
    const posteriorState = state.lastState;  // Vector2
    const posteriorCovariance = state.lastCovariance!;  // Matrix2x2

    // Build transition matrix F
    const F: number[][] = [[1, timeDeltaDays], [0, 1]];

    // Get process noise Q from kalman_params
    const kalmanParams = state.kalmanParams!;  // Already checked above
    const Q: number[][] = ensureFloatFromDecimal(kalmanParams.transitionCovariance);

    // Predict state: x_pred = F * x_posterior
    const predictedState: number[] = [
      F[0]![0]! * posteriorState[0] + F[0]![1]! * posteriorState[1],
      F[1]![0]! * posteriorState[0] + F[1]![1]! * posteriorState[1],
    ];

    // Predict covariance: P_pred = F * P_posterior * F' + Q
    // First: F * P_posterior
    const FP: number[][] = [
      [
        F[0]![0]! * posteriorCovariance[0]![0]! + F[0]![1]! * posteriorCovariance[1]![0]!,
        F[0]![0]! * posteriorCovariance[0]![1]! + F[0]![1]! * posteriorCovariance[1]![1]!,
      ],
      [
        F[1]![0]! * posteriorCovariance[0]![0]! + F[1]![1]! * posteriorCovariance[1]![0]!,
        F[1]![0]! * posteriorCovariance[0]![1]! + F[1]![1]! * posteriorCovariance[1]![1]!,
      ],
    ];

    // Then: (F * P_posterior) * F'
    const FPFt: number[][] = [
      [
        FP[0]![0]! * F[0]![0]! + FP[0]![1]! * F[0]![1]!,
        FP[0]![0]! * F[1]![0]! + FP[0]![1]! * F[1]![1]!,
      ],
      [
        FP[1]![0]! * F[0]![0]! + FP[1]![1]! * F[0]![1]!,
        FP[1]![0]! * F[1]![0]! + FP[1]![1]! * F[1]![1]!,
      ],
    ];

    // Finally: F * P_posterior * F' + Q
    const predictedCovariance: number[][] = [
      [FPFt[0]![0]! + Q[0]![0]!, FPFt[0]![1]! + Q[0]![1]!],
      [FPFt[1]![0]! + Q[1]![0]!, FPFt[1]![1]! + Q[1]![1]!],
    ];

    // Extract predicted weight (first element of state vector)
    const predictedWeight = predictedState[0] ?? null;

    // Calculate innovation covariance for the measurement
    // S = H * P_pred * H' + R, where H = [1, 0] for weight observation
    // Since H = [1, 0], this simplifies to P_pred[0,0] + R
    const R = parseFloat(String(kalmanParams.observationCovariance[0]![0]!));
    const innovationCovariance = predictedCovariance[0]![0]! + R;

    return [predictedWeight, innovationCovariance];
  }

  /**
   * Get adaptive covariances that start loose after reset and tighten over time.
   * This helps the filter adapt quickly to new weight patterns after a gap.
   */
  static getAdaptiveCovariances(
    measurementsSinceReset: number,
    config: any
  ): { weight: number; trend: number } {
    const baseWeightCov = config.transition_covariance_weight ??
      KALMAN_DEFAULTS.transition_covariance_weight;
    const baseTrendCov = config.transition_covariance_trend ??
      KALMAN_DEFAULTS.transition_covariance_trend;

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
  resetTimestamp: Date | null | undefined,
  currentTimestamp: Date,
  baseConfig: any,
  adaptiveDays: number = 7,
  state?: ProcessorState
): any {
  if (!resetTimestamp) {
    return baseConfig;
  }

  const reset = typeof resetTimestamp === 'string'
    ? new Date(resetTimestamp)
    : resetTimestamp;

  const daysSinceReset = (currentTimestamp.getTime() - reset.getTime()) / (86400.0 * 1000);

  // Initialize adaptation_days (may be overridden by reset_parameters)
  let adaptationDaysValue = adaptiveDays;
  let resetParams: any = {}; // Default empty reset params

  // Check if we have custom reset parameters in state
  if (state && state.resetParameters) {
    resetParams = state.resetParameters;
    adaptationDaysValue = resetParams.adaptation_days ?? adaptiveDays;

    // Get multipliers from reset parameters
    const initialVarMult = resetParams.initial_variance_multiplier ?? 5;
    const weightNoiseMult = resetParams.weight_noise_multiplier ?? 20;
    const trendNoiseMult = resetParams.trend_noise_multiplier ?? 200;
    const obsNoiseMult = resetParams.observation_noise_multiplier ?? 0.5;

    // Apply multipliers to base config
    const baseInitialVar = baseConfig.initial_variance ?? 0.361;
    const baseWeightCov = baseConfig.transition_covariance_weight ?? 0.016;
    const baseTrendCov = baseConfig.transition_covariance_trend ?? 0.0001;
    const baseObsCov = baseConfig.observation_covariance ?? 3.4;

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
    const measurementsSince = state?.measurementsSinceReset ?? 0;

    // Use measurement-based decay if available, otherwise time-based
    let decayFactor: number;
    if (measurementsSince > 0) {
      decayFactor = 1.0 - Math.exp(-measurementsSince / decayRate);
    } else {
      decayFactor = Math.min(1.0, daysSinceReset / adaptationDaysValue);
    }

    // Interpolate between adaptive and base parameters
    const result: any = {};
    for (const key of Object.keys(baseConfig)) {
      if (key in adaptiveParams) {
        const adaptiveValue = (adaptiveParams as any)[key];
        const baseValue = baseConfig[key];
        result[key] = adaptiveValue + decayFactor * (baseValue - adaptiveValue);
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

    // Simple time-based decay
    const decayFactor = Math.min(1.0, daysSinceReset / adaptationDaysValue);

    // Interpolate between adaptive and base parameters
    const result: any = {};
    for (const key of Object.keys(baseConfig)) {
      if (key in adaptiveParams) {
        const adaptiveValue = (adaptiveParams as any)[key];
        const baseValue = baseConfig[key];
        result[key] = adaptiveValue + decayFactor * (baseValue - adaptiveValue);
      } else {
        result[key] = baseConfig[key];
      }
    }

    return result;
  }
}
