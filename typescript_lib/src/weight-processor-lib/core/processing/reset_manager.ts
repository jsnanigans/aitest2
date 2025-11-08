/**
 * Reset Manager for parameterized reset handling.
 * Supports hard, initial, and soft resets with different adaptation parameters.
 */

import type { KalmanState } from '../database/base.js';
import type { ResetParameters, ResetEvent } from './kalman.js';

/**
 * Types of resets with different adaptation strategies.
 */
export enum ResetType {
  HARD = 'hard',      // 30+ day gaps - aggressive adaptation
  INITIAL = 'initial', // First measurement - most aggressive adaptation
  SOFT = 'soft',      // Manual data entry - gentle adaptation
}

/**
 * Manual data sources that can trigger soft resets
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
 * Configuration interface matching Python config structure
 */
export interface ResetConfig {
  kalman?: {
    reset?: {
      hard?: {
        enabled?: boolean;
        gap_threshold_days?: number;
        [key: string]: any;
      };
      soft?: {
        enabled?: boolean;
        min_weight_change_kg?: number;
        cooldown_days?: number;
        trigger_sources?: string[];
        [key: string]: any;
      };
      initial?: {
        [key: string]: any;
      };
      [key: string]: any;
    };
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Result of reset operation
 */
export interface ResetResult {
  newState: KalmanState;
  resetEvent: ResetEvent;
}

/**
 * Manages different types of resets with configurable parameters.
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
    state: KalmanState | null,
    weight: number,
    timestamp: Date,
    source: string,
    config: ResetConfig
  ): ResetType | null {
    // 1. Check for initial reset (no Kalman params yet)
    if (!state || !state.kalman_params) {
      return ResetType.INITIAL;
    }

    // 2. Check for hard reset (30+ day gap)
    const hardConfig = config.kalman?.reset?.hard ?? {};
    if (hardConfig.enabled !== false) {
      const lastTimestamp = state.last_accepted_timestamp ?? state.last_timestamp;
      if (lastTimestamp) {
        const gapDays = (timestamp.getTime() - lastTimestamp.getTime()) / (86400 * 1000);
        const threshold = hardConfig.gap_threshold_days ?? 30;
        if (gapDays >= threshold) {
          return ResetType.HARD;
        }
      }
    }

    // 3. Check for soft reset (manual data with significant change)
    const softConfig = config.kalman?.reset?.soft ?? {};
    if (softConfig.enabled !== false) {
      const triggerSources = softConfig.trigger_sources ?? [];
      if (MANUAL_DATA_SOURCES.has(source) || triggerSources.includes(source)) {
        const lastWeight = state.last_raw_weight ?? state.last_accepted_weight;
        if (lastWeight !== undefined && lastWeight !== null) {
          const weightChange = Math.abs(weight - lastWeight);
          const minChange = softConfig.min_weight_change_kg ?? 5;

          if (weightChange >= minChange) {
            const cooldownDays = softConfig.cooldown_days ?? 3;
            const lastReset = ResetManager.getLastResetTimestamp(state);
            if (
              !lastReset ||
              (timestamp.getTime() - lastReset.getTime()) / (86400 * 1000) > cooldownDays
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
      const timestamp = lastReset.timestamp;
      if (timestamp) {
        return timestamp instanceof Date ? timestamp : new Date(timestamp);
      }
    }

    const resetTimestamp = state.reset_timestamp;
    if (resetTimestamp) {
      return resetTimestamp instanceof Date ? resetTimestamp : new Date(resetTimestamp);
    }

    return null;
  }

  /**
   * Get parameters for specific reset type from config.
   *
   * Returns dict with adaptation parameters.
   * Loads all defaults from config.json (no hardcoded values).
   */
  static getResetParameters(
    resetType: ResetType,
    config: ResetConfig
  ): ResetParameters {
    // Try to load defaults from config.json FIRST
    // Only fall back to passed config if global config not available
    let resetConfig = config.kalman?.reset?.[resetType] ?? {};

    // If config doesn't have these values, try to load from global config.json
    if (Object.keys(resetConfig).length === 0) {
      try {
        // Dynamic import to avoid circular dependencies
        const configModule = require('../config.js');
        const globalConfig = configModule.loadConfig();

        if (globalConfig.kalman?.reset?.[resetType]) {
          resetConfig = globalConfig.kalman.reset[resetType];
        } else {
          throw new Error(`Reset config for ${resetType} not found in config.json`);
        }
      } catch (error) {
        throw new Error(
          `Failed to load reset parameters for ${resetType} from config.json. ` +
          `Config must be provided or config.json must exist. Error: ${error}`
        );
      }
    }

    // Build params from config (NO hardcoded defaults)
    const params: ResetParameters = {
      initial_variance_multiplier: resetConfig.initial_variance_multiplier,
      weight_noise_multiplier: resetConfig.weight_noise_multiplier,
      trend_noise_multiplier: resetConfig.trend_noise_multiplier,
      observation_noise_multiplier: resetConfig.observation_noise_multiplier,
      adaptation_measurements: resetConfig.adaptation_measurements,
      adaptation_days: resetConfig.adaptation_days,
      adaptation_decay_rate: resetConfig.adaptation_decay_rate,
      // Quality scoring params - use if provided, otherwise undefined
      quality_acceptance_threshold: resetConfig.quality_acceptance_threshold,
      quality_safety_weight: resetConfig.quality_safety_weight,
      quality_plausibility_weight: resetConfig.quality_plausibility_weight,
      quality_consistency_weight: resetConfig.quality_consistency_weight,
      quality_reliability_weight: resetConfig.quality_reliability_weight,
    };

    // Validate required params are present
    const required = [
      'initial_variance_multiplier',
      'weight_noise_multiplier',
      'trend_noise_multiplier',
      'observation_noise_multiplier',
      'adaptation_measurements',
      'adaptation_days',
      'adaptation_decay_rate',
    ];

    for (const key of required) {
      if (params[key as keyof ResetParameters] === undefined) {
        throw new Error(
          `Required reset parameter '${key}' missing for reset type '${resetType}' in config`
        );
      }
    }

    return params;
  }

  /**
   * Perform reset with appropriate parameters.
   *
   * Returns:
   *   Tuple of (new_state, reset_event)
   */
  static performReset(
    state: KalmanState,
    resetType: ResetType,
    timestamp: Date,
    weight: number,
    source: string,
    config: ResetConfig
  ): ResetResult {
    // Get parameters for this reset type
    const resetParams = ResetManager.getResetParameters(resetType, config);

    // Calculate gap if applicable
    let gapDays: number | null = null;
    const lastTimestamp = state.last_accepted_timestamp ?? state.last_timestamp;
    if (lastTimestamp) {
      gapDays = (timestamp.getTime() - lastTimestamp.getTime()) / (86400 * 1000);
    }

    // Create reset event
    const resetEvent: ResetEvent = {
      timestamp,
      type: resetType,
      source,
      weight,
      last_weight: state.last_raw_weight ?? undefined,
      gap_days: gapDays ?? undefined,
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
      adaptation_state: {},
      version: (state.version ?? 0) + 1,
    };

    return { newState, resetEvent };
  }

  /**
   * Generate human-readable reset reason.
   */
  static getResetReason(
    resetType: ResetType,
    gapDays: number | null,
    weight: number,
    state: KalmanState
  ): string {
    if (resetType === ResetType.INITIAL) {
      return 'initial_measurement';
    } else if (resetType === ResetType.HARD) {
      return gapDays !== null ? `gap_exceeded_${Math.floor(gapDays)}_days` : 'gap_exceeded';
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
   * Returns:
   *   Tuple of (is_adaptive, parameters_dict)
   */
  static isInAdaptivePeriod(
    state: KalmanState,
    timestamp: Date
  ): [boolean, ResetParameters | null] {
    const resetTimestamp = state.reset_timestamp;
    if (!resetTimestamp) {
      return [false, null];
    }

    const resetTimestampDate = resetTimestamp instanceof Date
      ? resetTimestamp
      : new Date(resetTimestamp);

    const resetParams = state.reset_parameters;
    if (!resetParams) {
      return [false, null];
    }

    // Check measurements-based
    const measurementsSince = state.measurements_since_reset ?? 0;
    const adaptationMeasurements = resetParams.adaptation_measurements ?? 10;

    // Check time-based
    const daysSince = (timestamp.getTime() - resetTimestampDate.getTime()) / (86400 * 1000);
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

    const resetTimestampDate = resetTimestamp instanceof Date
      ? resetTimestamp
      : new Date(resetTimestamp);

    const resetParams = state.reset_parameters ?? {};
    const decayRate = resetParams.adaptation_decay_rate ?? 3;

    // Use measurements-based decay
    const measurementsSince = state.measurements_since_reset ?? 0;
    const factor = 1.0 - Math.exp(-measurementsSince / decayRate);

    return Math.min(1.0, Math.max(0.0, factor));
  }
}
