/**
 * Reset Manager for parameterized reset handling.
 * Supports hard, initial, and soft resets with different adaptation parameters.
 */

import type { ProcessorState, ResetEvent, ResetParameters } from '../../models';

/**
 * Types of resets with different adaptation strategies.
 */
export enum ResetType {
  INITIAL = 'initial', // First measurement - most aggressive adaptation
  HARD = 'hard',       // 30+ day gaps - aggressive adaptation
  SOFT = 'soft',       // Manual data entry - gentle adaptation
}

/**
 * Sources that trigger soft resets when they contain manual data
 */
export const MANUAL_DATA_SOURCES = new Set([
  'internal-questionnaire',
  'initial-questionnaire',
  'questionnaire',
  'user-upload',
  'care-team-upload',
  'care-team-entry',
]);

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
    state: ProcessorState | null,
    weight: number,
    timestamp: Date,
    source: string,
    config: any
  ): ResetType | null {
    // 1. Check for initial reset (no Kalman params yet)
    if (!state || !state.kalmanParams) {
      return ResetType.INITIAL;
    }

    // 2. Check for hard reset (30+ day gap)
    const hardConfig = config?.kalman?.reset?.hard ?? {};
    if (hardConfig.enabled !== false) { // Default true
      const lastTimestamp = state.lastAcceptedTimestamp || state.lastTimestamp;
      if (lastTimestamp) {
        let last: Date;
        if (typeof lastTimestamp === 'string') {
          last = new Date(lastTimestamp);
        } else if (lastTimestamp instanceof Date) {
          last = lastTimestamp;
        } else {
          // Handle case where it might be stored as a plain object
          last = new Date(lastTimestamp);
        }

        if (!(last instanceof Date) || isNaN(last.getTime())) {
          console.warn(`Invalid lastTimestamp value: ${lastTimestamp}`);
          return null;
        }

        const gapDays = (timestamp.getTime() - last.getTime()) / (86400.0 * 1000);
        const threshold = hardConfig.gap_threshold_days ?? 30;
        if (gapDays >= threshold) {
          return ResetType.HARD;
        }
      }
    }

    // 3. Check for soft reset (manual data with significant change)
    const softConfig = config?.kalman?.reset?.soft ?? {};
    if (softConfig.enabled !== false) { // Default true
      const triggerSources = new Set(softConfig.trigger_sources ?? []);
      if (MANUAL_DATA_SOURCES.has(source) || triggerSources.has(source)) {
        const lastWeight = state.lastRawWeight ?? state.lastAcceptedWeight;
        if (lastWeight !== null && lastWeight !== undefined) {
          const weightChange = Math.abs(weight - lastWeight);
          const minChange = softConfig.min_weight_change_kg ?? 5;

          if (weightChange >= minChange) {
            const cooldownDays = softConfig.cooldown_days ?? 3;
            const lastReset = ResetManager.getLastResetTimestamp(state);
            if (
              !lastReset ||
              (timestamp.getTime() - lastReset.getTime()) / (86400.0 * 1000) > cooldownDays
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
  static getLastResetTimestamp(state: ProcessorState): Date | null {
    const resetEvents = state.resetEvents ?? [];
    if (resetEvents.length > 0) {
      const lastReset = resetEvents[resetEvents.length - 1];
      const timestamp = lastReset.timestamp;
      if (timestamp) {
        return typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
      }
    }

    const resetTimestamp = state.resetTimestamp;
    if (resetTimestamp) {
      return typeof resetTimestamp === 'string' ? new Date(resetTimestamp) : resetTimestamp;
    }

    return null;
  }

  /**
   * Get parameters for specific reset type from config.
   *
   * Returns dict with adaptation parameters using new naming convention.
   */
  static getResetParameters(resetType: ResetType, config: any): ResetParameters {
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
        quality_safety_weight: 0.50,
        quality_plausibility_weight: 0.05,
        quality_consistency_weight: 0.05,
        quality_reliability_weight: 0.40,
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
        quality_plausibility_weight: 0.10,
        quality_consistency_weight: 0.10,
        quality_reliability_weight: 0.35,
      },
      [ResetType.SOFT]: {
        // Multipliers for Kalman parameters
        initial_variance_multiplier: 2,
        weight_noise_multiplier: 5,
        trend_noise_multiplier: 20,
        observation_noise_multiplier: 0.7,
        // Adaptation duration
        adaptation_measurements: 15,
        adaptation_days: 10,
        adaptation_decay_rate: 4,
        // Quality scoring
        quality_acceptance_threshold: 0.45,
        quality_safety_weight: 0.40,
        quality_plausibility_weight: 0.15,
        quality_consistency_weight: 0.15,
        quality_reliability_weight: 0.30,
      },
    };

    // Get from config or use defaults
    const resetConfig = config?.kalman?.reset?.[resetType] ?? {};
    const defaultParams = defaults[resetType];

    // Merge config with defaults
    const params: ResetParameters = {} as ResetParameters;
    for (const [key, defaultValue] of Object.entries(defaultParams)) {
      (params as any)[key] = resetConfig[key] ?? defaultValue;
    }

    return params;
  }

  /**
   * Perform reset with appropriate parameters.
   *
   * Returns [new_state, reset_event]
   */
  static performReset(
    state: ProcessorState,
    resetType: ResetType,
    timestamp: Date,
    weight: number,
    source: string,
    config: any
  ): [ProcessorState, ResetEvent] {
    // Get parameters for this reset type
    const resetParams = ResetManager.getResetParameters(resetType, config);

    // Calculate gap if applicable
    let gapDays: number | null = null;
    const lastTimestamp = state.lastAcceptedTimestamp || state.lastTimestamp;
    if (lastTimestamp) {
      let last: Date;
      if (typeof lastTimestamp === 'string') {
        last = new Date(lastTimestamp);
      } else if (lastTimestamp instanceof Date) {
        last = lastTimestamp;
      } else {
        last = new Date(lastTimestamp);
      }

      if (last instanceof Date && !isNaN(last.getTime())) {
        gapDays = (timestamp.getTime() - last.getTime()) / (86400.0 * 1000);
      }
    }

    // Create reset event
    const resetEvent: ResetEvent = {
      timestamp: timestamp instanceof Date ? timestamp.toISOString() : timestamp,
      resetType: resetType,
      resetReason: ResetManager.getResetReason(resetType, gapDays, weight, state),
      previousWeight: state.lastRawWeight ?? undefined,
      newWeight: weight,
      gapDays: gapDays ?? undefined,
      metadata: {
        source,
        parameters: resetParams,
      },
    };

    // Create new state with reset
    const newState: ProcessorState = {
      kalmanParams: null,
      lastState: null,
      lastCovariance: null,
      measurementsSinceReset: 0,
      resetType,
      resetParameters: resetParams,
      resetTimestamp: timestamp,
      resetEvents: [...(state.resetEvents ?? []), resetEvent],
      lastTimestamp: state.lastTimestamp,
      lastSource: state.lastSource,
      lastRawWeight: state.lastRawWeight,
      lastAcceptedTimestamp: state.lastAcceptedTimestamp,
      measurementHistory: [],
    };

    return [newState, resetEvent];
  }

  /**
   * Generate human-readable reset reason.
   */
  static getResetReason(
    resetType: ResetType,
    gapDays: number | null,
    weight: number,
    state: ProcessorState
  ): string {
    if (resetType === ResetType.INITIAL) {
      return 'initial_measurement';
    } else if (resetType === ResetType.HARD) {
      return gapDays ? `gap_exceeded_${Math.round(gapDays)}_days` : 'gap_exceeded';
    } else if (resetType === ResetType.SOFT) {
      const lastWeight = state.lastRawWeight;
      if (lastWeight !== null && lastWeight !== undefined) {
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
   * Returns [is_adaptive, parameters_dict]
   */
  static isInAdaptivePeriod(
    state: ProcessorState,
    timestamp: Date
  ): [boolean, ResetParameters | null] {
    const resetTimestamp = state.resetTimestamp;
    if (!resetTimestamp) {
      return [false, null];
    }

    const reset = typeof resetTimestamp === 'string'
      ? new Date(resetTimestamp)
      : resetTimestamp;

    const resetParams = state.resetParameters;
    if (!resetParams) {
      return [false, null];
    }

    // Check measurements-based
    const measurementsSince = state.measurementsSinceReset ?? 0;
    const adaptationMeasurements = resetParams.adaptation_measurements ?? 10;

    // Check time-based
    const daysSince = (timestamp.getTime() - reset.getTime()) / (86400.0 * 1000);
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
  static getAdaptiveFactor(state: ProcessorState, timestamp: Date): number {
    const resetTimestamp = state.resetTimestamp;
    if (!resetTimestamp) {
      return 1.0;
    }

    const resetParams = state.resetParameters;
    const decayRate = resetParams?.adaptation_decay_rate ?? 3;

    // Use measurements-based decay
    const measurementsSince = state.measurementsSinceReset ?? 0;
    const factor = 1.0 - Math.exp(-measurementsSince / decayRate);

    return Math.min(1.0, Math.max(0.0, factor));
  }
}
