/**
 * Simplified weight processor with flattened pipeline.
 * Single processing function with clear flow.
 *
 * Ported from Python: weight_values/src/core/processing/processor.py
 */

import type { ProcessorState, ProcessResult } from '../../models';
import type { Config } from '../../config';
import type { StateStore } from '../database/base';
import { KalmanFilterManager } from './kalman';
import { ResetManager, ResetType } from './reset_manager';
import { ResetTransaction, ResetOperation } from './reset_transaction';
import { DataQualityPreprocessor } from './validation';
import { UnifiedQualityScorer } from './unified_quality_scorer';
import { PersistenceValidator } from './persistence_validator';
import { CircuitBreaker, CircuitOpenError } from './circuit_breaker';
import { ensureFloat, ensureNumericTypes } from './type_conversion';
import { KALMAN_DEFAULTS, PHYSIOLOGICAL_LIMITS, getNoiseMultiplier } from '../../constants';

// Circuit breaker for reset operations (module level)
const _reset_circuit_breaker = new CircuitBreaker({
  failureThreshold: 3,
  timeout: 60,
  successThreshold: 2,
  name: 'reset_operations'
});

// Verbose logging helpers
const VERBOSE_LOGGING = process.env.VERBOSE_LOGGING === "true";

function _log(message: string): void {
    if (VERBOSE_LOGGING) {
        console.log(`[TS] ${message}`);
    }
}

function _formatNum(value: number | null | undefined): string {
    if (value === null || value === undefined) {
        return "null";
    }
    return value.toFixed(6);
}

function _formatVec(vec: number[][] | number[] | null | undefined): string {
    if (!vec) return "null";
    const flat = Array.isArray(vec[0])
        ? (vec as number[][]).flat()
        : vec as number[];
    return `[${flat.map(v => v.toFixed(6)).join(', ')}]`;
}

/**
 * Get the timestamp of the most recent reset event.
 */
function get_reset_timestamp(state: ProcessorState): Date | null {
  const reset_events = state.resetEvents || [];
  if (reset_events.length === 0) {
    return null;
  }

  const last_reset = reset_events[reset_events.length - 1];
  const reset_timestamp = last_reset.timestamp;

  if (reset_timestamp) {
    if (typeof reset_timestamp === 'string') {
      return new Date(reset_timestamp);
    }
    return reset_timestamp;
  }

  return null;
}

/**
 * Get adaptive Kalman parameters that gradually transition from
 * loose (adaptive) to tight (normal) configuration after a reset.
 *
 * Uses multipliers from reset parameters to scale base config values.
 */
function get_adaptive_kalman_params(
  reset_timestamp: Date | null,
  current_timestamp: Date,
  base_config: Record<string, any>,
  adaptive_days: number = 7,
  state: ProcessorState | null = null
): Record<string, any> {
  if (!reset_timestamp) {
    return base_config;
  }

  const days_since_reset = (current_timestamp.getTime() - reset_timestamp.getTime()) / (86400 * 1000);

  // Initialize adaptation_days (may be overridden by reset_parameters)
  let adaptation_days_value = adaptive_days;
  let reset_params: Record<string, any> = {};

  // Check if we have custom reset parameters in state
  if (state && state.resetParameters) {
    reset_params = state.resetParameters;
    adaptation_days_value = reset_params.adaptation_days ?? adaptive_days;

    // Get multipliers from reset parameters
    const initial_var_mult = reset_params.initial_variance_multiplier ?? 5;
    const weight_noise_mult = reset_params.weight_noise_multiplier ?? 20;
    const trend_noise_mult = reset_params.trend_noise_multiplier ?? 200;
    const obs_noise_mult = reset_params.observation_noise_multiplier ?? 0.5;

    // Apply multipliers to base config
    const base_initial_var = base_config.initial_variance ?? 0.361;
    const base_weight_cov = base_config.transition_covariance_weight ?? 0.016;
    const base_trend_cov = base_config.transition_covariance_trend ?? 0.0001;
    const base_obs_cov = base_config.observation_covariance ?? 3.4;

    const adaptive_params = {
      initial_variance: base_initial_var * initial_var_mult,
      transition_covariance_weight: base_weight_cov * weight_noise_mult,
      transition_covariance_trend: base_trend_cov * trend_noise_mult,
      observation_covariance: base_obs_cov * obs_noise_mult
    };

    // Check if we're still in adaptation period
    if (days_since_reset >= adaptation_days_value) {
      return base_config;
    }

    // Calculate decay factor based on days since reset
    const decay_rate = reset_params.adaptation_decay_rate ?? 2.5;
    const measurements_since = state.measurementsSinceReset ?? 0;

    let decay_factor: number;
    // Use measurement-based decay if available, otherwise time-based
    if (measurements_since > 0) {
      decay_factor = 1.0 - Math.exp(-measurements_since / decay_rate);
    } else {
      decay_factor = Math.min(1.0, days_since_reset / adaptation_days_value);
    }

    // Interpolate between adaptive and base parameters
    const result: Record<string, any> = {};
    for (const key in base_config) {
      if (key in adaptive_params) {
        const adaptive_value = adaptive_params[key as keyof typeof adaptive_params];
        const base_value = base_config[key];
        result[key] = adaptive_value * (1 - decay_factor) + base_value * decay_factor;
      } else {
        result[key] = base_config[key];
      }
    }

    return result;
  } else {
    // Use default adaptive parameters (shouldn't happen with proper reset)
    const adaptive_params = {
      initial_variance: 5.0,
      transition_covariance_weight: 0.5,
      transition_covariance_trend: 0.01,
      observation_covariance: 2.0
    };

    // Check if we're still in adaptation period
    if (days_since_reset >= adaptation_days_value) {
      return base_config;
    }

    const decay_factor = Math.min(1.0, days_since_reset / adaptation_days_value);

    // Interpolate between adaptive and base parameters
    const result: Record<string, any> = {};
    for (const key in base_config) {
      if (key in adaptive_params) {
        const adaptive_value = adaptive_params[key as keyof typeof adaptive_params];
        const base_value = base_config[key];
        result[key] = adaptive_value * (1 - decay_factor) + base_value * decay_factor;
      } else {
        result[key] = base_config[key];
      }
    }

    return result;
  }
}

/**
 * Create periodic snapshot if interval has elapsed since last snapshot.
 *
 * This function checks if enough time has passed since the last snapshot
 * and creates a new one if needed. This ensures replay functionality has
 * adequate state history even for users without recent resets.
 */
function _maybe_create_periodic_snapshot(
  db: StateStore,
  userId: string,
  timestamp: Date,
  config: Config
): boolean {
  try {
    // Get snapshot configuration
    const snapshot_config = config.snapshot || {};
    const snapshot_interval_hours = snapshot_config.interval_hours || 24;
    const periodic_enabled = snapshot_config.periodic_enabled !== false;

    if (!periodic_enabled) {
      return false;
    }

    // Get the latest snapshot for this user
    const latest_snapshot = db.get_latest_snapshot?.(userId);

    // Create snapshot if none exists (initial snapshot)
    if (!latest_snapshot) {
      db.save_state_snapshot(userId, timestamp);
      return true;
    }

    // Check time since last snapshot
    let last_snapshot_time = latest_snapshot.lastTimestamp;
    if (!last_snapshot_time) {
      // Fallback: if no lastTimestamp in snapshot, create new one
      db.save_state_snapshot(userId, timestamp);
      return true;
    }

    // Ensure last_snapshot_time is a Date
    if (typeof last_snapshot_time === 'string') {
      last_snapshot_time = new Date(last_snapshot_time.replace('Z', '+00:00'));
    }

    // Calculate hours since last snapshot
    const hours_since = (timestamp.getTime() - last_snapshot_time.getTime()) / (3600 * 1000);

    // Create snapshot if interval elapsed
    if (hours_since >= snapshot_interval_hours) {
      db.save_state_snapshot(userId, timestamp);
      return true;
    }

    return false;

  } catch (e) {
    const error = e as Error;
    console.warn(`Failed to create periodic snapshot for ${userId}: ${error.message}`);
    return false;
  }
}

/**
 * Handle reset operations with transaction safety and circuit breaker.
 */
async function _handle_reset_with_transaction(
  userId: string,
  state: ProcessorState,
  reset_type: ResetType,
  timestamp: Date,
  weight: number,
  source: string,
  config: Config
): Promise<[ProcessorState, Record<string, any> | null, boolean]> {
  try {
    // Try through circuit breaker first
    return await _reset_circuit_breaker.call(
      _perform_transactional_reset,
      userId,
      state,
      reset_type,
      timestamp,
      weight,
      source,
      config
    );
  } catch (e) {
    if (e instanceof CircuitOpenError) {
      console.error(`Reset circuit open for user ${userId}: ${e.message}`);
    } else {
      console.error(`Reset failed for user ${userId}: ${(e as Error).message}`);
    }
    // Return original state without reset
    return [state, null, false];
  }
}

/**
 * Perform reset with transaction management.
 */
function _perform_transactional_reset(
  userId: string,
  state: ProcessorState,
  reset_type: ResetType,
  timestamp: Date,
  weight: number,
  source: string,
  config: Config
): [ProcessorState, Record<string, any> | null, boolean] {
  const txn = new ResetTransaction(userId);

  try {

    // Save original state for potential rollback
    txn.saveOriginalState(ResetOperation.STATE_UPDATE, state);

    // Step 1: Perform the actual reset
    const reset_type_value = typeof reset_type === 'string' ? reset_type : reset_type;

    const [new_state, reset_event] = ResetManager.performReset(
      state,
      reset_type,
      timestamp,
      weight,
      source,
      config
    );

    // Save checkpoint and validate
    txn.saveCheckpoint(ResetOperation.STATE_UPDATE, new_state);
    if (!txn.validateCheckpoint(ResetOperation.STATE_UPDATE)) {
      throw new Error(`State validation failed after ${reset_type_value} reset`);
    }

    txn.markCompleted(ResetOperation.STATE_UPDATE);

    // Step 2: Validate Kalman reset (kalmanParams should be None)
    const kalman_state = {
      kalmanParams: new_state.kalmanParams,
      resetParameters: new_state.resetParameters,
      measurementsSinceReset: new_state.measurementsSinceReset || 0,
      resetType: new_state.resetType,
      resetTimestamp: new_state.resetTimestamp,
      userId: new_state.userId
    };

    txn.saveCheckpoint(ResetOperation.KALMAN_RESET, kalman_state);
    if (!txn.validateCheckpoint(ResetOperation.KALMAN_RESET)) {
      throw new Error('Kalman state validation failed after reset');
    }

    txn.markCompleted(ResetOperation.KALMAN_RESET);

    // All operations succeeded
    txn.commit();
    return [new_state, reset_event, true];

  } catch (e) {
    const error = e as Error;
    console.error(`Reset transaction failed for user ${userId}: ${error.message}`);
    console.error(`Stack: ${error.stack}`);

    // Transaction will automatically rollback
    txn.rollback();

    // Return original state
    const original_state = txn.getOriginalState(ResetOperation.STATE_UPDATE);
    if (original_state) {
      return [original_state as ProcessorState, null, false];
    }
    return [state, null, false];
  }
}

/**
 * Process a single weight measurement through the complete pipeline.
 *
 * Single function that:
 * 1. Cleans and validates data
 * 2. Manages Kalman state
 * 3. Applies filtering
 * 4. Returns comprehensive result
 */
export async function processMeasurement(
  userId: string,
  weight: number,
  timestamp: Date,
  source: string,
  config: Config,
  db: StateStore,
  unit: string = 'kg',
  user_height_m?: number
): Promise<ProcessResult> {
  // Ensure weight is a number
  weight = ensureFloat(weight);

  // Log input header
  _log("=".repeat(80));
  _log(`Processing measurement for user: ${userId.substring(0, 12)}...`);
  _log(`Weight: ${_formatNum(weight)}, Unit: ${unit}, Timestamp: ${timestamp.toISOString()}, Source: ${source}`);

  // Step 1: Data cleaning and preprocessing
  _log("Step 1: Data cleaning and preprocessing");
  // Use provided height or default for preprocessing
  const height_for_preprocessing = user_height_m ?? PHYSIOLOGICAL_LIMITS.DEFAULT_HEIGHT_M;
  const [cleaned_weight, preprocess_metadata] = DataQualityPreprocessor.preprocess(
    weight,
    source,
    timestamp,
    userId,
    unit,
    height_for_preprocessing
  );

  if (cleaned_weight !== null) {
    _log(`Cleaned weight: ${_formatNum(cleaned_weight)}`);
  } else {
    _log(`Preprocessing rejected: ${preprocess_metadata.rejected || preprocess_metadata.rejection_reason || 'Unknown reason'}`);
  }

  // If preprocessing rejected the measurement
  if (cleaned_weight === null) {
    _log("Result: REJECTED");
    _log(`  stage: preprocessing`);
    _log("=".repeat(80));
    return {
      accepted: false,
      rejected: true,
      timestamp,
      source,
      raw_weight: weight,
      reason: preprocess_metadata.rejected || preprocess_metadata.rejection_reason || 'Preprocessing failed',
      stage: 'preprocessing',
      metadata: preprocess_metadata
    } as ProcessResult;
  }

  // Step 2: Load or create user state
  _log("Step 2: Load or create user state");
  let state = db.get_state(userId);
  if (state === null) {
    _log("Creating new state (no existing state)");
    state = db.create_initial_state();
  } else {
    _log("State exists");
    if (state.lastRawWeight !== undefined) {
      _log(`  last_raw_weight: ${_formatNum(state.lastRawWeight)}`);
    }
    if (state.lastTimestamp) {
      const lastTs = state.lastTimestamp instanceof Date
        ? state.lastTimestamp
        : new Date(state.lastTimestamp as any);

      if (lastTs && !isNaN(lastTs.getTime())) {
        _log(`  last_timestamp: ${lastTs.toISOString()}`);
      } else {
        _log(`  last_timestamp: <invalid date>`);
      }
    }
    if (state.kalmanParams) {
      _log(`  kalman_params: present`);
    }
  }

  // Use the same height we used for preprocessing
  const user_height = height_for_preprocessing;

  // Step 3: Check for any type of reset using ResetManager
  _log("Step 3: Check for reset");
  const kalman_config = config.kalman || {};

  // Check if reset is needed (only if reset features are enabled)
  const reset_type = ResetManager.shouldTriggerReset(
    state,
    cleaned_weight,
    timestamp,
    source,
    config
  );

  let reset_event: Record<string, any> | null = null;
  let reset_occurred = false;

  if (reset_type) {
    _log(`Reset needed: type=${reset_type}`);
    // Perform the reset with transaction safety
    [state, reset_event, reset_occurred] = await _handle_reset_with_transaction(
      userId,
      state,
      reset_type,
      timestamp,
      cleaned_weight,
      source,
      config
    );
    if (reset_occurred && reset_event) {
      _log(`  Reset completed: reason=${reset_event.resetReason || 'unknown'}, gap_days=${reset_event.gapDays || 0}`);
    }
  } else {
    _log("No reset needed");
  }

  // Step 4: Initialize Kalman if needed
  _log("Step 4: Initialize Kalman if needed");
  let kalman_already_updated = false;
  let result: ProcessResult | null = null;

  if (!state.kalmanParams) {
    _log("Initializing Kalman filter");
    // Check if this is a post-reset initialization
    // For initial measurements, treat current timestamp as "reset" to get adaptive params
    const reset_timestamp = reset_occurred ? get_reset_timestamp(state) : timestamp;

    // Get adaptive Kalman config if within post-reset period
    const adaptive_kalman_config = get_adaptive_kalman_params(
      reset_timestamp,
      timestamp,
      kalman_config,
      7, // adaptive_days
      state
    );
    _log(`  Using adaptive parameters: Q_weight=${_formatNum(adaptive_kalman_config.transition_covariance_weight)}, Q_trend=${_formatNum(adaptive_kalman_config.transition_covariance_trend)}`);

    // Get adaptive noise for this source
    const adaptive_config = config.adaptive_noise || {};
    const noise_multiplier = getNoiseMultiplier(source, config.sources);
    const observation_covariance =
      (adaptive_kalman_config.observation_covariance || 3.49) * noise_multiplier;
    _log(`  noise_multiplier: ${_formatNum(noise_multiplier)}`);
    _log(`  observation_covariance: ${_formatNum(observation_covariance)}`);

    const kalman_state = KalmanFilterManager.initializeImmediate(
      cleaned_weight,
      timestamp,
      adaptive_kalman_config,
      observation_covariance
    );
    _log(`  Initial state: ${_formatVec(kalman_state.lastState)}`);

    // Merge Kalman state with existing state to preserve reset parameters
    Object.assign(state, kalman_state);

    // DO NOT call updateState here - initializeImmediate already set the state
    // with the first measurement. Calling updateState would process it twice!

    result = KalmanFilterManager.createResult(
      state,
      cleaned_weight,
      timestamp,
      source,
      true,
      observation_covariance
    );

    // Add metadata
    result.stage = 'initialization';
    result.preprocessing = preprocess_metadata;
    result.noise_multiplier = noise_multiplier;

    // Add reset event info if it occurred (flattened for visualization)
    if (reset_occurred && reset_event) {
      result.was_reset = true;
      result.reset_reason = reset_event.resetReason || 'unknown';
      result.resetType = reset_event.resetType || 'unknown';
      result.gap_days = reset_event.gapDays || 0;
      // Also keep nested structure for backward compatibility
      result.reset_event = {
        type: reset_event.resetType || 'unknown',
        gap_days: reset_event.gapDays,
        reason: reset_event.resetReason || 'unknown'
      };
    }

    // Mark that we've already done the Kalman update
    kalman_already_updated = true;

    // Continue to quality validation - no early return during initialization
  }

  // Step 5: Quality scoring (replaces physiological validation)
  _log("Step 5: Quality scoring");
  const quality_config = config.quality_scoring || {};

  // Get previous weight and time diff
  let previous_weight: number | null = null;
  let time_diff_hours: number | null = null;

  // Try to get previous weight from Kalman state
  if (state) {
    const [current_weight] = KalmanFilterManager.getCurrentStateValues(state);
    if (current_weight !== null) {
      previous_weight = current_weight;
    } else if (state.lastRawWeight !== null && state.lastRawWeight !== undefined) {
      previous_weight = ensureFloat(state.lastRawWeight);
    }

    // Get time diff
    if (state.lastTimestamp) {
      let prev_time: Date;
      if (typeof state.lastTimestamp === 'string') {
        prev_time = new Date(state.lastTimestamp);
      } else if (state.lastTimestamp instanceof Date) {
        prev_time = state.lastTimestamp;
      } else {
        prev_time = new Date(state.lastTimestamp);
      }

      if (prev_time instanceof Date && !isNaN(prev_time.getTime())) {
        time_diff_hours = (timestamp.getTime() - prev_time.getTime()) / (3600 * 1000);
      }
    }
  }

  // Get recent weights for statistical analysis
  const recent_weights: number[] = [];
  if (state && state.measurementHistory) {
    const history = state.measurementHistory;
    if (Array.isArray(history)) {
      for (const h of history.slice(-20)) {
        if (h.weight !== undefined) {
          recent_weights.push(ensureFloat(h.weight));
        }
      }
    }
  }

  // Use unified Kalman-centric quality scorer
  // Get Kalman prediction using proper predict step
  let kalman_prediction: number | null = null;
  let innovation_covariance: number | null = null;

  if (state && state.kalmanParams) {
    // Use the proper Kalman predict step to get prediction BEFORE update
    [kalman_prediction, innovation_covariance] = KalmanFilterManager.predictNextState(
      state,
      timestamp
    );

    // Apply source-specific noise multiplier to innovation covariance if needed
    if (innovation_covariance !== null) {
      // The predictNextState already includes base observation noise
      // We need to adjust for source-specific multiplier
      const noise_multiplier = getNoiseMultiplier(source, config.sources);
      if (noise_multiplier !== 1.0) {
        // Adjust innovation covariance for source reliability
        // Remove base R, apply multiplier, add back
        const kalman_params = state.kalmanParams;
        const base_obs_cov = ensureFloat(
          kalman_params.observationCovariance?.[0]?.[0] || KALMAN_DEFAULTS.observation_covariance
        );
        // innovation_cov = P_pred[0,0] + R
        // We need: P_pred[0,0] + (R * multiplier)
        const predicted_cov_00 = innovation_covariance - base_obs_cov;
        innovation_covariance = predicted_cov_00 + base_obs_cov * noise_multiplier;

      }
    }
  }

  // Get recent timestamps if available
  const recent_timestamps: Date[] = [];
  if (state && state.measurementHistory) {
    const history = state.measurementHistory;
    if (Array.isArray(history)) {
      for (const h of history.slice(-20)) {
        if (h.timestamp) {
          const ts = typeof h.timestamp === 'string' ? new Date(h.timestamp) : h.timestamp;
          recent_timestamps.push(ts);
        }
      }
    }
  }

  // Create unified scorer instance with source profiles
  const unified_scorer = new UnifiedQualityScorer(quality_config, config.sources);

  // Log quality scoring inputs
  if (kalman_prediction !== null) {
    _log(`  kalman_prediction: ${_formatNum(kalman_prediction)}`);
  }
  if (innovation_covariance !== null) {
    _log(`  innovation_covariance: ${_formatNum(innovation_covariance)}`);
  }
  if (previous_weight !== null) {
    _log(`  previous_weight: ${_formatNum(previous_weight)}`);
  }
  if (time_diff_hours !== null) {
    _log(`  time_diff_hours: ${_formatNum(time_diff_hours)}`);
  }

  // Add current timestamp to kalman_state for time-based decay calculation
  const kalman_state_with_timestamp = { ...state, current_timestamp: timestamp };

  // Calculate quality score
  const quality_score = unified_scorer.calculateQualityScore({
    weight: cleaned_weight,
    source,
    kalmanState: kalman_state_with_timestamp,
    kalmanPrediction: kalman_prediction,
    innovationCovariance: innovation_covariance,
    previousWeight: previous_weight,
    timeDiffHours: time_diff_hours,
    recentWeights: recent_weights,
    recentTimestamps: recent_timestamps,
    userHeightM: user_height
  });

  _log(`  quality_score.overall: ${_formatNum(quality_score.overall)}`);
  _log(`  quality_components: ${JSON.stringify(quality_score.components)}`);

  if (!quality_score.accepted) {
    _log(`  Rejected by quality scorer: ${quality_score.rejectionReason}`);
    _log("Result: REJECTED");
    _log(`  stage: unified_quality_scoring`);
    _log("=".repeat(80));
    return {
      accepted: false,
      timestamp,
      raw_weight: weight,
      cleaned_weight,
      source,
      reason: quality_score.rejectionReason,
      stage: 'unified_quality_scoring',
      quality_score: quality_score.overall,
      quality_components: quality_score.components,
      quality_details: quality_score.toDict()
    } as ProcessResult;
  }

  // Store quality score for later use
  const quality_score_value = quality_score.overall;
  const quality_components = quality_score.components;

  // Only do Kalman update if not already done during initialization
  if (!kalman_already_updated) {
    _log("Step 6: Kalman update");
    // Check if we should use adaptive parameters (within 7 days of reset)
    const reset_timestamp = get_reset_timestamp(state);
    const adaptive_kalman_config = get_adaptive_kalman_params(
      reset_timestamp,
      timestamp,
      kalman_config,
      7, // adaptive_days
      state
    );

    // Update state's kalman_params with adaptive values
    if (
      reset_timestamp &&
      (timestamp.getTime() - reset_timestamp.getTime()) / (86400 * 1000) < 7
    ) {
      // Only update if we have the adaptive values
      if (
        adaptive_kalman_config.transition_covariance_weight !== undefined &&
        adaptive_kalman_config.transition_covariance_trend !== undefined
      ) {
        state.kalmanParams!.transition_covariance = [
          [adaptive_kalman_config.transition_covariance_weight, 0],
          [0, adaptive_kalman_config.transition_covariance_trend]
        ];
      }
    }

    const noise_multiplier = getNoiseMultiplier(source, config.sources);
    const observation_covariance =
      (adaptive_kalman_config.observation_covariance || 3.49) * noise_multiplier;

    _log(`  Using adaptive parameters: Q_weight=${_formatNum(adaptive_kalman_config.transition_covariance_weight)}, Q_trend=${_formatNum(adaptive_kalman_config.transition_covariance_trend)}`);
    _log(`  observation_covariance: ${_formatNum(observation_covariance)}`);

    // Apply trend limiting before update
    let [current_weight, current_trend] = KalmanFilterManager.getCurrentStateValues(state);
    if (state.lastState) {
      _log(`  State before update: ${_formatVec(state.lastState)}`);
    }
    if (current_trend !== null) {
      _log(`  Trend before limiting: ${_formatNum(current_trend)}`);
      // Limit trend to ±5kg/week (±0.714kg/day)
      const max_daily_trend = 0.714; // 5kg/week
      if (Math.abs(current_trend) > max_daily_trend) {
        // Clamp the trend in the state before update
        const limited_trend = current_trend > 0 ? max_daily_trend : -max_daily_trend;
        if (state.lastState !== null && state.lastState !== undefined) {
          const last_state = state.lastState;
          if (Array.isArray(last_state) && last_state.length >= 2) {
            // Handle both 1D and 2D arrays
            if (Array.isArray(last_state[0])) {
              // 2D array
              (last_state as number[][])[last_state.length - 1][1] = limited_trend;
            } else {
              // 1D array
              (last_state as number[])[1] = limited_trend;
            }
          }
        }
      }
    }

    state = KalmanFilterManager.updateState(
      state,
      cleaned_weight,
      timestamp,
      source,
      {},
      observation_covariance
    );

    if (state.lastState) {
      _log(`  State after update: ${_formatVec(state.lastState)}`);
    }

    // Apply trend limiting after update
    [current_weight, current_trend] = KalmanFilterManager.getCurrentStateValues(state);
    if (current_trend !== null) {
      _log(`  Trend after update (before limiting): ${_formatNum(current_trend)}`);
    }
    if (current_trend !== null && Math.abs(current_trend) > 0.714) {
      _log(`  Limiting trend from ${_formatNum(current_trend)} to ±0.714`);
      // Clamp the trend after update
      const limited_trend = current_trend > 0 ? 0.714 : -0.714;
      if (state.lastState !== null && state.lastState !== undefined) {
        const last_state = state.lastState;
        if (Array.isArray(last_state) && last_state.length >= 2) {
          // Handle both 1D and 2D arrays
          if (Array.isArray(last_state[0])) {
            // 2D array
            (last_state as number[][])[last_state.length - 1][1] = limited_trend;
          } else {
            // 1D array
            (last_state as number[])[1] = limited_trend;
          }
        }
      }
    }

    result = KalmanFilterManager.createResult(
      state,
      cleaned_weight,
      timestamp,
      source,
      true,
      observation_covariance
    );
  }

  // Ensure result exists
  if (!result) {
    throw new Error('Result should be set by now');
  }

  // Step 8: Add comprehensive metadata
  result.preprocessing = preprocess_metadata;
  // Set noise multiplier if not already set
  if (result.noise_multiplier === undefined) {
    result.noise_multiplier = getNoiseMultiplier(source, config.sources);
  }
  result.stage = 'accepted';

  // Add quality score if available
  // Always add quality score for accepted measurements
  result.quality_score = quality_score_value;
  result.quality_components = quality_components;

  // Add reset event info if it occurred
  if (reset_occurred && reset_event) {
    result.reset_event = {
      type: reset_event.type || 'unknown',
      gap_days: reset_event.gap_days,
      reason: reset_event.reason || 'unknown'
    };
  }

  // Calculate BMI details
  const implied_bmi = cleaned_weight / (user_height ** 2);
  result.bmi_details = {
    user_height_m: user_height,
    implied_bmi: Math.round(implied_bmi * 10) / 10,
    original_weight: weight,
    original_unit: unit,
    cleaned_weight
  };

  // Update measurement history for quality scoring
  if (!state.measurementHistory) {
    state.measurementHistory = [];
  }

  state.measurementHistory.push({
    weight: cleaned_weight,
    timestamp: timestamp.toISOString(),
    quality_score: quality_score_value,
    source
  });

  // Keep only last 30 measurements
  state.measurementHistory = state.measurementHistory.slice(-30);

  // Save updated state - Main successful processing path
  // Increment measurements counter for adaptation tracking
  state.measurementsSinceReset = (state.measurementsSinceReset || 0) + 1;
  state.lastSource = source;
  state.lastTimestamp = timestamp; // Keep for backward compatibility
  state.lastAcceptedTimestamp = timestamp;
  state.lastRawWeight = cleaned_weight; // Track for soft reset detection

  // Update temporal baseline for continuous tracking
  state = unified_scorer.update_temporal_baseline(state, cleaned_weight, timestamp);

  // Validate state before persistence
  const [is_valid, error_msg] = PersistenceValidator.validateState(
    state,
    userId,
    'successful_processing'
  );

  if (is_valid) {
    // Get previous state for change detection
    const previous_state = db.get_state(userId);
    const [should_persist, audit_msg] = PersistenceValidator.shouldPersist(
      state,
      previous_state,
      userId,
      'successful_processing'
    );

    if (should_persist) {
      db.save_state(userId, state);
      PersistenceValidator.createAuditLog(
        userId,
        'persist',
        state,
        true,
        'successful_processing',
        null
      );

      // Save snapshot after reset for replay functionality
      if (reset_occurred) {
        try {
          db.save_state_snapshot(userId, timestamp);
        } catch (e) {
          const error = e as Error;
          console.warn(`Failed to save post-reset snapshot for ${userId}: ${error.message}`);
          // Continue processing even if snapshot fails
        }
      }

      // Create periodic snapshot if interval has elapsed
      // This ensures replay has adequate state history even without resets
      _maybe_create_periodic_snapshot(db, userId, timestamp, config);
    } else {
      // Log why we're not persisting
      PersistenceValidator.createAuditLog(
        userId,
        'skip',
        state,
        true,
        audit_msg,
        null
      );
    }
  } else {
    // Log validation failure
    PersistenceValidator.create_audit_log(
      userId,
      'validate_failed',
      state,
      false,
      'successful_processing',
      error_msg
    );
  }

  // Log final result
  _log(`Result: ${result.accepted ? 'ACCEPTED' : 'REJECTED'}`);
  if (result.kalman_estimate !== undefined && result.kalman_estimate !== null) {
    _log(`  kalman_estimate: ${_formatNum(result.kalman_estimate)}`);
  }
  if (result.kalman_uncertainty !== undefined && result.kalman_uncertainty !== null) {
    _log(`  kalman_uncertainty: ${_formatNum(result.kalman_uncertainty)}`);
  }
  if (result.quality_score !== undefined && result.quality_score !== null) {
    _log(`  quality_score: ${_formatNum(result.quality_score)}`);
  }
  _log(`  stage: ${result.stage || 'unknown'}`);
  _log("=".repeat(80));

  return result;
}

/**
 * Backward compatibility wrapper for processMeasurement.
 */
export async function process_weight_enhanced(
  userId: string,
  weight: number,
  timestamp: Date,
  source: string,
  processing_config: Record<string, any>,
  kalman_config: Record<string, any>,
  db: StateStore,
  unit: string = 'kg'
): Promise<ProcessResult> {
  const config: Config = {
    processing: processing_config,
    kalman: kalman_config
  } as Config;

  // Handle nested config for adaptive noise
  if ('config' in processing_config) {
    Object.assign(config, processing_config.config);
  }

  return await processMeasurement(userId, weight, timestamp, source, config, db, unit);
}
