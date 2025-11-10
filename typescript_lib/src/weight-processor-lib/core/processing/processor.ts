/**
 * Simplified weight processor with flattened pipeline.
 * Single processing function with clear flow.
 *
 * TypeScript port of python_lib/src/weight_processor_lib/core/processing/processor.py
 */

import { KALMAN_DEFAULTS, PHYSIOLOGICAL_LIMITS, getNoiseMultiplier } from '../constants.js';
import type { KalmanState } from '../database/base.js';
import {
  KalmanFilterManager,
  getAdaptiveKalmanParams,
  getResetTimestamp,
  KalmanResult,
  ResetParameters,
  ResetEvent
} from './kalman.js';
import { ResetType } from './reset_manager.js';
import { DataQualityPreprocessor } from './validation.js';
import { UnifiedQualityScorer, QualityScore } from './unified_quality_scorer.js';
import { ensureFloat, ensureNumericTypes, deserializeState } from './type_conversion.js';
import { StateStore } from '../database/base.js';

// TODO: These modules need to be ported from Python
// For now, we'll use simplified implementations

/**
 * Circuit breaker state enum
 */
enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

/**
 * Circuit breaker error
 */
class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

/**
 * Simple circuit breaker implementation
 * TODO: Port full implementation from circuit_breaker.py
 */
class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private successCount: number = 0;
  private lastFailureTime: Date | null = null;
  private lastError: Error | null = null;

  constructor(
    private failureThreshold: number = 3,
    private timeout: number = 60,
    private successThreshold: number = 2,
    private name: string = 'circuit'
  ) {}

  call<T>(func: (...args: any[]) => T, ...args: any[]): T {
    if (this.state === CircuitState.OPEN) {
      if (this.shouldAttemptRecovery()) {
        this.state = CircuitState.HALF_OPEN;
        this.successCount = 0;
      } else {
        throw new CircuitOpenError(
          `Circuit '${this.name}' open due to ${this.failureCount} failures`
        );
      }
    }

    try {
      const result = func(...args);
      this.onSuccess();
      return result;
    } catch (e) {
      this.onFailure(e as Error);
      throw e;
    }
  }

  private shouldAttemptRecovery(): boolean {
    if (!this.lastFailureTime) return false;
    const now = new Date();
    const elapsed = (now.getTime() - this.lastFailureTime.getTime()) / 1000;
    return elapsed >= this.timeout;
  }

  private onSuccess(): void {
    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;
      if (this.successCount >= this.successThreshold) {
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
      }
    } else {
      this.failureCount = 0;
    }
  }

  private onFailure(error: Error): void {
    this.failureCount++;
    this.lastFailureTime = new Date();
    this.lastError = error;

    if (this.failureCount >= this.failureThreshold) {
      this.state = CircuitState.OPEN;
    }
  }
}

/**
 * Simple persistence validator
 * TODO: Port full implementation from persistence_validator.py
 */
class PersistenceValidator {
  private static readonly REQUIRED_FIELDS = new Set(['last_state', 'kalman_params', 'last_timestamp']);
  private static readonly MAX_WEIGHT_KG = 500;
  private static readonly MIN_WEIGHT_KG = 10;

  static validateState(
    state: KalmanState,
    userId: string,
    reason: string = 'unknown'
  ): [boolean, string | null] {
    if (!state) {
      return [false, 'State is None or empty'];
    }

    // Check required fields
    for (const field of this.REQUIRED_FIELDS) {
      if (!(field in state)) {
        return [false, `Missing required field: ${field}`];
      }
    }

    // Validate weight if present
    if (state.last_raw_weight !== undefined && state.last_raw_weight !== null) {
      if (state.last_raw_weight < this.MIN_WEIGHT_KG || state.last_raw_weight > this.MAX_WEIGHT_KG) {
        return [false, `Weight ${state.last_raw_weight} outside valid range`];
      }
    }

    return [true, null];
  }

  static shouldPersist(
    state: KalmanState,
    previousState: KalmanState | null,
    userId: string,
    reason: string = 'unknown'
  ): [boolean, string] {
    // Simple implementation - always persist valid state
    return [true, reason];
  }

  static createAuditLog(
    userId: string,
    action: string,
    state: KalmanState,
    success: boolean,
    reason: string,
    error: string | null
  ): void {
    // Simple implementation - just log
    console.debug(`[PersistenceValidator] ${action} for ${userId}: ${reason}${error ? ` - ${error}` : ''}`);
  }
}

/**
 * Simple reset manager
 * TODO: Port full implementation from reset_manager.py
 */
class ResetManager {
  static shouldTriggerReset(
    state: KalmanState,
    weight: number,
    timestamp: Date,
    source: string,
    config: Record<string, any>
  ): ResetType | null {
    // Simple implementation - check for time gaps
    if (!state.last_timestamp) {
      return ResetType.INITIAL;
    }

    const gapDays = (timestamp.getTime() - state.last_timestamp.getTime()) / (1000 * 60 * 60 * 24);
    const resetConfig = config.reset || {};
    const timeGapDays = resetConfig.time_gap_days || 30;

    if (gapDays >= timeGapDays) {
      return ResetType.HARD;
    }

    return null;
  }

  static performReset(
    state: KalmanState,
    resetType: ResetType,
    timestamp: Date,
    weight: number,
    source: string,
    config: Record<string, any>
  ): [KalmanState, ResetEvent] {
    const newState = { ...state };

    // Clear Kalman state
    newState.kalman_params = null;
    newState.last_state = undefined;
    newState.last_covariance = undefined;
    newState.measurements_since_reset = 0;
    newState.reset_timestamp = timestamp;
    newState.reset_type = resetType;

    // FIX: Add reset_parameters from config with defaults
    // Note: config structure is { kalman: { reset: { hard: {...}, soft: {...} } } }
    const kalmanResetConfig = config.kalman?.reset || config.reset || {};
    const resetConfigFromFile = kalmanResetConfig[resetType] || {};

    // Merge with defaults (match Python's get_reset_parameters behavior)
    const defaultResetParams: Record<string, any> = {
      initial: {
        initial_variance_multiplier: 10,
        weight_noise_multiplier: 50,
        trend_noise_multiplier: 500,
        observation_noise_multiplier: 0.3,
        adaptation_measurements: 20,
        adaptation_days: 21,
        adaptation_decay_rate: 1.5,
      },
      hard: {
        initial_variance_multiplier: 5,
        weight_noise_multiplier: 20,
        trend_noise_multiplier: 200,
        observation_noise_multiplier: 0.5,
        adaptation_measurements: 10,
        adaptation_days: 7,
        adaptation_decay_rate: 2.5,
      },
      soft: {
        initial_variance_multiplier: 2,
        weight_noise_multiplier: 5,
        trend_noise_multiplier: 20,
        observation_noise_multiplier: 0.7,
        adaptation_measurements: 15,
        adaptation_days: 10,
        adaptation_decay_rate: 4,
      },
    };

    const defaults = defaultResetParams[resetType] || {};
    const resetConfig = { ...defaults, ...resetConfigFromFile };
    newState.reset_parameters = resetConfig;

    const resetEvent: ResetEvent = {
      timestamp,
      type: resetType,
      source,
      weight,
      last_weight: state.last_raw_weight || undefined,
      gap_days: state.last_timestamp
        ? (timestamp.getTime() - state.last_timestamp.getTime()) / (1000 * 60 * 60 * 24)
        : undefined,
      reason: `${resetType} reset triggered`,
      parameters: resetConfig as ResetParameters,
    };

    return [newState, resetEvent];
  }
}

/**
 * Reset operation enum
 */
enum ResetOperation {
  STATE_UPDATE = 'state_update',
  KALMAN_RESET = 'kalman_reset',
}

/**
 * Simple reset transaction
 * TODO: Port full implementation from reset_transaction.py
 */
class ResetTransaction {
  private originalStates: Map<ResetOperation, any> = new Map();
  private checkpoints: Map<ResetOperation, any> = new Map();
  private completed: Set<ResetOperation> = new Set();
  private rolledBack: boolean = false;

  constructor(private userId: string) {}

  saveOriginalState(operation: ResetOperation, state: any): void {
    this.originalStates.set(operation, JSON.parse(JSON.stringify(state)));
  }

  saveCheckpoint(operation: ResetOperation, state: any): void {
    this.checkpoints.set(operation, JSON.parse(JSON.stringify(state)));
  }

  validateCheckpoint(operation: ResetOperation): boolean {
    const checkpoint = this.checkpoints.get(operation);
    return checkpoint !== null && checkpoint !== undefined;
  }

  markCompleted(operation: ResetOperation): void {
    this.completed.add(operation);
  }

  getOriginalState(operation: ResetOperation): any {
    return this.originalStates.get(operation);
  }

  rollback(): void {
    this.rolledBack = true;
  }
}

/**
 * Processing result interface
 */
export interface ProcessingResult {
  accepted: boolean;
  rejected?: boolean;
  timestamp: Date;
  source: string;
  raw_weight: number;
  cleaned_weight?: number;
  filtered_weight?: number;
  trend?: number;
  trend_weekly?: number;
  confidence?: number;
  innovation?: number;
  normalized_innovation?: number;
  kalman_confidence_upper?: number;
  kalman_confidence_lower?: number;
  kalman_variance?: number | null;
  prediction_error?: number | null;
  reason?: string;
  stage?: string;
  metadata?: Record<string, any>;
  preprocessing?: Record<string, any>;
  noise_multiplier?: number;
  quality_score?: number;
  quality_components?: Record<string, number>;
  quality_details?: Record<string, any>;
  reset_event?: {
    type: string;
    gap_days?: number;
    reason: string;
  };
  was_reset?: boolean;
  reset_reason?: string;
  reset_type?: string;
  gap_days?: number;
  bmi_details?: {
    user_height_m: number;
    implied_bmi: number;
    original_weight: number;
    original_unit: string;
    cleaned_weight: number;
  };
  warning?: string;
}

/**
 * Normalize state arrays to correct shape (Matrix objects).
 * With Matrix objects, we just ensure we have exactly 2 items in each array.
 */
function normalizeStateArrays(state: KalmanState): KalmanState {
  if (state.last_state) {
    const lastState = state.last_state;

    // If only 1 Matrix, duplicate it
    if (lastState.length === 1) {
      state.last_state = [lastState[0].clone(), lastState[0].clone()];
    }
    // If >2 matrices, keep only last 2
    else if (lastState.length > 2) {
      state.last_state = lastState.slice(-2);
    }
  }

  if (state.last_covariance) {
    const lastCov = state.last_covariance;

    // If only 1 matrix, duplicate it
    if (lastCov.length === 1) {
      state.last_covariance = [lastCov[0].clone(), lastCov[0].clone()];
    }
    // If >2 matrices, keep only last 2
    else if (lastCov.length > 2) {
      state.last_covariance = lastCov.slice(-2);
    }
  }

  return state;
}

/**
 * Validate state Matrix shapes.
 */
function validateStateShapes(state: KalmanState, userId: string): void {
  if (state.last_state) {
    if (state.last_state.length !== 2) {
      throw new Error(`Invalid last_state array length for ${userId}: ${state.last_state.length}, expected 2`);
    }
    // Check that each is a Matrix with correct dimensions (2x1 column vector)
    for (let i = 0; i < state.last_state.length; i++) {
      const mat = state.last_state[i];
      if (mat.rows !== 2 || mat.columns !== 1) {
        throw new Error(`Invalid last_state Matrix shape for ${userId}: [${mat.rows}, ${mat.columns}], expected [2, 1]`);
      }
    }
  }

  if (state.last_covariance) {
    if (state.last_covariance.length !== 2) {
      throw new Error(`Invalid last_covariance array length for ${userId}: ${state.last_covariance.length}, expected 2`);
    }
    // Check that each is a Matrix with correct dimensions (2x2)
    for (let i = 0; i < state.last_covariance.length; i++) {
      const mat = state.last_covariance[i];
      if (mat.rows !== 2 || mat.columns !== 2) {
        throw new Error(`Invalid last_covariance Matrix shape for ${userId}: [${mat.rows}, ${mat.columns}], expected [2, 2]`);
      }
    }
  }
}

/**
 * Create periodic snapshot if interval has elapsed.
 */
async function maybeCreatePeriodicSnapshot(
  db: StateStore,
  userId: string,
  timestamp: Date,
  config: Record<string, any>
): Promise<boolean> {
  try {
    const snapshotConfig = config.snapshot || {};
    const snapshotIntervalHours = snapshotConfig.interval_hours || 24;
    const periodicEnabled = snapshotConfig.periodic_enabled !== false;

    if (!periodicEnabled) {
      return false;
    }

    // Get latest snapshot
    const latestSnapshot = await db.getLatestSnapshot(userId);

    // Create snapshot if none exists
    if (!latestSnapshot) {
      await db.saveStateSnapshot(userId, timestamp);
      console.debug(`Created initial periodic snapshot for user ${userId}`);
      return true;
    }

    // Check time since last snapshot
    let lastSnapshotTime = latestSnapshot.last_timestamp;
    if (!lastSnapshotTime) {
      await db.saveStateSnapshot(userId, timestamp);
      console.debug(`Created periodic snapshot for user ${userId} (no timestamp in last snapshot)`);
      return true;
    }

    // Ensure lastSnapshotTime is a Date
    if (typeof lastSnapshotTime === 'string') {
      lastSnapshotTime = new Date(lastSnapshotTime);
    }

    // Calculate hours since last snapshot
    const hoursSince = (timestamp.getTime() - lastSnapshotTime.getTime()) / (1000 * 60 * 60);

    // Create snapshot if interval elapsed
    if (hoursSince >= snapshotIntervalHours) {
      await db.saveStateSnapshot(userId, timestamp);
      console.debug(`Created periodic snapshot for user ${userId} (${hoursSince.toFixed(1)} hours since last)`);
      return true;
    }

    return false;
  } catch (e) {
    console.warn(`Failed to create periodic snapshot for ${userId}:`, e);
    return false;
  }
}

/**
 * Handle reset with transaction safety and circuit breaker.
 */
function handleResetWithTransaction(
  userId: string,
  state: KalmanState,
  resetType: ResetType,
  timestamp: Date,
  weight: number,
  source: string,
  config: Record<string, any>,
  circuitBreaker: CircuitBreaker
): [KalmanState, ResetEvent | null, boolean] {
  try {
    return circuitBreaker.call(
      performTransactionalReset,
      userId,
      state,
      resetType,
      timestamp,
      weight,
      source,
      config
    );
  } catch (e) {
    if (e instanceof CircuitOpenError) {
      console.error(`Reset circuit open for user ${userId}:`, e);
    } else {
      console.error(`Reset failed for user ${userId}:`, e);
    }
    return [state, null, false];
  }
}

/**
 * Perform reset with transaction management.
 */
function performTransactionalReset(
  userId: string,
  state: KalmanState,
  resetType: ResetType,
  timestamp: Date,
  weight: number,
  source: string,
  config: Record<string, any>
): [KalmanState, ResetEvent | null, boolean] {
  const txn = new ResetTransaction(userId);

  try {
    // Save original state
    txn.saveOriginalState(ResetOperation.STATE_UPDATE, state);

    // Perform reset
    console.info(`Applying ${resetType} reset for user ${userId}`);

    const [newState, resetEvent] = ResetManager.performReset(
      state,
      resetType,
      timestamp,
      weight,
      source,
      config
    );

    // Save checkpoint and validate
    txn.saveCheckpoint(ResetOperation.STATE_UPDATE, newState);
    if (!txn.validateCheckpoint(ResetOperation.STATE_UPDATE)) {
      throw new Error(`State validation failed after ${resetType} reset`);
    }
    txn.markCompleted(ResetOperation.STATE_UPDATE);

    // Validate Kalman reset
    const kalmanState = {
      kalman_params: newState.kalman_params,
      reset_parameters: newState.reset_parameters,
      measurements_since_reset: newState.measurements_since_reset || 0,
      reset_type: newState.reset_type,
      reset_timestamp: newState.reset_timestamp,
    };

    txn.saveCheckpoint(ResetOperation.KALMAN_RESET, kalmanState);
    if (!txn.validateCheckpoint(ResetOperation.KALMAN_RESET)) {
      throw new Error('Kalman state validation failed after reset');
    }
    txn.markCompleted(ResetOperation.KALMAN_RESET);

    console.info(`Reset transaction completed successfully for user ${userId}`);
    return [newState, resetEvent, true];
  } catch (e) {
    console.error(`Reset transaction failed for user ${userId}:`, e);
    txn.rollback();

    const originalState = txn.getOriginalState(ResetOperation.STATE_UPDATE);
    if (originalState) {
      return [originalState, null, false];
    }
    return [state, null, false];
  }
}

// Circuit breaker for reset operations (module level)
const resetCircuitBreaker = new CircuitBreaker(3, 60, 2, 'reset_operations');

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
  config: Record<string, any>,
  unit: string = 'kg',
  db: StateStore | null = null,
  userHeightM: number | null = null
): Promise<ProcessingResult> {
  // Ensure weight is a number
  weight = ensureFloat(weight);

  if (!db) {
    throw new Error('Database instance required');
  }

  // Step 1: Data cleaning and preprocessing
  const [cleanedWeight, preprocessMetadata] = DataQualityPreprocessor.preprocess(
    weight,
    source,
    timestamp,
    userId,
    unit,
    userHeightM
  );

  // If preprocessing rejected the measurement
  if (cleanedWeight === null) {
    return {
      accepted: false,
      rejected: true,
      timestamp,
      source,
      raw_weight: weight,
      reason: preprocessMetadata.rejected || 'Preprocessing failed',
      stage: 'preprocessing',
      metadata: preprocessMetadata,
      quality_score: 0.0,  // Rejected during preprocessing
    };
  }

  // Step 2: Load or create user state
  let state: KalmanState = await db.getState(userId) || db.createInitialState();

  // Deserialize date fields if state was loaded from DB (JSON serialization converts Dates to strings)
  state = deserializeState(state);

  // If state was loaded from DB, normalize and validate it
  if (state.kalman_params || state.last_state) {
    // Ensure all numeric values are proper types
    state = ensureNumericTypes(state);
    // Normalize arrays to standard (2,2) shape
    state = normalizeStateArrays(state);
    // Validate shapes are correct
    validateStateShapes(state, userId);
  }

  // Use provided height or default
  const userHeight = userHeightM !== null ? userHeightM : PHYSIOLOGICAL_LIMITS.DEFAULT_HEIGHT_M;

  // Step 3: Check for any type of reset
  const kalmanConfig = config.kalman || {};

  const resetType = ResetManager.shouldTriggerReset(
    state,
    cleanedWeight,
    timestamp,
    source,
    config
  );

  let resetEvent: ResetEvent | null = null;
  let resetOccurred = false;

  if (resetType) {
    [state, resetEvent, resetOccurred] = handleResetWithTransaction(
      userId,
      state,
      resetType,
      timestamp,
      cleanedWeight,
      source,
      config,
      resetCircuitBreaker
    );
  }

  // Step 4: Initialize Kalman if needed
  let kalmanAlreadyUpdated = false;
  let result: KalmanResult | null = null;

  if (!state.kalman_params) {
    // Check if this is a post-reset initialization
    // FIX: Use state.reset_timestamp directly instead of getResetTimestamp
    // which relies on reset_events that may not be populated yet
    const resetTimestamp = resetOccurred ? state.reset_timestamp : timestamp;

    // Get adaptive Kalman config if within post-reset period
    const adaptiveKalmanConfig = getAdaptiveKalmanParams(
      resetTimestamp,
      timestamp,
      kalmanConfig,
      7,
      state
    );

    // Get adaptive noise for this source (NO hardcoded defaults)
    const adaptiveConfig = config.adaptive_noise || {};
    const noiseMultiplier = getNoiseMultiplier(source);

    if (!adaptiveKalmanConfig.observation_covariance) {
      throw new Error('observation_covariance missing from Kalman config. Must be loaded from config.json.');
    }
    const observationCovariance = adaptiveKalmanConfig.observation_covariance * noiseMultiplier;

    const kalmanState = KalmanFilterManager.initializeImmediate(
      cleanedWeight,
      timestamp,
      adaptiveKalmanConfig,
      observationCovariance
    );

    // Merge Kalman state with existing state
    Object.assign(state, kalmanState);

    result = KalmanFilterManager.createResult(
      state,
      cleanedWeight,
      timestamp,
      source,
      true,
      observationCovariance
    );

    // Add metadata
    const processingResult: ProcessingResult = {
      ...(result || {}),
      accepted: true,
      timestamp,
      source,
      raw_weight: weight,
      stage: 'initialization',
      preprocessing: preprocessMetadata,
      noise_multiplier: noiseMultiplier,
      quality_score: 0.8,  // Initial measurement, will be updated by quality scoring
    };

    // Add reset event info if it occurred
    if (resetOccurred && resetEvent) {
      processingResult.was_reset = true;
      processingResult.reset_reason = resetEvent.reason;
      processingResult.reset_type = resetEvent.type;
      processingResult.gap_days = resetEvent.gap_days || 0;
      processingResult.reset_event = {
        type: resetEvent.type,
        gap_days: resetEvent.gap_days,
        reason: resetEvent.reason,
      };
    }

    kalmanAlreadyUpdated = true;

    // Continue to quality validation - store result for later use
  }

  // Step 5: Quality scoring
  const processingConfig = config.processing || {};
  const qualityConfig = config.quality_scoring || {};

  // Get previous weight and time diff
  let previousWeight: number | null = null;
  let timeDiffHours: number | null = null;

  if (state) {
    const [currentWeight] = KalmanFilterManager.getCurrentStateValues(state);
    if (process.env.VERBOSE_LOGGING) {
      console.log(`[Processor] getCurrentStateValues returned: ${currentWeight}`);
    }
    if (currentWeight !== null) {
      previousWeight = currentWeight;
    } else if (state.last_raw_weight !== undefined && state.last_raw_weight !== null) {
      previousWeight = ensureFloat(state.last_raw_weight);
    }

    // Get time diff
    if (state.last_timestamp) {
      let prevTime = state.last_timestamp;
      if (typeof prevTime === 'string') {
        prevTime = new Date(prevTime);
      }
      timeDiffHours = (timestamp.getTime() - prevTime.getTime()) / (1000 * 60 * 60);
    }
  }

  // Get recent weights for statistical analysis
  const recentWeights: number[] = [];
  if (state && state.measurement_history) {
    const history = state.measurement_history;
    if (Array.isArray(history)) {
      history.slice(-20).forEach((h: any) => {
        if (h.weight !== undefined) {
          recentWeights.push(ensureFloat(h.weight));
        }
      });
    }
  }

  // Get Kalman prediction
  let kalmanPrediction: number | null = null;
  let innovationCovariance: number | null = null;

  if (state && state.kalman_params) {
    [kalmanPrediction, innovationCovariance] = KalmanFilterManager.predictNextState(state, timestamp);

    if (process.env.VERBOSE_LOGGING) {
      console.log(`[Processor] predictNextState returned: prediction=${kalmanPrediction}, covariance=${innovationCovariance}`);
    }

    // Apply source-specific noise multiplier to innovation covariance
    if (innovationCovariance !== null) {
      const noiseMultiplier = getNoiseMultiplier(source);
      if (noiseMultiplier !== 1.0) {
        const kalmanParams = state.kalman_params;
        const baseObsCov = kalmanParams.observation_covariance?.[0]?.[0] || KALMAN_DEFAULTS.observation_covariance;
        const predictedCov00 = innovationCovariance - ensureFloat(baseObsCov);
        innovationCovariance = predictedCov00 + (ensureFloat(baseObsCov) * noiseMultiplier);
      }
    }
  }

  // Get recent timestamps
  const recentTimestamps: Date[] = [];
  if (state && state.measurement_history) {
    const history = state.measurement_history;
    if (Array.isArray(history)) {
      history.slice(-20).forEach((h: any) => {
        if (h.timestamp) {
          const ts = typeof h.timestamp === 'string' ? new Date(h.timestamp) : h.timestamp;
          recentTimestamps.push(ts);
        }
      });
    }
  }

  // Create unified scorer instance
  const unifiedScorer = new UnifiedQualityScorer(qualityConfig);

  // Add current timestamp to kalman_state for time-based decay calculation
  const kalmanStateWithTimestamp = state ? { ...state, current_timestamp: timestamp } : {} as any;

  // Calculate quality score
  const qualityScore = unifiedScorer.calculateQualityScore({
    weight: cleanedWeight,
    source,
    kalmanState: kalmanStateWithTimestamp,
    kalmanPrediction: kalmanPrediction ?? undefined,
    innovationCovariance: innovationCovariance ?? undefined,
    previousWeight: previousWeight ?? undefined,
    timeDiffHours: timeDiffHours ?? undefined,
    recentWeights,
    recentTimestamps,
    userHeightM: userHeight
  });

  if (!qualityScore.accepted) {
    return {
      accepted: false,
      timestamp,
      raw_weight: weight,
      cleaned_weight: cleanedWeight,
      source,
      reason: qualityScore.rejectionReason || 'Quality check failed',
      stage: 'unified_quality_scoring',
      quality_score: qualityScore.overall,
      quality_components: qualityScore.components,
      quality_details: qualityScore,
    };
  }

  // Store quality score for later use
  const qualityScoreValue = qualityScore.overall;
  const qualityComponents = qualityScore.components;

  // Only do Kalman update if not already done during initialization
  if (!kalmanAlreadyUpdated) {
    // Check if we should use adaptive parameters
    const resetTimestamp = getResetTimestamp(state);
    const adaptiveKalmanConfig = getAdaptiveKalmanParams(
      resetTimestamp,
      timestamp,
      kalmanConfig,
      7,
      state
    );

    // Update state's kalman_params with adaptive values if within 7 days of reset
    if (resetTimestamp) {
      const daysSinceReset = (timestamp.getTime() - resetTimestamp.getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceReset < 7) {
        if (
          'transition_covariance_weight' in adaptiveKalmanConfig &&
          'transition_covariance_trend' in adaptiveKalmanConfig
        ) {
          state.kalman_params!.transition_covariance = [
            [adaptiveKalmanConfig.transition_covariance_weight, 0],
            [0, adaptiveKalmanConfig.transition_covariance_trend],
          ];
        }
      }
    }

    const noiseMultiplier = getNoiseMultiplier(source);

    if (!adaptiveKalmanConfig.observation_covariance) {
      throw new Error('observation_covariance missing from Kalman config. Must be loaded from config.json.');
    }
    const observationCovariance = adaptiveKalmanConfig.observation_covariance * noiseMultiplier;

    // Apply trend limiting before update
    let [currentWeight, currentTrend] = KalmanFilterManager.getCurrentStateValues(state);
    if (currentTrend !== null) {
      const maxDailyTrend = 0.714; // 5kg/week
      if (Math.abs(currentTrend) > maxDailyTrend) {
        const limitedTrend = currentTrend > 0 ? maxDailyTrend : -maxDailyTrend;
        if (state.last_state) {
          // Update the trend in the last state Matrix
          const lastStateMatrix = state.last_state[state.last_state.length - 1];
          lastStateMatrix.set(1, 0, limitedTrend); // Set row 1, col 0
        }
      }
    }

    state = KalmanFilterManager.updateState(
      state,
      cleanedWeight,
      timestamp,
      source,
      {},
      observationCovariance
    );

    // Apply trend limiting after update
    [currentWeight, currentTrend] = KalmanFilterManager.getCurrentStateValues(state);
    if (currentTrend !== null && Math.abs(currentTrend) > 0.714) {
      const limitedTrend = currentTrend > 0 ? 0.714 : -0.714;
      if (state.last_state) {
        // Update the trend in the last state Matrix
        const lastStateMatrix = state.last_state[state.last_state.length - 1];
        lastStateMatrix.set(1, 0, limitedTrend); // Set row 1, col 0
      }
    }

    result = KalmanFilterManager.createResult(
      state,
      cleanedWeight,
      timestamp,
      source,
      true,
      observationCovariance
    );
  }

  // Step 6: Add comprehensive metadata
  const finalResult: ProcessingResult = {
    ...result!,
    preprocessing: preprocessMetadata,
    stage: 'accepted',
    quality_score: qualityScoreValue,
    quality_components: qualityComponents,
  };

  // Set noise multiplier if not already set
  if (!finalResult.noise_multiplier) {
    finalResult.noise_multiplier = getNoiseMultiplier(source);
  }

  // Add reset event info if it occurred (flattened for visualization + nested for compatibility)
  if (resetOccurred && resetEvent) {
    finalResult.was_reset = true;
    finalResult.reset_reason = resetEvent.reason;
    finalResult.reset_type = resetEvent.type;
    finalResult.gap_days = resetEvent.gap_days || 0;
    finalResult.reset_event = {
      type: resetEvent.type,
      gap_days: resetEvent.gap_days,
      reason: resetEvent.reason,
    };
  }

  // Calculate BMI details
  const impliedBmi = cleanedWeight / (userHeight ** 2);
  finalResult.bmi_details = {
    user_height_m: userHeight,
    implied_bmi: Math.round(impliedBmi * 10) / 10,
    original_weight: weight,
    original_unit: unit,
    cleaned_weight: cleanedWeight,
  };

  // Update measurement history
  if (!state.measurement_history) {
    state.measurement_history = [];
  }

  state.measurement_history.push({
    weight: cleanedWeight,
    timestamp: timestamp.toISOString(),
    quality_score: qualityScoreValue,
    source,
  });

  // Keep only last 30 measurements
  state.measurement_history = state.measurement_history.slice(-30);

  // Save updated state - Main successful processing path
  state.measurements_since_reset = (state.measurements_since_reset || 0) + 1;
  state.last_source = source;
  state.last_timestamp = timestamp;
  state.last_accepted_timestamp = timestamp;
  state.last_raw_weight = cleanedWeight;

  // Update temporal baseline for continuous tracking
  state = unifiedScorer.updateTemporalBaseline(state, cleanedWeight, timestamp) as KalmanState;

  // Validate state before persistence
  const [isValid, errorMsg] = PersistenceValidator.validateState(state, userId, 'successful_processing');

  if (isValid) {
    // Get previous state for change detection
    const previousState = await db.getState(userId);
    const [shouldPersist, auditMsg] = PersistenceValidator.shouldPersist(
      state,
      previousState,
      userId,
      'successful_processing'
    );

    if (shouldPersist) {
      await db.saveState(userId, state);
      PersistenceValidator.createAuditLog(
        userId,
        'persist',
        state,
        true,
        'successful_processing',
        null
      );

      // Save snapshot after reset for replay functionality
      if (resetOccurred) {
        try {
          await db.saveStateSnapshot(userId, timestamp);
          console.debug(`Saved post-reset snapshot for user ${userId} at ${timestamp}`);
        } catch (e) {
          console.warn(`Failed to save post-reset snapshot for ${userId}:`, e);
        }
      }

      // Create periodic snapshot if interval has elapsed
      await maybeCreatePeriodicSnapshot(db, userId, timestamp, config);
    } else {
      PersistenceValidator.createAuditLog(userId, 'skip', state, true, auditMsg, null);
    }
  } else {
    PersistenceValidator.createAuditLog(
      userId,
      'validate_failed',
      state,
      false,
      'successful_processing',
      errorMsg
    );
  }

  return finalResult;
}

/**
 * Backward compatibility wrapper for process_measurement.
 */
export async function processWeightEnhanced(
  userId: string,
  weight: number,
  timestamp: Date,
  source: string,
  processingConfig: Record<string, any>,
  kalmanConfig: Record<string, any>,
  unit: string = 'kg',
  db: StateStore | null = null
): Promise<ProcessingResult | null> {
  const config = {
    processing: processingConfig,
    kalman: kalmanConfig,
  };

  // Handle nested config for adaptive noise
  if ('config' in processingConfig) {
    Object.assign(config, processingConfig.config);
  }

  return processMeasurement(userId, weight, timestamp, source, config, unit, db);
}
