/**
 * Core data models for weight processing
 *
 * These are the internal models used by the processing engine.
 * API models (request/response) would be separate if needed.
 */

import type { Matrix2x2, Vector2 } from './core/math/matrix';

/**
 * Weight measurement input
 */
export interface Measurement {
  measurementId?: string;
  deviceId: string;
  userId: string;
  weightKg: number; // Always in kg internally
  timestamp: string | Date; // ISO string or Date object
  source: string;
  metadata?: MeasurementMetadata;
}

/**
 * Measurement metadata
 */
export interface MeasurementMetadata {
  qualityScore?: number;
  accepted?: boolean;
  rejectionReason?: string;
  rawWeight?: number;
  filteredWeight?: number;
  sourceReliability?: string;
  [key: string]: any; // Allow additional metadata
}

/**
 * Kalman filter parameters
 */
export interface KalmanParams {
  initialStateMean: Vector2;
  initialStateCovariance: Matrix2x2;
  transitionCovariance: Matrix2x2;
  observationCovariance: number[][];  // 1x1 matrix (stored as [[value]])
}

/**
 * Kalman filter state
 */
export interface KalmanState {
  x: Vector2; // State vector [weight, velocity]
  P: Matrix2x2; // Covariance matrix
  lastTimestamp: string;
  measurementsCount: number;
  F?: Matrix2x2; // Transition matrix (optional, can be computed)
  H?: Matrix2x2; // Observation matrix (optional, usually fixed)
  Q?: Matrix2x2; // Process noise covariance (optional, from config)
  R?: number; // Measurement noise covariance (optional, from config)
}

/**
 * Reset event information
 */
export interface ResetEvent {
  resetType: ResetType;
  resetReason: string;
  timestamp: string;
  previousWeight?: number;
  newWeight?: number;
  gapDays?: number;
  weightChangeKg?: number;
  metadata?: Record<string, any>;
}

/**
 * Reset types
 */
export enum ResetType {
  INITIAL = 'initial',
  HARD = 'hard',
  SOFT = 'soft',
  NONE = 'none',
}

/**
 * Reset parameters for Kalman filter adaptation
 */
export interface ResetParameters {
  initialVarianceMultiplier: number;
  weightNoiseMultiplier: number;
  trendNoiseMultiplier: number;
  observationNoiseMultiplier: number;
  adaptationMeasurements: number;
  adaptationDays: number;
  adaptationDecayRate: number;
}

/**
 * Processor state (per user/device)
 */
export interface ProcessorState {
  deviceId?: string;
  userId: string;
  kalmanParams?: KalmanParams | null;
  lastState?: Vector2 | null;
  lastCovariance?: Matrix2x2 | null;
  lastTimestamp?: Date | string | null;
  lastRawWeight?: number | null;
  lastSource?: string | null;
  lastAcceptedTimestamp?: Date | string | null;
  measurementsSinceReset?: number;
  resetType?: ResetType | null;
  resetParameters?: ResetParameters | null;
  resetTimestamp?: Date | string | null;
  resetEvents?: ResetEvent[];
  measurementHistory?: MeasurementHistoryEntry[];
  createdAt?: string;
  updatedAt?: string;
  metadata?: Record<string, any>;
}

/**
 * Quality score components (for transparency)
 */
export interface QualityComponents {
  kalmanFit: number;
  temporalConsistency: number;
  anomalyDetection: number;
  sourceReliability: number;
  trendAlignment: number;
  combinedScore: number;
}

/**
 * Quality metadata (additional quality information)
 */
export interface QualityMetadata {
  innovation?: number; // Prediction error
  innovationVariance?: number;
  chiSquared?: number;
  pValue?: number;
  mahalanobisDistance?: number;
  timeSinceLastMeasurementHours?: number;
  weightChangeKg?: number;
  withinPhysiologicalLimits?: boolean;
  sourcePriority?: number;
  [key: string]: any;
}

/**
 * Measurement processing result
 */
export interface ProcessResult {
  measurementId?: string;
  deviceId?: string;
  userId: string;
  timestamp: string;
  source?: string;
  rawWeight: number;
  filteredWeight?: number;
  accepted: boolean;
  qualityScore: number;
  qualityComponents?: QualityComponents;
  qualityMetadata?: QualityMetadata;
  rejectionReason?: string;
  kalmanEstimate?: number;
  kalmanVariance?: number;
  kalmanUncertainty?: number;
  resetTriggered?: boolean;
  resetType?: ResetType;
  processingStage?: string;
  innovation?: number;
  normalizedInnovation?: number;
  confidence?: number;
  trend?: number;
  trendWeekly?: number;
  predictionError?: number;
  kalmanConfidenceUpper?: number;
  kalmanConfidenceLower?: number;
  metadata?: Record<string, any>;
}

/**
 * Batch processing response
 */
export interface ProcessResponseData {
  deviceId?: string;
  userId: string;
  measurementsProcessed: number;
  measurementsAccepted: number;
  measurementsRejected: number;
  results: ProcessResult[];
  finalState?: ProcessorState;
  stateUpdate?: any;
  metadata?: Record<string, any>;
}

/**
 * State snapshot for replay
 */
export interface StateSnapshot {
  snapshotId: string;
  deviceId?: string;
  userId: string;
  timestamp: string;
  state: ProcessorState;
  measurementsInWindow?: Measurement[];
  createdAt: string;
}

/**
 * Measurement history entry
 */
export interface MeasurementHistoryEntry {
  measurementId?: string;
  timestamp: string | Date;
  weight: number;
  source: string;
  accepted?: boolean;
  qualityScore?: number;
  unit?: string;
  metadata?: Record<string, any>;
}

/**
 * Replay buffer info
 */
export interface ReplayBufferInfo {
  userId: string;
  deviceId?: string;
  bufferSize: number;
  oldestTimestamp?: string;
  newestTimestamp?: string;
  readyForReplay: boolean;
}

/**
 * Replay result
 */
export interface ReplayResult {
  success: boolean;
  userId: string;
  deviceId?: string;
  measurementsReplayed: number;
  measurementsAccepted: number;
  measurementsRejected: number;
  outliersDetected: string[]; // measurement IDs
  stateRestoredTo?: string; // timestamp
  results: ProcessResult[];
  error?: string;
}

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Circuit breaker state
 */
export interface CircuitBreakerState {
  state: 'closed' | 'open' | 'half_open';
  failureCount: number;
  lastFailureTime?: string;
  lastSuccessTime?: string;
}

/**
 * Type guard to check if measurement is valid
 */
export function isValidMeasurement(obj: any): obj is Measurement {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    (typeof obj.deviceId === 'string' || typeof obj.device_id === 'string') &&
    (typeof obj.userId === 'string' || typeof obj.user_id === 'string') &&
    (typeof obj.weightKg === 'number' || typeof obj.weight_kg === 'number') &&
    (typeof obj.timestamp === 'string' || obj.timestamp instanceof Date) &&
    typeof obj.source === 'string'
  );
}

/**
 * Type guard for processor state
 */
export function isValidProcessorState(obj: any): obj is ProcessorState {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    (typeof obj.userId === 'string' || typeof obj.user_id === 'string')
  );
}

/**
 * Create an initial processor state
 */
export function createInitialState(deviceId: string, userId: string): ProcessorState {
  const now = new Date().toISOString();
  return {
    deviceId,
    userId,
    kalmanParams: null,
    lastState: null,
    lastCovariance: null,
    lastTimestamp: null,
    lastRawWeight: null,
    lastSource: null,
    lastAcceptedTimestamp: null,
    measurementsSinceReset: 0,
    resetType: null,
    resetParameters: null,
    resetTimestamp: null,
    resetEvents: [],
    measurementHistory: [],
    createdAt: now,
    updatedAt: now,
  };
}

/**
 * Create a process result
 */
export function createProcessResult(
  measurement: Measurement,
  accepted: boolean,
  qualityScore: number,
  rejectionReason?: string
): ProcessResult {
  return {
    measurementId: measurement.measurementId,
    deviceId: measurement.deviceId,
    userId: measurement.userId,
    timestamp: typeof measurement.timestamp === 'string'
      ? measurement.timestamp
      : measurement.timestamp.toISOString(),
    rawWeight: measurement.weightKg,
    accepted,
    qualityScore: qualityScore,
    rejectionReason: rejectionReason,
  };
}
