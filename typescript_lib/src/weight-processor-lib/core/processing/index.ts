/**
 * Processing module exports.
 * Core weight processing logic including Kalman filtering, quality scoring, and validation.
 */

// Main processor
export { processMeasurement, processWeightEnhanced } from './processor.js';
export type { ProcessingResult } from './processor.js';

// Kalman filter
export { KalmanFilter } from './kalman_filter.js';
export { KalmanFilterManager } from './kalman.js';
export type { KalmanParams, ResetParameters, ResetEvent } from './kalman.js';

// Validation
export {
  PhysiologicalValidator,
  BMIValidator,
  ThresholdCalculator,
  DataQualityPreprocessor
} from './validation.js';

// Quality scoring
export { UnifiedQualityScorer } from './unified_quality_scorer.js';
export type { QualityScore } from './unified_quality_scorer.js';

// Outlier detection
export { OutlierDetector } from './outlier_detection.js';

// Circuit breaker
export { CircuitBreaker, MultiCircuitBreaker, CircuitState, CircuitOpenError } from './circuit_breaker.js';

// State management
export { StateValidator, ResetOperation } from './state_validator.js';
export { PersistenceValidator } from './persistence_validator.js';
export { ResetTransaction, atomicReset, atomicResetSync } from './reset_transaction.js';
export { ResetManager, ResetType } from './reset_manager.js';

// Type conversion utilities
export { ensureFloat, ensureNumericTypes, prepareMeasurementForProcessing } from './type_conversion.js';
