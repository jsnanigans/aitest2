/**
 * Weight Processor TypeScript Library
 *
 * Core infrastructure-agnostic weight processing library with Kalman filtering,
 * quality scoring, and comprehensive validation.
 *
 * This is a 1:1 port of the Python `python_lib` implementation.
 */

// Export everything from core
export * from './weight-processor-lib/core/index.js';

// Re-export commonly used items for convenience
export {
  processMeasurement,
  InMemoryStore,
  KalmanFilterManager,
  UnifiedQualityScorer,
  PhysiologicalValidator,
  BMIValidator
} from './weight-processor-lib/core/index.js';

export type {
  ProcessingResult,
  QualityScore,
  KalmanState,
  StateStore
} from './weight-processor-lib/core/index.js';
