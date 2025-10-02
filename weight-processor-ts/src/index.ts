/**
 * @9amhealth/weight-processor
 *
 * TypeScript weight processing library with Kalman filtering and quality scoring
 *
 * ## Features
 * - Adaptive Kalman filtering for weight measurements
 * - Multi-component quality scoring system
 * - Automatic reset detection and management
 * - Replay mechanism for retroactive data quality improvements
 * - In-memory state storage
 * - Pure TypeScript with no dependencies
 *
 * ## Example Usage
 *
 * ### Basic Processing
 * ```typescript
 * import { WeightProcessorService, ProcessorStateDB, ConfigManager } from '@9amhealth/weight-processor';
 *
 * // Initialize components
 * const config = ConfigManager.loadConfig();
 * const db = new ProcessorStateDB();
 * const service = new WeightProcessorService(db, config);
 *
 * // Process a single measurement
 * const result = service.process_single('user123', {
 *   weight: 75.5,
 *   unit: 'kg',
 *   timestamp: new Date(),
 *   source: 'scale'
 * });
 *
 * if (result.accepted) {
 *   console.log('Accepted:', result.kalman_estimate);
 * } else {
 *   console.log('Rejected:', result.reason);
 * }
 * ```
 *
 * ### Batch Processing
 * ```typescript
 * // Process multiple measurements
 * const measurements = [
 *   { weight: 75.5, unit: 'kg', timestamp: new Date('2024-01-01'), source: 'scale' },
 *   { weight: 75.3, unit: 'kg', timestamp: new Date('2024-01-02'), source: 'scale' }
 * ];
 *
 * const batchResult = service.process_batch('user123', measurements);
 * console.log(`Accepted: ${batchResult.measurements_accepted}/${batchResult.measurements_processed}`);
 * ```
 *
 * @packageDocumentation
 */

// ============================================================================
// Core Processing
// ============================================================================

/**
 * Core processing pipeline components
 * - Kalman filtering
 * - Quality scoring
 * - Reset management
 * - Validation
 */
export * from './core/processing';

// ============================================================================
// State Management
// ============================================================================

/**
 * State storage and management
 * - In-memory state storage
 * - State snapshots for replay
 * - State interfaces
 */
export * from './core/database';

// ============================================================================
// Replay System
// ============================================================================

/**
 * Replay mechanism for data quality improvements
 * - Replay buffer management
 * - Outlier detection
 * - Replay orchestration
 */
export * from './core/replay';

// ============================================================================
// Mathematical Utilities
// ============================================================================

/**
 * Matrix operations and statistical functions
 * - 2x2 matrix operations
 * - Statistical calculations
 * - Linear regression
 */
export * from './core/math';

// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration management
 * - TOML config loading
 * - Configuration types
 */
export * from './config';

// ============================================================================
// Service Layer
// ============================================================================

/**
 * High-level service interface
 * - WeightProcessorService for easy integration
 * - Batch processing
 * - State management
 */
export * from './services';

// ============================================================================
// Models and Types
// ============================================================================

/**
 * TypeScript type definitions
 * - ProcessorState
 * - ProcessResult
 * - Measurement types
 * - Quality score types
 */
export * from './models';

/**
 * Constants and configuration defaults
 * - Physiological limits
 * - Kalman defaults
 * - Supported units
 */
export * from './constants';

/**
 * Utility functions
 * - Deep copy
 * - Date parsing
 * - Type conversion
 */
export * from './utils';
