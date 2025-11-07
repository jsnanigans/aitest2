/**
 * Batch Outlier Detection for Retrospective Data Quality Processing
 *
 * Implements statistical methods to identify outliers in buffered measurement batches:
 * - IQR Method: Interquartile Range-based outlier detection
 * - Modified Z-Score: Median Absolute Deviation-based detection
 * - Temporal Consistency: Rate-of-change based detection
 *
 * All methods are designed for batch analysis (not streaming) and focus on
 * identifying measurements that would cause Kalman filter instability.
 */

import type { StateStore, KalmanState } from '../database/base';
import { isEffectivelyZero } from '../utils';

/**
 * Measurement structure for outlier detection
 */
export interface Measurement {
  weight: number;
  timestamp: Date;
  source?: string;
  metadata?: {
    quality_score?: number;
    accepted?: boolean;
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Configuration for OutlierDetector
 */
export interface OutlierDetectorConfig {
  iqr_multiplier?: number;
  z_score_threshold?: number;
  temporal_max_change_percent?: number;
  min_measurements_for_analysis?: number;
  quality_score_threshold?: number;
  kalman_deviation_threshold?: number;
}

/**
 * Outlier analysis report
 */
export interface OutlierAnalysis {
  total_measurements: number;
  outlier_count: number;
  outlier_percentage: number;
  outlier_details: OutlierDetail[];
  median_weight?: number;
  weight_std?: number;
  weight_range?: [number, number];
}

/**
 * Detail about a single outlier
 */
export interface OutlierDetail {
  index: number;
  timestamp: Date;
  weight: number;
  deviation_from_median: number;
  source: string;
  change_from_previous?: number;
  percent_change_from_previous?: number;
}

/**
 * Statistical outlier detection for batch measurement analysis.
 * Designed to identify problematic measurements before Kalman processing.
 */
export class OutlierDetector {
  private config: OutlierDetectorConfig;
  private db: StateStore | null;

  // Configurable thresholds
  private iqrMultiplier: number;
  private zScoreThreshold: number;
  private temporalMaxChangePercent: number;
  private minMeasurementsForAnalysis: number;
  private qualityScoreThreshold: number;
  private kalmanDeviationThreshold: number;

  /**
   * Initialize outlier detector with configuration.
   *
   * @param config - Configuration dict with thresholds and method settings
   * @param db - Database instance for accessing Kalman states
   */
  constructor(config?: OutlierDetectorConfig, db?: StateStore) {
    this.config = config || {};
    this.db = db || null;

    // Default thresholds (can be overridden by config)
    this.iqrMultiplier = this.config.iqr_multiplier ?? 1.5;
    this.zScoreThreshold = this.config.z_score_threshold ?? 3.0;
    this.temporalMaxChangePercent = this.config.temporal_max_change_percent ?? 0.3;
    this.minMeasurementsForAnalysis = this.config.min_measurements_for_analysis ?? 5;

    // Quality score threshold - measurements with quality > this are never outliers
    this.qualityScoreThreshold = this.config.quality_score_threshold ?? 0.7;

    // Kalman prediction deviation threshold (as percentage)
    this.kalmanDeviationThreshold = this.config.kalman_deviation_threshold ?? 0.15;
  }

  /**
   * Detect outliers in a batch of measurements using multiple methods.
   * Respects quality scores and uses Kalman prediction deviation.
   *
   * @param measurements - List of measurement objects with 'weight', 'timestamp', 'metadata'
   * @param userId - Optional user identifier for accessing Kalman state
   * @returns Set of indices that are considered outliers
   */
  async detectOutliers(measurements: Measurement[], userId?: string): Promise<Set<number>> {
    if (measurements.length < this.minMeasurementsForAnalysis) {
      return new Set();
    }

    // Extract weights and sort by timestamp
    const sortedMeasurements = [...measurements].sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );
    const weights = sortedMeasurements.map((m) => m.weight);

    // First, identify high-quality measurements that should never be marked as outliers
    const protectedIndices = new Set<number>();

    // Apply quality override
    for (let i = 0; i < sortedMeasurements.length; i++) {
      const measurement = sortedMeasurements[i];
      const metadata = measurement.metadata || {};

      // Check if measurement has quality score
      const qualityScore = metadata.quality_score;
      if (qualityScore !== undefined && qualityScore > this.qualityScoreThreshold) {
        protectedIndices.add(i);
      }

      // Also protect measurements that were explicitly accepted
      if (metadata.accepted === true) {
        protectedIndices.add(i);
      }
    }

    // Collect potential outliers from statistical methods
    const statisticalOutliers = new Set<number>();

    // Method 1: IQR-based detection
    const iqrOutliers = this._detectIqrOutliers(weights);
    iqrOutliers.forEach((idx) => statisticalOutliers.add(idx));

    // Method 2: Modified Z-score detection
    const zscoreOutliers = this._detectZscoreOutliers(weights);
    zscoreOutliers.forEach((idx) => statisticalOutliers.add(idx));

    // Method 3: Temporal consistency check
    const temporalOutliers = this._detectTemporalOutliers(sortedMeasurements);
    temporalOutliers.forEach((idx) => statisticalOutliers.add(idx));

    // Method 4: Kalman prediction deviation (if database and userId available)
    let kalmanOutliers = new Set<number>();
    if (this.db && userId) {
      kalmanOutliers = await this._detectKalmanOutliers(sortedMeasurements, userId);
    }

    // AND logic: A measurement is only an outlier if:
    // 1. It's NOT protected by high quality score, AND
    // 2. It fails BOTH statistical tests AND Kalman prediction (if available)
    const finalOutliers = new Set<number>();

    for (let idx = 0; idx < sortedMeasurements.length; idx++) {
      // Skip if protected by quality score
      if (protectedIndices.has(idx)) {
        continue;
      }

      // Check if it fails statistical tests
      if (!statisticalOutliers.has(idx)) {
        continue;
      }

      // If we have Kalman predictions, also require it to fail that test
      if (kalmanOutliers.size > 0) {
        if (!kalmanOutliers.has(idx)) {
          continue;
        }
      }

      // This measurement is an outlier
      finalOutliers.add(idx);
    }

    return finalOutliers;
  }

  /**
   * Detect outliers using Interquartile Range method.
   *
   * @param weights - List of weight values
   * @returns Set of indices that are IQR outliers
   */
  private _detectIqrOutliers(weights: number[]): Set<number> {
    if (weights.length < 4) {
      // Need at least 4 points for quartiles
      return new Set();
    }

    const q1 = this._percentile(weights, 25);
    const q3 = this._percentile(weights, 75);
    const iqr = q3 - q1;

    const lowerBound = q1 - this.iqrMultiplier * iqr;
    const upperBound = q3 + this.iqrMultiplier * iqr;

    const outliers = new Set<number>();
    for (let i = 0; i < weights.length; i++) {
      const weight = weights[i];
      if (weight < lowerBound || weight > upperBound) {
        outliers.add(i);
      }
    }

    return outliers;
  }

  /**
   * Detect outliers using Modified Z-Score method (median-based).
   * More robust than standard z-score for datasets with outliers.
   *
   * @param weights - List of weight values
   * @returns Set of indices that are Z-score outliers
   */
  private _detectZscoreOutliers(weights: number[]): Set<number> {
    if (weights.length < 3) {
      return new Set();
    }

    const median = this._median(weights);

    // Median Absolute Deviation (MAD)
    const deviations = weights.map((w) => Math.abs(w - median));
    const mad = this._median(deviations);

    if (isEffectivelyZero(mad)) {
      // All values are identical (within epsilon)
      return new Set();
    }

    // Modified Z-scores
    const modifiedZScores = weights.map((w) => (0.6745 * (w - median)) / mad);

    const outliers = new Set<number>();
    for (let i = 0; i < modifiedZScores.length; i++) {
      if (Math.abs(modifiedZScores[i]) > this.zScoreThreshold) {
        outliers.add(i);
      }
    }

    return outliers;
  }

  /**
   * Detect outliers based on temporal consistency (rate of change).
   * Identifies measurements with impossible rate of change.
   *
   * @param sortedMeasurements - Measurements sorted by timestamp
   * @returns Set of indices that are temporal outliers
   */
  private _detectTemporalOutliers(sortedMeasurements: Measurement[]): Set<number> {
    if (sortedMeasurements.length < 2) {
      return new Set();
    }

    const outliers = new Set<number>();

    for (let i = 1; i < sortedMeasurements.length; i++) {
      const prevMeasurement = sortedMeasurements[i - 1];
      const currMeasurement = sortedMeasurements[i];

      const weightDiff = Math.abs(currMeasurement.weight - prevMeasurement.weight);
      const timeDiffMs =
        currMeasurement.timestamp.getTime() - prevMeasurement.timestamp.getTime();

      // Skip if measurements are very close in time (< 1 hour)
      if (timeDiffMs < 3600000) {
        // 3600000 ms = 1 hour
        continue;
      }

      // Calculate percentage change
      const prevWeight = prevMeasurement.weight;
      if (prevWeight > 0) {
        const percentChange = weightDiff / prevWeight;

        // Flag if change exceeds threshold
        if (percentChange > this.temporalMaxChangePercent) {
          outliers.add(i);
        }
      }
    }

    return outliers;
  }

  /**
   * Detect outliers based on deviation from Kalman filter predictions.
   *
   * @param sortedMeasurements - Measurements sorted by timestamp
   * @param userId - User identifier for accessing Kalman state
   * @returns Set of indices that deviate significantly from Kalman predictions
   */
  private async _detectKalmanOutliers(
    sortedMeasurements: Measurement[],
    userId: string
  ): Promise<Set<number>> {
    if (!this.db) {
      return new Set();
    }

    const outliers = new Set<number>();

    // Get user's Kalman state
    const userState = await this.db.getState(userId);
    if (!userState) {
      // No user state available
      return new Set();
    }

    const lastState = userState.last_state;
    if (lastState === null) {
      // No Kalman state available, can't do prediction-based detection
      return new Set();
    }

    // Get state history if available for more accurate predictions
    const stateHistory = (userState as any).state_history || [];

    for (let i = 0; i < sortedMeasurements.length; i++) {
      const measurement = sortedMeasurements[i];
      const weight = measurement.weight;
      const timestamp = measurement.timestamp;

      // Find the closest state snapshot before this measurement
      let predictedWeight: number | null = null;

      if (stateHistory.length > 0) {
        // Look for state snapshot just before this measurement
        for (let j = stateHistory.length - 1; j >= 0; j--) {
          const snapshot = stateHistory[j];
          const snapshotTs = snapshot.timestamp;

          if (snapshotTs !== undefined && snapshotTs !== null) {
            try {
              let snapshotTime: number;
              if (snapshotTs instanceof Date) {
                snapshotTime = snapshotTs.getTime();
              } else if (typeof snapshotTs === 'string') {
                snapshotTime = new Date(snapshotTs).getTime();
              } else if (typeof snapshotTs === 'number') {
                snapshotTime = snapshotTs;
              } else {
                continue;
              }

              if (snapshotTime < timestamp.getTime()) {
                if (snapshot.state) {
                  // Use the weight component of the state (first element)
                  if (Array.isArray(snapshot.state) && snapshot.state.length > 0) {
                    predictedWeight = snapshot.state[0];
                    break;
                  }
                }
              }
            } catch {
              // If comparison fails, skip this snapshot
              continue;
            }
          }
        }
      }

      // Fall back to last state if no snapshot found
      if (predictedWeight === null) {
        if (Array.isArray(lastState) && lastState.length > 0 && Array.isArray(lastState[0])) {
          predictedWeight = lastState[0][0];
        }
      }

      // Check deviation from prediction
      if (predictedWeight !== null) {
        try {
          const weightVal = Number(predictedWeight);
          if (weightVal > 0) {
            const deviation = Math.abs(weight - weightVal) / weightVal;
            if (deviation > this.kalmanDeviationThreshold) {
              outliers.add(i);
            }
          }
        } catch {
          // If can't convert to number, skip this measurement
          continue;
        }
      }
    }

    return outliers;
  }

  /**
   * Analyze detected outliers and provide detailed information.
   *
   * @param measurements - Original measurements list
   * @param outlierIndices - Set of outlier indices
   * @returns Analysis report object
   */
  analyzeOutliers(measurements: Measurement[], outlierIndices: Set<number>): OutlierAnalysis {
    if (outlierIndices.size === 0 || measurements.length === 0) {
      return {
        total_measurements: measurements.length,
        outlier_count: 0,
        outlier_percentage: 0.0,
        outlier_details: [],
      };
    }

    const sortedMeasurements = [...measurements].sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );
    const weights = sortedMeasurements.map((m) => m.weight);

    const outlierDetails: OutlierDetail[] = [];
    const sortedIndices = Array.from(outlierIndices).sort((a, b) => a - b);

    for (const idx of sortedIndices) {
      if (idx < sortedMeasurements.length) {
        const measurement = sortedMeasurements[idx];
        const weight = measurement.weight;

        const detail: OutlierDetail = {
          index: idx,
          timestamp: measurement.timestamp,
          weight: weight,
          deviation_from_median: weight - this._median(weights),
          source: measurement.source || 'unknown',
        };

        // Add context from neighboring measurements
        if (idx > 0) {
          const prevWeight = sortedMeasurements[idx - 1].weight;
          detail.change_from_previous = weight - prevWeight;
          detail.percent_change_from_previous =
            prevWeight > 0 ? Math.abs(weight - prevWeight) / prevWeight : 0;
        }

        outlierDetails.push(detail);
      }
    }

    return {
      total_measurements: measurements.length,
      outlier_count: outlierIndices.size,
      outlier_percentage: (outlierIndices.size / measurements.length) * 100,
      outlier_details: outlierDetails,
      median_weight: this._median(weights),
      weight_std: this._std(weights),
      weight_range: [Math.min(...weights), Math.max(...weights)],
    };
  }

  /**
   * Get measurements with outliers removed.
   *
   * @param measurements - Original measurements list
   * @param userId - Optional user identifier for Kalman-based outlier detection
   * @returns Tuple of [clean_measurements, outlier_indices]
   */
  async getCleanMeasurements(
    measurements: Measurement[],
    userId?: string
  ): Promise<[Measurement[], Set<number>]> {
    const outlierIndices = await this.detectOutliers(measurements, userId);

    if (outlierIndices.size === 0) {
      return [[...measurements], new Set()];
    }

    // Sort by timestamp to maintain chronological order
    const sortedMeasurements = [...measurements].sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );

    const cleanMeasurements: Measurement[] = [];
    for (let i = 0; i < sortedMeasurements.length; i++) {
      if (!outlierIndices.has(i)) {
        cleanMeasurements.push({ ...sortedMeasurements[i] });
      }
    }

    return [cleanMeasurements, outlierIndices];
  }

  /**
   * Update detector configuration.
   *
   * @param newConfig - New configuration dictionary
   */
  updateConfig(newConfig: OutlierDetectorConfig): void {
    Object.assign(this.config, newConfig);

    // Update thresholds
    this.iqrMultiplier = this.config.iqr_multiplier ?? this.iqrMultiplier;
    this.zScoreThreshold = this.config.z_score_threshold ?? this.zScoreThreshold;
    this.temporalMaxChangePercent =
      this.config.temporal_max_change_percent ?? this.temporalMaxChangePercent;
    this.minMeasurementsForAnalysis =
      this.config.min_measurements_for_analysis ?? this.minMeasurementsForAnalysis;
  }

  /**
   * Get current configuration.
   *
   * @returns Current configuration object
   */
  getConfig(): OutlierDetectorConfig {
    return {
      iqr_multiplier: this.iqrMultiplier,
      z_score_threshold: this.zScoreThreshold,
      temporal_max_change_percent: this.temporalMaxChangePercent,
      min_measurements_for_analysis: this.minMeasurementsForAnalysis,
    };
  }

  // ============================================================================
  // Statistical Helper Methods
  // ============================================================================

  /**
   * Calculate median of an array of numbers.
   */
  private _median(values: number[]): number {
    if (values.length === 0) {
      return 0;
    }

    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    } else {
      return sorted[mid];
    }
  }

  /**
   * Calculate percentile of an array of numbers.
   */
  private _percentile(values: number[], percentile: number): number {
    if (values.length === 0) {
      return 0;
    }

    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;

    if (lower === upper) {
      return sorted[lower];
    }

    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  /**
   * Calculate standard deviation of an array of numbers.
   */
  private _std(values: number[]): number {
    if (values.length === 0) {
      return 0;
    }

    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map((val) => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;

    return Math.sqrt(variance);
  }
}
