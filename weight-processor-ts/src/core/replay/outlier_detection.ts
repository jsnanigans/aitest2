/**
 * Batch Outlier Detection for Retrospective Data Quality Processing
 *
 * Implements statistical methods to identify outliers in buffered measurement batches:
 * - IQR Method: Interquartile Range-based outlier detection
 * - Modified Z-Score: Median Absolute Deviation-based detection
 * - Temporal Consistency: Rate-of-change based detection
 * - Kalman Prediction: Deviation from Kalman filter predictions
 *
 * All methods are designed for batch analysis (not streaming) and focus on
 * identifying measurements that would cause Kalman filter instability.
 *
 * Ported from Python: weight_values/src/core/processing/outlier_detection.py
 */

import type { StateStore } from '../database/base';
import { percentile, median, std } from '../math/statistics';

interface OutlierConfig {
  iqr_multiplier?: number;
  z_score_threshold?: number;
  temporal_max_change_percent?: number;
  min_measurements_for_analysis?: number;
  quality_score_threshold?: number;
  kalman_deviation_threshold?: number;
}

/**
 * Statistical outlier detection for batch measurement analysis.
 * Designed to identify problematic measurements before Kalman processing.
 */
export class OutlierDetector {
  private config: OutlierConfig;
  private db: StateStore | null;
  private iqr_multiplier: number;
  private z_score_threshold: number;
  private temporal_max_change_percent: number;
  private min_measurements_for_analysis: number;
  private quality_score_threshold: number;
  private kalman_deviation_threshold: number;

  /**
   * Initialize outlier detector with configuration.
   */
  constructor(config?: OutlierConfig, db?: StateStore) {
    this.config = config || {};
    this.db = db || null;

    // Default thresholds (can be overridden by config)
    this.iqr_multiplier = this.config.iqr_multiplier || 1.5;
    this.z_score_threshold = this.config.z_score_threshold || 3.0;
    this.temporal_max_change_percent = this.config.temporal_max_change_percent || 0.30;
    this.min_measurements_for_analysis = this.config.min_measurements_for_analysis || 5;

    // Quality score threshold - measurements with quality > this are never outliers
    this.quality_score_threshold = this.config.quality_score_threshold || 0.7;

    // Kalman prediction deviation threshold (as percentage)
    this.kalman_deviation_threshold = this.config.kalman_deviation_threshold || 0.15;
  }

  /**
   * Detect outliers in a batch of measurements using multiple methods.
   * Respects quality scores and uses Kalman prediction deviation.
   */
  detect_outliers(
    measurements: Array<Record<string, any>>,
    user_id?: string
  ): Set<number> {
    if (measurements.length < this.min_measurements_for_analysis) {
      return new Set();
    }

    // Extract weights and sort by timestamp
    const sorted_measurements = [...measurements].sort((a, b) => {
      const ts_a = a.timestamp as Date;
      const ts_b = b.timestamp as Date;
      return ts_a.getTime() - ts_b.getTime();
    });

    const weights = sorted_measurements.map((m) => m.weight as number);

    // First, identify high-quality measurements that should never be marked as outliers
    const protected_indices = new Set<number>();

    // Apply quality override
    for (let i = 0; i < sorted_measurements.length; i++) {
      const measurement = sorted_measurements[i];
      const metadata = measurement.metadata || {};

      // Check if measurement has quality score
      const quality_score = metadata.quality_score;
      if (quality_score !== undefined && quality_score > this.quality_score_threshold) {
        protected_indices.add(i);
      }

      // Also protect measurements that were explicitly accepted
      if (metadata.accepted === true) {
        protected_indices.add(i);
      }
    }

    // Collect potential outliers from statistical methods
    const statistical_outliers = new Set<number>();

    // Method 1: IQR-based detection
    const iqr_outliers = this._detect_iqr_outliers(weights);
    for (const idx of iqr_outliers) {
      statistical_outliers.add(idx);
    }

    // Method 2: Modified Z-score detection
    const zscore_outliers = this._detect_zscore_outliers(weights);
    for (const idx of zscore_outliers) {
      statistical_outliers.add(idx);
    }

    // Method 3: Temporal consistency check
    const temporal_outliers = this._detect_temporal_outliers(sorted_measurements);
    for (const idx of temporal_outliers) {
      statistical_outliers.add(idx);
    }

    // Method 4: Kalman prediction deviation (if database and user_id available)
    let kalman_outliers = new Set<number>();
    if (this.db && user_id) {
      kalman_outliers = this._detect_kalman_outliers(sorted_measurements, user_id);
    }

    // AND logic: A measurement is only an outlier if:
    // 1. It's NOT protected by high quality score, AND
    // 2. It fails BOTH statistical tests AND Kalman prediction (if available)
    const final_outliers = new Set<number>();

    for (let idx = 0; idx < sorted_measurements.length; idx++) {
      // Skip if protected by quality score
      if (protected_indices.has(idx)) {
        continue;
      }

      // Check if it fails statistical tests
      if (!statistical_outliers.has(idx)) {
        continue;
      }

      // If we have Kalman predictions, also require it to fail that test
      if (kalman_outliers.size > 0) {
        if (!kalman_outliers.has(idx)) {
          continue;
        }
      }

      // This measurement is an outlier
      final_outliers.add(idx);
    }

    return final_outliers;
  }

  /**
   * Detect outliers using Interquartile Range method.
   */
  private _detect_iqr_outliers(weights: number[]): Set<number> {
    if (weights.length < 4) {
      // Need at least 4 points for quartiles
      return new Set();
    }

    const q1 = percentile(weights, 25);
    const q3 = percentile(weights, 75);
    const iqr = q3 - q1;

    const lower_bound = q1 - this.iqr_multiplier * iqr;
    const upper_bound = q3 + this.iqr_multiplier * iqr;

    const outliers = new Set<number>();
    for (let i = 0; i < weights.length; i++) {
      if (weights[i] < lower_bound || weights[i] > upper_bound) {
        outliers.add(i);
      }
    }

    return outliers;
  }

  /**
   * Detect outliers using Modified Z-Score method (median-based).
   * More robust than standard z-score for datasets with outliers.
   */
  private _detect_zscore_outliers(weights: number[]): Set<number> {
    if (weights.length < 3) {
      return new Set();
    }

    const median_val = median(weights);

    // Median Absolute Deviation (MAD)
    const abs_deviations = weights.map((w) => Math.abs(w - median_val));
    const mad = median(abs_deviations);

    if (mad === 0) {
      // All values are identical
      return new Set();
    }

    // Modified Z-scores
    const modified_z_scores = weights.map((w) => (0.6745 * (w - median_val)) / mad);

    const outliers = new Set<number>();
    for (let i = 0; i < modified_z_scores.length; i++) {
      if (Math.abs(modified_z_scores[i]) > this.z_score_threshold) {
        outliers.add(i);
      }
    }

    return outliers;
  }

  /**
   * Detect outliers based on temporal consistency (rate of change).
   * Identifies measurements with impossible rate of change.
   */
  private _detect_temporal_outliers(
    sorted_measurements: Array<Record<string, any>>
  ): Set<number> {
    if (sorted_measurements.length < 2) {
      return new Set();
    }

    const outliers = new Set<number>();

    for (let i = 1; i < sorted_measurements.length; i++) {
      const prev_measurement = sorted_measurements[i - 1];
      const curr_measurement = sorted_measurements[i];

      const weight_diff = Math.abs(
        (curr_measurement.weight as number) - (prev_measurement.weight as number)
      );
      const time_diff_ms =
        (curr_measurement.timestamp as Date).getTime() -
        (prev_measurement.timestamp as Date).getTime();

      // Skip if measurements are very close in time (< 1 hour)
      if (time_diff_ms < 3600 * 1000) {
        continue;
      }

      // Calculate percentage change
      const prev_weight = prev_measurement.weight as number;
      if (prev_weight > 0) {
        const percent_change = weight_diff / prev_weight;

        // Flag if change exceeds threshold
        if (percent_change > this.temporal_max_change_percent) {
          outliers.add(i);
        }
      }
    }

    return outliers;
  }

  /**
   * Detect outliers based on deviation from Kalman filter predictions.
   */
  private _detect_kalman_outliers(
    sorted_measurements: Array<Record<string, any>>,
    user_id: string
  ): Set<number> {
    if (!this.db) {
      return new Set();
    }

    const outliers = new Set<number>();

    // Get user's Kalman state
    const user_state = this.db.get_state(user_id);
    if (!user_state) {
      // No user state available
      return new Set();
    }

    const last_state = user_state.lastState;
    if (last_state === null) {
      // No Kalman state available, can't do prediction-based detection
      return new Set();
    }

    // Get state history if available for more accurate predictions
    const state_history = (user_state as any).state_history || [];

    for (let i = 0; i < sorted_measurements.length; i++) {
      const measurement = sorted_measurements[i];
      const weight = measurement.weight as number;
      const timestamp = measurement.timestamp as Date;

      // Find the closest state snapshot before this measurement
      let predicted_weight: number | null = null;

      if (state_history.length > 0) {
        // Look for state snapshot just before this measurement
        for (let j = state_history.length - 1; j >= 0; j--) {
          const snapshot = state_history[j];
          const snapshot_ts = snapshot.timestamp;

          if (snapshot_ts !== undefined && snapshot_ts !== null) {
            try {
              const snapshot_time =
                typeof snapshot_ts === 'string' ? new Date(snapshot_ts) : snapshot_ts;

              if (snapshot_time < timestamp) {
                if (snapshot.state) {
                  // Use the weight component of the state (first element)
                  if (Array.isArray(snapshot.state)) {
                    predicted_weight = snapshot.state[0] as number;
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
      if (predicted_weight === null) {
        if (Array.isArray(last_state) && last_state.length > 0) {
          // Handle both 1D and 2D arrays
          if (Array.isArray(last_state[0])) {
            predicted_weight = last_state[0][0] as number;
          } else {
            predicted_weight = last_state[0] as number;
          }
        }
      }

      // Check deviation from prediction
      if (predicted_weight !== null) {
        try {
          const weight_val = Number(predicted_weight);
          if (weight_val > 0) {
            const deviation = Math.abs(weight - weight_val) / weight_val;
            if (deviation > this.kalman_deviation_threshold) {
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
   */
  analyze_outliers(
    measurements: Array<Record<string, any>>,
    outlier_indices: Set<number>
  ): Record<string, any> {
    if (outlier_indices.size === 0 || measurements.length === 0) {
      return {
        total_measurements: measurements.length,
        outlier_count: 0,
        outlier_percentage: 0.0,
        outlier_details: []
      };
    }

    const sorted_measurements = [...measurements].sort((a, b) => {
      const ts_a = a.timestamp as Date;
      const ts_b = b.timestamp as Date;
      return ts_a.getTime() - ts_b.getTime();
    });

    const weights = sorted_measurements.map((m) => m.weight as number);
    const median_weight = median(weights);

    const outlier_details: Array<Record<string, any>> = [];
    for (const idx of Array.from(outlier_indices).sort((a, b) => a - b)) {
      if (idx < sorted_measurements.length) {
        const measurement = sorted_measurements[idx];
        const weight = measurement.weight as number;

        const detail: Record<string, any> = {
          index: idx,
          timestamp: measurement.timestamp,
          weight,
          deviation_from_median: weight - median_weight,
          source: measurement.source || 'unknown'
        };

        // Add context from neighboring measurements
        if (idx > 0) {
          const prev_weight = sorted_measurements[idx - 1].weight as number;
          detail.change_from_previous = weight - prev_weight;
          detail.percent_change_from_previous =
            prev_weight > 0 ? Math.abs(weight - prev_weight) / prev_weight : 0;
        }

        outlier_details.push(detail);
      }
    }

    return {
      total_measurements: measurements.length,
      outlier_count: outlier_indices.size,
      outlier_percentage: (outlier_indices.size / measurements.length) * 100,
      outlier_details,
      median_weight,
      weight_std: std(weights),
      weight_range: [Math.min(...weights), Math.max(...weights)]
    };
  }

  /**
   * Get measurements with outliers removed.
   */
  get_clean_measurements(
    measurements: Array<Record<string, any>>,
    user_id?: string
  ): [Array<Record<string, any>>, Set<number>] {
    const outlier_indices = this.detect_outliers(measurements, user_id);

    if (outlier_indices.size === 0) {
      return [[...measurements], new Set()];
    }

    // Sort by timestamp to maintain chronological order
    const sorted_measurements = [...measurements].sort((a, b) => {
      const ts_a = a.timestamp as Date;
      const ts_b = b.timestamp as Date;
      return ts_a.getTime() - ts_b.getTime();
    });

    const clean_measurements: Array<Record<string, any>> = [];
    for (let i = 0; i < sorted_measurements.length; i++) {
      if (!outlier_indices.has(i)) {
        clean_measurements.push({ ...sorted_measurements[i] });
      }
    }

    return [clean_measurements, outlier_indices];
  }

  /**
   * Update detector configuration.
   */
  update_config(new_config: OutlierConfig): void {
    Object.assign(this.config, new_config);

    // Update thresholds
    this.iqr_multiplier = this.config.iqr_multiplier || this.iqr_multiplier;
    this.z_score_threshold = this.config.z_score_threshold || this.z_score_threshold;
    this.temporal_max_change_percent =
      this.config.temporal_max_change_percent || this.temporal_max_change_percent;
    this.min_measurements_for_analysis =
      this.config.min_measurements_for_analysis || this.min_measurements_for_analysis;
  }

  /**
   * Get current configuration.
   */
  get_config(): OutlierConfig {
    return {
      iqr_multiplier: this.iqr_multiplier,
      z_score_threshold: this.z_score_threshold,
      temporal_max_change_percent: this.temporal_max_change_percent,
      min_measurements_for_analysis: this.min_measurements_for_analysis
    };
  }
}
