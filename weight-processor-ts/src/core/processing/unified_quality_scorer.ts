/**
 * Unified Kalman-centric quality scoring system.
 * Replaces dual validation with single Kalman-deviation-based quality scorer.
 */

import { chi2Cdf, linearRegression, mean, std } from '../math/statistics';
import {
  PHYSIOLOGICAL_LIMITS,
  DEFAULT_SOURCE_PROFILE,
} from '../../constants';
import type { ProcessorState, QualityScore as QualityScoreType, QualityComponents } from '../../models';

/**
 * Container for quality score and its components
 */
export class QualityScore implements QualityScoreType {
  overall: number;
  components: QualityComponents;
  threshold: number;
  accepted: boolean;
  rejectionReason?: string;
  metadata: Record<string, any>;

  constructor(
    overall: number,
    components: QualityComponents,
    threshold: number = 0.46,
    metadata: Record<string, any> = {}
  ) {
    this.overall = overall;
    this.components = components;
    this.threshold = threshold;
    this.metadata = metadata;
    this.accepted = overall >= threshold;

    if (!this.accepted && !this.rejectionReason) {
      if (Object.keys(components).length > 0) {
        const minComponent = Object.entries(components).reduce((min, [name, score]) =>
          score < min[1] ? [name, score] : min
        );
        this.rejectionReason = `Quality score ${overall.toFixed(2)} below threshold ${threshold} (weakest: ${minComponent[0]}=${minComponent[1].toFixed(2)})`;
      } else {
        this.rejectionReason = `Quality score ${overall.toFixed(2)} below threshold ${threshold} (no components calculated)`;
      }
    }
  }

  toDict(): Record<string, any> {
    return {
      overall: this.overall,
      components: this.components,
      threshold: this.threshold,
      accepted: this.accepted,
      rejectionReason: this.rejectionReason,
      metadata: this.metadata,
    };
  }
}

/**
 * Unified Kalman-centric quality scoring system.
 * Primary signal is deviation from Kalman prediction.
 */
export class UnifiedQualityScorer {
  // Default component weights (must sum to 1.0)
  static readonly DEFAULT_WEIGHTS: Record<string, number> = {
    kalman_fit: 0.40,
    temporal_consistency: 0.30,
    anomaly_detection: 0.20,
    source_reliability: 0.05,
    trend_alignment: 0.05,
  };

  // Time-based thresholds for temporal consistency
  static readonly TEMPORAL_THRESHOLDS = {
    '6h': 3.0,
    '24h': 2.0,
    sustained: 2.0,
  };

  // Rapid measurement detection thresholds
  static readonly DUPLICATE_THRESHOLD_SECONDS = 5;
  static readonly RAPID_THRESHOLD_MINUTES = 5;
  static readonly BURST_WINDOW_MINUTES = 30;
  static readonly BURST_COUNT_THRESHOLD = 5;
  static readonly MAX_1MIN_CHANGE_KG = 0.5;
  static readonly MAX_5MIN_CHANGE_KG = 1.0;

  private config: any;
  private weights: Record<string, number>;
  private threshold: number;
  private temporalThresholds: Record<string, number>;
  private currentSource?: string;
  private sourceProfiles?: Record<string, any>;

  constructor(config?: any, sourceProfiles?: Record<string, any>) {
    this.config = config || {};
    this.sourceProfiles = sourceProfiles;

    // Get component weights from config
    this.weights = { ...UnifiedQualityScorer.DEFAULT_WEIGHTS, ...(this.config.component_weights || {}) };

    // Normalize weights to sum to 1.0
    const weightSum = Object.values(this.weights).reduce((sum, w) => sum + w, 0);
    if (weightSum > 0) {
      for (const key of Object.keys(this.weights)) {
        this.weights[key] /= weightSum;
      }
    }

    // Get thresholds
    this.threshold = this.config.threshold ?? 0.46;
    this.temporalThresholds = { ...UnifiedQualityScorer.TEMPORAL_THRESHOLDS, ...(this.config.temporal_thresholds || {}) };
  }

  /**
   * Calculate unified quality score with Kalman-centric approach.
   */
  calculateQualityScore(params: {
    weight: number;
    source: string;
    kalmanState?: ProcessorState | null;
    kalmanPrediction?: number | null;
    innovationCovariance?: number | null;
    previousWeight?: number | null;
    timeDiffHours?: number | null;
    recentWeights?: number[] | null;
    recentTimestamps?: Date[] | null;
    userHeightM?: number | null;
  }): QualityScore {
    const {
      weight,
      source,
      kalmanState,
      kalmanPrediction,
      innovationCovariance,
      previousWeight,
      timeDiffHours,
      recentWeights,
      recentTimestamps,
      userHeightM,
    } = params;

    const components: any = {};
    const metadata: any = {};

    // Store current source
    this.currentSource = source;

    // 1. Kalman Fit Component
    if (this.weights.kalman_fit > 0) {
      const [kalmanScore, kalmanMeta] = this.calculateKalmanFit(
        weight,
        kalmanPrediction,
        innovationCovariance,
        kalmanState
      );
      if (kalmanScore !== null && kalmanScore !== undefined && !isNaN(kalmanScore)) {
        components.kalman_fit = kalmanScore;
      }
      metadata.kalman_fit = kalmanMeta;
    }

    // 2. Temporal Consistency
    if (this.weights.temporal_consistency > 0) {
      const [temporalScore, temporalMeta] = this.calculateTemporalConsistency(
        weight,
        previousWeight,
        timeDiffHours,
        recentWeights,
        recentTimestamps
      );
      if (temporalScore !== null && temporalScore !== undefined && !isNaN(temporalScore)) {
        components.temporal_consistency = temporalScore;
      }
      metadata.temporal_consistency = temporalMeta;
    }

    // 3. Anomaly Detection
    if (this.weights.anomaly_detection > 0) {
      const currentTs = (kalmanState as any)?.current_timestamp;
      const [anomalyScore, anomalyMeta] = this.calculateAnomalyDetection(
        weight,
        recentWeights,
        recentTimestamps,
        userHeightM,
        currentTs
      );
      if (anomalyScore !== null && anomalyScore !== undefined && !isNaN(anomalyScore)) {
        components.anomaly_detection = anomalyScore;
      }
      metadata.anomaly_detection = anomalyMeta;
    }

    // 4. Source Reliability
    if (this.weights.source_reliability > 0) {
      const sourceScore = this.calculateSourceReliability(source);
      if (sourceScore !== null && sourceScore !== undefined && !isNaN(sourceScore)) {
        components.source_reliability = sourceScore;
      }
      metadata.source_reliability = { source, score: sourceScore };
    }

    // 5. Trend Alignment
    if (this.weights.trend_alignment > 0) {
      const [trendScore, trendMeta] = this.calculateTrendAlignment(
        weight,
        kalmanState,
        recentWeights
      );
      // Only set if we got a valid number (not null/undefined/NaN)
      if (trendScore !== null && trendScore !== undefined && !isNaN(trendScore)) {
        components.trend_alignment = trendScore;
      }
      metadata.trend_alignment = trendMeta;
    }

    // Calculate overall score
    const useHarmonic = this.config.use_harmonic_mean ?? false;
    const overall = useHarmonic
      ? this.calculateWeightedHarmonicMean(components)
      : this.calculateWeightedGeometricMean(components);

    return new QualityScore(overall, components, this.threshold, metadata);
  }

  /**
   * Calculate how well measurement fits Kalman prediction.
   */
  calculateKalmanFit(
    weight: number,
    kalmanPrediction: number | null | undefined,
    innovationCovariance: number | null | undefined,
    kalmanState?: ProcessorState | null
  ): [number, Record<string, any>] {
    const metadata: any = {};

    if (kalmanPrediction === null || kalmanPrediction === undefined ||
        innovationCovariance === null || innovationCovariance === undefined) {
      metadata.reason = 'No Kalman prediction available';
      return [0.5, metadata];
    }

    const innovation = weight - kalmanPrediction;
    metadata.innovation = innovation;
    metadata.prediction = kalmanPrediction;

    let covarianceValue = innovationCovariance <= 0 ? 1.0 : innovationCovariance;
    const normalizedInnovation = Math.abs(innovation) / Math.sqrt(covarianceValue);
    metadata.normalized_innovation = normalizedInnovation;

    // Chi-squared test
    const chiSquared = normalizedInnovation ** 2;
    const pValue = 1 - chi2Cdf(chiSquared, 1);
    metadata.chi_squared = chiSquared;
    metadata.p_value = pValue;

    // Check for adaptive period
    let inAdaptivePeriod = false;
    if (kalmanState) {
      const measurementsSinceReset = kalmanState.measurementsSinceReset ?? 100;
      const resetParams = kalmanState.resetParameters;
      const adaptationMeasurements = resetParams?.adaptation_measurements ?? 10;
      if (measurementsSinceReset < adaptationMeasurements) {
        inAdaptivePeriod = true;
        metadata.adaptive_period = true;
      }
    }

    // Convert to quality score
    let score: number;
    if (inAdaptivePeriod) {
      score = Math.exp(-0.2 * normalizedInnovation);
    } else {
      score = Math.exp(-0.5 * normalizedInnovation);
    }

    // Apply time-based decay for gap tolerance
    let daysSinceLast = 0;
    if (kalmanState && kalmanState.lastTimestamp) {
      const lastTimestamp = typeof kalmanState.lastTimestamp === 'string'
        ? new Date(kalmanState.lastTimestamp)
        : kalmanState.lastTimestamp;
      const currentTimestamp = (kalmanState as any).current_timestamp
        ? (typeof (kalmanState as any).current_timestamp === 'string'
          ? new Date((kalmanState as any).current_timestamp)
          : (kalmanState as any).current_timestamp)
        : new Date();
      daysSinceLast = (currentTimestamp.getTime() - lastTimestamp.getTime()) / (86400.0 * 1000);
      metadata.days_since_last = daysSinceLast;
    }

    if (daysSinceLast > 0) {
      const decayFactor = Math.min(1.0, daysSinceLast / 30.0);
      const adjustedScore = score + (1.0 - score) * decayFactor;
      metadata.decay_factor = decayFactor;
      metadata.original_score = score;
      score = adjustedScore;
    }

    score = Math.max(0.0, Math.min(1.0, score));
    metadata.score = score;

    return [score, metadata];
  }

  /**
   * Calculate temporal consistency using continuous exponential function.
   */
  calculateTemporalConsistency(
    weight: number,
    previousWeight: number | null | undefined,
    timeDiffHours: number | null | undefined,
    recentWeights?: number[] | null,
    recentTimestamps?: Date[] | null
  ): [number, Record<string, any>] {
    const metadata: any = {};

    if (previousWeight === null || previousWeight === undefined ||
        timeDiffHours === null || timeDiffHours === undefined) {
      metadata.reason = 'No previous weight for comparison';
      return [0.7, metadata];
    }

    const weightChange = Math.abs(weight - previousWeight);
    const absTimeDiff = Math.abs(timeDiffHours);
    const cappedTime = Math.min(absTimeDiff, 336); // Cap at 2 weeks
    const maxAcceptableChange = 0.5 + 4.5 * (1 - Math.exp(-cappedTime / 48));

    metadata.max_acceptable_change = maxAcceptableChange;
    metadata.actual_change = weightChange;
    metadata.time_diff_hours = timeDiffHours;

    let score: number;
    if (weightChange <= maxAcceptableChange) {
      score = 0.8 + 0.2 * Math.exp(-weightChange / maxAcceptableChange);
    } else {
      const excessRatio = (weightChange - maxAcceptableChange) / maxAcceptableChange;
      score = 0.8 * Math.exp(-excessRatio);
    }

    if (timeDiffHours > 168) {
      score = Math.max(score, 0.4);
      metadata.gap_adjustment = true;
    }

    score = Math.max(0.2, Math.min(1.0, score));
    return [score, metadata];
  }

  /**
   * Enhanced anomaly detection with time-aware physiological limits.
   */
  calculateAnomalyDetection(
    weight: number,
    recentWeights?: number[] | null,
    recentTimestamps?: Date[] | null,
    userHeightM?: number | null,
    currentTimestamp?: Date | null
  ): [number, Record<string, any>] {
    const metadata: any = {};
    let score = 1.0;

    // 1. Check absolute physiological bounds
    if (weight < PHYSIOLOGICAL_LIMITS.ABSOLUTE_MIN_WEIGHT) {
      metadata.outside_absolute_min = true;
      return [0.0, metadata];
    }
    if (weight > PHYSIOLOGICAL_LIMITS.ABSOLUTE_MAX_WEIGHT) {
      metadata.outside_absolute_max = true;
      return [0.0, metadata];
    }

    // Check suspicious bounds
    if (weight < PHYSIOLOGICAL_LIMITS.SUSPICIOUS_MIN_WEIGHT) {
      metadata.below_suspicious_min = true;
      score *= 0.3;
    } else if (weight > PHYSIOLOGICAL_LIMITS.SUSPICIOUS_MAX_WEIGHT) {
      metadata.above_suspicious_max = true;
      score *= 0.3;
    }

    // 2. Time-aware change detection
    if (recentWeights && recentTimestamps && recentWeights.length > 0 && recentTimestamps.length > 0) {
      const minLen = Math.min(recentWeights.length, recentTimestamps.length);
      const weights = recentWeights.slice(-minLen);
      const timestamps = recentTimestamps.slice(-minLen);

      if (weights.length > 0) {
        const previousWeight = weights[weights.length - 1];
        const weightChange = Math.abs(weight - previousWeight);

        if (timestamps.length >= 1) {
          const current = currentTimestamp || new Date();
          const prevTimestamp = timestamps[timestamps.length - 1];

          const timeDiffSeconds = (current.getTime() - prevTimestamp.getTime()) / 1000;
          const timeDiffMinutes = timeDiffSeconds / 60.0;
          const timeDiffHours = timeDiffSeconds / 3600.0;

          // Duplicate detection
          if (timeDiffSeconds < UnifiedQualityScorer.DUPLICATE_THRESHOLD_SECONDS) {
            if (weightChange < 0.05) {
              metadata.rejected_reason = 'duplicate_measurement';
              metadata.time_diff_seconds = timeDiffSeconds;
              return [0.0, metadata];
            } else if (weightChange < 0.2) {
              score *= 0.8;
              metadata.rapid_but_different = true;
            }
          } else if (timeDiffMinutes < UnifiedQualityScorer.RAPID_THRESHOLD_MINUTES) {
            // Rapid measurement handling
            let sourceFactor = 1.0;
            if (this.currentSource) {
              if (this.currentSource.toLowerCase().includes('device')) {
                sourceFactor = 1.5;
              } else if (this.currentSource.toLowerCase().includes('manual') ||
                         this.currentSource.toLowerCase().includes('upload')) {
                sourceFactor = 1.2;
              }
            }

            const maxAllowed = (0.5 + 0.5 * (1 - Math.exp(-timeDiffMinutes / 2))) * sourceFactor;

            if (weightChange > maxAllowed * 2) {
              metadata.rejected_reason = 'rapid_impossible_change';
              metadata.time_diff_minutes = timeDiffMinutes;
              metadata.change_kg = weightChange;
              return [0.0, metadata];
            } else if (weightChange > maxAllowed) {
              const excessRatio = (weightChange - maxAllowed) / maxAllowed;
              const rapidPenalty = Math.exp(-excessRatio);
              score *= rapidPenalty;
              metadata.rapid_measurement_penalty = rapidPenalty;
            }
          }

          // Burst pattern detection
          if (timestamps.length >= UnifiedQualityScorer.BURST_COUNT_THRESHOLD) {
            let burstCount = 1;
            const lookback = timestamps.slice(-(UnifiedQualityScorer.BURST_COUNT_THRESHOLD + 2));
            for (const ts of lookback) {
              if ((current.getTime() - ts.getTime()) / 60000 <= UnifiedQualityScorer.BURST_WINDOW_MINUTES) {
                burstCount++;
              }
            }

            if (burstCount >= UnifiedQualityScorer.BURST_COUNT_THRESHOLD) {
              metadata.burst_pattern_detected = true;
              metadata.burst_count = burstCount;
              const burstPenalty = Math.max(0.6, 1.0 - (burstCount - 4) * 0.1);
              score *= burstPenalty;
              metadata.burst_penalty = burstPenalty;
            }
          }

          metadata.time_diff_hours = timeDiffHours;

          // Physiological limits
          const maxChange = this.calculateMaxPhysiologicalChange(timeDiffHours);
          metadata.max_physiological_change = maxChange;
          metadata.actual_change = weightChange;

          if (maxChange <= 0) {
            if (weightChange > 0.1) {
              score *= 0.1;
              metadata.same_time_penalty = true;
            }
          } else if (weightChange > maxChange) {
            const excessRatio = (weightChange - maxChange) / maxChange;
            if (excessRatio > 2.0) {
              metadata.rejected_reason = 'physiological_limit_exceeded';
              return [0.0, metadata];
            }
            const physioPenalty = Math.exp(-excessRatio);
            score *= physioPenalty;
            metadata.physiological_penalty = physioPenalty;
          }
        }
      }
    }

    score = Math.max(0.0, Math.min(1.0, score));
    return [score, metadata];
  }

  /**
   * Calculate source reliability based on source profiles.
   */
  calculateSourceReliability(source: string): number {
    // Get source-specific profile from config or use default
    const profile = this.sourceProfiles?.[source] || DEFAULT_SOURCE_PROFILE;
    const noiseMultiplier = profile.noise_multiplier ?? 1.0;

    // Invert and normalize to [0, 1]
    // Higher noise multiplier = lower reliability
    // noise_multiplier ranges from 0.5 (excellent) to 3.0 (poor)
    let reliability = 1.0 - ((noiseMultiplier - 0.5) / 2.5);
    reliability = Math.max(0.2, Math.min(1.0, reliability));

    return reliability;
  }

  /**
   * Calculate alignment with established trend using linear regression.
   */
  calculateTrendAlignment(
    weight: number,
    kalmanState?: ProcessorState | null,
    recentWeights?: number[] | null
  ): [number, Record<string, any>] {
    const metadata: any = {};

    if (!recentWeights || recentWeights.length < 5) {
      metadata.reason = 'Insufficient data for trend';
      return [0.8, metadata];
    }

    let weights = recentWeights;

    // Use Kalman filtered weights if available
    if (kalmanState && (kalmanState as any).measurementHistory) {
      const history = (kalmanState as any).measurementHistory;
      if (Array.isArray(history) && history.length >= 5) {
        const kalmanWeights = history.slice(-10).map((h: any) => h.filtered_weight || h.weight).filter((w: any) => w !== undefined);
        if (kalmanWeights.length >= 5) {
          weights = kalmanWeights;
        }
      }
    }

    // Linear regression
    const x = Array.from({ length: weights.length }, (_, i) => i);
    const [slope, intercept] = linearRegression(x, weights);
    const predictedNext = slope * weights.length + intercept;

    metadata.trend_slope = slope;
    metadata.predicted = predictedNext;

    // Calculate deviation
    const deviation = Math.abs(weight - predictedNext);
    const trendLine = x.map(xi => slope * xi + intercept);
    const residuals = weights.map((w, i) => w - trendLine[i]);
    let stdDev = std(residuals);

    const trendConfig = this.config.trend_alignment || {};
    const minStdDev = trendConfig.trend_min_std_dev ?? 0.5;
    if (stdDev < minStdDev) {
      stdDev = minStdDev;
    }

    metadata.deviation = deviation;
    metadata.std_dev = stdDev;

    const normalizedDeviation = deviation / stdDev;
    const k = trendConfig.trend_decay_constant ?? 0.3;
    let score = Math.exp(-k * normalizedDeviation);
    score = Math.max(0.3, score);

    return [score, metadata];
  }

  /**
   * Calculate maximum physiological weight change based on time elapsed.
   */
  private calculateMaxPhysiologicalChange(timeHours: number): number {
    if (timeHours <= 0) return 0.0;

    // Ultra short-term (< 1 minute)
    if (timeHours < 0.0167) {
      return PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1MIN || 0.5;
    }

    // Very short-term (< 5 minutes)
    if (timeHours < 0.0833) {
      const max1min = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1MIN || 0.5;
      const max5min = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_5MIN || 1.0;
      const minutes = timeHours * 60;
      return max1min + (max5min - max1min) * (minutes - 1) / 4;
    }

    // Short-term (< 1 hour)
    if (timeHours < 1) {
      const max5min = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_5MIN || 0.3;
      const max1h = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1H || 1.0;
      const minutes = timeHours * 60;
      if (minutes <= 5) return max5min;
      return max5min + (max1h - max5min) * Math.log(minutes / 5) / Math.log(12);
    }

    // Hours (1-6 hours)
    if (timeHours <= 6) {
      const baseChange = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1H || 1.0;
      const max6h = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_6H || 3.0;
      const additional = (max6h - baseChange) * Math.log(1 + (timeHours - 1)) / Math.log(6);
      return baseChange + additional;
    }

    // Day (6-24 hours)
    if (timeHours <= 24) {
      const baseChange = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_6H || 3.0;
      const max24h = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_24H || 4.0;
      const additional = (max24h - baseChange) * Math.log(1 + (timeHours - 6) / 6) / Math.log(4);
      return baseChange + additional;
    }

    // Week (1-7 days)
    if (timeHours <= 168) {
      const days = timeHours / 24;
      const dailyMax = PHYSIOLOGICAL_LIMITS.MAX_DAILY_CHANGE_KG || 2.0;
      const weeklyMax = PHYSIOLOGICAL_LIMITS.MAX_WEEKLY_CHANGE_KG || 3.5;
      return Math.min(weeklyMax, dailyMax * Math.sqrt(days));
    }

    // Long-term (> 1 week)
    const days = timeHours / 24;
    const weeklyMax = PHYSIOLOGICAL_LIMITS.MAX_WEEKLY_CHANGE_KG || 3.5;
    const sustainedDaily = PHYSIOLOGICAL_LIMITS.MAX_SUSTAINED_DAILY_KG || 0.5;
    return days <= 7 ? weeklyMax : weeklyMax + (days - 7) * sustainedDaily;
  }

  /**
   * Calculate weighted geometric mean of component scores.
   */
  private calculateWeightedGeometricMean(components: Record<string, number>): number {
    if (Object.keys(components).length === 0) return 0.0;

    const epsilon = 1e-10;
    let product = 1.0;
    let weightSum = 0.0;

    for (const [componentName, score] of Object.entries(components)) {
      const weight = this.weights[componentName] || 0.0;
      if (weight > 0) {
        const clampedScore = Math.max(epsilon, Math.min(1.0, score));
        product *= Math.pow(clampedScore, weight);
        weightSum += weight;
      }
    }

    if (weightSum > 0) {
      const overall = Math.pow(product, 1.0 / weightSum);
      return Math.max(0.0, Math.min(1.0, overall));
    }

    return 0.0;
  }

  /**
   * Calculate weighted harmonic mean of component scores.
   */
  private calculateWeightedHarmonicMean(components: Record<string, number>): number {
    if (Object.keys(components).length === 0) return 0.0;

    const epsilon = 1e-10;
    let sumWeightedInverse = 0.0;
    let weightSum = 0.0;

    for (const [componentName, score] of Object.entries(components)) {
      const weight = this.weights[componentName] || 0.0;
      if (weight > 0) {
        const clampedScore = Math.max(epsilon, Math.min(1.0, score));
        sumWeightedInverse += weight / clampedScore;
        weightSum += weight;
      }
    }

    if (weightSum > 0 && sumWeightedInverse > 0) {
      const overall = weightSum / sumWeightedInverse;
      return Math.max(0.0, Math.min(1.0, overall));
    }

    return 0.0;
  }

  /**
   * Update rolling temporal baseline for continuity across measurements.
   */
  update_temporal_baseline(state: ProcessorState, weight: number, timestamp: Date): ProcessorState {
    const baseline = state.temporal_baseline || {};

    if (baseline.lastWeight && baseline.lastTimestamp) {
      let last_ts = baseline.lastTimestamp;
      if (typeof last_ts === 'string') {
        last_ts = new Date(last_ts);
      }

      const time_diff = (timestamp.getTime() - last_ts.getTime()) / (1000 * 3600); // hours
      if (time_diff > 0) {
        const weight_change = Math.abs(weight - baseline.lastWeight);
        const daily_rate = weight_change / Math.max(time_diff / 24, 0.1);

        // Exponential moving average with α=0.3
        const prev_rate = baseline.rolling_avg_change_rate || daily_rate;
        baseline.rolling_avg_change_rate = 0.3 * daily_rate + 0.7 * prev_rate;
      }
    }

    baseline.lastWeight = weight;
    baseline.lastTimestamp = timestamp instanceof Date ? timestamp.toISOString() : timestamp;

    state.temporal_baseline = baseline;
    return state;
  }
}
