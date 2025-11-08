/**
 * Unified Kalman-centric quality scoring system.
 * Replaces dual validation with single Kalman-deviation-based quality scorer.
 */

import {
  BMI_LIMITS,
  DEFAULT_PROFILE,
  PHYSIOLOGICAL_LIMITS,
  SOURCE_PROFILES,
  type SourceProfile,
} from '../constants';
import { base as statsBase } from '@stdlib/stats';
import { base as mathBase } from '@stdlib/math';
import { validateNumber, isNaN as stdlibIsNaN, isFinite as stdlibIsFinite } from '../stdlib-utils';

/**
 * Container for quality score and its components.
 */
export interface QualityScore {
  overall: number;
  components: Record<string, number>;
  threshold: number;
  accepted: boolean;
  rejectionReason?: string;
  metadata: Record<string, any>;
}

/**
 * Create a QualityScore object with proper initialization.
 */
export function createQualityScore(
  overall: number,
  components: Record<string, number>,
  threshold: number = 0.46,
  metadata: Record<string, any> = {}
): QualityScore {
  const accepted = overall >= threshold;
  let rejectionReason: string | undefined;

  if (!accepted && !rejectionReason) {
    if (Object.keys(components).length > 0) {
      // Find weakest component
      const entries = Object.entries(components);
      const minComponent = entries.reduce((min, curr) =>
        curr[1] < min[1] ? curr : min
      );
      rejectionReason =
        `Quality score ${overall.toFixed(2)} below threshold ${threshold} ` +
        `(weakest: ${minComponent[0]}=${minComponent[1].toFixed(2)})`;
    } else {
      rejectionReason =
        `Quality score ${overall.toFixed(2)} below threshold ${threshold} ` +
        `(no components calculated)`;
    }
  }

  return {
    overall,
    components,
    threshold,
    accepted,
    rejectionReason,
    metadata,
  };
}

/**
 * Configuration for UnifiedQualityScorer.
 */
export interface QualityScorerConfig {
  componentWeights?: Record<string, number>;
  threshold?: number;
  temporalThresholds?: Record<string, number>;
  useHarmonicMean?: boolean;
  trendAlignment?: {
    trendMinStdDev?: number;
    trendDecayConstant?: number;
  };
}

/**
 * Unified Kalman-centric quality scoring system.
 * Primary signal is deviation from Kalman prediction.
 */
export class UnifiedQualityScorer {
  // Default component weights (must sum to 1.0)
  private static readonly DEFAULT_WEIGHTS: Record<string, number> = {
    kalman_fit: 0.4, // Primary signal
    temporal_consistency: 0.3,
    anomaly_detection: 0.2,
    source_reliability: 0.05,
    trend_alignment: 0.05,
  };

  // Time-based thresholds for temporal consistency
  private static readonly TEMPORAL_THRESHOLDS: Record<string, number> = {
    '6h': 3.0, // 3kg in 6 hours
    '24h': 2.0, // 2kg in 24 hours
    sustained: 2.0, // 2kg/day sustained
  };

  // Anomaly patterns
  private static readonly UNIT_CONFUSION_FACTORS = [2.2, 0.454, 10.0, 0.1]; // kg/lbs, lbs/kg, decimal errors
  private static readonly BMI_RANGE: [number, number] = [15.0, 50.0]; // Common BMI range

  // Rapid measurement detection thresholds
  private static readonly DUPLICATE_THRESHOLD_SECONDS = 5;
  private static readonly RAPID_THRESHOLD_MINUTES = 5;
  private static readonly BURST_WINDOW_MINUTES = 30;
  private static readonly BURST_COUNT_THRESHOLD = 5;
  private static readonly MAX_1MIN_CHANGE_KG = 0.5;
  private static readonly MAX_5MIN_CHANGE_KG = 1.0;

  private readonly config: QualityScorerConfig;
  private readonly weights: Record<string, number>;
  private readonly threshold: number;
  private readonly temporalThresholds: Record<string, number>;
  private currentSource?: string;

  constructor(config: QualityScorerConfig = {}) {
    this.config = config;

    // Get component weights from config
    this.weights = { ...UnifiedQualityScorer.DEFAULT_WEIGHTS, ...config.componentWeights };

    // Normalize weights to sum to 1.0
    const weightSum = Object.values(this.weights).reduce((sum, w) => sum + w, 0);
    if (weightSum > 0) {
      for (const key in this.weights) {
        this.weights[key] = this.weights[key] / weightSum;
      }
    }

    // Get thresholds
    this.threshold = config.threshold ?? 0.46;
    this.temporalThresholds = {
      ...UnifiedQualityScorer.TEMPORAL_THRESHOLDS,
      ...config.temporalThresholds,
    };
  }

  /**
   * Calculate unified quality score with Kalman-centric approach.
   */
  calculateQualityScore(params: {
    weight: number;
    source: string;
    kalmanState?: Record<string, any>;
    kalmanPrediction?: number;
    innovationCovariance?: number;
    previousWeight?: number;
    timeDiffHours?: number;
    recentWeights?: number[];
    recentTimestamps?: Date[];
    userHeightM?: number;
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

    const components: Record<string, number> = {};
    const metadata: Record<string, any> = {};

    // Store current source for use in anomaly detection
    this.currentSource = source;

    // Only calculate components with non-zero weights
    // 1. Kalman Fit Component
    if ((this.weights.kalman_fit ?? 0) > 0) {
      const [kalmanScore, kalmanMeta] = this.calculateKalmanFit(
        weight,
        kalmanPrediction,
        innovationCovariance,
        kalmanState
      );
      components.kalman_fit = kalmanScore;
      metadata.kalman_fit = kalmanMeta;
    }

    // 2. Temporal Consistency
    if ((this.weights.temporal_consistency ?? 0) > 0) {
      const [temporalScore, temporalMeta] = this.calculateTemporalConsistency(
        weight,
        previousWeight,
        timeDiffHours,
        recentWeights,
        recentTimestamps
      );
      components.temporal_consistency = temporalScore;
      metadata.temporal_consistency = temporalMeta;
    }

    // 3. Anomaly Detection
    if ((this.weights.anomaly_detection ?? 0) > 0) {
      // Try to get current timestamp from kalman_state or recent data
      let currentTs: Date | undefined;
      if (kalmanState?.current_timestamp) {
        currentTs =
          kalmanState.current_timestamp instanceof Date
            ? kalmanState.current_timestamp
            : new Date(kalmanState.current_timestamp);
      }

      const [anomalyScore, anomalyMeta] = this.calculateAnomalyDetection(
        weight,
        recentWeights,
        recentTimestamps,
        userHeightM,
        currentTs
      );
      components.anomaly_detection = anomalyScore;
      metadata.anomaly_detection = anomalyMeta;
    }

    // 4. Source Reliability
    if ((this.weights.source_reliability ?? 0) > 0) {
      const sourceScore = this.calculateSourceReliability(source);
      components.source_reliability = sourceScore;
      metadata.source_reliability = { source, score: sourceScore };
    }

    // 5. Trend Alignment
    if ((this.weights.trend_alignment ?? 0) > 0) {
      const [trendScore, trendMeta] = this.calculateTrendAlignment(
        weight,
        kalmanState,
        recentWeights
      );
      components.trend_alignment = trendScore;
      metadata.trend_alignment = trendMeta;
    }

    // Debug: Log component values to identify invalid ones
    if (process.env.VERBOSE_LOGGING) {
      const componentStatus = Object.entries(components).map(([name, value]) => {
        const isValid = typeof value === 'number' && validateNumber(value);
        return `${name}=${value} (${isValid ? 'valid' : 'INVALID'})`;
      }).join(', ');
      console.log(`[QualityScorer] Components: ${componentStatus}`);
    }

    // Calculate overall score using configured mean type
    const useHarmonic = this.config.useHarmonicMean ?? false;
    const overall = useHarmonic
      ? this.calculateWeightedHarmonicMean(components)
      : this.calculateWeightedGeometricMean(components);

    if (process.env.VERBOSE_LOGGING) {
      console.log(`[QualityScorer] Overall score: ${overall}`);
    }

    return createQualityScore(overall, components, this.threshold, metadata);
  }

  /**
   * Calculate how well measurement fits Kalman prediction.
   * Uses Mahalanobis distance and chi-squared test.
   * Applies time-based decay: importance decreases over time since last measurement.
   */
  private calculateKalmanFit(
    weight: number,
    kalmanPrediction?: number,
    innovationCovariance?: number,
    kalmanState?: Record<string, any>
  ): [number, Record<string, any>] {
    const metadata: Record<string, any> = {};

    // Debug: Check if values are NaN (not just undefined)
    if (process.env.VERBOSE_LOGGING) {
      const predStatus = kalmanPrediction === undefined ? 'undefined' :
                        stdlibIsNaN(kalmanPrediction) ? 'NaN' :
                        kalmanPrediction;
      const covStatus = innovationCovariance === undefined ? 'undefined' :
                       stdlibIsNaN(innovationCovariance) ? 'NaN' :
                       innovationCovariance;
      console.log(`[KalmanFit] prediction=${predStatus}, covariance=${covStatus}`);
    }

    // If no Kalman prediction available, return neutral score using stdlib validation
    if (!validateNumber(kalmanPrediction) || !validateNumber(innovationCovariance)) {
      metadata.reason = 'No Kalman prediction available';
      return [0.5, metadata];
    }

    // Calculate innovation (prediction error)
    const innovation = weight - kalmanPrediction;
    metadata.innovation = innovation;
    metadata.prediction = kalmanPrediction;

    // Handle zero or very small covariance
    let covarianceValue = innovationCovariance;
    if (covarianceValue <= 0) {
      covarianceValue = 1.0;
    }

    // Normalize innovation (Mahalanobis distance)
    const normalizedInnovation = Math.abs(innovation) / Math.sqrt(covarianceValue);
    metadata.normalized_innovation = normalizedInnovation;

    // Chi-squared test (df=1 for univariate)
    const chiSquared = normalizedInnovation ** 2;
    // For df=1, chi2.cdf(x, 1) = erf(sqrt(x/2))
    const pValue = 1 - this.erf(Math.sqrt(chiSquared / 2));
    metadata.chi_squared = chiSquared;
    metadata.p_value = pValue;

    // Check for adaptive period (relax thresholds)
    let inAdaptivePeriod = false;
    if (kalmanState) {
      const measurementsSinceReset = kalmanState.measurements_since_reset ?? 100;
      const resetParams = kalmanState.reset_parameters ?? {};
      const adaptationMeasurements = resetParams.adaptation_measurements ?? 10;
      if (measurementsSinceReset < adaptationMeasurements) {
        inAdaptivePeriod = true;
        metadata.adaptive_period = true;
      }
    }

    // Convert to quality score
    let score: number;
    if (inAdaptivePeriod) {
      // More forgiving during adaptation
      score = Math.exp(-0.2 * normalizedInnovation); // Slower decay
    } else {
      // Standard scoring
      score = Math.exp(-0.5 * normalizedInnovation); // Exponential decay
    }

    // Apply time-based decay for gap tolerance
    // After gaps, Kalman predictions become less reliable
    let daysSinceLast = 0;
    if (kalmanState?.last_timestamp) {
      const lastTimestamp =
        kalmanState.last_timestamp instanceof Date
          ? kalmanState.last_timestamp
          : new Date(kalmanState.last_timestamp);
      // Get current timestamp from state or use now as fallback
      const currentTimestamp =
        kalmanState.current_timestamp instanceof Date
          ? kalmanState.current_timestamp
          : kalmanState.current_timestamp
          ? new Date(kalmanState.current_timestamp)
          : new Date();
      daysSinceLast = (currentTimestamp.getTime() - lastTimestamp.getTime()) / 86400000;
      metadata.days_since_last = daysSinceLast;
    }

    // Apply decay factor based on time gap (but NOT during adaptation)
    if (daysSinceLast > 0 && !inAdaptivePeriod) {
      const decayFactor = Math.min(1.0, daysSinceLast / 30.0); // Linear decay over 30 days
      // Blend towards 1.0 (full acceptance) as time increases
      const adjustedScore = score + (1.0 - score) * decayFactor;
      metadata.decay_factor = decayFactor;
      metadata.original_score = score;
      score = adjustedScore;
    } else if (inAdaptivePeriod) {
      // During adaptation, no time decay - rely on Kalman uncertainty
      metadata.decay_skipped = 'adaptation_period';
    }

    // Ensure score is in [0, 1]
    score = Math.max(0.0, Math.min(1.0, score));
    metadata.score = score;

    return [score, metadata];
  }

  /**
   * Calculate temporal consistency using continuous exponential function.
   * Eliminates step functions that cause artificial cycles.
   */
  private calculateTemporalConsistency(
    weight: number,
    previousWeight?: number,
    timeDiffHours?: number,
    recentWeights?: number[],
    recentTimestamps?: Date[]
  ): [number, Record<string, any>] {
    const metadata: Record<string, any> = {};

    // Debug: Check if values are NaN (not just undefined)
    if (process.env.VERBOSE_LOGGING) {
      const prevStatus = previousWeight === undefined ? 'undefined' :
                        stdlibIsNaN(previousWeight) ? 'NaN' :
                        previousWeight;
      const timeStatus = timeDiffHours === undefined ? 'undefined' :
                        stdlibIsNaN(timeDiffHours) ? 'NaN' :
                        timeDiffHours;
      console.log(`[TemporalConsistency] previousWeight=${prevStatus}, timeDiffHours=${timeStatus}`);
    }

    // If no previous weight, return neutral score using stdlib validation
    if (!validateNumber(previousWeight) || !validateNumber(timeDiffHours)) {
      metadata.reason = 'No previous weight for comparison';
      return [0.7, metadata];
    }

    const weightChange = Math.abs(weight - previousWeight);

    // Exponential growth of acceptable change over time
    // Starts at 0.5kg for immediate, grows to ~5kg at 7 days
    const absTimeDiff = Math.abs(timeDiffHours);
    // Cap the exponent to prevent overflow for very large time differences
    const cappedTime = Math.min(absTimeDiff, 336); // Cap at 2 weeks (336 hours)
    const maxAcceptableChange = 0.5 + 4.5 * (1 - Math.exp(-cappedTime / 48));

    metadata.max_acceptable_change = maxAcceptableChange;
    metadata.actual_change = weightChange;
    metadata.time_diff_hours = timeDiffHours;

    // Smooth scoring based on deviation from acceptable
    let score: number;
    if (weightChange <= maxAcceptableChange) {
      // Within acceptable range: high score with smooth decay
      score = 0.8 + 0.2 * Math.exp(-weightChange / maxAcceptableChange);
    } else {
      // Beyond acceptable: exponential penalty
      const excessRatio = (weightChange - maxAcceptableChange) / maxAcceptableChange;
      score = 0.8 * Math.exp(-excessRatio);
    }

    // Check for adaptive period from kalman state (more lenient during adaptation)
    if (timeDiffHours > 168) {
      // More than a week gap
      score = Math.max(score, 0.4);
      metadata.gap_adjustment = true;
    }

    // Clamp between 0.2 and 1.0
    score = Math.max(0.2, Math.min(1.0, score));

    return [score, metadata];
  }

  /**
   * Enhanced anomaly detection with time-aware physiological limits.
   */
  private calculateAnomalyDetection(
    weight: number,
    recentWeights?: number[],
    recentTimestamps?: Date[],
    userHeightM?: number,
    currentTimestamp?: Date
  ): [number, Record<string, any>] {
    const metadata: Record<string, any> = {};
    let score = 1.0;

    // 1. Check absolute physiological bounds
    if (weight < PHYSIOLOGICAL_LIMITS.ABSOLUTE_MIN_WEIGHT) {
      metadata.outside_absolute_min = true;
      return [0.0, metadata]; // Reject outright
    }

    if (weight > PHYSIOLOGICAL_LIMITS.ABSOLUTE_MAX_WEIGHT) {
      metadata.outside_absolute_max = true;
      return [0.0, metadata]; // Reject outright
    }

    // Check suspicious bounds (softer penalty)
    if (weight < PHYSIOLOGICAL_LIMITS.SUSPICIOUS_MIN_WEIGHT) {
      metadata.below_suspicious_min = true;
      score *= 0.3;
    } else if (weight > PHYSIOLOGICAL_LIMITS.SUSPICIOUS_MAX_WEIGHT) {
      metadata.above_suspicious_max = true;
      score *= 0.3;
    }

    // 2. Time-aware change detection
    if (recentWeights && recentTimestamps) {
      // Ensure we have matching lengths
      const minLen = Math.min(recentWeights.length, recentTimestamps.length);
      const weights = recentWeights.slice(-minLen);
      const timestamps = recentTimestamps.slice(-minLen);

      if (weights.length > 0) {
        const previousWeight = weights[weights.length - 1];
        const weightChange = Math.abs(weight - previousWeight);

        // Calculate time difference
        if (timestamps.length >= 1) {
          // Use provided timestamp or fall back to now
          const currentTs = currentTimestamp ?? new Date();

          let prevTimestamp = timestamps[timestamps.length - 1];
          if (!(prevTimestamp instanceof Date)) {
            prevTimestamp = new Date(prevTimestamp);
          }

          // Calculate time differences
          const timeDiffSeconds = (currentTs.getTime() - prevTimestamp.getTime()) / 1000;
          const timeDiffMinutes = timeDiffSeconds / 60.0;
          const timeDiffHours = timeDiffSeconds / 3600.0;

          // Check for minute-level precision (likely from manual entry)
          const hasMinutePrecision =
            currentTs.getSeconds() === 0 &&
            currentTs.getMilliseconds() === 0 &&
            prevTimestamp.getSeconds() === 0 &&
            prevTimestamp.getMilliseconds() === 0;

          // Enhanced rapid-fire measurement detection
          if (timeDiffSeconds < UnifiedQualityScorer.DUPLICATE_THRESHOLD_SECONDS) {
            // Check if weight is essentially the same (within 50g)
            if (weightChange < 0.05) {
              metadata.rejected_reason = 'duplicate_measurement';
              metadata.time_diff_seconds = timeDiffSeconds;
              metadata.threshold_seconds = UnifiedQualityScorer.DUPLICATE_THRESHOLD_SECONDS;
              return [0.0, metadata]; // Reject as duplicate
            }
            // Allow small variations (scale noise) even in rapid succession
            else if (weightChange < 0.2) {
              score *= 0.8; // Minor penalty for rapid but different reading
              metadata.rapid_but_different = true;
            }
          } else if (timeDiffMinutes < UnifiedQualityScorer.RAPID_THRESHOLD_MINUTES) {
            // Calculate adaptive threshold based on time and source
            let sourceFactor = 1.0;
            if (this.currentSource) {
              const sourceLower = this.currentSource.toLowerCase();
              if (sourceLower.includes('device')) {
                sourceFactor = 1.5; // 50% more lenient for devices
              } else if (sourceLower.includes('manual') || sourceLower.includes('upload')) {
                sourceFactor = 1.2; // 20% more lenient for manual
              }
            }

            // Smooth exponential growth of allowed change
            let maxAllowed = 0.5 + 0.5 * (1 - Math.exp(-timeDiffMinutes / 2));
            maxAllowed *= sourceFactor;

            if (weightChange > maxAllowed * 2) {
              // Only reject if WAY over (2x)
              metadata.rejected_reason = 'rapid_impossible_change';
              metadata.time_diff_minutes = timeDiffMinutes;
              metadata.change_kg = weightChange;
              metadata.max_allowed_change = maxAllowed;
              return [0.0, metadata]; // Reject as impossible
            } else if (weightChange > maxAllowed) {
              // Over threshold but not impossible - apply gradual penalty
              const excessRatio = (weightChange - maxAllowed) / maxAllowed;
              const rapidPenalty = Math.exp(-excessRatio); // Smoother penalty
              score *= rapidPenalty;
              metadata.rapid_measurement_penalty = rapidPenalty;
              metadata.time_diff_minutes = timeDiffMinutes;
              metadata.exceeded_soft_threshold = true;
            } else {
              // Within acceptable range for short-term change
              const timePenalty =
                0.9 + 0.1 * (timeDiffMinutes / UnifiedQualityScorer.RAPID_THRESHOLD_MINUTES);
              score *= timePenalty;
              metadata.minor_time_penalty = timePenalty;
            }
          }

          // Additional check: Look for burst patterns
          if (timestamps.length >= UnifiedQualityScorer.BURST_COUNT_THRESHOLD) {
            let burstCount = 1; // Start with current measurement
            const lookbackStart = Math.max(
              0,
              timestamps.length - (UnifiedQualityScorer.BURST_COUNT_THRESHOLD + 2)
            );
            for (let i = lookbackStart; i < timestamps.length; i++) {
              let ts = timestamps[i];
              if (!(ts instanceof Date)) {
                ts = new Date(ts);
              }
              if (
                (currentTs.getTime() - ts.getTime()) / 60000 <=
                UnifiedQualityScorer.BURST_WINDOW_MINUTES
              ) {
                burstCount++;
              }
            }

            if (burstCount >= UnifiedQualityScorer.BURST_COUNT_THRESHOLD) {
              metadata.burst_pattern_detected = true;
              metadata.burst_count = burstCount;
              metadata.burst_window_minutes = UnifiedQualityScorer.BURST_WINDOW_MINUTES;

              // Less aggressive penalty
              const burstPenalty = Math.max(0.6, 1.0 - (burstCount - 4) * 0.1);
              score *= burstPenalty;
              metadata.burst_penalty = burstPenalty;
            }
          }

          metadata.time_diff_hours = timeDiffHours;

          // Time-based physiological limits
          const maxChange = this.calculateMaxPhysiologicalChange(timeDiffHours);
          metadata.max_physiological_change = maxChange;
          metadata.actual_change = weightChange;

          // Track individual penalty components for weighted average
          const penaltyComponents: number[] = [];
          const penaltyWeights: number[] = [];

          // Apply penalty based on deviation from max allowed
          if (maxChange <= 0) {
            // Same timestamp or negative time diff
            if (weightChange > 0.1) {
              score *= 0.1; // Heavy penalty
              metadata.same_time_penalty = true;
            }
          } else if (weightChange > maxChange) {
            // Calculate severity of violation
            const excessRatio = (weightChange - maxChange) / maxChange;
            metadata.excess_ratio = excessRatio;

            if (excessRatio > 1.0) {
              // More than double the max
              penaltyComponents.push(0.0); // Impossible change
              penaltyWeights.push(2.0);
              metadata.impossible_change = true;
            } else if (excessRatio > 0.5) {
              // 50% over max
              penaltyComponents.push(0.2);
              penaltyWeights.push(1.5);
              metadata.very_unlikely_change = true;
            } else {
              // More gradual penalty curve
              const penaltyScore = Math.max(0.4, 0.7 - excessRatio * 0.5);
              penaltyComponents.push(penaltyScore);
              penaltyWeights.push(1.0);
              metadata.unlikely_change = true;
            }
          } else {
            // No physiological penalty
            penaltyComponents.push(1.0);
            penaltyWeights.push(1.0);
          }

          // 3. Check for percentage-based changes
          if (timeDiffHours > 72 && timeDiffHours <= 720) {
            // Between 3-30 days
            const percentChange = (weightChange / previousWeight) * 100;
            const maxMonthlyPercent = PHYSIOLOGICAL_LIMITS.MAX_MONTHLY_PERCENT;

            // More generous time scaling
            const timeFactor = Math.max(0.2, Math.min(1.0, Math.sqrt(timeDiffHours / 720)));
            const allowedPercent = maxMonthlyPercent * timeFactor;

            metadata.percent_change = percentChange;
            metadata.allowed_percent = allowedPercent;

            if (percentChange > allowedPercent) {
              const excessPercentRatio = (percentChange - allowedPercent) / allowedPercent;
              metadata.excess_percent_ratio = excessPercentRatio;

              if (excessPercentRatio > 2.0) {
                penaltyComponents.push(0.0);
                penaltyWeights.push(2.0);
                metadata.impossible_percent_change = true;
              } else if (excessPercentRatio > 1.0) {
                penaltyComponents.push(0.15);
                penaltyWeights.push(1.2);
                metadata.extreme_percent_change = true;
              } else if (excessPercentRatio > 0.5) {
                penaltyComponents.push(0.35);
                penaltyWeights.push(0.8);
                metadata.high_percent_change = true;
              } else {
                const penaltyScore = Math.max(0.5, 0.8 - excessPercentRatio * 0.4);
                penaltyComponents.push(penaltyScore);
                penaltyWeights.push(0.6);
                metadata.suspicious_percent_change = true;
              }
            } else {
              // No percentage penalty
              penaltyComponents.push(1.0);
              penaltyWeights.push(0.5);
            }
          }

          // 4. Check for sustained vs. fluctuation patterns
          if (weights.length >= 3) {
            const sustainedScore = this.checkSustainedPattern(weight, weights, timestamps);
            metadata.sustained_pattern_score = sustainedScore;
            penaltyComponents.push(sustainedScore);
            penaltyWeights.push(0.7);
          }

          // Calculate weighted average of penalties
          if (penaltyComponents.length > 0) {
            const totalWeight = penaltyWeights.reduce((sum, w) => sum + w, 0);
            if (totalWeight > 0) {
              let weightedScore = 0;
              for (let i = 0; i < penaltyComponents.length; i++) {
                weightedScore += penaltyComponents[i] * penaltyWeights[i];
              }
              weightedScore /= totalWeight;

              // Apply a minimum floor
              if (weightedScore > 0) {
                weightedScore = Math.max(0.25, weightedScore);
              }
              score *= weightedScore;
              metadata.penalty_method = 'weighted_average';
              metadata.penalty_components = penaltyComponents;
              metadata.penalty_weights = penaltyWeights;
            }
          }
        }
      }
    }

    return [Math.max(0.0, Math.min(1.0, score)), metadata];
  }

  /**
   * Calculate maximum physiological weight change based on time elapsed.
   */
  private calculateMaxPhysiologicalChange(timeHours: number): number {
    if (timeHours <= 0) {
      return 0.0;
    }

    // Ultra short-term (< 1 minute): Scale variance + positioning
    if (timeHours < 0.0167) {
      // 1 minute
      return PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1MIN;
    }

    // Very short-term (< 5 minutes): Scale variance + water/bathroom
    else if (timeHours < 0.0833) {
      // 5 minutes
      const max1min = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1MIN;
      const max5min = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_5MIN;
      // Linear interpolation
      const minutes = timeHours * 60;
      return max1min + ((max5min - max1min) * (minutes - 1)) / 4;
    }

    // Short-term (< 1 hour): Limited by water/food intake
    else if (timeHours < 1) {
      const max5min = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_5MIN;
      const max1h = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1H;
      const minutes = timeHours * 60;
      if (minutes <= 5) {
        return max5min;
      } else {
        // Logarithmic growth from 5 min to 1 hour
        return max5min + (max1h - max5min) * (Math.log(minutes / 5) / Math.log(12));
      }
    }

    // Hours (1-6 hours): Water, food, exercise effects
    else if (timeHours <= 6) {
      const baseChange = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_1H;
      const max6h = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_6H;
      const additional = ((max6h - baseChange) * Math.log(1 + (timeHours - 1))) / Math.log(6);
      return baseChange + additional;
    }

    // Day (6-24 hours): Full daily fluctuation
    else if (timeHours <= 24) {
      const baseChange = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_6H;
      const max24h = PHYSIOLOGICAL_LIMITS.MAX_CHANGE_24H;
      const additional =
        ((max24h - baseChange) * Math.log(1 + (timeHours - 6) / 6)) / Math.log(4);
      return baseChange + additional;
    }

    // Week (1-7 days): Compound changes with realistic limits
    else if (timeHours <= 168) {
      // 7 days
      const days = timeHours / 24;
      const dailyMax = PHYSIOLOGICAL_LIMITS.MAX_DAILY_CHANGE_KG;
      const weeklyMax = PHYSIOLOGICAL_LIMITS.MAX_WEEKLY_CHANGE_KG;
      return Math.min(weeklyMax, dailyMax * Math.sqrt(days));
    }

    // Long-term (> 1 week): Sustainable rates only
    else {
      const days = timeHours / 24;
      const weeklyMax = PHYSIOLOGICAL_LIMITS.MAX_WEEKLY_CHANGE_KG;
      const sustainedDaily = PHYSIOLOGICAL_LIMITS.MAX_SUSTAINED_DAILY_KG;

      // First week at aggressive rate, then sustainable rate
      if (days <= 7) {
        return weeklyMax;
      } else {
        // Additional sustainable change after first week
        return weeklyMax + (days - 7) * sustainedDaily;
      }
    }
  }

  /**
   * Check if changes follow a sustained pattern vs. erratic fluctuations.
   */
  private checkSustainedPattern(
    currentWeight: number,
    recentWeights: number[],
    recentTimestamps: Date[]
  ): number {
    if (recentWeights.length < 3) {
      return 1.0; // Not enough data
    }

    // Look at last 5 measurements or available data
    const lookback = Math.min(5, recentWeights.length);
    const weights = [...recentWeights.slice(-lookback), currentWeight];

    // Calculate successive differences
    const differences: number[] = [];
    for (let i = 0; i < weights.length - 1; i++) {
      differences.push(weights[i + 1] - weights[i]);
    }

    // Check consistency of direction (all gains or all losses)
    const positive = differences.filter((d) => d > 0.1).length;
    const negative = differences.filter((d) => d < -0.1).length;

    // Consistent direction is more believable
    if (positive === differences.length || negative === differences.length) {
      return 1.0; // Perfectly consistent
    } else {
      // Mixed directions - calculate variance
      const variance = this.variance(differences);
      const meanAbsChange = this.mean(differences.map(Math.abs));

      if (meanAbsChange > 0) {
        // Coefficient of variation
        const cv = Math.sqrt(variance) / meanAbsChange;
        // Lower CV = more consistent
        const consistencyScore = Math.exp(-cv * 0.5);
        return consistencyScore;
      } else {
        return 1.0;
      }
    }
  }

  /**
   * Calculate source reliability based on SOURCE_PROFILES.
   */
  private calculateSourceReliability(source: string): number {
    const profile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;

    // Convert noise_multiplier to reliability score
    const noiseMultiplier = profile.noise_multiplier;

    // Invert and normalize to [0, 1]
    // noise_multiplier range: 0.5 (best) to 3.0 (worst)
    let reliability = 1.0 - (noiseMultiplier - 0.5) / 2.5;
    reliability = Math.max(0.2, Math.min(1.0, reliability)); // Clamp to [0.2, 1.0]

    return reliability;
  }

  /**
   * Calculate alignment with established trend using linear regression.
   */
  private calculateTrendAlignment(
    weight: number,
    kalmanState?: Record<string, any>,
    recentWeights?: number[]
  ): [number, Record<string, any>] {
    const metadata: Record<string, any> = {};

    // Need at least 5 measurements for trend
    if (!recentWeights || recentWeights.length < 5) {
      metadata.reason = 'Insufficient data for trend';
      return [0.8, metadata]; // Neutral-high score
    }

    let weightsToUse = recentWeights;

    // Get recent Kalman states if available
    if (kalmanState?.measurement_history) {
      const history = kalmanState.measurement_history;
      if (Array.isArray(history) && history.length >= 5) {
        // Use Kalman filtered weights for trend
        const kalmanWeights: number[] = [];
        for (const h of history.slice(-10)) {
          const w = h.filtered_weight ?? h.weight;
          if (typeof w === 'number') {
            kalmanWeights.push(w);
          }
        }
        if (kalmanWeights.length >= 5) {
          weightsToUse = kalmanWeights;
        }
      }
    }

    // Perform linear regression on recent weights
    const x: number[] = [];
    const y = weightsToUse;
    for (let i = 0; i < y.length; i++) {
      x.push(i);
    }

    // Calculate trend line using polyfit
    const [slope, intercept] = this.polyfit(x, y, 1);
    const predictedNext = slope * y.length + intercept;

    metadata.trend_slope = slope;
    metadata.predicted = predictedNext;

    // Calculate deviation from trend
    const deviation = Math.abs(weight - predictedNext);

    // Expected variance around trend (use std of residuals)
    const trendLine: number[] = x.map((xi) => slope * xi + intercept);
    const residuals: number[] = y.map((yi, i) => yi - trendLine[i]);
    let stdDev = this.std(residuals);

    // Ensure minimum std_dev to avoid division by zero
    const trendConfig = this.config.trendAlignment ?? {};
    const minStdDev = trendConfig.trendMinStdDev ?? 0.5;
    if (stdDev < minStdDev) {
      stdDev = minStdDev;
    }

    metadata.deviation = deviation;
    metadata.std_dev = stdDev;

    // Score based on deviation from trend
    const normalizedDeviation = deviation / stdDev;

    // More gradual scoring: use exponential decay
    const k = trendConfig.trendDecayConstant ?? 0.3;
    let score = Math.exp(-k * normalizedDeviation);

    // Ensure minimum score of 0.3
    score = Math.max(0.3, score);

    return [score, metadata];
  }

  /**
   * Calculate weighted geometric mean of component scores.
   */
  private calculateWeightedGeometricMean(components: Record<string, number>): number {
    if (Object.keys(components).length === 0) {
      return 0.0;
    }

    // Ensure all scores are positive
    const epsilon = 1e-10;

    let product = 1.0;
    let weightSum = 0.0;

    for (const [componentName, score] of Object.entries(components)) {
      const weight = this.weights[componentName] ?? 0.0;
      if (weight > 0) {
        // Handle NaN, undefined, null, and invalid scores using stdlib validation
        if (!validateNumber(score)) {
          // Skip invalid scores or use neutral value
          continue;
        }
        // Clamp score to avoid numerical issues
        const clampedScore = Math.max(epsilon, Math.min(1.0, score));
        product *= clampedScore ** weight;
        weightSum += weight;
      }
    }

    let overall: number;
    if (weightSum > 0) {
      // Normalize by weight sum
      overall = product ** (1.0 / weightSum);
    } else {
      overall = 0.0;
    }

    return Math.max(0.0, Math.min(1.0, overall));
  }

  /**
   * Calculate weighted harmonic mean of component scores.
   */
  private calculateWeightedHarmonicMean(components: Record<string, number>): number {
    if (Object.keys(components).length === 0) {
      return 0.0;
    }

    // Ensure all scores are positive
    const epsilon = 1e-10;

    let weightedSum = 0.0;
    let weightSum = 0.0;

    for (const [componentName, score] of Object.entries(components)) {
      const weight = this.weights[componentName] ?? 0.0;
      if (weight > 0) {
        // Handle NaN, undefined, null, and invalid scores using stdlib validation
        if (!validateNumber(score)) {
          // Skip invalid scores
          continue;
        }
        // Clamp score to avoid numerical issues
        const clampedScore = Math.max(epsilon, Math.min(1.0, score));
        weightedSum += weight / clampedScore;
        weightSum += weight;
      }
    }

    // Ensure we have valid values before division
    if (weightSum > 0 && weightedSum > epsilon) {
      return weightSum / weightedSum;
    } else if (weightSum > 0) {
      return epsilon;
    } else {
      return 0.0;
    }
  }

  /**
   * Update rolling temporal baseline for continuity across measurements.
   */
  updateTemporalBaseline(
    state: Record<string, any>,
    weight: number,
    timestamp: Date
  ): Record<string, any> {
    const baseline = state.temporal_baseline ?? {};

    if (baseline.last_weight && baseline.last_timestamp) {
      const lastTs =
        baseline.last_timestamp instanceof Date
          ? baseline.last_timestamp
          : new Date(baseline.last_timestamp);

      const timeDiff = (timestamp.getTime() - lastTs.getTime()) / 3600000; // hours
      if (timeDiff > 0) {
        const weightChange = Math.abs(weight - baseline.last_weight);
        const dailyRate = weightChange / Math.max(timeDiff / 24, 0.1);

        // Exponential moving average with α=0.3
        const prevRate = baseline.rolling_avg_change_rate ?? dailyRate;
        baseline.rolling_avg_change_rate = 0.3 * dailyRate + 0.7 * prevRate;
      }
    }

    baseline.last_weight = weight;
    baseline.last_timestamp = timestamp.toISOString();

    state.temporal_baseline = baseline;
    return state;
  }

  // Helper math functions

  /**
   * Error function using stdlib for improved accuracy.
   */
  private erf(x: number): number {
    return (mathBase as any).special.erf(x);
  }

  /**
   * Calculate mean of an array using stdlib.
   */
  private mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return (statsBase as any).mean(arr.length, arr, 1);
  }

  /**
   * Calculate variance of an array using stdlib.
   */
  private variance(arr: number[]): number {
    if (arr.length === 0) return 0;
    // Using correction=0 for population variance (matching original implementation)
    return (statsBase as any).variance(arr.length, 0, arr, 1);
  }

  /**
   * Calculate standard deviation of an array using stdlib.
   */
  private std(arr: number[]): number {
    if (arr.length === 0) return 0;
    // Using correction=0 for population stdev (matching original implementation)
    return (statsBase as any).stdev(arr.length, 0, arr, 1);
  }

  /**
   * Calculate median of an array using stdlib.
   */
  private median(arr: number[]): number {
    if (arr.length === 0) return 0;
    // stdlib's mediansorted requires a sorted array
    const sorted = [...arr].sort((a, b) => a - b);
    return (statsBase as any).mediansorted(sorted.length, sorted, 1);
  }

  /**
   * Simple polynomial fitting (degree 1 for linear regression).
   */
  private polyfit(x: number[], y: number[], degree: number): number[] {
    if (degree !== 1) {
      throw new Error('Only linear regression (degree=1) is supported');
    }

    const n = x.length;
    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = y.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
    const sumXX = x.reduce((sum, val) => sum + val * val, 0);

    const denominator = n * sumXX - sumX * sumX;

    // Handle zero variance case (all x values are the same or nearly same)
    if (Math.abs(denominator) < 1e-10) {
      // Return horizontal line at mean y value
      const meanY = sumY / n;
      return [0, meanY];  // slope=0, intercept=mean
    }

    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    return [slope, intercept];
  }
}

/**
 * Test utility for maintaining measurement history.
 * NOT used in production (processor is stateless).
 */
export class MeasurementHistory {
  private readonly maxSize: number;
  private readonly weights: number[] = [];
  private readonly timestamps: Date[] = [];
  private readonly qualityScores: number[] = [];

  constructor(maxSize: number = 20) {
    this.maxSize = maxSize;
  }

  /**
   * Add a measurement to history.
   */
  add(weight: number, timestamp: Date, qualityScore: number): void {
    this.weights.push(weight);
    this.timestamps.push(timestamp);
    this.qualityScores.push(qualityScore);

    // Maintain max size
    if (this.weights.length > this.maxSize) {
      this.weights.shift();
      this.timestamps.shift();
      this.qualityScores.shift();
    }
  }

  /**
   * Get recent weights above quality threshold.
   */
  getRecentWeights(minQuality: number = 0.6): number[] {
    const result: number[] = [];
    for (let i = 0; i < this.weights.length; i++) {
      if (this.qualityScores[i] >= minQuality) {
        result.push(this.weights[i]);
      }
    }
    return result;
  }

  /**
   * Calculate statistics for recent measurements.
   */
  getStatistics(): Record<string, number> {
    if (this.weights.length === 0) {
      return {};
    }

    const sorted = [...this.weights].sort((a, b) => a - b);
    const sum = this.weights.reduce((s, w) => s + w, 0);
    const mean = sum / this.weights.length;
    const variance =
      this.weights.reduce((s, w) => s + (w - mean) ** 2, 0) / this.weights.length;

    return {
      mean,
      std: Math.sqrt(variance),
      median:
        sorted.length % 2 === 0
          ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
          : sorted[Math.floor(sorted.length / 2)],
      min: Math.min(...this.weights),
      max: Math.max(...this.weights),
      count: this.weights.length,
    };
  }
}
