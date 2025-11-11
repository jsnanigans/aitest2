/**
 * Unified validation and data quality module for weight processing.
 * Combines physiological validation, BMI detection, and data preprocessing.
 */

import {
  SOURCE_PROFILES,
  DEFAULT_PROFILE,
  BMI_LIMITS,
  PHYSIOLOGICAL_LIMITS,
  SUPPORTED_WEIGHT_UNITS,
  getSourceReliability,
  getNoiseMultiplier,
  categorizeRejectionEnhanced,
  getRejectionSeverity,
} from '../constants';
import { base as statsBase } from '@stdlib/stats';

/**
 * Helper function to calculate mean of an array using stdlib
 */
function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return (statsBase as any).mean(values.length, values, 1);
}

/**
 * Helper function to calculate standard deviation using stdlib
 */
function std(values: number[]): number {
  if (values.length === 0) return 0;
  // Using correction=0 for population stdev (matching original implementation)
  return (statsBase as any).stdev(values.length, 0, values, 1);
}

/**
 * Helper function to calculate median using stdlib
 */
function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return (statsBase as any).mediansorted(sorted.length, sorted, 1);
}

/**
 * Validates weight measurements against physiological constraints.
 */
export class PhysiologicalValidator {
  static readonly ABSOLUTE_MIN_WEIGHT = PHYSIOLOGICAL_LIMITS.ABSOLUTE_MIN_WEIGHT;
  static readonly ABSOLUTE_MAX_WEIGHT = PHYSIOLOGICAL_LIMITS.ABSOLUTE_MAX_WEIGHT;
  static readonly SUSPICIOUS_MIN_WEIGHT = PHYSIOLOGICAL_LIMITS.SUSPICIOUS_MIN_WEIGHT;
  static readonly SUSPICIOUS_MAX_WEIGHT = PHYSIOLOGICAL_LIMITS.SUSPICIOUS_MAX_WEIGHT;

  static readonly MAX_DAILY_CHANGE_KG = PHYSIOLOGICAL_LIMITS.MAX_DAILY_CHANGE_KG;
  static readonly MAX_WEEKLY_CHANGE_KG = PHYSIOLOGICAL_LIMITS.MAX_WEEKLY_CHANGE_KG;
  static readonly TYPICAL_DAILY_VARIATION_KG = PHYSIOLOGICAL_LIMITS.TYPICAL_DAILY_VARIATION_KG;

  /**
   * Check if weight is within absolute physiological limits.
   */
  static validateAbsoluteLimits(weight: number): [boolean, string | null] {
    if (weight < PhysiologicalValidator.ABSOLUTE_MIN_WEIGHT) {
      return [
        false,
        `Weight ${weight.toFixed(1)}kg below absolute minimum ${PhysiologicalValidator.ABSOLUTE_MIN_WEIGHT}kg`,
      ];
    }
    if (weight > PhysiologicalValidator.ABSOLUTE_MAX_WEIGHT) {
      return [
        false,
        `Weight ${weight.toFixed(1)}kg above absolute maximum ${PhysiologicalValidator.ABSOLUTE_MAX_WEIGHT}kg`,
      ];
    }
    return [true, null];
  }

  /**
   * Check if weight is in suspicious range.
   */
  static checkSuspiciousRange(weight: number): string | null {
    if (weight < PhysiologicalValidator.SUSPICIOUS_MIN_WEIGHT) {
      return `Weight ${weight.toFixed(1)}kg suspiciously low`;
    }
    if (weight > PhysiologicalValidator.SUSPICIOUS_MAX_WEIGHT) {
      return `Weight ${weight.toFixed(1)}kg suspiciously high`;
    }
    return null;
  }

  /**
   * Validate rate of weight change.
   *
   * Returns:
   *   [is_valid, rejection_reason, daily_change_rate]
   */
  static validateRateOfChange(
    currentWeight: number,
    previousWeight: number,
    timeDiffHours: number,
    source: string | null = null
  ): [boolean, string | null, number] {
    if (timeDiffHours <= 0) {
      return [true, null, 0.0];
    }

    const weightDiff = Math.abs(currentWeight - previousWeight);
    const dailyRate = (weightDiff / timeDiffHours) * 24;

    const sourceProfile = source
      ? (SOURCE_PROFILES[source] ?? DEFAULT_PROFILE)
      : DEFAULT_PROFILE;
    const maxDailyChange = sourceProfile.max_daily_change_kg ?? PhysiologicalValidator.MAX_DAILY_CHANGE_KG;

    if (dailyRate > maxDailyChange) {
      const hoursStr = timeDiffHours < 24
        ? `${timeDiffHours.toFixed(1)}h`
        : `${(timeDiffHours / 24).toFixed(1)}d`;
      return [
        false,
        `Change of ${weightDiff.toFixed(1)}kg in ${hoursStr} exceeds max rate`,
        dailyRate,
      ];
    }

    return [true, null, dailyRate];
  }

  /**
   * Analyze measurement patterns for anomalies.
   *
   * Args:
   *   measurements: Array of [timestamp, weight] tuples
   *   windowHours: Time window to analyze
   *
   * Returns:
   *   Object with pattern analysis results
   */
  static checkMeasurementPattern(
    measurements: Array<[Date, number]>,
    windowHours: number = 24
  ): Record<string, any> {
    if (measurements.length < 2) {
      return { sufficient_data: false };
    }

    // Sort by timestamp
    const sorted = [...measurements].sort((a, b) => a[0].getTime() - b[0].getTime());

    const now = sorted[sorted.length - 1][0];
    const windowStart = new Date(now.getTime() - windowHours * 60 * 60 * 1000);
    const recent = sorted.filter(([t, _]) => t >= windowStart);

    if (recent.length < 2) {
      return { sufficient_data: false };
    }

    const weights = recent.map(([_, w]) => w);
    const meanWeight = mean(weights);
    const stdWeight = std(weights);

    let oscillationCount = 0;
    for (let i = 1; i < weights.length - 1; i++) {
      if (
        (weights[i] > weights[i - 1] && weights[i] > weights[i + 1]) ||
        (weights[i] < weights[i - 1] && weights[i] < weights[i + 1])
      ) {
        oscillationCount++;
      }
    }

    return {
      sufficient_data: true,
      mean: meanWeight,
      std: stdWeight,
      cv: meanWeight > 0 ? stdWeight / meanWeight : 0,
      measurement_count: recent.length,
      oscillation_ratio: weights.length > 2 ? oscillationCount / (weights.length - 2) : 0,
      range: Math.max(...weights) - Math.min(...weights),
      suspicious_pattern: stdWeight > PhysiologicalValidator.TYPICAL_DAILY_VARIATION_KG * 2,
    };
  }

  /**
   * Comprehensive validation combining all checks.
   *
   * Returns:
   *   Object with validation results and metadata
   */
  static validateComprehensive(
    weight: number,
    previousWeight: number | null = null,
    timeDiffHours: number | null = null,
    source: string | null = null,
    recentMeasurements: Array<[Date, number]> | null = null
  ): Record<string, any> {
    const result: Record<string, any> = {
      valid: true,
      weight: weight,
      checks: [],
      warnings: [],
      rejection_reason: null,
    };

    // Always perform physiological validation (safety critical)
    const [isValid, reason] = PhysiologicalValidator.validateAbsoluteLimits(weight);
    if (!isValid) {
      result.valid = false;
      result.rejection_reason = reason;
      return result;
    }
    result.checks.push('absolute_limits');

    const warning = PhysiologicalValidator.checkSuspiciousRange(weight);
    if (warning) {
      result.warnings.push(warning);
    }

    // Check rate limiting
    if (previousWeight !== null && timeDiffHours !== null) {
      const [rateValid, rateReason, rate] = PhysiologicalValidator.validateRateOfChange(
        weight,
        previousWeight,
        timeDiffHours,
        source
      );
      if (!rateValid) {
        result.valid = false;
        result.rejection_reason = rateReason;
        result.daily_change_rate = rate;
        return result;
      }
      result.checks.push('rate_of_change');
      result.daily_change_rate = rate;
    }

    if (recentMeasurements) {
      const patternAnalysis = PhysiologicalValidator.checkMeasurementPattern(
        recentMeasurements
      );
      if (patternAnalysis.sufficient_data) {
        result.pattern_analysis = patternAnalysis;
        if (patternAnalysis.suspicious_pattern) {
          result.warnings.push('Suspicious measurement pattern detected');
        }
      }
    }

    return result;
  }
}

/**
 * Validates BMI-related measurements and detects BMI vs weight confusion.
 */
export class BMIValidator {
  static readonly BMI_RANGE: [number, number] = [BMI_LIMITS.IMPOSSIBLE_LOW, BMI_LIMITS.IMPOSSIBLE_HIGH];
  static readonly WEIGHT_RANGE: [number, number] = [
    PHYSIOLOGICAL_LIMITS.ABSOLUTE_MIN_WEIGHT,
    PHYSIOLOGICAL_LIMITS.ABSOLUTE_MAX_WEIGHT,
  ];

  /**
   * Calculate BMI from weight and height.
   */
  static calculateBMI(weightKg: number, heightM: number): number {
    if (heightM <= 0) {
      return 0;
    }
    return weightKg / (heightM ** 2);
  }

  /**
   * Check if a value is likely BMI rather than weight.
   *
   * Args:
   *   value: The numeric value to check
   *   unit: The stated unit ('kg', 'lb', etc.)
   *
   * Returns:
   *   True if value is likely BMI
   */
  static isLikelyBMI(value: number, unit: string = 'kg'): boolean {
    const unitLower = unit.toLowerCase();

    if (['bmi', 'kg/m2', 'kg/m^2'].includes(unitLower)) {
      return true;
    }

    if (['kg', 'kilogram', 'kilograms'].includes(unitLower)) {
      if (15 <= value && value <= 50) {
        return true;
      }
    }

    return false;
  }

  /**
   * Convert BMI to weight given height.
   */
  static convertBMIToWeight(bmi: number, heightM: number): number {
    return bmi * (heightM ** 2);
  }

  /**
   * Validate if BMI is within physiological limits.
   */
  static validateBMI(bmi: number): [boolean, string | null] {
    if (bmi < BMI_LIMITS.IMPOSSIBLE_LOW) {
      return [false, `BMI ${bmi.toFixed(1)} below physiological minimum`];
    }
    if (bmi > BMI_LIMITS.IMPOSSIBLE_HIGH) {
      return [false, `BMI ${bmi.toFixed(1)} above physiological maximum`];
    }
    return [true, null];
  }

  /**
   * Categorize BMI into standard categories.
   */
  static categorizeBMI(bmi: number): string {
    if (bmi < BMI_LIMITS.UNDERWEIGHT) {
      return 'underweight';
    } else if (bmi < BMI_LIMITS.OVERWEIGHT) {
      return 'normal';
    } else if (bmi < BMI_LIMITS.OBESE) {
      return 'overweight';
    } else {
      return 'obese';
    }
  }

  /**
   * Detect if value is BMI and convert to weight if needed.
   *
   * Args:
   *   value: The input value
   *   unit: The stated unit
   *   heightM: User's height in meters
   *   source: Data source (some sources are more likely to send BMI)
   *
   * Returns:
   *   [weight_kg, was_converted, metadata]
   */
  static detectAndConvert(
    value: number,
    unit: string,
    heightM: number,
    source: string | null = null
  ): [number, boolean, Record<string, any>] {
    const metadata: Record<string, any> = {
      original_value: value,
      original_unit: unit,
      height_m: heightM,
    };

    const unitLower = unit ? unit.toLowerCase() : 'kg';

    if (['lb', 'lbs', 'pound', 'pounds'].includes(unitLower)) {
      const weightKg = value * 0.453592;
      metadata.conversion = `${value.toFixed(1)} lb to ${weightKg.toFixed(1)} kg`;
      return [weightKg, false, metadata];
    }

    if (['st', 'stone', 'stones'].includes(unitLower)) {
      const weightKg = value * 6.35029;
      metadata.conversion = `${value.toFixed(1)} st to ${weightKg.toFixed(1)} kg`;
      return [weightKg, false, metadata];
    }

    if (BMIValidator.isLikelyBMI(value, unit)) {
      const weightKg = BMIValidator.convertBMIToWeight(value, heightM);

      if (
        PHYSIOLOGICAL_LIMITS.ABSOLUTE_MIN_WEIGHT <= weightKg &&
        weightKg <= PHYSIOLOGICAL_LIMITS.ABSOLUTE_MAX_WEIGHT
      ) {
        metadata.detected_as_bmi = true;
        metadata.conversion = `BMI ${value.toFixed(1)} to weight ${weightKg.toFixed(1)} kg`;
        metadata.confidence = (source && source.toLowerCase().includes('connectivehealth'))
          ? 'high'
          : 'medium';
        return [weightKg, true, metadata];
      }
    }

    return [value, false, metadata];
  }

  /**
   * Validate consistency between weight and implied BMI.
   *
   * Returns:
   *   Object with validation results
   */
  static validateWeightBMIConsistency(
    weightKg: number,
    heightM: number,
    source: string | null = null
  ): Record<string, any> {
    const bmi = BMIValidator.calculateBMI(weightKg, heightM);

    const result: Record<string, any> = {
      weight_kg: weightKg,
      height_m: heightM,
      bmi: bmi,
      bmi_category: BMIValidator.categorizeBMI(bmi),
      valid: true,
      warnings: [],
    };

    const [isValid, reason] = BMIValidator.validateBMI(bmi);
    if (!isValid) {
      result.valid = false;
      result.rejection_reason = reason;
      return result;
    }

    if (bmi < BMI_LIMITS.SUSPICIOUS_LOW) {
      result.warnings.push(`BMI ${bmi.toFixed(1)} suspiciously low`);
    } else if (bmi > BMI_LIMITS.SUSPICIOUS_HIGH) {
      result.warnings.push(`BMI ${bmi.toFixed(1)} suspiciously high`);
    }

    if (source && source.toLowerCase().includes('iglucose')) {
      result.warnings.push('High-outlier source detected');
      result.high_risk = true;
    }

    return result;
  }

  /**
   * Estimate user height from pairs of weights and BMIs.
   *
   * Args:
   *   weightBMIPairs: Array of [weight_kg, bmi] tuples
   *
   * Returns:
   *   Estimated height in meters or null if insufficient data
   */
  static estimateHeightFromWeightsAndBMIs(
    weightBMIPairs: Array<[number, number]>
  ): number | null {
    if (weightBMIPairs.length < 2) {
      return null;
    }

    const heights: number[] = [];
    for (const [weight, bmi] of weightBMIPairs) {
      if (bmi > 0) {
        const height = Math.sqrt(weight / bmi);
        if (1.0 <= height && height <= 2.5) {
          heights.push(height);
        }
      }
    }

    if (heights.length === 0) {
      return null;
    }

    return median(heights);
  }

  /**
   * Detect patterns of unit confusion in measurements.
   *
   * Args:
   *   measurements: Array of [timestamp, value, unit] tuples
   *   heightM: User's height in meters
   *
   * Returns:
   *   Analysis of potential unit confusion patterns
   */
  static detectUnitConfusion(
    measurements: Array<[Date, number, string]>,
    heightM: number
  ): Record<string, any> {
    if (measurements.length < 3) {
      return { sufficient_data: false };
    }

    const potentialBMIs: number[] = [];
    const potentialWeights: number[] = [];

    for (const [_, value, unit] of measurements) {
      if (15 <= value && value <= 50) {
        potentialBMIs.push(value);
      }
      if (40 <= value && value <= 300) {
        potentialWeights.push(value);
      }
    }

    const bmiRatio = potentialBMIs.length / measurements.length;
    const weightRatio = potentialWeights.length / measurements.length;

    const result: Record<string, any> = {
      sufficient_data: true,
      total_measurements: measurements.length,
      potential_bmi_count: potentialBMIs.length,
      potential_weight_count: potentialWeights.length,
      bmi_ratio: bmiRatio,
      weight_ratio: weightRatio,
    };

    if (bmiRatio > 0.3) {
      result.likely_confusion = 'frequent_bmi_values';
      result.recommendation = 'Check source configuration for unit settings';
    }

    if (potentialBMIs.length > 0 && potentialWeights.length > 0) {
      const bmiMean = mean(potentialBMIs);
      const impliedWeight = bmiMean * (heightM ** 2);
      const weightMean = mean(potentialWeights);

      if (Math.abs(impliedWeight - weightMean) < 10) {
        result.pattern_detected = 'consistent_bmi_weight_relationship';
        result.confidence = 'high';
      }
    }

    return result;
  }
}

/**
 * Calculate adaptive thresholds for weight validation.
 */
export class ThresholdCalculator {
  /**
   * Calculate adaptive threshold based on source and time gap.
   *
   * Args:
   *   source: Data source identifier
   *   timeGapHours: Hours since last measurement
   *   baseWeight: Reference weight for percentage calculations
   *   measurementNoise: Base measurement noise in kg
   *
   * Returns:
   *   Threshold in kg
   */
  static calculateAdaptiveThreshold(
    source: string,
    timeGapHours: number,
    baseWeight: number,
    measurementNoise: number = 0.5
  ): number {
    const sourceProfile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;

    const baseThreshold = sourceProfile.base_threshold_kg ?? 2.0;

    let timeFactor = 1.0 + (timeGapHours / 24.0) * 0.5;
    timeFactor = Math.min(timeFactor, 3.0);

    const weightFactor = 1.0 + (baseWeight / 100.0) * 0.1;

    const noiseFactor = 1.0 + measurementNoise;

    let threshold = baseThreshold * timeFactor * weightFactor * noiseFactor;

    const maxThreshold = sourceProfile.max_threshold_kg ?? 10.0;
    threshold = Math.min(threshold, maxThreshold);

    return threshold;
  }

  /**
   * Calculate threshold based on recent rate of change.
   *
   * Args:
   *   recentChanges: Array of recent weight changes
   *   timeGapHours: Hours since last measurement
   *   source: Data source identifier
   *
   * Returns:
   *   Rate-based threshold in kg
   */
  static calculateRateBasedThreshold(
    recentChanges: number[],
    timeGapHours: number,
    source: string
  ): number {
    if (recentChanges.length === 0) {
      return ThresholdCalculator.calculateAdaptiveThreshold(
        source,
        timeGapHours,
        70.0
      );
    }

    const absChanges = recentChanges.map(Math.abs);
    const meanChange = mean(absChanges);
    const stdChange = recentChanges.length > 1 ? std(recentChanges) : meanChange;

    const expectedChange = meanChange * (timeGapHours / 24.0);

    let threshold = expectedChange + 2 * stdChange;

    const sourceProfile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;
    const maxThreshold = sourceProfile.max_threshold_kg ?? 10.0;

    return Math.min(threshold, maxThreshold);
  }

  /**
   * Adjust threshold based on confidence level.
   *
   * Args:
   *   confidence: Confidence level (0-1)
   *   baseThreshold: Base threshold in kg
   *   source: Data source identifier
   *
   * Returns:
   *   Adjusted threshold in kg
   */
  static calculateConfidenceBasedThreshold(
    confidence: number,
    baseThreshold: number,
    source: string
  ): number {
    const sourceProfile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;

    let multiplier: number;
    if (confidence > 0.8) {
      multiplier = 0.8; // high_confidence_multiplier default
    } else if (confidence > 0.5) {
      multiplier = 1.0;
    } else {
      multiplier = 1.5; // low_confidence_multiplier default
    }

    return baseThreshold * multiplier;
  }

  /**
   * Get rejection threshold for specific source and category.
   *
   * Args:
   *   source: Data source identifier
   *   category: Rejection category
   *
   * Returns:
   *   Rejection threshold in kg
   */
  static getRejectionThreshold(source: string, category: string = 'default'): number {
    const sourceProfile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;

    const categoryThresholds: Record<string, number> = {
      spike: 5.0,
      drift: 3.0,
      noise: 2.0,
      default: sourceProfile.base_threshold_kg ?? 2.0,
    };

    return categoryThresholds[category] ?? categoryThresholds.default;
  }
}

/**
 * Pre-process and clean data before Kalman filtering.
 */
export class DataQualityPreprocessor {
  static readonly DEFAULT_HEIGHT_M = PHYSIOLOGICAL_LIMITS.DEFAULT_HEIGHT_M;

  /**
   * Convert height to meters.
   */
  private static convertHeightToMeters(value: number, unit: string): number {
    const unitLower = unit ? unit.toLowerCase() : 'm';

    if (unitLower.includes('cm') || unitLower.includes('centimeter')) {
      return value / 100.0;
    } else if (unitLower.includes('in') || unitLower.includes('inch')) {
      return value * 0.0254;
    } else if (unitLower.includes('ft') || unitLower.includes('feet')) {
      return value * 0.3048;
    } else if (unitLower.includes('m') || unitLower.includes('meter')) {
      return value;
    } else {
      return value / 100.0;
    }
  }

  /**
   * Get user's height in meters, using provided value or default.
   */
  static getUserHeight(userId: string | null, userHeightM: number | null = null): number {
    // Use provided height if available, otherwise use default
    return userHeightM ?? DataQualityPreprocessor.DEFAULT_HEIGHT_M;
  }

  /**
   * Clean and standardize weight data with STRICT unit validation.
   * NO BMI detection, NO unit assumptions.
   *
   * Args:
   *   weight: The weight value
   *   source: Data source identifier
   *   timestamp: Measurement timestamp
   *   userId: User identifier for height lookup
   *   unit: Unit of the weight measurement
   *   userHeightM: User's height in meters
   *
   * Returns:
   *   [cleaned_weight, metadata] or [null, metadata] if rejected
   */
  static preprocess(
    weight: number,
    source: string,
    timestamp: Date,
    userId: string | null = null,
    unit: string = 'kg',
    userHeightM: number | null = null
  ): [number | null, Record<string, any>] {
    const metadata: Record<string, any> = {
      original_weight: weight,
      original_unit: unit,
      source: source,
      timestamp: timestamp.getTime(),  // Convert to milliseconds for Python compatibility
      corrections: [],
      warnings: [],
      checks_passed: [],
    };

    // STRICT UNIT VALIDATION - reject if not in whitelist
    if (!unit) {
      metadata.rejected = 'Missing unit - cannot process without explicit unit';
      return [null, metadata];
    }

    const unitLower = unit.toLowerCase().trim();

    // Check against whitelist
    if (!SUPPORTED_WEIGHT_UNITS.has(unitLower)) {
      metadata.rejected = `Unsupported unit: ${unit} - only ${Array.from(SUPPORTED_WEIGHT_UNITS).join(', ')} are supported`;
      return [null, metadata];
    }

    // Perform conversions for supported units
    if (['lb', 'lbs', 'pound', 'pounds'].includes(unitLower)) {
      const weightKg = weight * 0.453592;
      metadata.corrections.push(
        `Converted ${weight.toFixed(1)} ${unit} to ${weightKg.toFixed(1)} kg`
      );
      weight = weightKg;
    } else if (['st', 'stone', 'stones'].includes(unitLower)) {
      const weightKg = weight * 6.35029;
      metadata.corrections.push(
        `Converted ${weight.toFixed(1)} ${unit} to ${weightKg.toFixed(1)} kg`
      );
      weight = weightKg;
    } else if (['g', 'gram', 'grams'].includes(unitLower)) {
      const weightKg = weight / 1000.0;
      metadata.corrections.push(
        `Converted ${weight.toFixed(1)} ${unit} to ${weightKg.toFixed(1)} kg`
      );
      weight = weightKg;
    }
    // else: already in kg/kilogram/kilograms - no conversion needed

    // BMI validation (for rejection only, NO conversion)
    const userHeight = userId
      ? DataQualityPreprocessor.getUserHeight(userId, userHeightM)
      : DataQualityPreprocessor.DEFAULT_HEIGHT_M;
    const impliedBMI = weight / (userHeight ** 2);

    // Reject physiologically impossible BMI values
    if (impliedBMI < BMI_LIMITS.IMPOSSIBLE_LOW) {
      metadata.rejected = `Implied BMI ${impliedBMI.toFixed(1)} physiologically impossible (weight: ${weight.toFixed(1)}kg, height: ${userHeight.toFixed(2)}m)`;
      return [null, metadata];
    }

    if (impliedBMI > BMI_LIMITS.IMPOSSIBLE_HIGH) {
      metadata.rejected = `Implied BMI ${impliedBMI.toFixed(1)} physiologically impossible (weight: ${weight.toFixed(1)}kg, height: ${userHeight.toFixed(2)}m)`;
      return [null, metadata];
    }

    // Add warnings for suspicious BMI (but don't reject)
    if (impliedBMI < BMI_LIMITS.SUSPICIOUS_LOW) {
      metadata.warnings.push(`Implied BMI ${impliedBMI.toFixed(1)} suspiciously low`);
    }

    if (impliedBMI > BMI_LIMITS.SUSPICIOUS_HIGH) {
      metadata.warnings.push(`Implied BMI ${impliedBMI.toFixed(1)} suspiciously high`);
    }

    // Track BMI for metadata
    metadata.checks_passed.push('unit_validation');
    metadata.checks_passed.push('physiological_limits');
    metadata.implied_bmi = parseFloat(impliedBMI.toFixed(1));
    metadata.user_height_m = parseFloat(userHeight.toFixed(2));

    // Categorize BMI (informational only)
    if (impliedBMI < BMI_LIMITS.UNDERWEIGHT) {
      metadata.bmi_category = 'underweight';
    } else if (impliedBMI < BMI_LIMITS.OVERWEIGHT) {
      metadata.bmi_category = 'normal';
    } else if (impliedBMI < BMI_LIMITS.OBESE) {
      metadata.bmi_category = 'overweight';
    } else {
      metadata.bmi_category = 'obese';
    }

    // Flag high-risk sources
    // if (source.toLowerCase().includes('iglucose')) {
    //   metadata.warnings.push('High-outlier source - increased scrutiny');
    //   metadata.high_risk = true;
    // }

    return [weight, metadata];
  }
}
