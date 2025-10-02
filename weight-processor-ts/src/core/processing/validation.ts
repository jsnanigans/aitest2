/**
 * Unified validation and data quality module for weight processing.
 * Combines physiological validation, BMI detection, and data preprocessing.
 */

import {
  PHYSIOLOGICAL_LIMITS,
  BMI_LIMITS,
  DEFAULT_SOURCE_PROFILE,
  SUPPORTED_WEIGHT_UNITS,
} from '../../constants';

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
   * Returns [is_valid, rejection_reason, daily_change_rate]
   */
  static validateRateOfChange(
    currentWeight: number,
    previousWeight: number,
    timeDiffHours: number,
    source?: string
  ): [boolean, string | null, number] {
    if (timeDiffHours <= 0) {
      return [true, null, 0.0];
    }

    const weightDiff = Math.abs(currentWeight - previousWeight);
    const dailyRate = (weightDiff / timeDiffHours) * 24;

    const sourceProfile = DEFAULT_SOURCE_PROFILE; // TODO: Implement source-specific profiles
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
   * Comprehensive validation combining all checks.
   */
  static validateComprehensive(params: {
    weight: number;
    previousWeight?: number;
    timeDiffHours?: number;
    source?: string;
  }): {
    valid: boolean;
    weight: number;
    checks: string[];
    warnings: string[];
    rejectionReason?: string;
  } {
    const { weight, previousWeight, timeDiffHours, source } = params;

    const result = {
      valid: true,
      weight,
      checks: [] as string[],
      warnings: [] as string[],
      rejectionReason: undefined as string | undefined,
    };

    // Absolute limits check
    const [isValid, reason] = PhysiologicalValidator.validateAbsoluteLimits(weight);
    if (!isValid) {
      result.valid = false;
      result.rejectionReason = reason || undefined;
      return result;
    }
    result.checks.push('absolute_limits');

    // Suspicious range check
    const warning = PhysiologicalValidator.checkSuspiciousRange(weight);
    if (warning) {
      result.warnings.push(warning);
    }

    // Rate of change check
    if (previousWeight !== undefined && timeDiffHours !== undefined) {
      const [rateValid, rateReason] = PhysiologicalValidator.validateRateOfChange(
        weight,
        previousWeight,
        timeDiffHours,
        source
      );
      if (!rateValid) {
        result.valid = false;
        result.rejectionReason = rateReason || undefined;
        return result;
      }
      result.checks.push('rate_of_change');
    }

    return result;
  }
}

/**
 * Handles unit conversion and BMI detection.
 */
export class DataQualityPreprocessor {
  /**
   * Convert weight to kilograms from various units.
   */
  static convertToKg(weight: number, unit: string): number {
    const unitLower = unit.toLowerCase().trim();

    // Already in kg
    if (unitLower === 'kg' || unitLower === 'kilogram' || unitLower === 'kilograms') {
      return weight;
    }

    // Grams
    if (unitLower === 'g' || unitLower === 'gram' || unitLower === 'grams') {
      return weight / 1000;
    }

    // Pounds
    if (unitLower === 'lb' || unitLower === 'lbs' || unitLower === 'pound' || unitLower === 'pounds') {
      return weight * 0.453592;
    }

    // Stone
    if (unitLower === 'st' || unitLower === 'stone' || unitLower === 'stones') {
      return weight * 6.35029;
    }

    // Default: assume kg
    return weight;
  }

  /**
   * Check if value might be BMI instead of weight.
   */
  static checkBMI(value: number, userHeightM?: number): [boolean, number | null] {
    // BMI typically ranges from 15-50
    if (value < 15 || value > 50) {
      return [false, null];
    }

    // If we have height, calculate what weight this BMI would represent
    if (userHeightM && userHeightM > 0) {
      const calculatedWeight = value * (userHeightM ** 2);
      // Check if calculated weight is reasonable
      if (
        calculatedWeight >= PHYSIOLOGICAL_LIMITS.ABSOLUTE_MIN_WEIGHT &&
        calculatedWeight <= PHYSIOLOGICAL_LIMITS.ABSOLUTE_MAX_WEIGHT
      ) {
        return [true, calculatedWeight];
      }
    }

    return [false, null];
  }

  /**
   * Preprocess weight measurement: convert units, detect BMI, validate.
   * Returns [cleaned_weight, metadata] or [null, metadata] if rejected.
   */
  static preprocess(
    weight: number,
    source: string,
    timestamp: Date,
    userId: string,
    unit: string = 'kg',
    userHeightM?: number
  ): [number | null, Record<string, any>] {
    const metadata: any = {
      original_weight: weight,
      original_unit: unit,
      source,
      user_id: userId,
      timestamp: timestamp.toISOString(),
    };

    // 1. Validate unit
    if (!SUPPORTED_WEIGHT_UNITS.has(unit.toLowerCase())) {
      metadata.rejection_reason = `Unsupported unit: ${unit}`;
      metadata.stage = 'unit_validation';
      return [null, metadata];
    }

    // 2. Convert to kg
    let weightKg = DataQualityPreprocessor.convertToKg(weight, unit);
    metadata.weight_kg = weightKg;

    // 3. Check for BMI confusion
    if (userHeightM) {
      const [isBMI, calculatedWeight] = DataQualityPreprocessor.checkBMI(weightKg, userHeightM);
      if (isBMI && calculatedWeight) {
        metadata.bmi_detected = true;
        metadata.original_value_as_bmi = weightKg;
        metadata.calculated_weight = calculatedWeight;
        weightKg = calculatedWeight;
      }
    }

    // 4. Validate absolute limits
    const [isValid, reason] = PhysiologicalValidator.validateAbsoluteLimits(weightKg);
    if (!isValid) {
      metadata.rejection_reason = reason;
      metadata.stage = 'physiological_validation';
      return [null, metadata];
    }

    // 5. Check suspicious range (warning only)
    const warning = PhysiologicalValidator.checkSuspiciousRange(weightKg);
    if (warning) {
      metadata.warning = warning;
    }

    metadata.stage = 'preprocessing_complete';
    metadata.cleaned_weight = weightKg;

    return [weightKg, metadata];
  }
}

/**
 * Validates processor state structure.
 */
export class StateValidator {
  /**
   * Validate that state has required fields.
   */
  static validate(state: any, operation?: string): boolean {
    if (!state || typeof state !== 'object') {
      return false;
    }

    // Basic validation - state should be an object
    // More specific validation can be added based on operation type
    return true;
  }
}
