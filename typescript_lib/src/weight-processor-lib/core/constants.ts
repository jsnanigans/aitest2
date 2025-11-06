/**
 * Constants for weight stream processor.
 * Safety limits and immutable values.
 */

/**
 * Result from threshold calculation with explicit units.
 */
export interface ThresholdResult {
  value: number;
  unit: string;
  metadata?: Record<string, any>;
}

export function createThresholdResult(
  value: number,
  unit: string,
  metadata: Record<string, any> = {}
): ThresholdResult {
  return { value, unit, metadata };
}

// Physiological limits
export const PHYSIOLOGICAL_LIMITS = {
  ABSOLUTE_MIN_WEIGHT: 30.0,  // kg
  ABSOLUTE_MAX_WEIGHT: 400.0,  // kg
  SUSPICIOUS_MIN_WEIGHT: 40.0,  // kg
  SUSPICIOUS_MAX_WEIGHT: 300.0,  // kg
  DEFAULT_HEIGHT_M: 1.67,
  MAX_DAILY_CHANGE_KG: 2.0,  // Realistic daily fluctuation
  MAX_WEEKLY_CHANGE_KG: 3.5,  // Aggressive but possible weight loss/gain
  TYPICAL_DAILY_VARIATION_KG: 1.5,  // Normal daily variation
  MAX_SUSTAINED_DAILY_KG: 0.5,  // Sustainable long-term change
  MAX_CHANGE_1H: 1.0,  // Water/food intake immediate effect
  MAX_CHANGE_6H: 3.0,  // Half-day variation (increased for meal+water+exercise)
  MAX_CHANGE_24H: 4.0,  // Full day variation (meal cycles + exercise + hydration)
  MAX_CHANGE_1MIN: 0.5,  // Scale variance + positioning tolerance
  MAX_CHANGE_5MIN: 1.0,  // Water/bathroom + multiple measurements
  MAX_MONTHLY_PERCENT: 15,  // Maximum 15% body weight change per month
  LIMIT_TOLERANCE: 0.1,  // Optimized from 0.10
  SUSTAINED_TOLERANCE: 0.25,  // Optimized from 0.25
  SESSION_VARIANCE: 2,  // Optimized from 5.0
} as const;

// Supported weight units - STRICT WHITELIST
export const SUPPORTED_WEIGHT_UNITS = new Set([
  // Metric units
  'kg',
  'kilogram',
  'kilograms',
  'g',
  'gram',
  'grams',
  // Imperial units
  'lb',
  'lbs',
  'pound',
  'pounds',
  'st',
  'stone',
  'stones',
]);

// BMI limits
export const BMI_LIMITS = {
  CRITICAL_LOW: 15.0,
  SEVERE_LOW: 16.0,
  UNDERWEIGHT: 18.5,
  OVERWEIGHT: 25.0,
  OBESE: 30.0,
  SEVERE_OBESE: 35.0,
  MORBID_OBESE: 40.0,
  CRITICAL_HIGH: 50.0,
  IMPOSSIBLE_LOW: 17.0,
  IMPOSSIBLE_HIGH: 100.0,
  SUSPICIOUS_LOW: 20.0,
  SUSPICIOUS_HIGH: 70.0,
} as const;

// Source profiles configuration
// NOTE: In Python this is loaded from config.toml, but for TypeScript
// we'll include default profiles here
export interface SourceProfile {
  outlier_rate?: number;
  reliability: string;
  noise_multiplier: number;
  priority: number;
  base_threshold_kg?: number;
  max_threshold_kg?: number;
  max_daily_change_kg?: number;
}

export const SOURCE_PROFILES: Record<string, SourceProfile> = {
  'patient-device': {
    outlier_rate: 5.0,
    reliability: 'high',
    noise_multiplier: 0.7,
    priority: 1,
    base_threshold_kg: 1.5,
    max_threshold_kg: 5.0,
  },
  'care-team-upload': {
    outlier_rate: 2.0,
    reliability: 'very_high',
    noise_multiplier: 0.5,
    priority: 0,
    base_threshold_kg: 1.0,
    max_threshold_kg: 4.0,
  },
  'patient-upload': {
    outlier_rate: 10.0,
    reliability: 'medium',
    noise_multiplier: 1.0,
    priority: 2,
    base_threshold_kg: 2.0,
    max_threshold_kg: 6.0,
  },
  'internal-questionnaire': {
    outlier_rate: 15.0,
    reliability: 'low',
    noise_multiplier: 1.5,
    priority: 3,
    base_threshold_kg: 3.0,
    max_threshold_kg: 8.0,
  },
  'initial-questionnaire': {
    outlier_rate: 15.0,
    reliability: 'low',
    noise_multiplier: 1.5,
    priority: 3,
    base_threshold_kg: 3.0,
    max_threshold_kg: 8.0,
  },
  'questionnaire': {
    outlier_rate: 15.0,
    reliability: 'low',
    noise_multiplier: 1.5,
    priority: 3,
    base_threshold_kg: 3.0,
    max_threshold_kg: 8.0,
  },
  default: {
    outlier_rate: 20.0,
    reliability: 'unknown',
    noise_multiplier: 1.0,
    priority: 999,
    base_threshold_kg: 3.0,
    max_threshold_kg: 10.0,
  },
};

export const DEFAULT_PROFILE: SourceProfile = SOURCE_PROFILES.default;

// Questionnaire sources (for special handling)
export const QUESTIONNAIRE_SOURCES = new Set([
  'internal-questionnaire',
  'initial-questionnaire',
  'care-team-upload',
  'questionnaire',
]);

// Kalman defaults
export const KALMAN_DEFAULTS = {
  initial_variance: 0.364,
  transition_covariance_weight: 0.018,
  transition_covariance_trend: 0.00015,
  observation_covariance: 3.4,
} as const;

// Visualization marker symbols for source types
export const SOURCE_MARKER_SYMBOLS: Record<string, string> = {
  'care-team-upload': 'triangle-up',
  'patient-upload': 'circle',
  'internal-questionnaire': 'square',
  'initial-questionnaire': 'square',
  'patient-device': 'diamond',
  'https://connectivehealth.io': 'hexagon',
  'https://api.iglucose.com': 'hexagon',
  questionnaire: 'square',
  default: 'circle',
};

// Rejection severity color mapping
export const REJECTION_SEVERITY_COLORS: Record<string, string> = {
  Critical: '#8B0000',  // Dark red for impossible values
  High: '#CC0000',  // Medium-dark red for extreme deviations
  Medium: '#FF4444',  // Medium red for suspicious values
  Low: '#FF9999',  // Light red for minor issues
};

// Session detection
export const SESSION_TIMEOUT_MINUTES = 5.0;
export const SESSION_VARIANCE_THRESHOLD = 5.81;  // kg

// Helper functions

/**
 * Get priority for a source (lower number = higher priority).
 */
export function getSourcePriority(source: string): number {
  const profile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;
  return profile.priority;
}

/**
 * Get reliability classification for source.
 */
export function getSourceReliability(source: string): string {
  const profile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;
  return profile.reliability;
}

/**
 * Get Kalman filter measurement noise multiplier for source.
 */
export function getNoiseMultiplier(source: string): number {
  const profile = SOURCE_PROFILES[source] ?? DEFAULT_PROFILE;
  return profile.noise_multiplier;
}

/**
 * Enhanced categorization including BMI and unit issues.
 */
export function categorizeRejectionEnhanced(reason: string): string {
  const reasonLower = reason.toLowerCase();

  if (reasonLower.includes('bmi')) {
    return 'BMI_Detection';
  } else if (
    reasonLower.includes('unit') ||
    reasonLower.includes('pound') ||
    reasonLower.includes('conversion')
  ) {
    return 'Unit_Conversion';
  } else if (reasonLower.includes('physiological')) {
    return 'Physiological_Limit';
  } else if (reasonLower.includes('outside bounds')) {
    return 'Bounds';
  } else if (reasonLower.includes('extreme deviation')) {
    return 'Extreme';
  } else if (
    reasonLower.includes('session variance') ||
    reasonLower.includes('different user')
  ) {
    return 'Variance';
  } else if (reasonLower.includes('sustained')) {
    return 'Sustained';
  } else if (reasonLower.includes('daily fluctuation')) {
    return 'Daily';
  } else {
    return 'Other';
  }
}

/**
 * Determine severity of rejection.
 */
export function getRejectionSeverity(reason: string, weightChange: number = 0): string {
  const reasonLower = reason.toLowerCase();

  if (reasonLower.includes('impossible') || reasonLower.includes('physiologically impossible')) {
    return 'Critical';
  } else if (reasonLower.includes('extreme') || weightChange > 20) {
    return 'High';
  } else if (reasonLower.includes('suspicious') || weightChange > 10) {
    return 'Medium';
  } else {
    return 'Low';
  }
}
