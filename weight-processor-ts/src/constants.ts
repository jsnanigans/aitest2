/**
 * Constants for weight stream processor
 *
 * Safety limits and immutable values.
 * Configurable parameters are loaded from config.toml via ConfigManager.
 */

/**
 * Physiological limits for weight measurements (in kg)
 */
export const PHYSIOLOGICAL_LIMITS = {
  ABSOLUTE_MIN_WEIGHT: 30.0, // kg
  ABSOLUTE_MAX_WEIGHT: 400.0, // kg
  SUSPICIOUS_MIN_WEIGHT: 40.0, // kg
  SUSPICIOUS_MAX_WEIGHT: 300.0, // kg
  DEFAULT_HEIGHT_M: 1.67,
  MAX_DAILY_CHANGE_KG: 2.0, // Realistic daily fluctuation
  MAX_WEEKLY_CHANGE_KG: 3.5, // Aggressive but possible weight loss/gain
  TYPICAL_DAILY_VARIATION_KG: 1.5, // Normal daily variation
  MAX_SUSTAINED_DAILY_KG: 0.5, // Sustainable long-term change
  MAX_CHANGE_1H: 1.0, // Water/food intake immediate effect
  MAX_CHANGE_6H: 3.0, // Half-day variation (increased for meal+water+exercise)
  MAX_CHANGE_24H: 4.0, // Full day variation (meal cycles + exercise + hydration)
  MAX_CHANGE_1MIN: 0.5, // Scale variance + positioning tolerance
  MAX_CHANGE_5MIN: 1.0, // Water/bathroom + multiple measurements
  MAX_MONTHLY_PERCENT: 15, // Maximum 15% body weight change per month
  LIMIT_TOLERANCE: 0.1,
  SUSTAINED_TOLERANCE: 0.25,
  SESSION_VARIANCE: 2,
} as const;

/**
 * Supported weight units - STRICT WHITELIST
 */
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

/**
 * BMI limits for validation
 */
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

/**
 * Questionnaire sources (for special handling)
 */
export const QUESTIONNAIRE_SOURCES = new Set([
  'internal-questionnaire',
  'initial-questionnaire',
  'care-team-upload',
  'questionnaire',
  'QUESTIONNAIRE_ONBOARDING',
  'QUESTIONNAIRE_CHECKIN',
]);

/**
 * Default Kalman filter parameters
 *
 * NOTE: These are fallback defaults. Production values should be loaded from config.toml
 */
export const KALMAN_DEFAULTS = {
  initial_variance: 0.364,
  transition_covariance_weight: 0.018,
  transition_covariance_trend: 0.00015,
  observation_covariance: 3.4,
} as const;

/**
 * Default source profile for unknown sources
 */
export const DEFAULT_SOURCE_PROFILE = {
  outlier_rate: 20.0,
  reliability: 'unknown',
  noise_multiplier: 1.0,
  priority: 999,
  base_threshold_kg: 3.0,
  max_threshold_kg: 10.0,
} as const;

/**
 * Session detection parameters
 */
export const SESSION_TIMEOUT_MINUTES = 5.0;
export const SESSION_VARIANCE_THRESHOLD = 5.81; // kg

/**
 * Visualization marker symbols for source types
 */
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

/**
 * Rejection severity color mapping
 */
export const REJECTION_SEVERITY_COLORS: Record<string, string> = {
  Critical: '#8B0000', // Dark red for impossible values
  High: '#CC0000', // Medium-dark red for extreme deviations
  Medium: '#FF4444', // Medium red for suspicious values
  Low: '#FF9999', // Light red for minor issues
};

/**
 * Source profile type (will be loaded from config)
 */
export interface SourceProfile {
  outlier_rate: number;
  reliability: string;
  noise_multiplier: number;
  priority: number;
  base_threshold_kg?: number;
  max_threshold_kg?: number;
}

/**
 * Get priority for a source (lower number = higher priority)
 *
 * @param source Source identifier
 * @param sourceProfiles Source profiles loaded from config (optional)
 * @returns Priority value
 */
export function getSourcePriority(
  source: string,
  sourceProfiles?: Record<string, SourceProfile>
): number {
  if (!sourceProfiles) {
    return DEFAULT_SOURCE_PROFILE.priority;
  }
  const profile = sourceProfiles[source];
  return profile?.priority ?? DEFAULT_SOURCE_PROFILE.priority;
}

/**
 * Get reliability classification for source
 *
 * @param source Source identifier
 * @param sourceProfiles Source profiles loaded from config (optional)
 * @returns Reliability classification
 */
export function getSourceReliability(
  source: string,
  sourceProfiles?: Record<string, SourceProfile>
): string {
  if (!sourceProfiles) {
    return DEFAULT_SOURCE_PROFILE.reliability;
  }
  const profile = sourceProfiles[source];
  return profile?.reliability ?? DEFAULT_SOURCE_PROFILE.reliability;
}

/**
 * Get Kalman filter measurement noise multiplier for source
 *
 * @param source Source identifier
 * @param sourceProfiles Source profiles loaded from config (optional)
 * @returns Noise multiplier
 */
export function getNoiseMultiplier(
  source: string,
  sourceProfiles?: Record<string, SourceProfile>
): number {
  if (!sourceProfiles) {
    return DEFAULT_SOURCE_PROFILE.noise_multiplier;
  }
  const profile = sourceProfiles[source];
  return profile?.noise_multiplier ?? DEFAULT_SOURCE_PROFILE.noise_multiplier;
}

/**
 * Categorize rejection reason for analytics
 *
 * @param reason Rejection reason string
 * @returns Category string
 */
export function categorizeRejection(reason: string): string {
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
 * Determine severity of rejection
 *
 * @param reason Rejection reason string
 * @param weightChange Weight change amount (optional)
 * @returns Severity level
 */
export function getRejectionSeverity(reason: string, weightChange = 0): string {
  const reasonLower = reason.toLowerCase();

  if (
    reasonLower.includes('impossible') ||
    reasonLower.includes('physiologically impossible')
  ) {
    return 'Critical';
  } else if (reasonLower.includes('extreme') || weightChange > 20) {
    return 'High';
  } else if (reasonLower.includes('suspicious') || weightChange > 10) {
    return 'Medium';
  } else {
    return 'Low';
  }
}

/**
 * Check if a source is a questionnaire source
 *
 * @param source Source identifier
 * @returns True if source is a questionnaire
 */
export function isQuestionnaireSource(source: string): boolean {
  return QUESTIONNAIRE_SOURCES.has(source);
}
