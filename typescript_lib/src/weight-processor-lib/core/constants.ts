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
// NOTE: Loaded from config.json (converted from Python's config.toml)
import { loadConfig, type SourceConfig } from './config.js';

export interface SourceProfile {
  outlier_rate?: number;
  reliability: string;
  noise_multiplier: number;
  priority: number;
  base_threshold_kg?: number;
  max_threshold_kg?: number;
  max_daily_change_kg?: number;
  spike_threshold_kg?: number;
  drift_threshold_kg?: number;
  noise_threshold_kg?: number;
  high_confidence_multiplier?: number;
  low_confidence_multiplier?: number;
}

// Lazy-loaded source profiles from config.json
let _sourceProfiles: Record<string, SourceProfile> | null = null;

function ensureSourceProfilesLoaded(): Record<string, SourceProfile> {
  if (!_sourceProfiles) {
    const config = loadConfig();
    _sourceProfiles = config.sources as Record<string, SourceProfile>;
  }
  return _sourceProfiles;
}

// Export as a getter to ensure lazy loading
export const SOURCE_PROFILES = new Proxy({} as Record<string, SourceProfile>, {
  get(target, prop: string) {
    const profiles = ensureSourceProfilesLoaded();
    return profiles[prop];
  },
  ownKeys(target) {
    const profiles = ensureSourceProfilesLoaded();
    return Reflect.ownKeys(profiles);
  },
  getOwnPropertyDescriptor(target, prop) {
    const profiles = ensureSourceProfilesLoaded();
    return Reflect.getOwnPropertyDescriptor(profiles, prop);
  },
});

export const DEFAULT_PROFILE: SourceProfile = new Proxy({} as SourceProfile, {
  get(target, prop: string) {
    const profiles = ensureSourceProfilesLoaded();
    return profiles.default[prop as keyof SourceProfile];
  },
});

// Questionnaire sources (for special handling)
export const QUESTIONNAIRE_SOURCES = new Set([
  'internal-questionnaire',
  'initial-questionnaire',
  'care-team-upload',
  'questionnaire',
]);

// Kalman defaults - loaded from config.json
let _kalmanDefaults: any = null;

function ensureKalmanDefaultsLoaded() {
  if (!_kalmanDefaults) {
    const config = loadConfig();
    _kalmanDefaults = {
      initial_variance: config.kalman.initial_variance,
      transition_covariance_weight: config.kalman.transition_covariance_weight,
      transition_covariance_trend: config.kalman.transition_covariance_trend,
      observation_covariance: config.kalman.observation_covariance,
    };
  }
  return _kalmanDefaults;
}

export const KALMAN_DEFAULTS = new Proxy({} as any, {
  get(target, prop: string) {
    const defaults = ensureKalmanDefaultsLoaded();
    return defaults[prop];
  },
});

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
