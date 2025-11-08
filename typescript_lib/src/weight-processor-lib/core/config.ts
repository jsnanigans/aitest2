/**
 * Configuration loader for weight processor.
 * Loads config from config.json (converted from Python's config.toml).
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

// Get the directory of this file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface SourceConfig {
  outlier_rate?: number;
  reliability: string;
  noise_multiplier: number;
  priority: number;
  base_threshold_kg?: number;
  max_threshold_kg?: number;
  spike_threshold_kg?: number;
  drift_threshold_kg?: number;
  noise_threshold_kg?: number;
  max_daily_change_kg?: number;
  high_confidence_multiplier?: number;
  low_confidence_multiplier?: number;
}

export interface KalmanConfig {
  initial_variance: number;
  transition_covariance_weight: number;
  transition_covariance_trend: number;
  observation_covariance: number;
  reset?: {
    initial?: any;
    hard?: any;
    soft?: any;
  };
}

export interface QualityScoringConfig {
  use_harmonic_mean: boolean;
  threshold: number;
  component_weights: {
    kalman_fit: number;
    temporal_consistency: number;
    anomaly_detection: number;
    source_reliability: number;
    trend_alignment: number;
  };
  temporal?: {
    min_score: number;
    max_score: number;
    initial_threshold_kg: number;
    max_threshold_kg: number;
    time_constant_hours: number;
  };
  trend_alignment?: {
    trend_decay_constant: number;
    trend_min_std_dev: number;
  };
}

export interface Config {
  kalman: KalmanConfig;
  quality_scoring: QualityScoringConfig;
  processing: {
    quality_threshold: number;
  };
  adaptive_ranges?: any;
  sources: Record<string, SourceConfig>;
  replay?: any;
  snapshot?: any;
  database?: any;
  logging?: any;
}

let cachedConfig: Config | null = null;

/**
 * Load configuration from config.json.
 * Config is cached after first load.
 *
 * IMPORTANT: No hardcoded defaults - config.json MUST exist.
 * This ensures TypeScript always uses the same values as Python's config.toml.
 */
export function loadConfig(): Config {
  if (cachedConfig) {
    return cachedConfig;
  }

  try {
    // Try to load from typescript_lib root (../../.. from this file)
    const configPath = join(__dirname, '../../../config.json');
    const configData = readFileSync(configPath, 'utf-8');
    cachedConfig = JSON.parse(configData);

    // Validate required sections exist
    if (!cachedConfig.kalman || !cachedConfig.sources || !cachedConfig.quality_scoring) {
      throw new Error('config.json is missing required sections (kalman, sources, quality_scoring)');
    }

    return cachedConfig!;
  } catch (error) {
    // NO FALLBACK DEFAULTS - config.json must exist
    throw new Error(
      `Failed to load config.json. Config file is required and must be present at typescript_lib/config.json. ` +
      `This ensures TypeScript uses the exact same configuration as Python's config.toml. ` +
      `Error: ${error}`
    );
  }
}

/**
 * Get source configuration for a specific source.
 */
export function getSourceConfig(source: string): SourceConfig {
  const config = loadConfig();
  return config.sources[source] ?? config.sources.default;
}

/**
 * Get Kalman configuration.
 */
export function getKalmanConfig(): KalmanConfig {
  const config = loadConfig();
  return config.kalman;
}

/**
 * Get quality scoring configuration.
 */
export function getQualityScoringConfig(): QualityScoringConfig {
  const config = loadConfig();
  return config.quality_scoring;
}
