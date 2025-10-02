/**
 * Configuration Manager
 *
 * Loads and manages configuration from config.toml file.
 * Provides typed access to all configuration parameters.
 */

import * as TOML from '@iarna/toml';
import { readFileSync } from 'fs';
import { join } from 'path';
import type { SourceProfile } from '../constants';

/**
 * Kalman filter configuration
 */
export interface KalmanConfig {
  initial_variance: number;
  transition_covariance_weight: number;
  transition_covariance_trend: number;
  observation_covariance: number;
}

/**
 * Reset configuration for different reset types
 */
export interface ResetConfig {
  enabled: boolean;
  initial_variance_multiplier: number;
  weight_noise_multiplier: number;
  trend_noise_multiplier: number;
  observation_noise_multiplier: number;
  adaptation_measurements: number;
  adaptation_days: number;
  adaptation_decay_rate: number;
  gap_threshold_days?: number; // For hard reset
  min_weight_change_kg?: number; // For soft reset
  cooldown_days?: number; // For soft reset
  trigger_sources?: string[]; // For soft reset
}

/**
 * Quality scoring configuration
 */
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
  temporal: {
    min_score: number;
    max_score: number;
    initial_threshold_kg: number;
    max_threshold_kg: number;
    time_constant_hours: number;
  };
  trend_alignment: {
    trend_decay_constant: number;
    trend_min_std_dev: number;
  };
}

/**
 * Replay configuration
 */
export interface ReplayConfig {
  enabled: boolean;
  buffer_hours: number;
  trigger_mode: string;
  max_buffer_measurements: number;
  state_history_limit: number;
  outlier_detection: {
    iqr_multiplier: number;
    z_score_threshold: number;
    temporal_max_change_percent: number;
    min_measurements_for_analysis: number;
  };
  safety: {
    max_processing_time_seconds: number;
    require_rollback_confirmation: boolean;
    preserve_immediate_results: boolean;
  };
}

/**
 * Adaptive ranges configuration
 */
export interface AdaptiveRangesConfig {
  enabled: boolean;
  ema_alpha: number;
  noise_alpha: number;
  buffer_size: number;
  thresholds: {
    base_multiplier: number;
    min_threshold: number;
    max_threshold: number;
  };
  oscillation: {
    high_oscillation_score: number;
    moderate_oscillation_score: number;
  };
}

/**
 * Processing configuration
 */
export interface ProcessingConfig {
  quality_threshold: number;
}

/**
 * Snapshot configuration
 */
export interface SnapshotConfig {
  periodic_enabled: boolean;
  interval_hours: number;
  retention_days: number;
}

/**
 * Database configuration
 */
export interface DatabaseConfig {
  backend: string;
  table_name: string;
  region: string;
}

/**
 * Logging configuration
 */
export interface LoggingConfig {
  level: string;
  verbose: boolean;
}

/**
 * Complete configuration structure
 */
export interface Config {
  kalman: KalmanConfig & {
    reset: {
      initial: ResetConfig;
      hard: ResetConfig;
      soft: ResetConfig;
    };
  };
  quality_scoring: QualityScoringConfig;
  processing: ProcessingConfig;
  adaptive_ranges: AdaptiveRangesConfig;
  sources: Record<string, SourceProfile>;
  replay: ReplayConfig;
  snapshot: SnapshotConfig;
  database: DatabaseConfig;
  logging: LoggingConfig;
}

/**
 * ConfigManager class for loading and accessing configuration
 */
export class ConfigManager {
  private static instance: ConfigManager | null = null;
  private config: Config | null = null;

  private constructor() {}

  /**
   * Get singleton instance
   */
  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  /**
   * Load configuration from TOML file
   *
   * @param configPath Path to config.toml file (optional, defaults to ./config.toml)
   * @returns Loaded configuration
   */
  public loadConfig(configPath?: string): Config {
    if (this.config) {
      return this.config;
    }

    const path = configPath || join(process.cwd(), 'config.toml');

    try {
      const fileContent = readFileSync(path, 'utf-8');
      const parsed = TOML.parse(fileContent);
      this.config = parsed as unknown as Config;
      return this.config;
    } catch (error) {
      throw new Error(
        `Failed to load config from ${path}: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Get current configuration (must call loadConfig first)
   */
  public getConfig(): Config {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call loadConfig() first.');
    }
    return this.config;
  }

  /**
   * Reset the configuration (for testing)
   */
  public reset(): void {
    this.config = null;
  }

  /**
   * Static helper to load config
   */
  public static loadConfig(configPath?: string): Config {
    return ConfigManager.getInstance().loadConfig(configPath);
  }

  /**
   * Static helper to get config
   */
  public static getConfig(): Config {
    return ConfigManager.getInstance().getConfig();
  }
}

/**
 * Load configuration from file
 *
 * @param configPath Path to config.toml (optional)
 * @returns Loaded configuration
 */
export function loadConfig(configPath?: string): Config {
  return ConfigManager.loadConfig(configPath);
}

/**
 * Get current configuration
 *
 * @returns Current configuration
 */
export function getConfig(): Config {
  return ConfigManager.getConfig();
}
