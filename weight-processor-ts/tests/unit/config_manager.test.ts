/**
 * Unit tests for ConfigManager
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import { ConfigManager, loadConfig } from '../../src/config/config_manager';
import { join } from 'path';

describe('ConfigManager', () => {
  beforeEach(() => {
    // Reset singleton for each test
    ConfigManager.getInstance().reset();
  });

  it('should load configuration from file', () => {
    const configPath = join(process.cwd(), 'config.toml');
    const config = loadConfig(configPath);

    expect(config).toBeDefined();
    expect(config.kalman).toBeDefined();
    expect(config.sources).toBeDefined();
  });

  it('should load Kalman configuration', () => {
    const config = loadConfig();

    expect(config.kalman.initial_variance).toBeGreaterThan(0);
    expect(config.kalman.transition_covariance_weight).toBeGreaterThan(0);
    expect(config.kalman.observation_covariance).toBeGreaterThan(0);
  });

  it('should load reset configurations', () => {
    const config = loadConfig();

    expect(config.kalman.reset.initial).toBeDefined();
    expect(config.kalman.reset.hard).toBeDefined();
    expect(config.kalman.reset.soft).toBeDefined();

    expect(config.kalman.reset.initial.enabled).toBe(true);
    expect(config.kalman.reset.hard.enabled).toBe(true);
    expect(config.kalman.reset.soft.enabled).toBe(true);
  });

  it('should load quality scoring configuration', () => {
    const config = loadConfig();

    expect(config.quality_scoring.threshold).toBeGreaterThan(0);
    expect(config.quality_scoring.component_weights).toBeDefined();

    // Check weights sum to 1.0 (approximately)
    const weights = config.quality_scoring.component_weights;
    const sum =
      weights.kalman_fit +
      weights.temporal_consistency +
      weights.anomaly_detection +
      weights.source_reliability +
      weights.trend_alignment;

    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('should load source profiles', () => {
    const config = loadConfig();

    expect(config.sources).toBeDefined();
    expect(config.sources['care-team-upload']).toBeDefined();
    expect(config.sources['patient-upload']).toBeDefined();
    expect(config.sources['questionnaire']).toBeDefined();
    expect(config.sources['default']).toBeDefined();
  });

  it('should load source profile properties', () => {
    const config = loadConfig();
    const careTeamProfile = config.sources['care-team-upload'];

    expect(careTeamProfile).toBeDefined();
    expect(careTeamProfile?.reliability).toBe('excellent');
    expect(careTeamProfile?.priority).toBe(1);
    expect(careTeamProfile?.noise_multiplier).toBeGreaterThan(0);
  });

  it('should load replay configuration', () => {
    const config = loadConfig();

    expect(config.replay.enabled).toBe(true);
    expect(config.replay.buffer_hours).toBeGreaterThan(0);
    expect(config.replay.outlier_detection).toBeDefined();
  });

  it('should load adaptive ranges configuration', () => {
    const config = loadConfig();

    expect(config.adaptive_ranges.enabled).toBe(true);
    expect(config.adaptive_ranges.ema_alpha).toBeGreaterThan(0);
    expect(config.adaptive_ranges.thresholds).toBeDefined();
  });

  it('should be singleton', () => {
    const instance1 = ConfigManager.getInstance();
    const instance2 = ConfigManager.getInstance();

    expect(instance1).toBe(instance2);
  });

  it('should cache loaded config', () => {
    const config1 = loadConfig();
    const config2 = loadConfig();

    expect(config1).toBe(config2);
  });

  it('should throw error if config not loaded when accessing', () => {
    const manager = ConfigManager.getInstance();
    expect(() => manager.getConfig()).toThrow();
  });

  it('should load snapshot configuration', () => {
    const config = loadConfig();

    expect(config.snapshot.periodic_enabled).toBe(true);
    expect(config.snapshot.interval_hours).toBeGreaterThan(0);
    expect(config.snapshot.retention_days).toBeGreaterThan(0);
  });

  it('should load database configuration', () => {
    const config = loadConfig();

    expect(config.database.backend).toBe('memory');
    expect(config.database.table_name).toBeDefined();
  });
});
