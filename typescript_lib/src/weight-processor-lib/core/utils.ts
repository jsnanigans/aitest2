/**
 * Utility functions for weight stream processor.
 * Consolidates logging and general utilities.
 */

// ============================================================================
// Logging Utilities
// ============================================================================

export enum LogLevel {
  ERROR = 'ERROR',
  WARNING = 'WARNING',
  INFO = 'INFO',
  METRIC = 'METRIC',
}

export class StructuredLogger {
  /**
   * Simple structured logger for production use.
   */
  constructor(
    public name: string,
    public enabled: boolean = true
  ) {}

  private _log(level: LogLevel, message: string, extras: Record<string, any> = {}): void {
    if (!this.enabled) {
      return;
    }

    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      logger: this.name,
      message,
      ...extras,
    };

    if (level === LogLevel.ERROR) {
      console.error(JSON.stringify(logEntry));
    } else if (level === LogLevel.METRIC) {
      console.log(JSON.stringify(logEntry));
    }
  }

  error(message: string, extras: Record<string, any> = {}): void {
    this._log(LogLevel.ERROR, message, extras);
  }

  warning(message: string, extras: Record<string, any> = {}): void {
    this._log(LogLevel.WARNING, message, extras);
  }

  info(message: string, extras: Record<string, any> = {}): void {
    this._log(LogLevel.INFO, message, extras);
  }

  metric(metricName: string, value: number, tags: Record<string, any> = {}): void {
    this._log(LogLevel.METRIC, `Metric: ${metricName}`, {
      metric: metricName,
      value,
      tags,
    });
  }
}

export class PerformanceTimer {
  /**
   * Context manager for timing operations.
   */
  private startTime: Date | null = null;

  constructor(
    private logger: StructuredLogger,
    private operation: string
  ) {}

  start(): this {
    this.startTime = new Date();
    return this;
  }

  end(): void {
    if (this.startTime) {
      const durationMs = new Date().getTime() - this.startTime.getTime();
      this.logger.metric(`${this.operation}_duration_ms`, durationMs, {
        operation: this.operation,
      });
    }
  }
}

// Global logger instances
export const processorLogger = new StructuredLogger('processor');
export const validationLogger = new StructuredLogger('validation');
export const kalmanLogger = new StructuredLogger('kalman');

// ============================================================================
// Visualization Logging
// ============================================================================

export class VizLogger {
  /**
   * Simple logger for visualization modules.
   */
  constructor(public verbosity: number = 0) {}

  debug(msg: string): void {
    if (this.verbosity >= 2) {
      console.log(`[DEBUG] ${msg}`);
    }
  }

  info(msg: string): void {
    if (this.verbosity >= 1) {
      console.log(`[INFO] ${msg}`);
    }
  }

  warning(msg: string): void {
    console.warn(`[WARNING] ${msg}`);
  }

  error(msg: string): void {
    console.error(`[ERROR] ${msg}`);
  }
}

let _vizLogger: VizLogger | null = null;
let _verbosityLevel = 0;

export function getLogger(): VizLogger {
  /**
   * Get the global visualization logger instance.
   */
  if (_vizLogger === null) {
    _vizLogger = new VizLogger(_verbosityLevel);
  }
  return _vizLogger;
}

export function setVerbosity(level: number): void {
  /**
   * Set the global verbosity level.
   */
  _verbosityLevel = level;
  if (_vizLogger !== null) {
    _vizLogger.verbosity = level;
  }
}

// ============================================================================
// General Utilities
// ============================================================================

export function formatTimestamp(ts: Date | string | any): string {
  /**
   * Format timestamp for display.
   */
  if (typeof ts === 'string') {
    try {
      ts = new Date(ts);
    } catch {
      return String(ts);
    }
  }

  if (ts instanceof Date) {
    return ts.toISOString().replace('T', ' ').substring(0, 19);
  }

  return String(ts);
}

// ============================================================================
// Floating-Point Comparison Utilities
// ============================================================================

/**
 * Default epsilon for floating-point comparisons.
 * Set to 1e-10 to handle typical floating-point precision issues.
 */
export const DEFAULT_EPSILON = 1e-10;

/**
 * Compare two floating-point numbers for equality within epsilon tolerance.
 *
 * @param a - First number
 * @param b - Second number
 * @param epsilon - Tolerance (default: DEFAULT_EPSILON)
 * @returns true if |a - b| < epsilon
 */
export function floatsEqual(a: number, b: number, epsilon: number = DEFAULT_EPSILON): boolean {
  return Math.abs(a - b) < epsilon;
}

/**
 * Check if a floating-point number is effectively zero.
 *
 * @param value - Number to check
 * @param epsilon - Tolerance (default: DEFAULT_EPSILON)
 * @returns true if |value| < epsilon
 */
export function isEffectivelyZero(value: number, epsilon: number = DEFAULT_EPSILON): boolean {
  return Math.abs(value) < epsilon;
}

/**
 * Round a number to a specified number of significant digits.
 *
 * @param num - Number to round
 * @param significantDigits - Number of significant digits
 * @returns Rounded number
 */
export function roundToSignificant(num: number, significantDigits: number): number {
  if (num === 0) return 0;
  const multiplier = Math.pow(10, significantDigits - Math.floor(Math.log10(Math.abs(num))) - 1);
  return Math.round(num * multiplier) / multiplier;
}

/**
 * Clamp a value within epsilon of bounds.
 * Useful for ensuring values stay within [0, 1] despite floating-point errors.
 *
 * @param value - Value to clamp
 * @param min - Minimum value
 * @param max - Maximum value
 * @param epsilon - Tolerance (default: DEFAULT_EPSILON)
 * @returns Clamped value
 */
export function epsilonClamp(
  value: number,
  min: number,
  max: number,
  epsilon: number = DEFAULT_EPSILON
): number {
  // If value is within epsilon of bounds, snap to bounds
  if (floatsEqual(value, min, epsilon)) return min;
  if (floatsEqual(value, max, epsilon)) return max;

  // Otherwise clamp normally
  return Math.max(min, Math.min(max, value));
}

export function safeDivide(numerator: number, denominator: number, defaultValue: number = 0.0): number {
  /**
   * Safely divide two numbers, using epsilon comparison for zero check.
   */
  try {
    if (isEffectivelyZero(denominator)) {
      return defaultValue;
    }
    return numerator / denominator;
  } catch {
    return defaultValue;
  }
}

export interface ConfigValidationResult {
  isValid: boolean;
  errors: string[];
}

export function validateConfig(config: Record<string, any>): ConfigValidationResult {
  /**
   * Validate configuration structure and values.
   */
  const errors: string[] = [];

  // Check required sections
  const requiredSections = [
    'data',
    'processing',
    'kalman',
    'visualization',
    'logging',
    'quality_scoring',
  ];

  for (const section of requiredSections) {
    if (!(section in config)) {
      errors.push(`Missing required section: [${section}]`);
    }
  }

  // Validate data section
  if ('data' in config) {
    const data = config.data;
    if (!('csv_file' in data)) {
      errors.push('Missing required field: data.csv_file');
    }
    if (!('output_dir' in data)) {
      errors.push('Missing required field: data.output_dir');
    }
  }

  // Validate processing section
  if ('processing' in config) {
    const processing = config.processing;
    if ('extreme_threshold' in processing) {
      const threshold = processing.extreme_threshold;
      if (!(0 < threshold && threshold < 1)) {
        errors.push(`Invalid extreme_threshold: ${threshold} (must be between 0 and 1)`);
      }
    }
  }

  // Validate kalman section
  if ('kalman' in config) {
    const kalman = config.kalman;
    const requiredKalman = [
      'initial_variance',
      'transition_covariance_weight',
      'transition_covariance_trend',
      'observation_covariance',
    ];

    for (const field of requiredKalman) {
      if (!(field in kalman)) {
        errors.push(`Missing required Kalman field: ${field}`);
      } else if (kalman[field] <= 0) {
        errors.push(`Invalid Kalman ${field}: must be positive`);
      }
    }
  }

  // Validate quality scoring weights
  if ('quality_scoring' in config) {
    const qs = config.quality_scoring;
    if ('component_weights' in qs) {
      const weights = qs.component_weights;
      const total = Object.values(weights as Record<string, number>).reduce(
        (sum, w) => sum + (w as number),
        0
      );
      if (Math.abs(total - 1.0) > 0.001) {
        errors.push(`Quality scoring weights must sum to 1.0, got ${total.toFixed(3)}`);
      }
      for (const [name, weight] of Object.entries(weights as Record<string, number>)) {
        if (!(0 <= weight && weight <= 1)) {
          errors.push(`Invalid weight for ${name}: ${weight} (must be 0-1)`);
        }
      }
    }
  }

  // Validate visualization verbosity
  if ('visualization' in config) {
    const viz = config.visualization;
    if ('verbosity' in viz) {
      const validVerbosity = ['silent', 'minimal', 'normal', 'verbose'];
      if (!validVerbosity.includes(viz.verbosity)) {
        errors.push(
          `Invalid verbosity: ${viz.verbosity} (must be one of ${validVerbosity.join(', ')})`
        );
      }
    }
  }

  // Validate adaptive noise
  if ('adaptive_noise' in config) {
    const noise = config.adaptive_noise;
    if ('default_multiplier' in noise) {
      const multiplier = noise.default_multiplier;
      if (!(0.5 <= multiplier && multiplier <= 5.0)) {
        errors.push(
          `Invalid default_multiplier: ${multiplier} (should be between 0.5 and 5.0)`
        );
      }
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}
