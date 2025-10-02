/**
 * Statistical functions
 *
 * Implements statistical operations needed for quality scoring and outlier detection.
 * These functions provide equivalents to NumPy and SciPy statistical operations.
 */

/**
 * Calculate the arithmetic mean of an array
 *
 * @param values Array of numbers
 * @returns Mean value
 * @throws Error if array is empty
 */
export function mean(values: number[]): number {
  if (values.length === 0) {
    throw new Error('Cannot calculate mean of empty array');
  }
  const sum = values.reduce((acc, val) => acc + val, 0);
  return sum / values.length;
}

/**
 * Calculate the median of an array
 *
 * @param values Array of numbers
 * @returns Median value
 * @throws Error if array is empty
 */
export function median(values: number[]): number {
  if (values.length === 0) {
    throw new Error('Cannot calculate median of empty array');
  }

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1]! + sorted[mid]!) / 2;
  } else {
    return sorted[mid]!;
  }
}

/**
 * Calculate the variance of an array
 *
 * @param values Array of numbers
 * @param ddof Delta degrees of freedom (default 0 for population variance, 1 for sample variance)
 * @returns Variance
 * @throws Error if array is too small
 */
export function variance(values: number[], ddof = 0): number {
  if (values.length <= ddof) {
    throw new Error(`Cannot calculate variance with ${values.length} values and ddof=${ddof}`);
  }

  const avg = mean(values);
  const squaredDiffs = values.map((val) => Math.pow(val - avg, 2));
  const sum = squaredDiffs.reduce((acc, val) => acc + val, 0);
  return sum / (values.length - ddof);
}

/**
 * Calculate the standard deviation of an array
 *
 * @param values Array of numbers
 * @param ddof Delta degrees of freedom (default 0 for population std, 1 for sample std)
 * @returns Standard deviation
 */
export function std(values: number[], ddof = 0): number {
  return Math.sqrt(variance(values, ddof));
}

/**
 * Calculate a percentile of an array
 *
 * Uses linear interpolation between closest ranks (NumPy default method)
 *
 * @param values Array of numbers
 * @param percentile Percentile to calculate (0-100)
 * @returns Percentile value
 * @throws Error if array is empty or percentile is out of range
 */
export function percentile(values: number[], percentile: number): number {
  if (values.length === 0) {
    throw new Error('Cannot calculate percentile of empty array');
  }
  if (percentile < 0 || percentile > 100) {
    throw new Error(`Percentile must be between 0 and 100, got ${percentile}`);
  }

  const sorted = [...values].sort((a, b) => a - b);

  if (percentile === 0) {
    return sorted[0]!;
  }
  if (percentile === 100) {
    return sorted[sorted.length - 1]!;
  }

  // Linear interpolation (NumPy default method)
  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;

  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

/**
 * Perform linear regression (polyfit degree 1)
 *
 * Fits y = slope * x + intercept using least squares
 *
 * @param x Array of x values
 * @param y Array of y values
 * @returns Tuple of [slope, intercept]
 * @throws Error if arrays are empty, different lengths, or x has no variance
 */
export function linearRegression(x: number[], y: number[]): [number, number] {
  if (x.length === 0 || y.length === 0) {
    throw new Error('Cannot perform linear regression on empty arrays');
  }
  if (x.length !== y.length) {
    throw new Error(`x and y must have same length, got ${x.length} and ${y.length}`);
  }
  if (x.length < 2) {
    throw new Error('Need at least 2 points for linear regression');
  }

  const n = x.length;
  const meanX = mean(x);
  const meanY = mean(y);

  // Calculate slope: sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i]! - meanX;
    const dy = y[i]! - meanY;
    numerator += dx * dy;
    denominator += dx * dx;
  }

  if (Math.abs(denominator) < 1e-10) {
    throw new Error('x values have no variance, cannot perform regression');
  }

  const slope = numerator / denominator;
  const intercept = meanY - slope * meanX;

  return [slope, intercept];
}

/**
 * Error function (erf) using Abramowitz and Stegun approximation
 *
 * Maximum error: 1.5 × 10^-7
 * Formula 7.1.26 from "Handbook of Mathematical Functions"
 *
 * @param x Input value
 * @returns erf(x)
 */
export function erf(x: number): number {
  // Constants for Abramowitz and Stegun approximation
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  // Save the sign of x
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);

  // Abramowitz and Stegun formula
  const t = 1.0 / (1.0 + p * absX);
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;

  const erf = 1.0 - ((a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.exp(-absX * absX));

  return sign * erf;
}

/**
 * Normal (Gaussian) cumulative distribution function
 *
 * @param x Input value
 * @param mean Mean of the distribution (default 0)
 * @param stdDev Standard deviation (default 1)
 * @returns CDF value P(X <= x)
 */
export function normalCdf(x: number, mean = 0, stdDev = 1): number {
  const z = (x - mean) / stdDev;
  return 0.5 * (1 + erf(z / Math.sqrt(2)));
}

/**
 * Chi-squared cumulative distribution function
 *
 * Uses Gamma function approximation for chi-squared CDF
 * This is a simplified implementation for df=1 (one degree of freedom)
 * which is what's used in the quality scorer
 *
 * @param x Input value
 * @param df Degrees of freedom (currently only df=1 is implemented)
 * @returns CDF value P(X <= x)
 * @throws Error if df !== 1 (not implemented for other df values)
 */
export function chi2Cdf(x: number, df: number): number {
  if (df !== 1) {
    throw new Error(`chi2Cdf only implemented for df=1, got df=${df}`);
  }

  if (x <= 0) {
    return 0;
  }

  // For df=1: chi2(x) = 2 * normalCdf(sqrt(x)) - 1
  // This is because chi2(1) is the square of a standard normal variable
  const sqrtX = Math.sqrt(x);
  return 2 * normalCdf(sqrtX) - 1;
}

/**
 * Calculate Median Absolute Deviation (MAD)
 *
 * MAD = median(|x_i - median(x)|)
 *
 * @param values Array of numbers
 * @returns MAD value
 */
export function mad(values: number[]): number {
  const med = median(values);
  const deviations = values.map((val) => Math.abs(val - med));
  return median(deviations);
}

/**
 * Calculate modified Z-scores using MAD
 *
 * Modified Z-score = 0.6745 * (x_i - median) / MAD
 * More robust than standard Z-score for data with outliers
 *
 * @param values Array of numbers
 * @returns Array of modified Z-scores
 * @throws Error if MAD is zero
 */
export function modifiedZScores(values: number[]): number[] {
  const med = median(values);
  const madValue = mad(values);

  if (madValue === 0) {
    throw new Error('MAD is zero, cannot calculate modified Z-scores');
  }

  return values.map((val) => (0.6745 * (val - med)) / madValue);
}
