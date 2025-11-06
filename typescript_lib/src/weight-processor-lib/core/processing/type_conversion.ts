/**
 * Type conversion utilities for handling numeric conversions.
 */

export function ensureFloat(value: any): number {
  /**
   * Convert a value to number, handling various types.
   *
   * Args:
   *     value: Value to convert
   *
   * Returns:
   *     Number value
   */
  if (value === null || value === undefined) {
    return 0.0;
  }

  // Check if it's already a number
  if (typeof value === 'number') {
    return value;
  }

  // Try to convert anything else
  try {
    const num = Number(value);
    return isNaN(num) ? 0.0 : num;
  } catch {
    return 0.0;
  }
}

export function ensureNumericTypes(data: any): any {
  /**
   * Recursively ensure all numeric values in a data structure are proper types.
   *
   * Args:
   *     data: Data structure to process
   *
   * Returns:
   *     Data with proper numeric types
   */
  if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
    // Handle objects (dictionaries)
    const result: Record<string, any> = {};
    for (const [key, value] of Object.entries(data)) {
      if (
        [
          'weight',
          'filtered_weight',
          'raw_weight',
          'quality_score',
          'kalman_deviation',
          'temporal_consistency',
          'source_reliability',
        ].includes(key)
      ) {
        // These are numeric fields that should be numbers
        result[key] = ensureFloat(value);
      } else if (typeof value === 'object' && value !== null) {
        result[key] = ensureNumericTypes(value);
      } else {
        result[key] = value;
      }
    }
    return result;
  } else if (Array.isArray(data)) {
    // Handle arrays (lists)
    return data.map((item) => {
      if (typeof item === 'object' && item !== null) {
        return ensureNumericTypes(item);
      }
      return item;
    });
  }

  return data;
}

export function prepareMeasurementForProcessing(measurement: Record<string, any>): Record<string, any> {
  /**
   * Prepare a measurement for processing by ensuring proper types.
   *
   * Args:
   *     measurement: Measurement dictionary
   *
   * Returns:
   *     Measurement with proper types
   */
  const cleanMeasurement = { ...measurement };

  // Ensure weight is number
  if ('weight' in cleanMeasurement) {
    cleanMeasurement.weight = ensureFloat(cleanMeasurement.weight);
  }

  if ('raw_weight' in cleanMeasurement) {
    cleanMeasurement.raw_weight = ensureFloat(cleanMeasurement.raw_weight);
  }

  if ('filtered_weight' in cleanMeasurement) {
    cleanMeasurement.filtered_weight = ensureFloat(cleanMeasurement.filtered_weight);
  }

  if ('quality_score' in cleanMeasurement) {
    cleanMeasurement.quality_score = ensureFloat(cleanMeasurement.quality_score);
  }

  // Handle nested metadata
  if ('metadata' in cleanMeasurement && typeof cleanMeasurement.metadata === 'object') {
    cleanMeasurement.metadata = ensureNumericTypes(cleanMeasurement.metadata);
  }

  return cleanMeasurement;
}
