/**
 * Type conversion utilities for handling numeric conversions.
 */

/**
 * Convert a value to float, handling various numeric types.
 */
export function ensureFloat(value: any): number {
  if (value === null || value === undefined) {
    return 0.0;
  }

  if (typeof value === 'number') {
    return value;
  }

  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? 0.0 : parsed;
  }

  // Try to convert anything else
  try {
    const num = Number(value);
    return isNaN(num) ? 0.0 : num;
  } catch {
    return 0.0;
  }
}

/**
 * Recursively ensure all numeric values in a data structure are proper types.
 */
export function ensureNumericTypes(data: any): any {
  if (data === null || data === undefined) {
    return data;
  }

  if (typeof data === 'object' && !Array.isArray(data)) {
    const result: any = {};
    for (const [key, value] of Object.entries(data)) {
      const numericFields = new Set([
        'weight',
        'filtered_weight',
        'raw_weight',
        'quality_score',
        'kalman_deviation',
        'temporal_consistency',
        'source_reliability',
      ]);

      if (numericFields.has(key)) {
        result[key] = ensureFloat(value);
      } else if (typeof value === 'object') {
        result[key] = ensureNumericTypes(value);
      } else if (typeof value === 'string') {
        const num = parseFloat(value);
        result[key] = isNaN(num) ? value : num;
      } else {
        result[key] = value;
      }
    }
    return result;
  }

  if (Array.isArray(data)) {
    return data.map(item => {
      if (typeof item === 'object') {
        return ensureNumericTypes(item);
      } else if (typeof item === 'string') {
        const num = parseFloat(item);
        return isNaN(num) ? item : num;
      }
      return item;
    });
  }

  if (typeof data === 'string') {
    const num = parseFloat(data);
    return isNaN(num) ? data : num;
  }

  return data;
}

/**
 * Prepare a measurement for processing by ensuring proper types.
 */
export function prepareMeasurementForProcessing(measurement: Record<string, any>): Record<string, any> {
  const cleanMeasurement = { ...measurement };

  // Ensure weight is float
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
