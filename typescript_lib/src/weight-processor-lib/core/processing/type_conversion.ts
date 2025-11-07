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
  // Preserve Date objects - don't recurse into them
  if (data instanceof Date) {
    return data;
  }

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
      } else if (value instanceof Date) {
        // Preserve Date objects
        result[key] = value;
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
      if (item instanceof Date) {
        // Preserve Date objects
        return item;
      } else if (typeof item === 'object' && item !== null) {
        return ensureNumericTypes(item);
      }
      return item;
    });
  }

  return data;
}

export function deserializeState(state: any): any {
  /**
   * Deserialize state retrieved from storage, converting date strings back to Date objects.
   *
   * When state is serialized (e.g., via JSON.stringify in InMemoryStore), Date objects
   * become ISO strings. This function converts them back to Date objects.
   *
   * Args:
   *     state: State object from storage
   *
   * Returns:
   *     State with Date fields properly deserialized
   */
  if (!state || typeof state !== 'object') {
    return state;
  }

  const dateFields = ['last_timestamp', 'last_accepted_timestamp', 'reset_timestamp'];

  // Convert top-level date fields
  for (const field of dateFields) {
    if (field in state && state[field] !== null && state[field] !== undefined) {
      if (typeof state[field] === 'string') {
        state[field] = new Date(state[field]);
      }
    }
  }

  // Convert dates in measurement_history
  if (Array.isArray(state.measurement_history)) {
    state.measurement_history = state.measurement_history.map((measurement: any) => {
      if (measurement && typeof measurement === 'object') {
        if (measurement.timestamp && typeof measurement.timestamp === 'string') {
          measurement.timestamp = new Date(measurement.timestamp);
        }
      }
      return measurement;
    });
  }

  // Convert dates in reset_events
  if (Array.isArray(state.reset_events)) {
    state.reset_events = state.reset_events.map((event: any) => {
      if (event && typeof event === 'object') {
        if (event.timestamp && typeof event.timestamp === 'string') {
          event.timestamp = new Date(event.timestamp);
        }
      }
      return event;
    });
  }

  return state;
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
