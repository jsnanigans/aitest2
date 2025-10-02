/**
 * General utility functions
 */

/**
 * Deep copy an object or array
 *
 * Creates a deep clone of objects, arrays, Dates, and primitive types.
 * Note: Does not handle circular references, functions, or special objects.
 *
 * @param obj Object to copy
 * @returns Deep copy of the object
 */
export function deepCopy<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as T;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => deepCopy(item)) as T;
  }

  if (obj instanceof Map) {
    const copy = new Map();
    obj.forEach((value, key) => {
      copy.set(key, deepCopy(value));
    });
    return copy as T;
  }

  if (obj instanceof Set) {
    const copy = new Set();
    obj.forEach((value) => {
      copy.add(deepCopy(value));
    });
    return copy as T;
  }

  // Plain object
  const copy: any = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      copy[key] = deepCopy(obj[key]);
    }
  }
  return copy as T;
}

/**
 * Parse a timestamp from various formats
 *
 * Supports:
 * - ISO 8601 strings (with or without timezone)
 * - Date objects
 * - Unix timestamps (milliseconds)
 *
 * @param timestamp Input timestamp in various formats
 * @returns Date object
 * @throws Error if timestamp format is invalid
 */
export function parseTimestamp(timestamp: string | Date | number): Date {
  // Already a Date
  if (timestamp instanceof Date) {
    return timestamp;
  }

  // Unix timestamp (number)
  if (typeof timestamp === 'number') {
    return new Date(timestamp);
  }

  // ISO string
  if (typeof timestamp === 'string') {
    // Handle 'Z' timezone indicator
    const normalized = timestamp.replace('Z', '+00:00');

    const date = new Date(normalized);

    if (isNaN(date.getTime())) {
      throw new Error(`Invalid timestamp format: ${timestamp}`);
    }

    return date;
  }

  throw new Error(`Unsupported timestamp type: ${typeof timestamp}`);
}

/**
 * Ensure a value is converted to a float
 *
 * Handles various numeric types and converts them to float.
 * Returns 0.0 for null, undefined, or unconvertible values.
 *
 * @param value Value to convert
 * @returns Float value
 */
export function ensureFloat(value: any): number {
  if (value === null || value === undefined) {
    return 0.0;
  }

  // Already a number
  if (typeof value === 'number') {
    return value;
  }

  // String
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? 0.0 : parsed;
  }

  // Try to convert
  try {
    const converted = Number(value);
    return isNaN(converted) ? 0.0 : converted;
  } catch {
    return 0.0;
  }
}

/**
 * Calculate time difference in seconds between two timestamps
 *
 * @param timestamp1 First timestamp
 * @param timestamp2 Second timestamp
 * @returns Time difference in seconds (absolute value)
 */
export function timeDiffSeconds(
  timestamp1: string | Date | number,
  timestamp2: string | Date | number
): number {
  const date1 = parseTimestamp(timestamp1);
  const date2 = parseTimestamp(timestamp2);
  return Math.abs(date1.getTime() - date2.getTime()) / 1000;
}

/**
 * Calculate time difference in days between two timestamps
 *
 * @param timestamp1 First timestamp
 * @param timestamp2 Second timestamp
 * @returns Time difference in days (absolute value)
 */
export function timeDiffDays(
  timestamp1: string | Date | number,
  timestamp2: string | Date | number
): number {
  return timeDiffSeconds(timestamp1, timestamp2) / 86400;
}

/**
 * Add seconds to a timestamp
 *
 * @param timestamp Base timestamp
 * @param seconds Seconds to add (can be negative)
 * @returns New Date object
 */
export function addSeconds(timestamp: string | Date | number, seconds: number): Date {
  const date = parseTimestamp(timestamp);
  return new Date(date.getTime() + seconds * 1000);
}

/**
 * Add days to a timestamp
 *
 * @param timestamp Base timestamp
 * @param days Days to add (can be negative)
 * @returns New Date object
 */
export function addDays(timestamp: string | Date | number, days: number): Date {
  return addSeconds(timestamp, days * 86400);
}

/**
 * Format a Date as ISO 8601 string
 *
 * @param date Date to format
 * @returns ISO 8601 string
 */
export function toISOString(date: Date): string {
  return date.toISOString();
}

/**
 * Check if a value is a plain object (not an array, Date, etc.)
 *
 * @param value Value to check
 * @returns True if value is a plain object
 */
export function isPlainObject(value: any): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    !(value instanceof Date) &&
    !(value instanceof Map) &&
    !(value instanceof Set)
  );
}

/**
 * Recursively ensure all numeric values in a data structure are proper numbers
 *
 * Converts string numbers, handles null/undefined, and ensures clean numeric types.
 * Useful for preparing data from JSON or CSV for processing.
 *
 * @param data Data structure to process
 * @param numericFields Set of field names that should be converted to numbers
 * @returns Data with proper numeric types
 */
export function ensureNumericTypes(
  data: any,
  numericFields: Set<string> = new Set([
    'weight',
    'weight_kg',
    'filtered_weight',
    'raw_weight',
    'quality_score',
    'kalman_deviation',
    'temporal_consistency',
    'source_reliability',
  ])
): any {
  if (isPlainObject(data)) {
    const result: any = {};
    for (const [key, value] of Object.entries(data)) {
      if (numericFields.has(key)) {
        result[key] = ensureFloat(value);
      } else if (isPlainObject(value) || Array.isArray(value)) {
        result[key] = ensureNumericTypes(value, numericFields);
      } else {
        result[key] = value;
      }
    }
    return result;
  }

  if (Array.isArray(data)) {
    return data.map((item) => ensureNumericTypes(item, numericFields));
  }

  return data;
}

/**
 * Clamp a number between min and max values
 *
 * @param value Value to clamp
 * @param min Minimum value
 * @param max Maximum value
 * @returns Clamped value
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Check if two numbers are approximately equal within a tolerance
 *
 * @param a First number
 * @param b Second number
 * @param tolerance Tolerance (default 1e-10)
 * @returns True if numbers are approximately equal
 */
export function approxEqual(a: number, b: number, tolerance = 1e-10): boolean {
  return Math.abs(a - b) < tolerance;
}

/**
 * Generate a unique ID string
 *
 * @returns Unique ID
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}
