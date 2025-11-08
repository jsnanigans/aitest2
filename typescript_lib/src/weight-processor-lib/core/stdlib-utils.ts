/**
 * stdlib-js utility functions for validation and assertions.
 * Provides a clean wrapper around stdlib validation functions.
 */

import * as assert from '@stdlib/assert';

/**
 * Check if a value is a finite number (not NaN, not Infinity).
 * More robust than native isFinite().
 */
export function isFinite(value: any): value is number {
  return (assert as any).isFinite(value);
}

/**
 * Check if a value is NaN.
 */
export function isNaN(value: any): boolean {
  return (assert as any).isnan(value);
}

/**
 * Check if a value is a valid number (type check).
 */
export function isNumber(value: any): value is number {
  return (assert as any).isNumber(value);
}

/**
 * Check if an array contains only finite values.
 */
export function isFiniteArray(arr: any[]): boolean {
  return (assert as any).isFiniteArray(arr);
}

/**
 * Validate that a number is defined and finite (not NaN, not Infinity).
 * Common validation pattern for numeric calculations.
 */
export function validateNumber(value: number | undefined): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  return isFinite(value) && !isNaN(value);
}

/**
 * Validate a matrix (2D array) contains only finite values.
 * Useful for Kalman filter matrix validation.
 */
export function validateMatrix(matrix: number[][]): boolean {
  const flat = matrix.flat();
  return isFiniteArray(flat);
}

/**
 * Validate a 1D array contains only finite values.
 */
export function validateArray(arr: number[]): boolean {
  return isFiniteArray(arr);
}
