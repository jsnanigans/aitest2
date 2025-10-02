/**
 * Matrix operations for 2x2 matrices
 *
 * Specialized matrix operations for the Kalman filter implementation.
 * Since our state space is 2D (weight, velocity), we use optimized 2x2
 * matrix operations instead of a general-purpose matrix library.
 */

/**
 * 2x2 matrix type
 */
export type Matrix2x2 = [[number, number], [number, number]];

/**
 * 2x1 vector type
 */
export type Vector2 = [number, number];

/**
 * Create a 2x2 identity matrix
 *
 * @returns Identity matrix [[1, 0], [0, 1]]
 */
export function eye2(): Matrix2x2 {
  return [
    [1, 0],
    [0, 1],
  ];
}

/**
 * Multiply two 2x2 matrices: C = A * B
 *
 * @param a First matrix
 * @param b Second matrix
 * @returns Product matrix C
 */
export function multiply2x2(a: Matrix2x2, b: Matrix2x2): Matrix2x2 {
  return [
    [
      a[0][0] * b[0][0] + a[0][1] * b[1][0],
      a[0][0] * b[0][1] + a[0][1] * b[1][1],
    ],
    [
      a[1][0] * b[0][0] + a[1][1] * b[1][0],
      a[1][0] * b[0][1] + a[1][1] * b[1][1],
    ],
  ];
}

/**
 * Multiply 2x2 matrix by 2x1 vector: y = A * x
 *
 * @param a Matrix (2x2)
 * @param x Vector (2x1)
 * @returns Product vector y (2x1)
 */
export function multiplyVector2x2(a: Matrix2x2, x: Vector2): Vector2 {
  return [
    a[0][0] * x[0] + a[0][1] * x[1],
    a[1][0] * x[0] + a[1][1] * x[1],
  ];
}

/**
 * Transpose a 2x2 matrix: B = A^T
 *
 * @param a Input matrix
 * @returns Transposed matrix
 */
export function transpose2x2(a: Matrix2x2): Matrix2x2 {
  return [
    [a[0][0], a[1][0]],
    [a[0][1], a[1][1]],
  ];
}

/**
 * Add two 2x2 matrices: C = A + B
 *
 * @param a First matrix
 * @param b Second matrix
 * @returns Sum matrix C
 */
export function add2x2(a: Matrix2x2, b: Matrix2x2): Matrix2x2 {
  return [
    [a[0][0] + b[0][0], a[0][1] + b[0][1]],
    [a[1][0] + b[1][0], a[1][1] + b[1][1]],
  ];
}

/**
 * Subtract two 2x2 matrices: C = A - B
 *
 * @param a First matrix
 * @param b Second matrix
 * @returns Difference matrix C
 */
export function subtract2x2(a: Matrix2x2, b: Matrix2x2): Matrix2x2 {
  return [
    [a[0][0] - b[0][0], a[0][1] - b[0][1]],
    [a[1][0] - b[1][0], a[1][1] - b[1][1]],
  ];
}

/**
 * Multiply a 2x2 matrix by a scalar: B = s * A
 *
 * @param scalar Scalar value
 * @param a Input matrix
 * @returns Scaled matrix
 */
export function scalarMultiply2x2(scalar: number, a: Matrix2x2): Matrix2x2 {
  return [
    [scalar * a[0][0], scalar * a[0][1]],
    [scalar * a[1][0], scalar * a[1][1]],
  ];
}

/**
 * Invert a 2x2 matrix using analytical formula: B = A^{-1}
 *
 * For a 2x2 matrix:
 * A = [[a, b], [c, d]]
 * A^{-1} = (1/det) * [[d, -b], [-c, a]]
 *
 * where det = a*d - b*c
 *
 * @param a Input matrix
 * @returns Inverted matrix
 * @throws Error if matrix is singular (determinant = 0)
 */
export function invert2x2(a: Matrix2x2): Matrix2x2 {
  const det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

  if (Math.abs(det) < 1e-10) {
    throw new Error(
      `Matrix is singular or nearly singular (det = ${det}). Cannot invert.`
    );
  }

  const invDet = 1.0 / det;

  return [
    [invDet * a[1][1], invDet * -a[0][1]],
    [invDet * -a[1][0], invDet * a[0][0]],
  ];
}

/**
 * Calculate determinant of a 2x2 matrix
 *
 * @param a Input matrix
 * @returns Determinant value
 */
export function determinant2x2(a: Matrix2x2): number {
  return a[0][0] * a[1][1] - a[0][1] * a[1][0];
}

/**
 * Check if a 2x2 matrix is symmetric (A = A^T)
 *
 * @param a Input matrix
 * @param tolerance Numerical tolerance for comparison (default: 1e-10)
 * @returns True if matrix is symmetric
 */
export function isSymmetric2x2(a: Matrix2x2, tolerance = 1e-10): boolean {
  return Math.abs(a[0][1] - a[1][0]) < tolerance;
}

/**
 * Check if a 2x2 matrix is positive definite
 *
 * A symmetric 2x2 matrix is positive definite if:
 * 1. a[0][0] > 0 (first leading principal minor > 0)
 * 2. det(A) > 0 (second leading principal minor > 0)
 *
 * @param a Input matrix (must be symmetric)
 * @returns True if matrix is positive definite
 */
export function isPositiveDefinite2x2(a: Matrix2x2): boolean {
  if (!isSymmetric2x2(a)) {
    return false;
  }

  // Check first leading principal minor
  if (a[0][0] <= 0) {
    return false;
  }

  // Check determinant (second leading principal minor)
  const det = determinant2x2(a);
  return det > 0;
}

/**
 * Deep copy a 2x2 matrix
 *
 * @param a Input matrix
 * @returns Deep copy of the matrix
 */
export function copy2x2(a: Matrix2x2): Matrix2x2 {
  return [
    [a[0][0], a[0][1]],
    [a[1][0], a[1][1]],
  ];
}

/**
 * Deep copy a 2D vector
 *
 * @param v Input vector
 * @returns Deep copy of the vector
 */
export function copyVector2(v: Vector2): Vector2 {
  return [v[0], v[1]];
}

/**
 * Add two 2D vectors: c = a + b
 *
 * @param a First vector
 * @param b Second vector
 * @returns Sum vector
 */
export function addVector2(a: Vector2, b: Vector2): Vector2 {
  return [a[0] + b[0], a[1] + b[1]];
}

/**
 * Subtract two 2D vectors: c = a - b
 *
 * @param a First vector
 * @param b Second vector
 * @returns Difference vector
 */
export function subtractVector2(a: Vector2, b: Vector2): Vector2 {
  return [a[0] - b[0], a[1] - b[1]];
}

/**
 * Multiply a 2D vector by a scalar: b = s * a
 *
 * @param scalar Scalar value
 * @param a Input vector
 * @returns Scaled vector
 */
export function scalarMultiplyVector2(scalar: number, a: Vector2): Vector2 {
  return [scalar * a[0], scalar * a[1]];
}
