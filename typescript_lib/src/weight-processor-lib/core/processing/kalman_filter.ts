/**
 * Custom Kalman Filter implementation.
 *
 * Implements the standard Kalman filter algorithm as specified at:
 * https://en.wikipedia.org/wiki/Kalman_filter
 *
 * This is a minimal, dependency-free implementation that provides
 * the same interface used by our weight processing system.
 */

/**
 * Matrix operations helper functions
 */

function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const rowsA = a.length;
  const colsA = a[0].length;
  const colsB = b[0].length;

  const result: number[][] = Array(rowsA)
    .fill(0)
    .map(() => Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

function matrixTranspose(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const cols = matrix[0].length;

  const result: number[][] = Array(cols)
    .fill(0)
    .map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }

  return result;
}

function matrixAdd(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = a[0].length;

  const result: number[][] = Array(rows)
    .fill(0)
    .map(() => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }

  return result;
}

function matrixSubtract(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = a[0].length;

  const result: number[][] = Array(rows)
    .fill(0)
    .map(() => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }

  return result;
}

function matrixInverse2x2(matrix: number[][]): number[][] {
  const [[a, b], [c, d]] = matrix;
  const det = a * d - b * c;

  if (Math.abs(det) < 1e-10) {
    throw new Error('Matrix is singular and cannot be inverted');
  }

  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ];
}

function matrixInverse(matrix: number[][]): number[][] {
  const n = matrix.length;

  // Special case for 1x1 matrix
  if (n === 1) {
    return [[1 / matrix[0][0]]];
  }

  // Special case for 2x2 matrix (most common in Kalman filters)
  if (n === 2) {
    return matrixInverse2x2(matrix);
  }

  // General case using Gauss-Jordan elimination
  // Create augmented matrix [A | I]
  const augmented: number[][] = matrix.map((row, i) =>
    [...row, ...Array(n).fill(0).map((_, j) => (i === j ? 1 : 0))]
  );

  // Forward elimination
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
        maxRow = k;
      }
    }

    // Swap rows
    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

    // Check for singular matrix
    if (Math.abs(augmented[i][i]) < 1e-10) {
      throw new Error('Matrix is singular and cannot be inverted');
    }

    // Scale pivot row
    const pivot = augmented[i][i];
    for (let j = 0; j < 2 * n; j++) {
      augmented[i][j] /= pivot;
    }

    // Eliminate column
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = augmented[k][i];
        for (let j = 0; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }
  }

  // Extract inverse from augmented matrix
  return augmented.map((row) => row.slice(n));
}

function matrixVectorMultiply(matrix: number[][], vector: number[]): number[] {
  const rows = matrix.length;
  const result: number[] = Array(rows).fill(0);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < vector.length; j++) {
      result[i] += matrix[i][j] * vector[j];
    }
  }

  return result;
}

function vectorSubtract(a: number[], b: number[]): number[] {
  return a.map((val, i) => val - b[i]);
}

function vectorAdd(a: number[], b: number[]): number[] {
  return a.map((val, i) => val + b[i]);
}

function identity(n: number): number[][] {
  const result: number[][] = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    result[i][i] = 1;
  }

  return result;
}

/**
 * Standard Kalman Filter implementation.
 *
 * The Kalman filter is an optimal linear estimator that recursively estimates
 * the state of a linear dynamic system from noisy measurements.
 *
 * State-space model:
 *     x_k = F_k * x_{k-1} + w_k        (process model)
 *     z_k = H_k * x_k + v_k            (measurement model)
 *
 * where:
 *     x_k is the state vector at time k
 *     z_k is the measurement vector at time k
 *     F_k is the state transition matrix
 *     H_k is the observation matrix
 *     w_k ~ N(0, Q_k) is process noise
 *     v_k ~ N(0, R_k) is measurement noise
 */
export class KalmanFilter {
  private F: number[][];  // State transition matrix
  private H: number[][];  // Observation matrix
  private x: number[];    // Initial state mean
  private P: number[][];  // Initial state covariance
  private Q: number[][];  // Process noise covariance
  private R: number[][];  // Measurement noise covariance

  public readonly nStates: number;
  public readonly nObservations: number;

  /**
   * Create a new Kalman filter.
   *
   * @param transitionMatrices - State transition matrix F (n_states x n_states)
   * @param observationMatrices - Observation matrix H (n_observations x n_states)
   * @param initialStateMean - Initial state estimate x_0 (n_states,)
   * @param initialStateCovariance - Initial covariance P_0 (n_states x n_states)
   * @param transitionCovariance - Process noise covariance Q (n_states x n_states)
   * @param observationCovariance - Measurement noise covariance R (n_observations x n_observations)
   */
  constructor(
    transitionMatrices: number[][],
    observationMatrices: number[][],
    initialStateMean: number[],
    initialStateCovariance: number[][],
    transitionCovariance: number[][],
    observationCovariance: number[][]
  ) {
    this.F = transitionMatrices;
    this.H = observationMatrices;
    this.x = initialStateMean;
    this.P = initialStateCovariance;
    this.Q = transitionCovariance;
    this.R = observationCovariance;

    // Validate dimensions
    this.nStates = this.x.length;
    this.nObservations = this.H.length;

    if (this.F.length !== this.nStates || this.F[0].length !== this.nStates) {
      throw new Error('F must be n_states x n_states');
    }
    if (this.H.length !== this.nObservations || this.H[0].length !== this.nStates) {
      throw new Error('H must be n_obs x n_states');
    }
    if (this.P.length !== this.nStates || this.P[0].length !== this.nStates) {
      throw new Error('P must be n_states x n_states');
    }
    if (this.Q.length !== this.nStates || this.Q[0].length !== this.nStates) {
      throw new Error('Q must be n_states x n_states');
    }
    if (this.R.length !== this.nObservations || this.R[0].length !== this.nObservations) {
      throw new Error('R must be n_obs x n_obs');
    }
  }

  /**
   * Kalman filter prediction step.
   *
   * Predicts the next state and covariance using the process model:
   *     x̂_{k|k-1} = F * x_{k-1|k-1}
   *     P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
   *
   * @param stateMean - Current state estimate x_{k-1|k-1} (n_states,)
   * @param stateCovariance - Current covariance P_{k-1|k-1} (n_states x n_states)
   * @returns Tuple of [predicted_state_mean, predicted_state_covariance]
   */
  predict(
    stateMean: number[],
    stateCovariance: number[][]
  ): [number[], number[][]] {
    // x̂_{k|k-1} = F * x_{k-1|k-1}
    const predictedStateMean = matrixVectorMultiply(this.F, stateMean);

    // P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
    const FP = matrixMultiply(this.F, stateCovariance);
    const FPFt = matrixMultiply(FP, matrixTranspose(this.F));
    const predictedStateCovariance = matrixAdd(FPFt, this.Q);

    return [predictedStateMean, predictedStateCovariance];
  }

  /**
   * Kalman filter update (correction) step.
   *
   * Updates the predicted state with the measurement:
   *     ỹ_k = z_k - H * x̂_{k|k-1}                    (innovation)
   *     S_k = H * P_{k|k-1} * H^T + R                 (innovation covariance)
   *     K_k = P_{k|k-1} * H^T * S_k^{-1}              (Kalman gain)
   *     x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k              (updated state)
   *     P_{k|k} = (I - K_k * H) * P_{k|k-1}           (updated covariance)
   *
   * @param predictedStateMean - Predicted state x̂_{k|k-1} (n_states,)
   * @param predictedStateCovariance - Predicted covariance P_{k|k-1} (n_states x n_states)
   * @param observation - Measurement z_k (n_observations,)
   * @returns Tuple of [filtered_state_mean, filtered_state_covariance]
   */
  update(
    predictedStateMean: number[],
    predictedStateCovariance: number[][],
    observation: number[]
  ): [number[], number[][]] {
    // ỹ_k = z_k - H * x̂_{k|k-1}  (innovation/measurement residual)
    const Hx = matrixVectorMultiply(this.H, predictedStateMean);
    const innovation = vectorSubtract(observation, Hx);

    // S_k = H * P_{k|k-1} * H^T + R  (innovation covariance)
    const HP = matrixMultiply(this.H, predictedStateCovariance);
    const HPHt = matrixMultiply(HP, matrixTranspose(this.H));
    const innovationCovariance = matrixAdd(HPHt, this.R);

    // K_k = P_{k|k-1} * H^T * S_k^{-1}  (Kalman gain)
    const PHt = matrixMultiply(predictedStateCovariance, matrixTranspose(this.H));
    const Sinv = matrixInverse(innovationCovariance);
    const kalmanGain = matrixMultiply(PHt, Sinv);

    // x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k  (updated state estimate)
    const Ky = matrixVectorMultiply(kalmanGain, innovation);
    const filteredStateMean = vectorAdd(predictedStateMean, Ky);

    // P_{k|k} = (I - K_k * H) * P_{k|k-1}  (updated covariance)
    // Using Joseph form for numerical stability:
    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T
    const KH = matrixMultiply(kalmanGain, this.H);
    const I_KH = matrixSubtract(identity(this.nStates), KH);

    const I_KH_P = matrixMultiply(I_KH, predictedStateCovariance);
    const I_KH_P_I_KHt = matrixMultiply(I_KH_P, matrixTranspose(I_KH));

    const KR = matrixMultiply(kalmanGain, this.R);
    const KRKt = matrixMultiply(KR, matrixTranspose(kalmanGain));

    const filteredStateCovariance = matrixAdd(I_KH_P_I_KHt, KRKt);

    return [filteredStateMean, filteredStateCovariance];
  }

  /**
   * Perform one complete predict + update cycle.
   *
   * This is the interface expected by our existing code. It combines
   * the prediction and update steps into a single operation.
   *
   * @param filteredStateMean - Current posterior state x_{k-1|k-1} (n_states,)
   * @param filteredStateCovariance - Current posterior covariance P_{k-1|k-1} (n_states x n_states)
   * @param observation - New measurement z_k (n_observations,)
   * @returns Tuple of [new_filtered_state_mean, new_filtered_state_covariance]
   */
  filterUpdate(
    filteredStateMean: number[],
    filteredStateCovariance: number[][],
    observation: number[]
  ): [number[], number[][]] {
    // Predict step
    const [predictedMean, predictedCov] = this.predict(filteredStateMean, filteredStateCovariance);

    // Update step
    const [filteredMean, filteredCov] = this.update(predictedMean, predictedCov, observation);

    return [filteredMean, filteredCov];
  }

  /**
   * Apply Kalman filter to a sequence of observations.
   *
   * Processes multiple measurements sequentially, starting from the
   * initial state (x_0, P_0) provided in the constructor.
   *
   * @param observations - Array of measurements, shape (n_timesteps, n_observations)
   *                       or (n_observations,) for a single measurement
   * @returns Tuple of:
   *          - filtered_state_means: Array of state estimates, shape (n_timesteps, n_states)
   *          - filtered_state_covariances: Array of covariances, shape (n_timesteps, n_states, n_states)
   */
  filter(observations: number[][]): [number[][], number[][][]] {
    const nTimesteps = observations.length;

    // Initialize output arrays
    const filteredStateMeans: number[][] = [];
    const filteredStateCovariances: number[][][] = [];

    // Start with initial state
    let currentMean = [...this.x];
    let currentCov = this.P.map(row => [...row]);

    // Process each observation
    for (let t = 0; t < nTimesteps; t++) {
      // Predict
      const [predictedMean, predictedCov] = this.predict(currentMean, currentCov);

      // Update
      [currentMean, currentCov] = this.update(predictedMean, predictedCov, observations[t]);

      // Store results
      filteredStateMeans.push([...currentMean]);
      filteredStateCovariances.push(currentCov.map(row => [...row]));
    }

    return [filteredStateMeans, filteredStateCovariances];
  }
}
