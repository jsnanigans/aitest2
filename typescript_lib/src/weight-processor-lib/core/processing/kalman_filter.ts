/**
 * Kalman Filter implementation using ml-matrix library.
 *
 * Implements the standard Kalman filter algorithm as specified at:
 * https://en.wikipedia.org/wiki/Kalman_filter
 *
 * Uses ml-matrix for numerical stability and proven matrix operations.
 */

import { Matrix } from 'ml-matrix';
import { validateMatrix } from '../stdlib-utils';

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
  private F: Matrix;  // State transition matrix
  private H: Matrix;  // Observation matrix
  private x: Matrix;  // Initial state mean (column vector)
  private P: Matrix;  // Initial state covariance
  private Q: Matrix;  // Process noise covariance
  private R: Matrix;  // Measurement noise covariance

  public readonly nStates: number;
  public readonly nObservations: number;

  /**
   * Create a new Kalman filter.
   *
   * @param transitionMatrices - State transition matrix F (n_states x n_states)
   * @param observationMatrices - Observation matrix H (n_observations x n_states)
   * @param initialStateMean - Initial state estimate x_0 (n_states,) or (n_states x 1)
   * @param initialStateCovariance - Initial covariance P_0 (n_states x n_states)
   * @param transitionCovariance - Process noise covariance Q (n_states x n_states)
   * @param observationCovariance - Measurement noise covariance R (n_observations x n_observations)
   */
  constructor(
    transitionMatrices: number[][] | Matrix,
    observationMatrices: number[][] | Matrix,
    initialStateMean: number[] | number[][] | Matrix,
    initialStateCovariance: number[][] | Matrix,
    transitionCovariance: number[][] | Matrix,
    observationCovariance: number[][] | Matrix
  ) {
    // Convert to Matrix objects
    this.F = transitionMatrices instanceof Matrix ? transitionMatrices : new Matrix(transitionMatrices);
    this.H = observationMatrices instanceof Matrix ? observationMatrices : new Matrix(observationMatrices);
    this.Q = transitionCovariance instanceof Matrix ? transitionCovariance : new Matrix(transitionCovariance);
    this.R = observationCovariance instanceof Matrix ? observationCovariance : new Matrix(observationCovariance);
    this.P = initialStateCovariance instanceof Matrix ? initialStateCovariance : new Matrix(initialStateCovariance);

    // Handle state mean - convert 1D array to column vector
    if (initialStateMean instanceof Matrix) {
      this.x = initialStateMean;
    } else if (Array.isArray(initialStateMean) && !Array.isArray(initialStateMean[0])) {
      // Convert 1D array to column vector
      this.x = Matrix.columnVector(initialStateMean as number[]);
    } else {
      this.x = new Matrix(initialStateMean as number[][]);
    }

    // Validate dimensions
    this.nStates = this.x.rows;
    this.nObservations = this.H.rows;

    if (this.F.rows !== this.nStates || this.F.columns !== this.nStates) {
      throw new Error(`F must be n_states x n_states, got ${this.F.rows}x${this.F.columns}`);
    }
    if (this.H.rows !== this.nObservations || this.H.columns !== this.nStates) {
      throw new Error(`H must be n_obs x n_states, got ${this.H.rows}x${this.H.columns}`);
    }
    if (this.P.rows !== this.nStates || this.P.columns !== this.nStates) {
      throw new Error(`P must be n_states x n_states, got ${this.P.rows}x${this.P.columns}`);
    }
    if (this.Q.rows !== this.nStates || this.Q.columns !== this.nStates) {
      throw new Error(`Q must be n_states x n_states, got ${this.Q.rows}x${this.Q.columns}`);
    }
    if (this.R.rows !== this.nObservations || this.R.columns !== this.nObservations) {
      throw new Error(`R must be n_obs x n_obs, got ${this.R.rows}x${this.R.columns}`);
    }
  }

  /**
   * Kalman filter prediction step.
   *
   * Predicts the next state and covariance using the process model:
   *     x̂_{k|k-1} = F * x_{k-1|k-1}
   *     P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
   *
   * @param stateMean - Current state estimate x_{k-1|k-1}
   * @param stateCovariance - Current covariance P_{k-1|k-1}
   * @returns Tuple of [predicted_state_mean, predicted_state_covariance]
   */
  predict(
    stateMean: Matrix,
    stateCovariance: Matrix
  ): [Matrix, Matrix] {
    // Strict validation of inputs
    if (!(stateMean instanceof Matrix)) {
      throw new Error(
        `predict() stateMean must be a Matrix, got ${typeof stateMean}. ` +
        `Value: ${JSON.stringify(stateMean)}`
      );
    }
    if (!(stateCovariance instanceof Matrix)) {
      throw new Error(
        `predict() stateCovariance must be a Matrix, got ${typeof stateCovariance}. ` +
        `Value: ${JSON.stringify(stateCovariance)}`
      );
    }
    if (stateMean.rows !== this.nStates || stateMean.columns !== 1) {
      throw new Error(
        `predict() stateMean must be ${this.nStates}x1, got ${stateMean.rows}x${stateMean.columns}`
      );
    }
    if (stateCovariance.rows !== this.nStates || stateCovariance.columns !== this.nStates) {
      throw new Error(
        `predict() stateCovariance must be ${this.nStates}x${this.nStates}, got ${stateCovariance.rows}x${stateCovariance.columns}`
      );
    }

    // x̂_{k|k-1} = F * x_{k-1|k-1}
    const predictedStateMean = this.F.mmul(stateMean);

    // P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
    const predictedStateCovariance = this.F
      .mmul(stateCovariance)
      .mmul(this.F.transpose())
      .add(this.Q);

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
   * @param predictedStateMean - Predicted state x̂_{k|k-1}
   * @param predictedStateCovariance - Predicted covariance P_{k|k-1}
   * @param observation - Measurement z_k (column vector or 1D array)
   * @returns Tuple of [filtered_state_mean, filtered_state_covariance]
   */
  update(
    predictedStateMean: Matrix,
    predictedStateCovariance: Matrix,
    observation: number[] | Matrix
  ): [Matrix, Matrix] {
    // Strict validation of inputs
    if (!(predictedStateMean instanceof Matrix)) {
      throw new Error(
        `update() predictedStateMean must be a Matrix, got ${typeof predictedStateMean}. ` +
        `Value: ${JSON.stringify(predictedStateMean)}`
      );
    }
    if (!(predictedStateCovariance instanceof Matrix)) {
      throw new Error(
        `update() predictedStateCovariance must be a Matrix, got ${typeof predictedStateCovariance}. ` +
        `Value: ${JSON.stringify(predictedStateCovariance)}`
      );
    }
    if (!observation || (Array.isArray(observation) && observation.length === 0)) {
      throw new Error(
        `update() observation cannot be empty or undefined. Got: ${JSON.stringify(observation)}`
      );
    }

    // Convert observation to column vector if needed
    const z = observation instanceof Matrix
      ? observation
      : Matrix.columnVector(observation);

    // ỹ_k = z_k - H * x̂_{k|k-1}  (innovation/measurement residual)
    const innovation = z.sub(this.H.mmul(predictedStateMean));

    // S_k = H * P_{k|k-1} * H^T + R  (innovation covariance)
    const innovationCovariance = this.H
      .mmul(predictedStateCovariance)
      .mmul(this.H.transpose())
      .add(this.R);

    // K_k = P_{k|k-1} * H^T * S_k^{-1}  (Kalman gain)
    // IMPORTANT: Check if innovation covariance is invertible
    let innovationCovInverse: Matrix;
    try {
      innovationCovInverse = innovationCovariance.inverse();

      // Check for NaN or Infinity in the inverse using stdlib validation
      const invData = innovationCovInverse.to2DArray();
      const hasInvalidValues = !validateMatrix(invData);

      if (hasInvalidValues) {
        // Singular matrix - use pseudoinverse or return prediction unchanged
        if (process.env.VERBOSE_LOGGING) {
          console.log('[KalmanUpdate] Innovation covariance inverse has invalid values, using prediction as-is');
        }
        return [predictedStateMean, predictedStateCovariance];
      }
    } catch (e) {
      // Matrix is singular - return prediction unchanged
      if (process.env.VERBOSE_LOGGING) {
        console.log('[KalmanUpdate] Innovation covariance is singular, using prediction as-is');
      }
      return [predictedStateMean, predictedStateCovariance];
    }

    const kalmanGain = predictedStateCovariance
      .mmul(this.H.transpose())
      .mmul(innovationCovInverse);

    // x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k  (updated state estimate)
    const filteredStateMean = predictedStateMean.add(kalmanGain.mmul(innovation));

    // P_{k|k} = (I - K_k * H) * P_{k|k-1}  (updated covariance)
    // Using Joseph form for numerical stability:
    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T
    const I_KH = Matrix.eye(this.nStates).sub(kalmanGain.mmul(this.H));

    const filteredStateCovariance = I_KH
      .mmul(predictedStateCovariance)
      .mmul(I_KH.transpose())
      .add(kalmanGain.mmul(this.R).mmul(kalmanGain.transpose()));

    // Final NaN check on output using stdlib validation
    const filteredData = filteredStateMean.to2DArray();
    const covData = filteredStateCovariance.to2DArray();
    const hasNaN = !validateMatrix(filteredData) || !validateMatrix(covData);

    if (hasNaN) {
      if (process.env.VERBOSE_LOGGING) {
        console.log('[KalmanUpdate] NaN detected in output, returning prediction as-is');
      }
      return [predictedStateMean, predictedStateCovariance];
    }

    return [filteredStateMean, filteredStateCovariance];
  }

  /**
   * Perform one complete predict + update cycle.
   *
   * This is the interface expected by our existing code. It combines
   * the prediction and update steps into a single operation.
   *
   * @param filteredStateMean - Current posterior state x_{k-1|k-1}
   * @param filteredStateCovariance - Current posterior covariance P_{k-1|k-1}
   * @param observation - New measurement z_k
   * @returns Tuple of [new_filtered_state_mean, new_filtered_state_covariance]
   */
  filterUpdate(
    filteredStateMean: Matrix,
    filteredStateCovariance: Matrix,
    observation: number[] | Matrix
  ): [Matrix, Matrix] {
    // Strict validation of inputs
    if (!(filteredStateMean instanceof Matrix)) {
      throw new Error(
        `filterUpdate() filteredStateMean must be a Matrix, got ${typeof filteredStateMean}. ` +
        `Value: ${JSON.stringify(filteredStateMean)}`
      );
    }
    if (!(filteredStateCovariance instanceof Matrix)) {
      throw new Error(
        `filterUpdate() filteredStateCovariance must be a Matrix, got ${typeof filteredStateCovariance}. ` +
        `Value: ${JSON.stringify(filteredStateCovariance)}`
      );
    }
    if (!observation || (Array.isArray(observation) && observation.length === 0)) {
      throw new Error(
        `filterUpdate() observation cannot be empty or undefined. Got: ${JSON.stringify(observation)}`
      );
    }

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
   *                       or array of 1D arrays
   * @returns Tuple of:
   *          - filtered_state_means: Array of Matrix (column vectors)
   *          - filtered_state_covariances: Array of Matrix
   */
  filter(observations: number[][]): [Matrix[], Matrix[]] {
    const nTimesteps = observations.length;

    // Initialize output arrays
    const filteredStateMeans: Matrix[] = [];
    const filteredStateCovariances: Matrix[] = [];

    // Start with initial state
    let currentMean = this.x.clone();
    let currentCov = this.P.clone();

    // Process each observation
    for (let t = 0; t < nTimesteps; t++) {
      // Predict
      const [predictedMean, predictedCov] = this.predict(currentMean, currentCov);

      // Update
      [currentMean, currentCov] = this.update(predictedMean, predictedCov, observations[t]);

      // Store results
      filteredStateMeans.push(currentMean.clone());
      filteredStateCovariances.push(currentCov.clone());
    }

    return [filteredStateMeans, filteredStateCovariances];
  }
}
