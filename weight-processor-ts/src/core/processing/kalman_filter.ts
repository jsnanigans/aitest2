/**
 * Kalman Filter implementation
 *
 * Standard Kalman filter algorithm for weight estimation.
 * Specialized for 2D state space (weight, velocity).
 *
 * Based on: https://en.wikipedia.org/wiki/Kalman_filter
 */

import {
  type Matrix2x2,
  type Vector2,
  multiply2x2,
  multiplyVector2x2,
  transpose2x2,
  add2x2,
  subtract2x2,
  eye2,
  invert2x2,
  copy2x2,
  copyVector2,
} from '../math/matrix';

/**
 * Kalman Filter for 2D state space
 *
 * State-space model:
 *   x_k = F_k * x_{k-1} + w_k        (process model)
 *   z_k = H_k * x_k + v_k            (measurement model)
 *
 * where:
 *   x_k is the state vector at time k [weight, velocity]
 *   z_k is the measurement vector at time k [weight]
 *   F_k is the state transition matrix (2x2)
 *   H_k is the observation matrix (1x2, but we use 2x2 with second row zeros)
 *   w_k ~ N(0, Q_k) is process noise
 *   v_k ~ N(0, R_k) is measurement noise
 */
export class KalmanFilter {
  // Filter matrices
  public F: Matrix2x2; // State transition matrix
  public H: Matrix2x2; // Observation matrix (actually 1x2, but padded to 2x2)
  public Q: Matrix2x2; // Process noise covariance
  public R: number; // Measurement noise variance (scalar for 1D measurement)

  // State
  public x: Vector2; // State estimate [weight, velocity]
  public P: Matrix2x2; // State covariance

  /**
   * Create a new Kalman Filter
   *
   * @param transitionMatrix State transition matrix F (2x2)
   * @param observationMatrix Observation matrix H (2x2)
   * @param initialStateMean Initial state estimate x_0 (2x1)
   * @param initialStateCovariance Initial covariance P_0 (2x2)
   * @param transitionCovariance Process noise covariance Q (2x2)
   * @param observationCovariance Measurement noise variance R (scalar)
   */
  constructor(
    transitionMatrix: Matrix2x2,
    observationMatrix: Matrix2x2,
    initialStateMean: Vector2,
    initialStateCovariance: Matrix2x2,
    transitionCovariance: Matrix2x2,
    observationCovariance: number
  ) {
    this.F = copy2x2(transitionMatrix);
    this.H = copy2x2(observationMatrix);
    this.x = copyVector2(initialStateMean);
    this.P = copy2x2(initialStateCovariance);
    this.Q = copy2x2(transitionCovariance);
    this.R = observationCovariance;

    // Validate matrices
    this.validate();
  }

  /**
   * Validate matrix dimensions and properties
   */
  private validate(): void {
    // All matrices should be 2x2
    if (this.F.length !== 2 || this.F[0].length !== 2) {
      throw new Error('Transition matrix F must be 2x2');
    }
    if (this.H.length !== 2 || this.H[0].length !== 2) {
      throw new Error('Observation matrix H must be 2x2');
    }
    if (this.P.length !== 2 || this.P[0].length !== 2) {
      throw new Error('Covariance matrix P must be 2x2');
    }
    if (this.Q.length !== 2 || this.Q[0].length !== 2) {
      throw new Error('Process noise Q must be 2x2');
    }

    // State vector should be 2x1
    if (this.x.length !== 2) {
      throw new Error('State vector x must be 2x1');
    }

    // R should be positive
    if (this.R <= 0) {
      throw new Error('Measurement noise R must be positive');
    }
  }

  /**
   * Prediction step
   *
   * Predicts the next state and covariance:
   *   x̂_{k|k-1} = F * x_{k-1|k-1}
   *   P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
   *
   * @param stateMean Current state estimate
   * @param stateCovariance Current covariance
   * @returns Tuple of [predicted_state, predicted_covariance]
   */
  public predict(stateMean: Vector2, stateCovariance: Matrix2x2): [Vector2, Matrix2x2] {
    // x̂_{k|k-1} = F * x_{k-1|k-1}
    const predictedStateMean = multiplyVector2x2(this.F, stateMean);

    // P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
    const FT = transpose2x2(this.F);
    const temp = multiply2x2(this.F, stateCovariance);
    const FPFt = multiply2x2(temp, FT);
    const predictedStateCovariance = add2x2(FPFt, this.Q);

    return [predictedStateMean, predictedStateCovariance];
  }

  /**
   * Update (correction) step
   *
   * Updates the predicted state with the measurement:
   *   ỹ_k = z_k - H * x̂_{k|k-1}                    (innovation)
   *   S_k = H * P_{k|k-1} * H^T + R                 (innovation covariance)
   *   K_k = P_{k|k-1} * H^T * S_k^{-1}              (Kalman gain)
   *   x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k              (updated state)
   *   P_{k|k} = (I - K_k * H) * P_{k|k-1} * (I - K_k * H)^T + K_k * R * K_k^T  (Joseph form)
   *
   * @param predictedStateMean Predicted state
   * @param predictedStateCovariance Predicted covariance
   * @param observation Measurement [weight, 0] (second component unused)
   * @returns Tuple of [filtered_state, filtered_covariance]
   */
  public update(
    predictedStateMean: Vector2,
    predictedStateCovariance: Matrix2x2,
    observation: number
  ): [Vector2, Matrix2x2] {
    // Observation vector (we only measure weight, so second component is 0)
    const z: Vector2 = [observation, 0];

    // ỹ_k = z_k - H * x̂_{k|k-1}  (innovation)
    const Hx = multiplyVector2x2(this.H, predictedStateMean);
    const innovation: Vector2 = [z[0] - Hx[0], z[1] - Hx[1]];

    // S_k = H * P_{k|k-1} * H^T + R  (innovation covariance)
    // Since we only have one measurement, S is effectively a scalar (top-left element)
    const HT = transpose2x2(this.H);
    const HP = multiply2x2(this.H, predictedStateCovariance);
    const HPHT = multiply2x2(HP, HT);
    const innovationCovariance = HPHT[0][0]! + this.R;

    // K_k = P_{k|k-1} * H^T * S_k^{-1}  (Kalman gain)
    // This is a 2x1 vector (stored as 2x2 with second column zeros)
    const PHT = multiply2x2(predictedStateCovariance, HT);
    const innovationCovarianceInv = 1.0 / innovationCovariance;
    const K: Matrix2x2 = [
      [PHT[0][0]! * innovationCovarianceInv, 0],
      [PHT[1][0]! * innovationCovarianceInv, 0],
    ];

    // x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k  (updated state)
    const Ky = multiplyVector2x2(K, innovation);
    const filteredStateMean: Vector2 = [
      predictedStateMean[0] + Ky[0],
      predictedStateMean[1] + Ky[1],
    ];

    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T  (Joseph form for stability)
    const I = eye2();
    const KH = multiply2x2(K, this.H);
    const I_KH = subtract2x2(I, KH);
    const I_KH_T = transpose2x2(I_KH);

    // (I - K*H) * P * (I - K*H)^T
    const temp1 = multiply2x2(I_KH, predictedStateCovariance);
    const term1 = multiply2x2(temp1, I_KH_T);

    // K * R * K^T
    const KT = transpose2x2(K);
    const term2: Matrix2x2 = [
      [K[0][0]! * this.R * KT[0][0]!, K[0][0]! * this.R * KT[0][1]!],
      [K[1][0]! * this.R * KT[0][0]!, K[1][0]! * this.R * KT[0][1]!],
    ];

    const filteredStateCovariance = add2x2(term1, term2);

    return [filteredStateMean, filteredStateCovariance];
  }

  /**
   * Combined predict + update step
   *
   * Convenience method that combines prediction and update.
   *
   * @param filteredStateMean Current posterior state
   * @param filteredStateCovariance Current posterior covariance
   * @param observation New measurement
   * @returns Tuple of [new_state, new_covariance]
   */
  public filterUpdate(
    filteredStateMean: Vector2,
    filteredStateCovariance: Matrix2x2,
    observation: number
  ): [Vector2, Matrix2x2] {
    // Predict
    const [predictedMean, predictedCov] = this.predict(filteredStateMean, filteredStateCovariance);

    // Update
    const [newMean, newCov] = this.update(predictedMean, predictedCov, observation);

    return [newMean, newCov];
  }

  /**
   * Process a sequence of observations
   *
   * @param observations Array of weight measurements
   * @returns Arrays of filtered states and covariances
   */
  public filter(observations: number[]): [Vector2[], Matrix2x2[]] {
    const nTimesteps = observations.length;
    const filteredStateMeans: Vector2[] = [];
    const filteredStateCovariances: Matrix2x2[] = [];

    // Start with initial state
    let currentMean = copyVector2(this.x);
    let currentCov = copy2x2(this.P);

    // Process each observation
    for (let t = 0; t < nTimesteps; t++) {
      // Predict
      const [predictedMean, predictedCov] = this.predict(currentMean, currentCov);

      // Update
      const [newMean, newCov] = this.update(predictedMean, predictedCov, observations[t]!);

      // Store results
      filteredStateMeans.push(newMean);
      filteredStateCovariances.push(newCov);

      // Update current state
      currentMean = newMean;
      currentCov = newCov;
    }

    return [filteredStateMeans, filteredStateCovariances];
  }
}
