/**
 * Detailed step-by-step Kalman filter calculation debug
 */

import { Matrix } from 'ml-matrix';
import { KalmanFilter } from '../typescript_lib/src/weight-processor-lib/core/processing/kalman_filter';

function printMatrix(name: string, m: Matrix) {
  console.log(`${name}:`);
  console.log(`  Shape: ${m.rows}x${m.columns}`);
  const data = m.to2DArray();
  data.forEach((row, i) => {
    console.log(`  [${i}]: [${row.map(v => v.toFixed(10)).join(', ')}]`);
  });
}

async function debugDetailedKalman() {
  console.log('='.repeat(80));
  console.log('DETAILED KALMAN FILTER DEBUG - TypeScript');
  console.log('='.repeat(80));

  // Load config to get exact parameters
  const configPath = `${import.meta.dir}/../typescript_lib/config.json`;
  const config = await Bun.file(configPath).json();

  const kalmanConfig = config.kalman;
  const initialVariance = kalmanConfig.initial_variance;
  const transitionCovWeight = kalmanConfig.transition_covariance_weight;
  const transitionCovTrend = kalmanConfig.transition_covariance_trend;
  const observationCov = kalmanConfig.observation_covariance;

  console.log('\nConfiguration:');
  console.log(`  initial_variance: ${initialVariance}`);
  console.log(`  transition_covariance_weight: ${transitionCovWeight}`);
  console.log(`  transition_covariance_trend: ${transitionCovTrend}`);
  console.log(`  observation_covariance: ${observationCov}`);

  // Setup for measurement 1
  const weight1 = 70.0;
  console.log('\n' + '='.repeat(80));
  console.log('MEASUREMENT 1: Initialize at weight = 70.0 kg');
  console.log('='.repeat(80));

  const initialStateMean = Matrix.columnVector([weight1, 0]);
  const initialStateCovariance = new Matrix([[initialVariance, 0], [0, 0.001]]);

  printMatrix('Initial state mean', initialStateMean);
  printMatrix('Initial state covariance', initialStateCovariance);

  // After measurement 1, we have:
  const posteriorState1 = initialStateMean.clone();
  const posteriorCov1 = initialStateCovariance.clone();

  console.log('\nAfter measurement 1 (initialization):');
  printMatrix('Posterior state', posteriorState1);
  printMatrix('Posterior covariance', posteriorCov1);
  console.log(`Variance P[0,0]: ${posteriorCov1.get(0, 0)}`);

  // Setup for measurement 2
  const weight2 = 70.1;
  const timeDeltaDays = 1.0;

  console.log('\n' + '='.repeat(80));
  console.log('MEASUREMENT 2: Update at weight = 70.1 kg (1 day later)');
  console.log('='.repeat(80));

  // Build Kalman filter
  const F = new Matrix([[1, timeDeltaDays], [0, 1]]);
  const H = new Matrix([[1, 0]]);
  const Q = new Matrix([[transitionCovWeight, 0], [0, transitionCovTrend]]);
  const R = new Matrix([[observationCov]]);

  printMatrix('Transition matrix F', F);
  printMatrix('Observation matrix H', H);
  printMatrix('Process noise Q', Q);
  printMatrix('Measurement noise R', R);

  const kalman = new KalmanFilter(
    F,
    H,
    posteriorState1,
    posteriorCov1,
    Q,
    R
  );

  console.log('\n--- PREDICTION STEP ---');
  // Make new Matrix objects from the arrays
  const stateMeanForPredict = new Matrix(posteriorState1.to2DArray());
  const stateCovForPredict = new Matrix(posteriorCov1.to2DArray());
  const [predictedState, predictedCov] = kalman.predict(stateMeanForPredict, stateCovForPredict);

  printMatrix('Predicted state', predictedState);
  printMatrix('Predicted covariance', predictedCov);
  console.log(`Predicted weight: ${predictedState.get(0, 0)}`);
  console.log(`Predicted variance P_pred[0,0]: ${predictedCov.get(0, 0)}`);

  // Calculate innovation covariance S manually
  const S = H.mmul(predictedCov).mmul(H.transpose()).add(R);
  printMatrix('Innovation covariance S', S);
  console.log(`Innovation covariance S[0,0]: ${S.get(0, 0)}`);

  console.log('\n--- UPDATE STEP ---');
  const observation = Matrix.columnVector([weight2]);
  printMatrix('Observation z', observation);

  const [filteredState, filteredCov] = kalman.update(predictedState, predictedCov, observation);

  printMatrix('Filtered (posterior) state', filteredState);
  printMatrix('Filtered (posterior) covariance', filteredCov);
  console.log(`Filtered weight: ${filteredState.get(0, 0)}`);
  console.log(`Filtered variance P_post[0,0]: ${filteredCov.get(0, 0)}`);

  // Calculate innovation
  const innovation = weight2 - predictedState.get(0, 0);
  console.log(`\nInnovation (y): ${innovation}`);
  console.log(`Normalized innovation: ${Math.abs(innovation) / Math.sqrt(S.get(0, 0))}`);

  // Calculate Kalman gain manually for verification
  const K = predictedCov.mmul(H.transpose()).mmul(Matrix.inverse(S));
  printMatrix('Kalman gain K', K);

  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Initial variance (after meas 1): ${posteriorCov1.get(0, 0)}`);
  console.log(`Predicted variance (before meas 2): ${predictedCov.get(0, 0)}`);
  console.log(`Final variance (after meas 2): ${filteredCov.get(0, 0)}`);
  console.log(`\nExpected Python value: 4.00252950373697`);
  console.log(`TypeScript value: ${filteredCov.get(0, 0)}`);
  console.log(`Difference: ${(4.00252950373697 - filteredCov.get(0, 0)).toFixed(10)}`);
}

debugDetailedKalman().catch(console.error);
