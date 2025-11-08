#!/usr/bin/env bun
/**
 * Compare sequential states from TS and Python to find divergence point
 */

import { readFileSync } from "node:fs";

interface StateSnapshot {
  measurementIndex: number;
  measurementId: string;
  timestamp: string;
  weight: number;
  accepted: boolean;
  qualityScore: number;
  kalmanState: {
    weight: number;
    velocity: number;
  } | null;
  kalmanCovariance: number[][] | null;
  processNoise: number[][] | null;
}

function compareMatrices(a: number[][] | null, b: number[][] | null, tolerance = 1e-10): boolean {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  if (a.length !== b.length) return false;

  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) {
      if (Math.abs(a[i][j] - b[i][j]) > tolerance) return false;
    }
  }
  return true;
}

function main() {
  // Load both files
  const tsStates: StateSnapshot[] = JSON.parse(
    readFileSync("ts_sequential_states.json", "utf-8")
  );
  const pyStates: StateSnapshot[] = JSON.parse(
    readFileSync("py_sequential_states.json", "utf-8")
  );

  console.log("=== Sequential State Comparison ===\n");
  console.log(`TypeScript snapshots: ${tsStates.length}`);
  console.log(`Python snapshots: ${pyStates.length}\n`);

  if (tsStates.length !== pyStates.length) {
    console.log("❌ Different number of snapshots!");
    return;
  }

  let firstDivergence: number | null = null;
  const tolerance = 1e-6; // More lenient for quality scores
  const stateTolerance = 1e-10; // Strict for Kalman state

  for (let i = 0; i < tsStates.length; i++) {
    const ts = tsStates[i];
    const py = pyStates[i];

    // Check if measurement IDs match
    if (ts.measurementId !== py.measurementId) {
      console.log(`❌ Measurement ID mismatch at index ${i}`);
      console.log(`  TS: ${ts.measurementId}`);
      console.log(`  Py: ${py.measurementId}`);
      break;
    }

    // Compare acceptance
    const acceptanceMatches = ts.accepted === py.accepted;

    // Compare quality scores
    const qualityDiff = Math.abs((ts.qualityScore || 0) - (py.qualityScore || 0));
    const qualityMatches = qualityDiff < tolerance;

    // Compare Kalman state
    let stateMatches = true;
    let stateDiff = { weight: 0, velocity: 0 };
    if (ts.kalmanState && py.kalmanState) {
      stateDiff.weight = Math.abs(ts.kalmanState.weight - py.kalmanState.weight);
      stateDiff.velocity = Math.abs(ts.kalmanState.velocity - py.kalmanState.velocity);
      stateMatches =
        stateDiff.weight < stateTolerance && stateDiff.velocity < stateTolerance;
    } else if (ts.kalmanState !== py.kalmanState) {
      stateMatches = false;
    }

    // Compare covariance matrix
    const covarianceMatches = compareMatrices(
      ts.kalmanCovariance,
      py.kalmanCovariance,
      stateTolerance
    );

    // Compare process noise
    const processNoiseMatches = compareMatrices(
      ts.processNoise,
      py.processNoise,
      stateTolerance
    );

    const allMatch =
      acceptanceMatches &&
      qualityMatches &&
      stateMatches &&
      covarianceMatches &&
      processNoiseMatches;

    if (!allMatch && firstDivergence === null) {
      firstDivergence = i;
    }

    // Print details for first divergence and every 10th measurement
    if (i === firstDivergence || i % 10 === 0 || i < 5) {
      const prefix = i === firstDivergence ? "🔴 FIRST DIVERGENCE" : "✅";
      console.log(`${prefix} [${i}] ${ts.measurementId.substring(0, 8)}... @ ${ts.timestamp}`);
      console.log(`  Weight: ${ts.weight.toFixed(1)} kg`);
      console.log(
        `  Accepted: TS=${ts.accepted} Py=${py.accepted} ${acceptanceMatches ? "✓" : "✗"}`
      );
      const tsQuality = ts.qualityScore !== null ? ts.qualityScore.toFixed(6) : "null";
      const pyQuality = py.qualityScore !== null ? py.qualityScore.toFixed(6) : "null";
      console.log(
        `  Quality: TS=${tsQuality} Py=${pyQuality} diff=${qualityDiff.toFixed(9)} ${qualityMatches ? "✓" : "✗"}`
      );

      if (ts.kalmanState && py.kalmanState) {
        console.log(
          `  State.weight: TS=${ts.kalmanState.weight.toFixed(6)} Py=${py.kalmanState.weight.toFixed(6)} diff=${stateDiff.weight.toExponential(3)} ${stateDiff.weight < stateTolerance ? "✓" : "✗"}`
        );
        console.log(
          `  State.velocity: TS=${ts.kalmanState.velocity.toFixed(6)} Py=${py.kalmanState.velocity.toFixed(6)} diff=${stateDiff.velocity.toExponential(3)} ${stateDiff.velocity < stateTolerance ? "✓" : "✗"}`
        );
      }

      console.log(
        `  Covariance: ${covarianceMatches ? "✓" : "✗ DIFFERS"}`
      );
      console.log(
        `  ProcessNoise: ${processNoiseMatches ? "✓" : "✗ DIFFERS"}`
      );
      console.log();
    }
  }

  console.log("\n=== Summary ===");
  if (firstDivergence === null) {
    console.log("✅ All states match perfectly!");
  } else {
    console.log(`🔴 First divergence at measurement index: ${firstDivergence}`);
    console.log(`   Measurement ID: ${tsStates[firstDivergence].measurementId}`);
    console.log(`   Timestamp: ${tsStates[firstDivergence].timestamp}`);
    console.log(`   Weight: ${tsStates[firstDivergence].weight} kg`);
    console.log(
      `\n   This is measurement ${firstDivergence + 1} of ${tsStates.length}`
    );
  }

  // Count total divergences
  let totalDivergences = 0;
  for (let i = 0; i < tsStates.length; i++) {
    const ts = tsStates[i];
    const py = pyStates[i];
    if (
      ts.accepted !== py.accepted ||
      Math.abs(ts.qualityScore - py.qualityScore) >= tolerance
    ) {
      totalDivergences++;
    }
  }

  console.log(`\nTotal measurements with divergent results: ${totalDivergences}/${tsStates.length}`);
}

main();
