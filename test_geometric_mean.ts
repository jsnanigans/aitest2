#!/usr/bin/env bun
/**
 * Test geometric mean calculation with component exclusions
 */

function calcGeomMean(comps: Record<string, number>, weights: Record<string, number>): number {
  const epsilon = 1e-10;
  let product = 1.0;
  let weightSum = 0.0;

  for (const [name, score] of Object.entries(comps)) {
    const weight = weights[name] ?? 0.0;
    if (weight > 0) {
      const clampedScore = Math.max(epsilon, Math.min(1.0, score));
      product *= clampedScore ** weight;
      weightSum += weight;
    }
  }

  return weightSum > 0 ? product ** (1.0 / weightSum) : 0.0;
}

// Component scores for measurement 726b441f
const full = {
  kalman_fit: 0.867847,
  temporal_consistency: 0.361080,
  anomaly_detection: 0.714157,
  source_reliability: 0.800000,
  trend_alignment: 0.616598,
};

const weights = {
  kalman_fit: 0.30,
  temporal_consistency: 0.25,
  anomaly_detection: 0.25,
  source_reliability: 0.10,
  trend_alignment: 0.10,
};

console.log('\n=== Comparing Calculation Methods ===\n');

// Weighted arithmetic sum
let arithmeticSum = 0;
for (const [name, score] of Object.entries(full)) {
  const weight = weights[name];
  arithmeticSum += score * weight;
  console.log(`  ${name.padEnd(25)}: ${score.toFixed(6)} × ${weight.toFixed(2)} = ${(score * weight).toFixed(6)}`);
}

console.log(`\nWeighted Arithmetic Sum:  ${arithmeticSum.toFixed(15)}`);
console.log(`Weighted Geometric Mean:  ${calcGeomMean(full, weights).toFixed(15)}`);
console.log(`TypeScript reported:      0.628161937378212\n`);

// Check if it's using arithmetic with some adjustment
const diff = arithmeticSum - 0.628161937378212;
console.log(`Arithmetic - TS reported: ${diff.toFixed(15)}`);

// Test if it's a decay-adjusted geometric mean
const geomMean = calcGeomMean(full, weights);
console.log(`\nTesting decay adjustments on geometric mean:`);
for (let days = 0; days <= 5; days += 0.5) {
  const decayFactor = Math.min(1.0, days / 30.0);
  const adjusted = geomMean + (1.0 - geomMean) * decayFactor;
  const match = Math.abs(adjusted - 0.628161937378212) < 0.0001 ? ' *** MATCH ***' : '';
  console.log(`  ${days.toFixed(1)} days: ${adjusted.toFixed(15)}${match}`);
}
