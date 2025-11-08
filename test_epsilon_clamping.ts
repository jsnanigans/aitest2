#!/usr/bin/env bun
/**
 * Test effect of epsilon clamping on geometric mean
 */

console.log("\n=== Testing Epsilon Clamping Effect ===\n");

const components = {
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

const epsilon = 1e-10;

// WITHOUT clamping
let product1 = 1.0;
for (const [name, score] of Object.entries(components)) {
  const weight = weights[name];
  product1 *= score ** weight;
}
const gm1 = product1 ** (1.0 / 1.0);

console.log("WITHOUT clamping:");
console.log("  Geometric Mean: ", gm1.toFixed(15));

// WITH clamping (matching scorer logic)
let product2 = 1.0;
for (const [name, score] of Object.entries(components)) {
  const weight = weights[name];
  const clampedScore = Math.max(epsilon, Math.min(1.0, score));
  product2 *= clampedScore ** weight;
}
const gm2 = product2 ** (1.0 / 1.0);

console.log("\nWITH clamping:");
console.log("  Geometric Mean: ", gm2.toFixed(15));

// WITH clamping AND final clamping
const gm3 = Math.max(0.0, Math.min(1.0, gm2));

console.log("\nWITH clamping + final clamp:");
console.log("  Geometric Mean: ", gm3.toFixed(15));

console.log("\nExpected:         0.628161937378212");
console.log("Difference:       ", Math.abs(gm2 - 0.628161937378212).toFixed(15));

// Check each component score
console.log("\n\nComponent score analysis:");
for (const [name, score] of Object.entries(components)) {
  const clamped = Math.max(epsilon, Math.min(1.0, score));
  const changed = score !== clamped;
  console.log(`  ${name.padEnd(25)} ${score.toFixed(6)} → ${clamped.toFixed(6)} ${changed ? "CHANGED" : ""}`);
}
