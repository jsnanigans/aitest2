#!/usr/bin/env bun
/**
 * Direct quality scorer test - bypasses processor to test scoring logic directly
 */

import { UnifiedQualityScorer } from "./typescript_lib/src/weight-processor-lib/core/processing/unified_quality_scorer";

function main() {
  console.log("=== Direct Quality Scorer Test ===\n");

  // Create scorer with config
  const config = {
    quality_scoring: {
      threshold: 0.5,
      components: {
        kalman_fit: { weight: 0.3, enabled: true },
        temporal_consistency: { weight: 0.25, enabled: true },
        anomaly_detection: { weight: 0.25, enabled: true },
        source_reliability: { weight: 0.1, enabled: true },
        trend_alignment: { weight: 0.1, enabled: true },
      },
    },
  };

  const scorer = new UnifiedQualityScorer(config);

  // Test case 1: Measurement #1 (first measurement - should score high)
  console.log("Test 1: First measurement (measurement #0)");
  console.log("  Weight: 104.33 kg, Timestamp: 2025-01-14");

  const result1 = scorer.calculateQualityScore(
    104.32616, // weight
    "kg", // unit
    new Date("2025-01-14T00:00:00.000Z"), // timestamp
    "https://api.iglucose.com", // source
    {
      // Kalman fit params (after first measurement)
      innovation: 0,
      innovation_variance: 3.49,
      state_variance: 0.364,
    },
    // Historical weights for anomaly detection
    [104.32616],
    // Last measurement timestamp
    null,
    // Kalman velocity
    0
  );

  console.log(`  Quality Score: ${result1.overall_score.toFixed(15)}`);
  console.log(`  Components:`);
  for (const [name, score] of Object.entries(result1.component_scores)) {
    const weight = config.quality_scoring.components[name as keyof typeof config.quality_scoring.components]?.weight || 0;
    const contribution = (score as number) * weight;
    console.log(`    ${name}: ${(score as number).toFixed(6)} × ${weight} = ${contribution.toFixed(6)}`);
  }

  // Test case 2: Measurement #1 (divergent measurement!)
  console.log("\n\nTest 2: Divergent measurement (measurement #1)");
  console.log("  Weight: 104.33 kg, Timestamp: 2025-02-03 (480h = 20 days after #0)");

  const result2 = scorer.calculateQualityScore(
    104.33, // weight
    "kg", // unit
    new Date("2025-02-03T00:00:00.000Z"), // timestamp
    "https://api.iglucose.com", // source
    {
      // After processing measurement #0, Kalman state would be:
      // state = [104.32616, 0]
      // So for measurement #1 at 104.33:
      // prediction = 104.32616 (no velocity change)
      // innovation = 104.33 - 104.32616 = 0.00384
      innovation: 0.00384,
      innovation_variance: 3.49, // observation_covariance
      state_variance: 0.364, // initial_variance
    },
    // Historical weights
    [104.32616, 104.33],
    // Last measurement timestamp
    new Date("2025-01-14T00:00:00.000Z"),
    // Kalman velocity (should be ~0 after first measurement)
    0
  );

  console.log(`  Quality Score: ${result2.overall_score.toFixed(15)}`);
  console.log(`  Expected (TS): 0.665926676041184`);
  console.log(`  Expected (Py): 0.977765...`);
  console.log(`  Components:`);
  for (const [name, score] of Object.entries(result2.component_scores)) {
    const weight = config.quality_scoring.components[name as keyof typeof config.quality_scoring.components]?.weight || 0;
    const contribution = (score as number) * weight;
    console.log(`    ${name}: ${(score as number).toFixed(15)} × ${weight} = ${contribution.toFixed(15)}`);
  }

  // Write results
  const breakdown = {
    measurement: {
      index: 1,
      weight: 104.33,
      timestamp: "2025-02-03T00:00:00.000Z",
    },
    qualityScore: result2.overall_score,
    componentScores: result2.component_scores,
    componentContributions: Object.fromEntries(
      Object.entries(result2.component_scores).map(([name, score]) => {
        const weight = config.quality_scoring.components[name as keyof typeof config.quality_scoring.components]?.weight || 0;
        return [name, (score as number) * weight];
      })
    ),
  };

  Bun.write("ts_quality_scorer_direct.json", JSON.stringify(breakdown, null, 2));
  console.log("\n✅ Results written to ts_quality_scorer_direct.json");
}

main();
