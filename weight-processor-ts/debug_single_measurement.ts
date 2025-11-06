#!/usr/bin/env bun
/**
 * Debug script to check quality scoring for a single measurement
 */

import { UnifiedQualityScorer } from './src/core/processing/unified_quality_scorer';

// Test measurement: 43.2 kg (line 7 from test_user.csv)
// This is accepted in TS but rejected in Python
const weight = 43.2;
const source = 'https://api.iglucose.com';

const scorer = new UnifiedQualityScorer();

// Call with minimal context (first measurement)
const qualityScore = scorer.calculateQualityScore({
  weight,
  source,
  kalmanState: null,
  kalmanPrediction: null,
  innovationCovariance: null,
  previousWeight: null,
  timeDiffHours: null,
  recentWeights: null,
  recentTimestamps: null,
  userHeightM: null,
});

console.log('=== Quality Score for 43.2 kg ===\n');
console.log('Overall Score:', qualityScore.overall.toFixed(4));
console.log('Threshold:', qualityScore.threshold);
console.log('Accepted:', qualityScore.accepted);
console.log('\nComponents:');
for (const [key, value] of Object.entries(qualityScore.components)) {
  console.log(`  ${key}: ${value.toFixed(4)}`);
}
console.log('\nMetadata:');
console.log(JSON.stringify(qualityScore.metadata, null, 2));

// Also test with some context (later measurement with previous weight)
console.log('\n\n=== Quality Score for 43.2 kg (with previous weight 115.4 kg) ===\n');

const scorer2 = new UnifiedQualityScorer();
const qualityScore2 = scorer2.calculateQualityScore({
  weight: 43.2,
  source,
  kalmanState: null,
  kalmanPrediction: 115.4, // Previous weight
  innovationCovariance: 10.0,
  previousWeight: 115.4,
  timeDiffHours: 150, // ~6 days
  recentWeights: [115.4],
  recentTimestamps: [new Date('2025-03-26T12:30:46.000-05:00')],
  userHeightM: null,
});

console.log('Overall Score:', qualityScore2.overall.toFixed(4));
console.log('Threshold:', qualityScore2.threshold);
console.log('Accepted:', qualityScore2.accepted);
console.log('\nComponents:');
for (const [key, value] of Object.entries(qualityScore2.components)) {
  console.log(`  ${key}: ${value.toFixed(4)}`);
}
console.log('\nAnomaly Detection Metadata:');
console.log(JSON.stringify(qualityScore2.metadata.anomaly_detection, null, 2));
