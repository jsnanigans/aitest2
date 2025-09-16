# Weight Processing System - Quality Components Investigation Report

## Executive Summary

The quality scoring system evaluates weight measurements using four distinct components that assess different aspects of data reliability and validity. Each component serves a specific purpose in filtering out erroneous measurements while accepting legitimate weight changes.

## Quality Components Overview

### Component Weights
The system uses a weighted combination of four components:
- **Safety**: 35% weight - Most critical component
- **Plausibility**: 25% weight
- **Consistency**: 25% weight
- **Reliability**: 15% weight - Least weight but still important

The overall score is calculated using either:
- **Harmonic mean** (default): Penalizes low scores more severely
- **Arithmetic mean**: Standard weighted average

## Component Details

### 1. Safety Component (35% weight)
**Purpose**: Ensures measurements fall within physiologically possible limits for human weight.

**Calculation Method** (`_calculate_safety`, lines 171-201):
- **Hard Limits Check**:
  - Absolute minimum: 30 kg
  - Absolute maximum: 400 kg
  - Returns 0.0 if outside these bounds

- **Safe Range**:
  - Suspicious minimum: 40 kg
  - Suspicious maximum: 300 kg
  - Returns 1.0 if within safe range

- **Gradient Scoring**:
  - For weights between suspicious and absolute limits
  - Uses exponential decay: `score = exp(-3 * distance_ratio)`
  - Distance ratio measures how close to the absolute limit

- **BMI Validation**:
  - Calculates BMI using provided height (default 1.67m)
  - BMI < 15 or > 60: Multiplies score by 0.5 (50% penalty)
  - BMI < 18 or > 40: Multiplies score by 0.8 (20% penalty)

**Critical Threshold**: If safety score < 0.3, measurement is immediately rejected regardless of other components.

### 2. Plausibility Component (25% weight)
**Purpose**: Evaluates statistical likelihood of measurement based on historical data and trends.

**Calculation Method** (`_calculate_plausibility`, lines 203-266):
- **Historical Analysis**:
  - Uses up to 20 recent weight measurements
  - Calculates mean and standard deviation
  - Minimum std dev: 0.5 kg (prevents over-sensitivity)

- **Trend Detection**:
  - Performs linear regression on recent weights (≥4 measurements)
  - Calculates R² to measure trend strength
  - For strong trends (R² > 0.5):
    - Adjusts expected value based on trend projection
    - Increases acceptable variance for trending data
    - Minimum std dev increases to `max(1.0, |slope| * 3)` for trends

- **Z-Score Based Scoring**:
  - Z ≤ 1: Score = 1.0 (within 1 standard deviation)
  - Z ≤ 2: Score = 0.9
  - Z ≤ 3: Score = 0.7
  - Z > 3: Score = exponential decay from 0.5

- **Fallback Logic**:
  - With only previous weight: Assumes 2% standard deviation
  - No history: Returns 0.8 (moderate confidence)

### 3. Consistency Component (25% weight)
**Purpose**: Validates rate of change between consecutive measurements based on time elapsed.

**Calculation Method** (`_calculate_consistency`, lines 267-316):
- **Time-Based Thresholds**:

  **< 6 hours**:
  - Maximum change: 3.0 kg
  - Perfect score (1.0) if within limit
  - Exponential decay penalty for excess

  **6-24 hours**:
  - Maximum change: 2.0 kg for perfect score
  - Up to 4.0 kg: Linear penalty (0.1 per kg over limit)
  - > 4.0 kg: Percentage-based evaluation
    - ≤ 5% change: Score = 0.7
    - > 5% change: Exponential decay from 0.5

  **> 24 hours**:
  - Evaluates daily rate of change
  - Maximum daily rate: 2.0 kg/day for perfect score
  - 2.0-4.0 kg/day: Linear penalty
  - 4.0-6.44 kg/day: Progressive penalty to 0.2
  - > 6.44 kg/day (physiological max): Exponential decay from 0.2

- **No History**: Returns 0.8 (default moderate confidence)

### 4. Reliability Component (15% weight)
**Purpose**: Assigns confidence based on known characteristics of the data source.

**Calculation Method** (`_calculate_reliability`, lines 318-345):
- **Base Reliability Scores**:
  - Excellent: 1.0 (care-team-upload, patient-upload)
  - Good: 0.85 (patient-device, questionnaires)
  - Moderate: 0.7 (connectivehealth.io)
  - Poor: 0.5 (iglucose.com)
  - Unknown: 0.6 (unrecognized sources)

- **Outlier Rate Adjustment**:
  - < 5% outliers: No penalty (1.0x)
  - 5-20% outliers: 5% penalty (0.95x)
  - 20-50% outliers: 10% penalty (0.9x)
  - > 50% outliers: 20% penalty (0.8x)

- **Final Score**: `base_score * outlier_multiplier`

## Score Combination Methods

### Harmonic Mean (Default)
- Formula: `n / Σ(weight_i / score_i)`
- **Characteristics**:
  - Severely penalizes low component scores
  - Returns 0 if any component is 0
  - Ensures all components meet minimum standards
  - Better for safety-critical applications

### Arithmetic Mean (Alternative)
- Formula: `Σ(weight_i * score_i) / Σ(weight_i)`
- **Characteristics**:
  - Standard weighted average
  - Allows compensation between components
  - More forgiving of individual low scores

## Acceptance Logic

1. **Safety Gate**: Safety score must be ≥ 0.3 or measurement is immediately rejected
2. **Overall Threshold**: Combined score must be ≥ 0.6 (default configurable)
3. **Quality Override**: High quality scores (> 0.8) can override outlier detection in the main processor

## Feature Management

Each component can be individually enabled/disabled via the FeatureManager:
- `quality_safety`: Safety component
- `quality_plausibility`: Plausibility component
- `quality_consistency`: Consistency component
- `quality_reliability`: Reliability component

When disabled, components return a perfect score (1.0) and don't affect the overall calculation.

## Key Insights

1. **Safety First**: The safety component acts as a hard gate, preventing acceptance of physiologically impossible values regardless of other scores.

2. **Trend Awareness**: The plausibility component adapts to weight trends, allowing for legitimate weight loss/gain programs while filtering noise.

3. **Time Sensitivity**: The consistency component uses research-based thresholds (2-3% daily variation is normal) with time-appropriate limits.

4. **Source Intelligence**: The reliability component leverages empirical data about source accuracy, automatically giving more weight to trusted sources.

5. **Balanced Approach**: The harmonic mean ensures all components maintain minimum quality, preventing any single weak component from allowing bad data through.

## Configuration Parameters

From `QUALITY_SCORING_DEFAULTS`:
- Threshold: 0.6 (60% minimum score)
- Harmonic mean: Enabled by default
- Component weights: Optimized based on empirical testing
- Safety critical threshold: 0.3 (30% minimum)
- History window: 20 measurements

## Recommendations

1. **Keep harmonic mean enabled** for production systems to maintain strict quality standards
2. **Monitor component scores** to identify systematic issues with specific data sources
3. **Adjust source profiles** in `SOURCE_PROFILES` based on observed outlier rates
4. **Consider height updates** for more accurate BMI-based safety scoring
5. **Review rejection reasons** to fine-tune thresholds if legitimate data is being rejected