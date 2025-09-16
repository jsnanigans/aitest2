# Investigation: Retrospective Processing Overrides Quality Scores

## Bottom Line
**Root Cause**: Statistical outlier detection ignores quality scores completely
**Fix Location**: `src/outlier_detection.py:40-71` and `main.py:213`
**Confidence**: High

## What's Happening
The retrospective processing pipeline discards measurements with good quality scores because it relies purely on statistical outlier detection (IQR, Z-score, temporal consistency) without considering the original quality validation results. This causes valid measurements to be rejected and problematic ones to be accepted.

## Why It Happens
**Primary Cause**: Complete separation of quality scoring and outlier detection pipelines
**Trigger**: `main.py:213` - `outlier_detector.get_clean_measurements()` ignores quality metadata
**Decision Point**: `src/outlier_detection.py:40-71` - Three statistical methods run without quality input

## Evidence
- **Key File**: `src/outlier_detection.py:40` - `detect_outliers()` method only uses weight values
- **Search Used**: `rg "quality_score" src/outlier_detection.py` - Found 0 matches
- **Buffer Storage**: `main.py:192-197` - Quality scores stored in metadata but never used
- **Replay Config**: `src/replay_manager.py:298-310` - Reduces quality threshold to 0.25 during replay

## Critical Findings

### 1. Quality Scores Are Stored But Ignored
Measurements include quality metadata when buffered (`main.py:192-197`):
- `metadata.accepted`: Initial quality decision
- `metadata.rejection_reason`: Why it was rejected
But `outlier_detection.py` never reads this metadata.

### 2. Statistical Methods Override Quality
Three statistical methods run independently:
- IQR method (line 60-61)
- Modified Z-score (line 64-65) 
- Temporal consistency (line 68-69)
Any method can mark a high-quality measurement as outlier.

### 3. Kalman Predictions Not Used
Despite having Kalman predictions available, outlier detection doesn't compare measurements against predicted values - the most reliable signal for true outliers.

## Next Steps
1. Modify `OutlierDetector.detect_outliers()` to respect quality scores - never mark high-quality measurements as outliers
2. Add Kalman prediction deviation as primary outlier signal before statistical methods
3. Combine quality scores WITH statistical methods using AND logic, not OR

## Risks
- **Main Risk**: System accepting bad data while rejecting good data, causing trajectory jumps
- **Secondary**: Loss of valid historical data during retrospective processing
