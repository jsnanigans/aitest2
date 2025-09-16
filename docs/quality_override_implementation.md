# Quality Score Override Implementation

## Changes Implemented

This document describes the implementation of quality score override in the outlier detection system, as specified in the retrospective processing report.

## Key Changes

### 1. OutlierDetector Class Enhancement (`src/outlier_detection.py`)

#### Constructor Updates
- Added `db` parameter to access Kalman states for prediction-based outlier detection
- Added `quality_score_threshold` config (default: 0.7) - measurements above this are protected
- Added `kalman_deviation_threshold` config (default: 0.15) - 15% deviation threshold from Kalman prediction

#### New Method: `_detect_kalman_outliers`
- Compares measurements against Kalman filter predictions
- Uses state history snapshots for accurate predictions
- Marks measurements as outliers if they deviate >15% from predicted values

#### Modified Method: `detect_outliers`
- Now accepts optional `user_id` parameter for Kalman-based detection
- Implements quality score protection:
  - Measurements with quality_score > 0.7 are never marked as outliers
  - Measurements explicitly marked as 'accepted' are protected
- Uses AND logic for outlier determination:
  - Must fail statistical tests (IQR, Z-score, or temporal consistency)
  - AND must fail Kalman prediction test (if available)
  - AND must have low quality score (<0.7)

#### Updated Method: `get_clean_measurements`
- Now accepts optional `user_id` parameter
- Passes user_id through to detect_outliers for Kalman-based detection

### 2. Main Processing Updates (`main.py`)

#### OutlierDetector Initialization
- Now passes database instance to OutlierDetector constructor
- Enables Kalman prediction-based outlier detection

#### Metadata Enhancement
- Quality scores and components are now included in measurement metadata
- Ensures outlier detector has access to quality information

#### Retrospective Processing
- `_process_retrospective_buffer` now passes user_id to outlier detector
- Enables user-specific Kalman prediction comparisons

## Behavior Changes

### Before
- Statistical outlier detection ignored quality scores completely
- Any measurement could be marked as outlier based purely on statistics
- Valid high-quality measurements were being discarded
- Problematic low-quality measurements were sometimes accepted

### After
- High-quality measurements (score > 0.7) are NEVER marked as outliers
- Outlier detection uses AND logic:
  - Low quality score (<0.7)
  - AND statistical outlier (IQR, Z-score, or temporal)
  - AND Kalman prediction outlier (if state available)
- System trusts quality scores while still removing true outliers
- Kalman trajectories remain smooth

## Configuration

New configuration options in `config.toml`:

```toml
[retrospective.outlier_detection]
quality_score_threshold = 0.7  # Measurements above this are protected
kalman_deviation_threshold = 0.15  # 15% deviation from Kalman prediction
```

## Testing

A comprehensive test suite (`test_quality_override.py`) validates:
1. High-quality measurements are never marked as outliers
2. AND logic correctly combines quality and statistical checks
3. Low-quality statistical outliers are properly identified

## Benefits

1. **Preserves Valid Data**: High-quality measurements from trusted sources are retained
2. **Removes True Outliers**: Low-quality measurements with statistical anomalies are removed
3. **Smooth Trajectories**: Kalman filter predictions help identify true deviations
4. **Configurable Thresholds**: All thresholds can be tuned via configuration

## Next Steps

1. Monitor system behavior with new logic
2. Adjust quality_score_threshold if needed (currently 0.7)
3. Fine-tune kalman_deviation_threshold based on real data (currently 0.15)
4. Consider adding source-specific quality thresholds