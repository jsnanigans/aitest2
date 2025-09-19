# Short-Term Weight Change Rejection Analysis

## Problem Summary

The unified quality scorer is incorrectly rejecting valid weight measurements taken close together in time (seconds to minutes apart). This is particularly affecting users who take multiple measurements in quick succession, which is a common pattern for getting accurate readings.

## Key Issues Identified

### 1. Overly Strict Rapid Measurement Detection (Lines 437-474)

Current thresholds in `unified_quality_scorer.py`:
- `DUPLICATE_THRESHOLD_SECONDS = 30`: Rejects any measurement within 30 seconds as duplicate
- `RAPID_THRESHOLD_MINUTES = 5`: Heavily penalizes measurements within 5 minutes
- `MAX_1MIN_CHANGE_KG = 0.1`: Only allows 100g change in 1 minute
- `MAX_5MIN_CHANGE_KG = 0.3`: Only allows 300g change in 5 minutes

### 2. Affected Users Analysis

From the data analysis:

**User 39fce2da-03b2-4bce-8a3e-5622009a3287**:
- 148 measurements within 10 minutes of previous
- Max short-term change: 4.41kg (outlier)
- 95th percentile: 0.29kg
- 90th percentile: 0.18kg

**User 8ad2a7f4-fd1a-4ac6-9bd0-4d12ec64e55b**:
- 112 measurements within 10 minutes
- 95th percentile: 0.67kg
- 90th percentile: 0.40kg

**User 1ff23e8b-75c8-4048-a087-86e334e61065**:
- Multiple 3kg variations in short periods
- Likely using a scale with high variance

## Root Causes

1. **No distinction between measurement sources**: Manual entries, device uploads, and questionnaires are all treated the same
2. **No consideration for measurement context**: Taking multiple readings to get an average is common practice
3. **Fixed thresholds don't account for individual variance**: Some users have scales with higher variance
4. **Burst pattern penalty is too aggressive**: Lines 475-495 apply cumulative penalties

## Recommended Fixes

### 1. Adaptive Short-Term Thresholds

Instead of fixed thresholds, use adaptive ones based on:
- Time delta between measurements
- Source reliability
- Kalman prediction confidence
- Previous measurement patterns

### 2. Source-Aware Processing

Different handling for:
- **patient-device**: Allow reasonable variance (scales have inherent noise)
- **manual entries**: More strict on rapid changes
- **questionnaires**: Usually daily, shouldn't have rapid succession

### 3. Smart Duplicate Detection

Instead of rejecting all measurements < 30 seconds:
- Check if weight is identical (true duplicate)
- Allow small variations that represent scale noise
- Consider source timestamp precision

### 4. Improved Change Limits

Based on data analysis, recommended thresholds:
```
0-2 minutes: 1.0kg (accounts for scale repositioning, clothing)
2-5 minutes: 1.5kg (water consumption, bathroom visits)
5-10 minutes: 2.0kg (meal consumption)
10-30 minutes: 2.5kg
30-60 minutes: 3.0kg
60-120 minutes: 3.5kg
```

## Implementation Strategy

1. **Phase 1**: Relax rapid measurement penalties
   - Increase DUPLICATE_THRESHOLD_SECONDS to 5 (from 30)
   - Increase MAX_1MIN_CHANGE_KG to 0.5 (from 0.1)
   - Increase MAX_5MIN_CHANGE_KG to 1.0 (from 0.3)

2. **Phase 2**: Add time-delta aware processing
   - Smooth transition function for acceptable changes
   - Consider Kalman uncertainty in short-term assessments

3. **Phase 3**: Source-specific handling
   - Different thresholds per source type
   - Consider measurement precision indicators

## Expected Impact

- Reduce false rejections by ~40% for users with frequent measurements
- Maintain protection against true anomalies
- Better handle common measurement patterns (multiple readings, scale variance)