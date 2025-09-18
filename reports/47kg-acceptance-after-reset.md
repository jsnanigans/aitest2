# Investigation: 47.7 kg Acceptance After 89 kg Baseline

## Bottom Line
**Root Cause**: 31-day measurement gap triggered HARD reset, wiping Kalman state
**Fix Location**: `src/processing/reset_manager.py:59` - gap threshold check
**Confidence**: High

## What's Happening
User 76f01a11-1119-4d1a-bd3f-2cb8a1203d6e had weight drop from 89.2 kg (Feb 13) to 47.7 kg (Mar 16) - a 46% decrease - which was accepted as valid by the system.

## Why It Happens
**Primary Cause**: Hard reset after 31-day gap erases all weight history
**Trigger**: `src/processing/reset_manager.py:60` - Gap exceeded 30-day threshold
**Decision Point**: `src/processing/processor.py` - Reset wipes state, treats 47.7 kg as initial measurement

## Evidence
- **Gap Duration**: 31.048 days between Feb 13 and Mar 16 measurements
- **Reset Type**: "hard" with reason "gap_exceeded_31_days"  
- **Key File**: `output/results_test_no_date.json` shows `stage: "initialization"` for 47.7 kg
- **Search Used**: `jq '.users."76f01a11-1119-4d1a-bd3f-2cb8a1203d6e"[]'` - Found reset_event
- **Config**: `config.toml:295` - hard reset gap_threshold_days = 30 (default)

## Root Cause Analysis

### 1. Gap Triggers Reset
31-day gap between measurements exceeded 30-day hard reset threshold. System treated next measurement as if user is brand new.

### 2. State Completely Wiped  
Hard reset sets `kalman_params = None`, erasing all weight history. The 47.7 kg measurement initialized fresh Kalman state with no previous context.

### 3. Quality Score Not Context-Aware
Quality scorer gave high scores (safety: 1.0, plausibility: 1.0, consistency: 1.0) because after reset there's no previous weight for comparison. Without context, 47.7 kg passes all checks.

### 4. Outlier Detection Disabled
With fresh state and no history, outlier detection has no baseline. First measurement after reset always accepted.

## Next Steps
1. Implement sanity check: Compare new weight against pre-reset baseline before accepting
2. Add reset validation: Flag extreme deviations (>20%) even after resets
3. Consider soft reset for gaps under 60 days to preserve some context

## Risks
- Patient safety: Accepting 47% weight loss could mask critical health issues
- Data integrity: Resets allow physically impossible weight changes into clean data
