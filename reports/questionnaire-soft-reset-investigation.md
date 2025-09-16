# Investigation: Questionnaire Measurement Not Triggering Soft Reset

## Bottom Line
**Root Cause**: Weight change (3.99kg) below soft reset threshold (5kg)
**Fix Location**: `config.toml:130` - soft reset min_weight_change_kg
**Confidence**: High

## What's Happening
A questionnaire (manual entry) measurement for user 011366d4 shows 95.25kg but doesn't trigger a soft reset because the weight change from the previous measurement (91.26kg) is only 3.99kg, below the 5kg threshold required for soft resets.

## Why It Happens
**Primary Cause**: Soft reset requires BOTH manual source AND ≥5kg change
**Trigger**: `src/reset_manager.py:66-72` - Two-condition check
**Decision Point**: `src/reset_manager.py:70` - Weight change 3.99kg < 5kg threshold

## Evidence
- **Key File**: `src/reset_manager.py:63-77` - Soft reset logic with dual conditions
- **Data Analysis**: `grep 011366d4 data/2025-09-05_nocon.csv` - Shows weight progression
- **Test Verification**: Created test showing 3.99kg change doesn't trigger reset
- **Source Preservation**: Verified questionnaire source preserved through replay

## Detailed Analysis

### 1. Soft Reset Requirements
The system correctly identifies questionnaire as manual entry (`reset_manager.py:17-25`):
- 'internal-questionnaire' is in MANUAL_DATA_SOURCES
- Source preserved through buffer → outlier detection → replay

### 2. Weight Change Calculation
User 011366d4's measurements:
- 2025-01-27: 91.26kg (patient-device)
- 2025-01-28: 95.25kg (internal-questionnaire)
- Change: 3.99kg < 5kg threshold

### 3. Retrospective Processing Not At Fault
- Source correctly preserved: `measurement.get('source', 'retrospective-replay')`
- Replay manager passes original source to process_measurement
- Reset logic sees correct 'internal-questionnaire' source

### 4. Quality Override Unrelated
While quality override issue exists (see retrospective-quality-override.md), it doesn't affect reset logic. Resets check source and weight change only, not quality scores.

## Next Steps
1. Review if 5kg threshold appropriate for manual entries
2. Consider separate thresholds for different manual sources
3. Add logging when soft reset conditions partially met but threshold not reached

## Risks
- **Main Risk**: Legitimate manual corrections not triggering necessary recalibration
- **Secondary**: User confusion when manual entries don't reset as expected
