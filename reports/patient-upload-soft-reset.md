# Investigation: Patient Upload Soft Reset Triggering

## Bottom Line

**Root Cause**: `patient-upload` is hardcoded in `MANUAL_DATA_SOURCES` set which overrides config
**Fix Location**: `src/processing/reset_manager.py:21`
**Confidence**: High

## What's Happening

Patient Upload sources trigger soft resets despite configuration attempts to exclude them. The config.toml profile settings for `trigger_sources` are being ignored because `patient-upload` is hardcoded as a manual data source.

## Why It Happens

**Primary Cause**: Two-path source checking with hardcoded set taking precedence
**Trigger**: `src/processing/reset_manager.py:66` - `if source in MANUAL_DATA_SOURCES or source in soft_config.get('trigger_sources', [])`
**Decision Point**: `src/processing/reset_manager.py:17-25` - `MANUAL_DATA_SOURCES` includes `'patient-upload'`

## Evidence

- **Key File**: `src/processing/reset_manager.py:17-25` - Shows hardcoded set with `patient-upload`
- **Search Used**: `rg "MANUAL_DATA_SOURCES"` - Found hardcoded definition
- **Config Loader**: `src/config_loader.py:50` - Shows profile "moderate" sets trigger_sources to `["questionnaire", "care-team-upload"]` but doesn't include `patient-upload`
- **CSV Data**: `data/2025-09-05_nocon.csv` - Confirms source values are `patient-upload` (lowercase with hyphen)

## Next Steps

1. Remove `'patient-upload'` from `MANUAL_DATA_SOURCES` set in `src/processing/reset_manager.py`
2. Ensure all manual sources are controlled via config `trigger_sources` only
3. Update default config to explicitly list all desired manual sources

## Risks

- Removing from hardcoded set may affect existing behavior if configs aren't properly set
- Other code may depend on `MANUAL_DATA_SOURCES` for source classification
