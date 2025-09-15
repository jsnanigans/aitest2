# Investigation: Source-Specific Kalman Gap Handling

## Summary
The system has **source-specific handling for measurement noise** but **NO source-specific gap thresholds**. All sources use the same gap detection threshold (10 days by default), though different sources have different noise multipliers that affect Kalman filter adaptation after gaps.

## The Complete Story

### 1. Gap Detection - Universal Threshold
**Location**: `src/processor.py:38-68`
**What happens**: Gap detection uses a single threshold for all sources
```python
def handle_gap_detection(state, current_timestamp, config):
    gap_config = config.get('kalman', {}).get('gap_handling', {})
    gap_threshold = gap_config.get('gap_threshold_days', 10)  # Same for all sources
    
    if gap_days > gap_threshold:
        state['gap_buffer'] = {
            'active': True,
            'gap_days': gap_days,
            ...
        }
```
**Why it matters**: Every source type triggers gap handling at the same 10-day threshold

### 2. Source-Specific Noise Multipliers
**Location**: `src/processor.py:25-34`
**What happens**: Different sources have different noise multipliers
```python
SOURCE_NOISE_MULTIPLIERS = {
    "patient-upload": 1.0,           # Most reliable
    "care-team-upload": 1.2,
    "internal-questionnaire": 1.6,
    "initial-questionnaire": 1.6,
    "questionnaire": 1.6,
    "https://connectivehealth.io": 2.2,
    "patient-device": 2.5,
    "https://api.iglucose.com": 2.6  # Least reliable
}
```
**Root Cause**: Sources with higher noise multipliers are considered less reliable

### 3. Noise Application During Processing
**Location**: `src/processor.py:189-194`
**What happens**: Source-specific noise is applied during Kalman filtering
```python
if adaptive_config.get('enabled', False):
    default_multiplier = adaptive_config.get('default_multiplier', 1.0)
    noise_multiplier = SOURCE_NOISE_MULTIPLIERS.get(source, default_multiplier)
    
    kalman_result = KalmanFilterManager.update(
        state, cleaned_weight, timestamp, config, noise_multiplier
    )
```

### 4. Source Profiles for Validation
**Location**: `src/constants.py:76-190`
**What happens**: Comprehensive source profiles exist but aren't used for gap handling
```python
SOURCE_PROFILES = {
    'care-team-upload': {
        'base_threshold_kg': 2.0,
        'max_threshold_kg': 8.0,
        'max_daily_change_kg': 6.44,
        ...
    },
    'patient-device': {
        'base_threshold_kg': 2.5,
        'max_threshold_kg': 10.0,
        'max_daily_change_kg': 7.0,
        ...
    }
}
```
**Why it matters**: These profiles are used for validation thresholds, not gap detection

### 5. Test Configuration Suggests Intent
**Location**: `scripts/test_no_hard_reset.py:24`
**What happens**: Test config includes unused questionnaire-specific gap threshold
```python
'gap_handling': {
    'enabled': True,
    'gap_threshold_days': 10,
    'questionnaire_gap_threshold_days': 5,  # Not implemented
}
```
**Root Cause**: The test suggests an intent to have source-specific thresholds that was never implemented

## Key Insights

1. **Primary Cause**: Gap detection is source-agnostic - all sources use the same 10-day threshold
2. **Contributing Factors**: 
   - Source-specific noise multipliers affect Kalman adaptation AFTER gap detection
   - Source profiles exist for validation but not for gap handling
3. **Design Intent**: Tests suggest there was an intent to have source-specific gap thresholds (e.g., 5 days for questionnaires) but this was never implemented

## Evidence Trail

### Files Examined
- `src/processor.py`: Contains gap detection logic and noise multipliers
- `src/constants.py`: Defines source profiles with thresholds (unused for gaps)
- `src/kalman.py`: Handles adaptive parameters based on gap size, not source
- `scripts/test_no_hard_reset.py`: Shows intended but unimplemented source-specific thresholds

### Search Commands Used
```bash
rg "source.*gap|gap.*source|source_type|measurement_source" -g "*.py"
rg "calculate_adaptive_threshold|adaptive.*threshold|threshold.*source" -g "*.py"
rg "SOURCE_PROFILES|DEFAULT_PROFILE" -g "*.py"
rg "gap.*threshold|gap_threshold|gap.*days|gap_handling" -g "*.py"
rg "questionnaire.*gap|gap.*questionnaire" -g "*.py"
rg "SOURCE_NOISE_MULTIPLIERS" -g "*.py"
```

## Confidence Assessment
**Overall Confidence**: High
**Reasoning**: Direct code inspection shows no source-specific gap threshold logic exists
**Gaps**: None - the implementation is clear and consistent

## Alternative Explanations
None - the code definitively shows that gap detection uses a universal threshold regardless of source type.

## Recommendations

If source-specific gap thresholds are desired:

1. **Questionnaire Sources** (manual entry): Could use shorter gap threshold (5 days) since irregular entry is expected
2. **Device Sources** (automated): Could use standard threshold (10 days) for continuous monitoring
3. **High-Noise Sources** (api.iglucose.com): Could use longer threshold (15 days) to avoid false resets

Implementation would require:
- Modifying `handle_gap_detection()` to check source type
- Adding source-specific thresholds to config or SOURCE_PROFILES
- Testing different threshold combinations for optimal performance
