# Investigation: Config Feature Toggle Architecture

## Bottom Line
**Root Cause**: Inconsistent feature control with many hard-coded behaviors
**Fix Location**: `config.toml` structure and corresponding code readers
**Confidence**: High

## What's Happening
The system has partial feature controls through profiles and some explicit settings, but lacks comprehensive on/off switches for major features like outlier detection, validation layers, and state management.

## Why It Happens
**Primary Cause**: Features evolved organically without unified toggle design
**Trigger**: `config.toml:8` - Uses profile system but lacks granular controls
**Decision Point**: Code assumes features are always enabled with no disable option

## Evidence
- **Key File**: `src/processing/processor.py:223` - Only quality_scoring has explicit enable flag
- **Search Used**: `rg "get\('enabled'" src/` - Found only 8 enable checks total
- **Missing Controls**: No toggles for outlier detection, BMI validation, rate limiting

## Current Feature Control Mechanisms

### 1. Features With Explicit Controls
- **Quality Scoring**: `quality_config.get('enabled', False)` at processor.py:223
- **Adaptive Noise**: `adaptive_config.get('enabled', True)` at processor.py:176
- **Visualization**: `[visualization] enabled = true` in config.toml:23
- **Threading**: `[visualization.threading] enabled = true` in config.toml:28
- **Retrospective**: `[retrospective] enabled = true` in config.toml:123

### 2. Features With Indirect Controls
- **Reset Management**: Controlled via reset types (initial/hard/soft) with individual enables
- **Source Multipliers**: Controlled by adaptive_noise.enabled parent flag

### 3. Features Without Any Controls (Always On)
- **Outlier Detection**: Always runs, only threshold configurable
- **Physiological Validation**: Hard-coded limits, no bypass option
- **BMI Validation**: Always active if height available
- **Rate Limiting**: Always enforced (hourly/daily/weekly)
- **State Persistence**: Database always used
- **Kalman Filtering**: Core feature, cannot disable
- **Measurement History Buffer**: Always maintained

## Recommended Config Structure

```toml
# Feature Toggles Section
[features]
# Core Processing
kalman_filtering = true          # Cannot be disabled in practice
quality_scoring = true            # Already has control
outlier_detection = true          # NEW: Currently always on
state_persistence = true          # NEW: Currently always on

# Validation Features  
[features.validation]
enabled = true                    # Master switch for all validation
physiological_limits = true       # NEW: Hard safety limits
bmi_plausibility = true          # NEW: BMI-based checks
rate_limiting = true             # NEW: Time-based change limits
source_validation = true         # NEW: Source-specific rules

# Adaptive Features
[features.adaptive]
enabled = true                   # Already exists as adaptive_noise.enabled
noise_modeling = true            # Source-specific multipliers
parameter_decay = true           # NEW: Time-based adaptation
reset_management = true          # Currently spread across reset types

# State Management
[features.state]
persistence = true               # NEW: Database storage
history_buffer = true            # NEW: 30-measurement buffer
reset_tracking = true            # NEW: Reset event audit

# Data Quality
[features.quality]
scoring = true                   # Exists as quality_scoring.enabled
harmonic_mean = true            # Exists as quality_scoring.use_harmonic_mean
component_weights = true        # Allow custom weight overrides
override_outliers = true        # NEW: Quality can override statistical rejection

# Processing Controls
[features.processing]
retrospective_analysis = true   # Exists as retrospective.enabled
batch_outlier_detection = true  # NEW: Currently part of retrospective
extreme_threshold_check = true  # NEW: 15% Kalman deviation
```

## Implementation Challenges

### 1. Safety Concerns
Nancy Leveson: "Disabling physiological validation could cause patient harm. Some features must remain mandatory."
- Solution: Add warnings when disabling safety features
- Consider read-only flags for critical safety features

### 2. Interdependencies
- Quality scoring depends on validation components
- Outlier detection uses Kalman predictions
- State persistence required for meaningful processing
- Solution: Validate feature combinations, warn on invalid configs

### 3. Backward Compatibility
Barbara Liskov: "Existing configs must continue working without modification."
- Solution: All new toggles default to `true` (current behavior)
- Profile system remains primary interface

### 4. Performance Impact
- Checking feature flags adds overhead
- Solution: Cache feature flags at startup, not per-measurement

## Next Steps
1. Add feature toggle section to config.toml with safe defaults
2. Update processor.py to check outlier_detection.enabled flag
3. Make validation components individually controllable
4. Add config validation to warn about unsafe combinations

## Risks
- Users might disable critical safety features
- Invalid feature combinations could cause crashes
- Performance overhead from excessive flag checking
