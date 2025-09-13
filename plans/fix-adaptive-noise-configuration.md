# Plan: Fix Adaptive Noise and Add Configuration Support

## Summary
Fix the broken adaptive noise feature in the Kalman filter to properly adjust measurement noise based on data source reliability, and make it fully configurable via config.toml. Based on analysis of 709K+ measurements, implement optimized multipliers that reflect actual source reliability.

## Context
- **Source**: Investigation revealed adaptive noise is calculated but never applied during Kalman updates
- **Data Analysis**: Analyzed 709,246 measurements from 15,615 users across 7 sources
- **Key Finding**: Current assumptions about source reliability are wrong (patient-upload is most reliable, not care-team)
- **Assumptions**: 
  - The feature should improve accuracy by trusting reliable sources more
  - Configuration should allow disabling or tuning the feature
  - Backward compatibility with existing data must be maintained
- **Constraints**:
  - Must maintain stateless processor architecture
  - Cannot break existing state persistence
  - Should be toggleable without code changes

## Requirements

### Functional
1. Adaptive noise must be applied during every Kalman update, not just initialization
2. Source-specific noise multipliers must be configurable
3. Feature must be toggleable via config
4. Must preserve existing behavior when disabled
5. Should log when adaptive noise is applied for debugging
6. Must use data-driven multipliers from analysis (patient-upload as baseline)

### Non-functional
1. Performance impact must be minimal (<1% overhead)
2. Configuration must be intuitive and well-documented
3. Changes must be testable in isolation
4. Must not increase memory usage significantly
5. Should show measurable improvement in prediction accuracy (target: 5-10%)

## Alternatives

### Option A: Fix in Place (Minimal Change)
**Approach**: Modify KalmanFilterManager.update_state to accept observation_covariance parameter
- **Pros**: 
  - Minimal code changes
  - Easy to understand
  - Quick to implement
- **Cons**: 
  - Requires changing method signature
  - May break existing tests
  - Doesn't address state persistence of adapted values

### Option B: Dynamic Adaptation (Recommended)
**Approach**: Calculate and apply adaptive noise dynamically during each update
- **Pros**: 
  - Clean separation of concerns
  - No state format changes needed
  - Easy to toggle on/off
  - Can log adaptation decisions
- **Cons**: 
  - Slightly more computation per measurement
  - Requires passing source to Kalman methods

### Option C: State-Based Adaptation
**Approach**: Store adapted parameters in state and update them per measurement
- **Pros**: 
  - Preserves adaptation history
  - Could enable learning/adjustment over time
- **Cons**: 
  - Requires state format migration
  - More complex implementation
  - Harder to debug

## Recommendation
**Option B: Dynamic Adaptation** - Calculate and apply adaptive noise during each Kalman update based on the current source. This maintains simplicity while fixing the core issue.

## High-Level Design

### Architecture Changes
```
config.toml
    ↓
[adaptive_noise] section
    ↓
processor.py (reads config)
    ↓
kalman.py (applies multiplier during update)  ← KEY FIX
    ↓
Result with adapted confidence
```

### Affected Files
1. `config.toml` - Add [adaptive_noise] section with optimized multipliers
2. `src/models.py` - Make SOURCE_PROFILES configurable
3. `src/kalman.py` - **CRITICAL FIX**: Modify update_state to accept and use adaptive observation_covariance
4. `src/processor.py` - Calculate adapted noise and pass to Kalman update
5. `src/validation.py` - Use config for noise multipliers
6. `src/quality.py` - Remove redundant adaptation code

### Data Flow
1. Config defines enable flag and optimized source multipliers
2. Processor calculates adapted observation_covariance for current source
3. **Kalman.update_state receives and uses adapted value (not stored value)**
4. Confidence calculation uses actual adapted noise value
5. Result includes adaptation metadata for validation

## Implementation Plan (No Code)

### Step 1: Add Configuration Section with Optimized Values
- Add `[adaptive_noise]` section to config.toml
- Include `enabled = true/false` flag
- Add `[adaptive_noise.multipliers]` with data-driven values:
  - patient-upload = 1.0 (baseline, most reliable)
  - care-team-upload = 1.2 (not as reliable as assumed)
  - internal-questionnaire = 1.6
  - patient-device = 2.5
  - connectivehealth.io = 2.2
  - api.iglucose.com = 2.6
- Add `default_multiplier = 1.5` for unknown sources
- Document findings in comments

### Step 2: Fix the Core Bug in Kalman Update
- **CRITICAL**: Modify `KalmanFilterManager.update_state` signature to accept `observation_covariance` parameter
- Change line creating KalmanFilter to use passed value, not state's stored value
- Ensure backward compatibility with default parameter
- This is the minimal fix that makes everything work

### Step 3: Update Processor to Pass Adapted Noise
- In `process_weight_enhanced`, calculate adapted observation_covariance
- Pass adapted value to both `initialize_immediate` and `update_state`
- Remove redundant adaptation in quality.py
- Add debug logging when adaptation applied

### Step 4: Simplify and Clean Up
- Remove `AdaptiveKalmanConfig` class (redundant)
- Consolidate multiplier logic in one place
- Remove unused adaptation code from quality.py
- Ensure single source of truth for multipliers

### Step 5: Add Validation and Metrics
- Log multiplier used for each measurement
- Track average confidence per source
- Compare prediction error before/after
- Add metrics to database export

### Step 6: Create Comprehensive Tests
- Test that different sources produce different confidence
- Verify multipliers are actually applied
- Test with our optimized values vs old values
- Measure accuracy improvement (target: 5-10%)
- Test enable/disable toggle

## Validation & Rollout

### Test Strategy
1. **Unit Tests**:
   - Test multiplier calculation for each source
   - Test confidence changes with different multipliers
   - Test config loading and validation
   
2. **Integration Tests**:
   - Process same data with/without adaptation
   - Verify improved accuracy for mixed-source data
   - Test state persistence across restarts

3. **Performance Tests**:
   - Measure overhead of adaptation
   - Test with 1000+ measurements
   - Verify memory usage unchanged

### Manual QA Checklist
- [ ] Config loads correctly from toml
- [ ] Unknown sources use default multiplier
- [ ] Disabling feature preserves old behavior
- [ ] Adaptation logged appropriately
- [ ] Visualization shows adaptation metadata
- [ ] Database export includes adaptation info

### Rollout Plan
1. **Phase 1**: Deploy with feature disabled by default
2. **Phase 2**: Enable for test users, monitor metrics
3. **Phase 3**: Enable globally if metrics improve
4. **Rollback**: Set `enabled = false` in config

## Risks & Mitigations

### Risk 1: Performance Degradation
- **Mitigation**: Cache multipliers, benchmark before/after
- **Monitoring**: Track processing time per measurement
- **Note**: Single multiplication per measurement is negligible

### Risk 2: Unexpected Behavior Changes
- **Mitigation**: Feature flag allows instant disable
- **Monitoring**: Compare results with/without adaptation
- **Validation**: Test on known problematic users (e.g., 0040872d)

### Risk 3: Wrong Multiplier Values
- **Mitigation**: Values are data-driven from 709K measurements
- **Monitoring**: Track confidence and acceptance rates per source
- **Adjustment**: Config allows easy tuning without code changes

### Risk 4: Breaking Existing State
- **Mitigation**: Changes don't affect state format
- **Testing**: Verify state persistence works across versions
- **Fallback**: Stored observation_covariance remains unchanged

## Acceptance Criteria
1. ✓ Adaptive noise multipliers are applied during every Kalman update (not just init)
2. ✓ Feature can be enabled/disabled via config.toml
3. ✓ Source multipliers are configurable and use optimized values
4. ✓ Unknown sources use configurable default (1.5)
5. ✓ No performance degradation (< 1% overhead)
6. ✓ Tests show 5-10% improved accuracy for mixed-source data
7. ✓ Backward compatible when disabled
8. ✓ Adaptation metadata included in results
9. ✓ patient-upload shows highest confidence (lowest multiplier)
10. ✓ iglucose shows lower confidence but not as low as currently

## Out of Scope
- Machine learning for multiplier adjustment
- Per-user multiplier customization  
- Historical reprocessing of existing data
- UI for configuring multipliers
- Automatic multiplier optimization

## Open Questions
1. Should we allow per-user multiplier overrides?
2. Should multipliers be time-dependent (e.g., trust degrades over time)?
3. Should we track and report adaptation effectiveness metrics?
4. What should the default multipliers be for new/unknown sources?

## Configuration Example
```toml
[adaptive_noise]
# Enable adaptive measurement noise based on source reliability
# Optimized from analysis of 709K+ measurements across 15K+ users (2025-09-13)
enabled = true

# Default multiplier for unknown sources
default_multiplier = 1.5

# Log adaptation decisions (for debugging)
log_adaptations = false

[adaptive_noise.multipliers]
# Lower = more trusted, higher = less trusted
# Multiplier is applied to observation_covariance (base: 3.49)
# IMPORTANT: These values are data-driven, not assumptions!

# Most reliable (baseline) - 23K measurements from 5K users
"patient-upload" = 1.0

# Surprisingly less reliable than expected - only 2K measurements
"care-team-upload" = 1.2  

# Questionnaires - noisier than expected
"internal-questionnaire" = 1.6
"initial-questionnaire" = 1.6
"questionnaire" = 1.6

# Moderate reliability - 171K measurements
"https://connectivehealth.io" = 2.2

# Higher noise - 297K measurements (largest source)
"patient-device" = 2.5

# Highest noise but better than assumed - 195K measurements
"https://api.iglucose.com" = 2.6
```

## Review Notes

### Data-Driven Insights
- **Surprising finding**: care-team-upload is not the gold standard we assumed
- **patient-upload is most reliable**: Should be our baseline for comparison
- **iglucose better than expected**: Current 3.0x multiplier is too punitive
- **Questionnaires are noisy**: Despite being manual entry, show high variability

### Implementation Priority
1. **Fix the bug first** - One-line change in KalmanFilterManager.update_state
2. **Add config support** - Use the optimized multipliers from analysis
3. **Test thoroughly** - Especially on problematic users like 0040872d
4. **Monitor in production** - Track confidence scores per source

### Future Enhancements
- **Automatic learning**: Adjust multipliers based on prediction accuracy
- **Per-user adaptation**: Some users may have consistently good/bad sources
- **Time-based decay**: Old measurements could be trusted less
- **Process noise adaptation**: Currently only adapting measurement noise

### Success Metrics
- **5-10% accuracy improvement** expected based on source distribution
- **Higher confidence for patient-upload** (most common reliable source)
- **Better handling of iglucose** (large volume, currently over-penalized)
- **Reduced false rejections** from reliable sources