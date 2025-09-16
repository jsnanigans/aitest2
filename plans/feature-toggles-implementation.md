# Plan: Feature Toggle System Implementation

## Decision
**Approach**: Add comprehensive [features] section to config.toml with safety-first defaults
**Why**: Provide granular control over processing features while maintaining safety guarantees
**Risk Level**: Medium (config changes affect core processing logic)

## Implementation Steps

### Phase 1: Core Infrastructure (Priority: High)
1. **Create feature config structure** - Add new `[features]` section to `config.toml`
2. **Update config_loader.py** - Add feature toggle parsing and validation logic
3. **Create feature manager** - New `src/feature_manager.py` for centralized feature checks
4. **Add dependency validation** - Ensure dependent features can't be disabled independently

### Phase 2: Safety-Critical Features (Priority: Mandatory)
5. **Protect safety features** - Mark physiological validation as non-toggleable in code
6. **Update validation.py** - Add feature checks for non-safety validation layers
7. **Test safety guarantees** - Verify safety limits remain enforced

### Phase 3: Processing Features (Priority: High)
8. **Update processor.py** - Add feature checks for:
   - Kalman filtering (lines 200-250)
   - Quality scoring (lines 300-350)
   - Outlier detection (lines 400-450)
9. **Update outlier_detection.py** - Conditionally apply IQR, MAD, temporal checks
10. **Update quality_scorer.py** - Make component weights toggleable

### Phase 4: State Management Features (Priority: Medium)
11. **Update database.py** - Add feature checks for:
    - State persistence enable/disable
    - History buffer size configuration
    - Reset event tracking
12. **Update reset_manager.py** - Make reset types individually toggleable

### Phase 5: Optional Features (Priority: Low)
13. **Update visualization.py** - Already has toggle, ensure consistency
14. **Add retrospective toggles** - Refine existing retrospective section

## Files to Change

### New Files
- `src/feature_manager.py` - Centralized feature checking and dependency validation

### Modified Files
- `config.toml` - Add comprehensive [features] section
- `src/config_loader.py:150-200` - Parse and validate feature toggles
- `src/processing/processor.py:200-450` - Add feature conditionals
- `src/processing/validation.py:50-150` - Toggle non-safety validators
- `src/processing/outlier_detection.py:100-200` - Conditional detection methods
- `src/processing/quality_scorer.py:85-95` - Component toggle support
- `src/processing/reset_manager.py:40-80` - Reset type toggles
- `src/database/database.py:150-200` - State persistence toggles

## Config Structure

```toml
[features]
# SAFETY FEATURES (Cannot be disabled)
safety_validation = true  # Always true, config ignored
physiological_limits = true  # Always true, config ignored

# CORE PROCESSING (High Impact)
kalman_filtering = true
quality_scoring = true
outlier_detection = true
adaptive_noise = true

# VALIDATION LAYERS (Medium Impact)
[features.validation]
bmi_check = true
rate_limiting = true
temporal_consistency = true
session_variance = true

# OUTLIER DETECTION METHODS
[features.outlier]
iqr_method = true
mad_method = true
temporal_method = true
kalman_deviation = true
quality_override = true  # High quality can override outliers

# QUALITY SCORING COMPONENTS
[features.quality]
safety_component = true  # Tied to safety_validation
plausibility_component = true
consistency_component = true
reliability_component = true

# STATE MANAGEMENT
[features.state]
persistence = true
history_buffer = true
history_buffer_size = 30
reset_tracking = true

# RESET TYPES
[features.reset]
initial_reset = true
hard_reset = true
soft_reset = true
gap_detection = true

# RETROSPECTIVE PROCESSING
[features.retrospective]
enabled = true
buffer_processing = true
state_rollback = true
```

## Dependency Rules

```python
# In feature_manager.py
FEATURE_DEPENDENCIES = {
    'quality_scoring': ['kalman_filtering'],  # Quality needs Kalman predictions
    'outlier_detection': ['kalman_filtering'],  # Outlier needs innovations
    'features.outlier.kalman_deviation': ['kalman_filtering'],
    'features.quality.consistency_component': ['features.state.history_buffer'],
    'features.reset.soft_reset': ['quality_scoring'],  # Soft reset uses quality
    'features.retrospective.state_rollback': ['features.state.persistence'],
}

MANDATORY_FEATURES = [
    'safety_validation',
    'physiological_limits',
    'features.validation.rate_limiting'  # Prevent dangerous weight changes
]
```

## Acceptance Criteria
- [ ] All existing tests pass with default feature toggles
- [ ] Disabling non-mandatory features reduces processing time by >20%
- [ ] Safety features remain active regardless of config
- [ ] Feature dependencies are validated on config load
- [ ] Clear error messages for invalid feature combinations
- [ ] Backward compatibility maintained (missing section = all enabled)

## Risks & Mitigations

**Main Risk**: Users disable critical features and get poor results
**Mitigation**:
- Default all features to `true`
- Add warning logs when disabling important features
- Document recommended configurations for different use cases

**Secondary Risk**: Feature interaction bugs
**Mitigation**:
- Comprehensive integration tests with feature combinations
- Dependency validation prevents invalid states
- Phased rollout starting with low-risk features

## Migration Strategy

### Phase 1: Add Infrastructure (Week 1)
- Deploy feature manager without changing behavior
- All features default to enabled
- Log feature status without enforcement

### Phase 2: Enable Toggles (Week 2)
- Start respecting feature toggles in code
- Monitor for unexpected behavior
- Keep safety features mandatory

### Phase 3: Documentation (Week 3)
- Create feature combination guide
- Document performance implications
- Add troubleshooting section

## Testing Plan

```python
# Test cases to add in tests/test_feature_manager.py
def test_mandatory_features_cannot_be_disabled():
    """Verify safety features ignore config."""

def test_feature_dependencies_enforced():
    """Verify dependent features auto-enable."""

def test_backward_compatibility():
    """Old configs without [features] work."""

def test_performance_with_minimal_features():
    """Measure speedup with features disabled."""
```

## Out of Scope
- Runtime feature toggling (config is loaded once)
- Per-user feature configuration
- Feature A/B testing framework
- Dynamic feature discovery
- Feature usage analytics