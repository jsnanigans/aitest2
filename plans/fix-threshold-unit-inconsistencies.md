# Plan: Fix Threshold Unit Inconsistencies and Implement Adaptive System

## Summary
Fix critical unit mismatch bugs between enhanced and base processors, implement a unified adaptive threshold system with explicit units, and ensure consistent behavior across all weight validation paths.

## Context
- **Source**: User investigation of impossible weight dips for user `0040872d-333a-4ace-8c5a-b2fcd056e65a`
- **Root Cause**: Enhanced processor passes absolute kg values where base processor expects percentages
- **Impact**: Valid measurements incorrectly rejected, causing Kalman filter to extrapolate unrealistic values

### Assumptions
- Current bug causes valid measurements to be incorrectly accepted/rejected
- System should adapt thresholds based on source reliability
- Both absolute (kg) and relative (%) limits are needed for different scenarios
- Backward compatibility must be maintained

### Constraints
- Must maintain backward compatibility with existing data
- Cannot break the stateless processor architecture
- Should preserve existing physiological limit logic
- Minimal performance impact required

## Requirements

### Functional
- Fix immediate bug where enhanced processor passes kg values where percentages are expected
- Implement unified threshold calculation with explicit unit handling
- Adapt thresholds based on source reliability and time gaps
- Maintain physiological safety limits
- Provide clear unit documentation in all interfaces

### Non-functional
- Clear documentation of units in all interfaces
- Consistent naming convention for configuration parameters
- Testable threshold calculations
- Minimal performance impact (<10ms per calculation)
- Maintainable and extensible design

## Alternatives

### Option A: Minimal Fix - Convert Units at Interface
**Approach**: Only fix the immediate bug by converting kg to percentage in enhanced processor

**Pros**:
- Minimal code changes (5-10 lines)
- Quick to implement (1-2 hours)
- Low risk of breaking existing functionality
- Can be deployed immediately

**Cons**:
- Doesn't address systemic unit confusion
- Leaves multiple threshold systems in place
- Technical debt remains
- High risk of similar bugs in future

### Option B: Unified Threshold System with Explicit Units
**Approach**: Create new threshold calculator class with explicit unit parameters

**Pros**:
- Solves root cause of unit confusion
- Single source of truth for thresholds
- Extensible for future threshold types
- Clear API contracts
- Reduces future maintenance burden

**Cons**:
- More extensive changes (new file, multiple updates)
- Requires updating multiple files
- Higher testing burden
- 2-3 day implementation

### Option C: Type-Safe Units with Custom Classes
**Approach**: Create Weight and Percentage classes that prevent unit mixing at type level

**Pros**:
- Compile-time safety against unit errors
- Self-documenting code
- Impossible to mix units incorrectly
- Industry best practice for unit safety

**Cons**:
- Major refactoring required (weeks)
- Performance overhead from object creation
- May be over-engineering for this use case
- Requires team buy-in on new patterns

## Recommendation
**Option B: Unified Threshold System** - Provides the right balance of fixing the immediate issue while preventing future problems. The explicit unit handling will make the code more maintainable without requiring a complete rewrite.

**Rationale**: 
- Fixes the bug completely, not just symptoms
- Prevents similar bugs in future
- Reasonable implementation effort
- Makes system more maintainable
- Aligns with "do it right" principle without over-engineering

## High-Level Design

### Architecture Overview
```
┌─────────────────────────────────────┐
│     Enhanced Processor              │
│  (processor_enhanced.py)            │
│  - Handles source quality           │
│  - Requests thresholds with units   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Unified Threshold Calculator      │
│   (NEW: threshold_calculator.py)    │
│   - get_extreme_threshold()         │
│   - get_physiological_limit()       │
│   - convert_units()                 │
│   - source_reliability_profiles     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Base Processor                 │
│      (processor.py)                 │
│  - Validates with percentages       │
│  - Applies Kalman filter            │
└─────────────────────────────────────┘
```

### Data Flow
1. Enhanced processor receives measurement with source and weight
2. Calls unified threshold calculator with explicit unit request ('percentage' or 'kg')
3. Calculator considers source reliability, time gaps, and weight magnitude
4. Returns threshold in requested units with metadata
5. Enhanced processor passes correctly-typed values to base processor
6. Base processor validates using consistent units

### Key Interfaces
```python
# Threshold Calculator Interface
ThresholdCalculator.get_extreme_deviation_threshold(
    source: str,
    time_gap_days: float,
    current_weight: float,
    unit: Literal['percentage', 'kg']
) -> ThresholdResult

# ThresholdResult contains:
# - value: float (in requested units)
# - unit: str
# - metadata: dict (reasoning, source_reliability, etc.)
```

### Affected Modules
- `src/processor_enhanced.py` - Update to use unified calculator (~50 lines)
- `src/processor.py` - Update physiological limit calculation (~30 lines)
- `src/threshold_calculator.py` - NEW file for unified logic (~300 lines)
- `config.toml` - Update parameter names with unit suffixes
- `tests/test_threshold_consistency.py` - NEW comprehensive tests (~200 lines)
- `tests/test_threshold_calculator.py` - NEW unit tests (~150 lines)

## Implementation Plan (No Code)

### Step 1: Create Unified Threshold Calculator (Day 1)
**File**: `src/threshold_calculator.py`

**Components**:
- `ThresholdCalculator` class with all static methods
- Source reliability profiles (migrate from enhanced processor)
- Core methods:
  - `get_extreme_deviation_threshold()` - Main entry point for extreme checks
  - `get_physiological_limit()` - Physiological safety limits
  - `convert_threshold()` - Unit conversion helper
  - `_calculate_adaptive_threshold()` - Internal adaptive logic
- Comprehensive docstrings with unit specifications
- Type hints for all parameters and returns

**Key Decisions**:
- Use explicit `unit` parameter in all methods
- Return structured result with value, unit, and metadata
- Include confidence scores in metadata
- Log all threshold calculations for debugging

### Step 2: Fix Critical Bug in Enhanced Processor (Day 1)
**File**: `src/processor_enhanced.py`

**Changes**:
- Import new ThresholdCalculator
- Replace `AdaptiveOutlierDetector.get_adaptive_threshold()` call
- Use `ThresholdCalculator.get_extreme_deviation_threshold(unit='percentage')`
- Add threshold metadata to result
- Add debug logging for threshold values
- Preserve all existing functionality

**Backward Compatibility**:
- Keep AdaptiveOutlierDetector class but mark deprecated
- Add migration warnings in logs
- Support old config parameter names with warnings

### Step 3: Update Base Processor Physiological Limits (Day 2)
**File**: `src/processor.py`

**Changes**:
- Import ThresholdCalculator
- Refactor `_get_physiological_limit()` to use calculator
- Make return value units explicit in docstring
- Add unit conversion helpers if needed
- Ensure tolerance calculations remain correct

**Validation**:
- Verify all callers handle return values correctly
- Check that kg comparisons remain kg-to-kg
- Ensure percentage comparisons remain pct-to-pct

### Step 4: Update Configuration Naming (Day 2)
**Files**: `config.toml`, configuration loaders

**Parameter Renames**:
- `extreme_threshold` → `extreme_threshold_pct`
- `max_daily_change` → `max_daily_change_pct`
- `session_variance_threshold` → `session_variance_kg`
- `max_change_1h_absolute` → `max_change_1h_kg`
- `max_change_1h_percent` → `max_change_1h_pct`

**Migration Strategy**:
- Support both old and new names initially
- Log deprecation warnings for old names
- Document migration in README
- Provide config migration script

### Step 5: Implement Comprehensive Tests (Day 2-3)
**New Test Files**:

**`tests/test_threshold_consistency.py`**:
- Test unit conversion correctness
- Test source-based adaptation
- Test time-gap scaling
- Test edge cases (40kg and 200kg weights)
- Test backward compatibility
- Test interaction between thresholds

**`tests/test_threshold_calculator.py`**:
- Unit tests for each calculator method
- Test boundary conditions
- Test invalid inputs
- Test unit conversion accuracy
- Performance benchmarks

**Update Existing Tests**:
- Fix any tests broken by changes
- Add threshold unit assertions
- Verify user `0040872d` case is fixed

### Step 6: Add Debugging and Monitoring (Day 3)
**Enhancements**:
- Add threshold values to all result metadata
- Include both kg and percentage for transparency
- Add structured logging for threshold decisions
- Create analysis script for threshold tuning
- Add metrics collection for threshold effectiveness

**Debug Output Format**:
```json
{
  "threshold_decision": {
    "extreme_threshold_pct": 0.15,
    "extreme_threshold_kg": 11.25,
    "source_reliability": "moderate",
    "time_gap_factor": 1.2,
    "weight_reference": 75.0
  }
}
```

## Validation & Rollout

### Test Strategy

**Unit Tests**:
- All ThresholdCalculator methods
- Unit conversion accuracy
- Edge cases and boundaries
- Invalid input handling

**Integration Tests**:
- Full processing pipeline with new thresholds
- Real user data from problematic cases
- Various weight ranges and sources
- Time gap scenarios

**Regression Tests**:
- User `0040872d-333a-4ace-8c5a-b2fcd056e65a` specifically
- Top 100 users by measurement count
- Known edge cases from historical issues

**Performance Tests**:
- Threshold calculation < 10ms
- No memory leaks
- Batch processing performance

### Manual Verification Checklist
- [ ] Enhanced processor correctly converts units
- [ ] No more impossible weight dips in visualizations
- [ ] Thresholds scale appropriately with weight (40-200kg range)
- [ ] Source reliability affects thresholds as expected
- [ ] Physiological limits still enforced correctly
- [ ] Config migration works without data loss
- [ ] Backward compatibility maintained for existing configs
- [ ] Debug logs show correct threshold values and units
- [ ] Performance meets requirements (<10ms)

### Rollout Plan

**Phase 1: Shadow Mode (Day 4)**
- Deploy ThresholdCalculator but don't use it
- Log what decisions would be made
- Compare with existing system
- No user impact

**Phase 2: Canary Deployment (Day 5)**
- Enable for 5% of new measurements
- Monitor metrics closely
- Compare acceptance/rejection rates
- Roll back if issues detected

**Phase 3: Gradual Rollout (Day 6-7)**
- 25% → 50% → 75% → 100%
- Monitor at each stage
- Automated rollback on anomalies
- Full rollout over 2 days

**Phase 4: Cleanup (Week 2)**
- Remove deprecated code
- Update documentation
- Close related issues
- Post-mortem review

**Rollback Plan**:
- Feature flag for instant disable
- Old code paths remain for 30 days
- Database compatible with both versions
- Rollback script prepared and tested

## Risks & Mitigations

### Risk 1: Breaking Existing Processing
**Severity**: High  
**Probability**: Medium  
**Mitigation**:
- Extensive testing with production data snapshots
- Feature flag for gradual rollout
- Keep parallel old/new code paths initially
- Automated rollback on metric anomalies

### Risk 2: Performance Impact
**Severity**: Medium  
**Probability**: Low  
**Mitigation**:
- Profile threshold calculations
- Cache threshold values when possible
- Keep calculations simple and optimized
- Performance benchmarks in CI/CD

### Risk 3: Incorrect Unit Conversions
**Severity**: High  
**Probability**: Low  
**Mitigation**:
- Comprehensive unit tests
- Explicit unit parameters in all functions
- Runtime assertions for sanity checks
- Code review by multiple developers

### Risk 4: Source Reliability Profiles Incorrect
**Severity**: Medium  
**Probability**: Medium  
**Mitigation**:
- Base on empirical data analysis
- Make profiles configurable
- Monitor and adjust based on results
- A/B testing for validation

## Acceptance Criteria
- [ ] User `0040872d-333a-4ace-8c5a-b2fcd056e65a` shows realistic weight progression
- [ ] No unit mismatch errors in logs
- [ ] All tests pass including new threshold tests
- [ ] Threshold values in metadata show correct units
- [ ] Source-based adaptation working as designed
- [ ] No regression in processing accuracy for other users
- [ ] Performance requirements met (<10ms per calculation)
- [ ] Configuration migration successful
- [ ] Documentation updated and accurate
- [ ] Code review approved by senior developer

## Out of Scope
- Complete type system overhaul
- Database schema changes
- Changing the stateless architecture
- Modifying visualization components
- Implementing machine learning for thresholds
- Historical data reprocessing
- User-configurable thresholds
- Multi-language support

## Open Questions
1. **Should we add runtime unit validation (assertions)?**
   - Pro: Catches bugs early
   - Con: Performance impact
   - Recommendation: Yes, but only in debug mode

2. **What should the default thresholds be for unknown sources?**
   - Option: Use most conservative (strictest)
   - Option: Use median of known sources
   - Recommendation: Use median for balance

3. **Should we log all threshold decisions for analysis?**
   - Pro: Great for tuning and debugging
   - Con: Log volume concerns
   - Recommendation: Sample 10% in production

4. **How often should we update source reliability profiles?**
   - Option: Monthly based on data analysis
   - Option: Real-time adaptive
   - Recommendation: Monthly with manual triggers

5. **Should thresholds be user-configurable in the future?**
   - Pro: Personalization for edge cases
   - Con: Complexity and support burden
   - Recommendation: Consider for v2 if needed

## Review Cycle

### Self-Review Notes
- Verified that Option B provides best balance of fix completeness and implementation complexity
- Confirmed that backward compatibility approach will work
- Checked that test strategy covers all critical paths
- Ensured rollout plan has sufficient safety mechanisms
- Added specific metrics for monitoring
- Included performance requirements
- Clarified scope boundaries

### Stakeholder Feedback Points
- Review threshold values with data science team
- Validate source reliability profiles with data analysis
- Confirm rollout timeline with operations team
- Review API changes with downstream consumers

### Success Metrics
- **Bug Fix**: Zero instances of impossible weight dips
- **Accuracy**: <5% change in valid measurement rejection rate
- **Performance**: <10ms threshold calculation time
- **Reliability**: Zero threshold-related errors in production
- **Adoption**: 100% of measurements using new system within 7 days

## Appendix: Technical Details

### Current Bug Analysis
```python
# Current (BROKEN):
# Enhanced processor (line 517-519):
adapted_config['extreme_threshold'] = get_adaptive_threshold(source, time_gap_days)
# Returns: 3.0 (meant as 3kg)

# Base processor expects (line 167):
deviation = abs(weight - predicted) / predicted  # Returns: 0.13 (13%)
if deviation > extreme_threshold:  # Compares: 0.13 > 3.0 (always false!)
```

### Fixed Implementation Concept
```python
# Fixed:
threshold_kg = get_adaptive_threshold(source, time_gap_days)  # Returns: 3.0kg
threshold_pct = threshold_kg / current_weight  # Returns: 0.04 (4% for 75kg person)
adapted_config['extreme_threshold'] = threshold_pct  # Passes: 0.04
```

### Source Reliability Data
Based on analysis of 709,246 measurements:
- `care-team-upload`: 3.6 outliers per 1000
- `patient-upload`: 13.0 outliers per 1000
- `internal-questionnaire`: 14.0 outliers per 1000
- `patient-device`: 20.7 outliers per 1000
- `connectivehealth.io`: 35.8 outliers per 1000
- `api.iglucose.com`: 151.4 outliers per 1000