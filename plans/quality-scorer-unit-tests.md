# Plan: QualityScorer Comprehensive Unit Tests

## Decision
**Approach**: Implement comprehensive unit test suite for QualityScorer with safety-critical coverage
**Why**: Quality scoring determines measurement acceptance/rejection, directly impacting patient care
**Risk Level**: High - incorrect scoring could allow harmful data through or reject valid measurements

## Implementation Steps

### 1. Test Structure Setup
**File**: `tests/test_quality_scorer.py`
- Import QualityScorer, QualityScore from src/processing/quality_scorer
- Import constants from src/constants
- Setup test fixtures for various source profiles
- Mock FeatureManager for feature toggle testing

### 2. Component Score Testing

#### Safety Score Tests (`test_safety_score_*`)
**Coverage**: `_calculate_safety()` method
- `test_safety_score_absolute_limits` - Weight < 30kg or > 400kg returns 0.0
- `test_safety_score_safe_range` - 40-300kg returns 1.0
- `test_safety_score_suspicious_boundaries` - Exponential decay near limits
- `test_safety_score_bmi_penalties` - BMI < 15 or > 60 reduces score by 50%
- `test_safety_score_edge_cases` - Zero, negative, NaN weights

#### Plausibility Score Tests (`test_plausibility_score_*`)
**Coverage**: `_calculate_plausibility()` method
- `test_plausibility_no_history` - Returns 0.8 with no data
- `test_plausibility_with_previous_only` - Uses 2% std deviation baseline
- `test_plausibility_with_recent_weights` - Z-score calculation with history
- `test_plausibility_trend_detection` - R² > 0.5 adjusts expectations
- `test_plausibility_strong_trend` - R² > 0.8 uses projected values
- `test_plausibility_z_score_bands` - Score mapping (z≤1:1.0, z≤2:0.9, z≤3:0.7)

#### Consistency Score Tests (`test_consistency_score_*`)
**Coverage**: `_calculate_consistency()` method
- `test_consistency_no_previous` - Returns 0.8 with no history
- `test_consistency_within_6_hours` - 3kg max change threshold
- `test_consistency_within_24_hours` - 2kg max change, gradual penalties
- `test_consistency_daily_rate` - 2kg/day max sustained rate
- `test_consistency_percentage_fallback` - > 5% change triggers percentage mode
- `test_consistency_physiological_max` - 6.44kg/day absolute limit

#### Reliability Score Tests (`test_reliability_score_*`)
**Coverage**: `_calculate_reliability()` method
- `test_reliability_known_sources` - care-team (1.0), patient-device (0.7), iglucose (0.5)
- `test_reliability_unknown_source` - Default 0.6 score
- `test_reliability_outlier_adjustment` - High outlier rates reduce score
- `test_reliability_profile_loading` - SOURCE_PROFILES correctly applied

### 3. Score Integration Tests

#### Weighted Averaging Tests (`test_averaging_*`)
- `test_harmonic_mean_calculation` - Penalizes low component scores
- `test_harmonic_mean_zero_component` - Returns 0.0 if any component is 0
- `test_arithmetic_mean_calculation` - Standard weighted average
- `test_custom_weights` - Config overrides default weights
- `test_weight_normalization` - Handles missing/extra components

### 4. Acceptance Logic Tests

#### Threshold Tests (`test_threshold_*`)
- `test_default_threshold` - 0.6 default acceptance
- `test_custom_threshold` - Config override works
- `test_safety_critical_override` - Safety < 0.3 forces rejection
- `test_quality_override` - Score > 0.8 overrides outlier detection
- `test_rejection_reason_generation` - Correct reason with weakest component

### 5. Feature Toggle Tests

#### Component Disabling (`test_feature_toggle_*`)
- `test_safety_disabled` - Returns 1.0 when disabled
- `test_plausibility_disabled` - Returns 1.0 when disabled
- `test_consistency_disabled` - Returns 1.0 when disabled
- `test_reliability_disabled` - Returns 1.0 when disabled
- `test_all_features_disabled` - Still calculates overall score

### 6. Edge Cases & Error Handling

#### Invalid Input Tests (`test_invalid_*`)
- `test_negative_weight` - Handled gracefully
- `test_nan_weight` - Returns 0.0 safety score
- `test_inf_weight` - Returns 0.0 safety score
- `test_negative_time_diff` - Treated as no history
- `test_zero_height` - Avoid division by zero in BMI
- `test_empty_recent_weights` - Falls back to previous_weight

### 7. Integration Scenarios

#### Real-World Scenarios (`test_scenario_*`)
- `test_first_measurement` - No history, accepts reasonable weight
- `test_after_long_gap` - 30+ day gap handling
- `test_rapid_weight_loss` - Medical intervention detection
- `test_gradual_trend` - Consistent loss/gain pattern
- `test_data_entry_error` - Catches 10x input errors
- `test_source_switching` - Different reliability between sources

### 8. Performance & Memory Tests

#### Efficiency Tests (`test_performance_*`)
- `test_large_history_buffer` - 1000+ measurements
- `test_memory_usage` - No leaks with repeated calls
- `test_calculation_speed` - < 10ms per score

## Files to Change

- `tests/test_quality_scorer.py` - New comprehensive test file
- `tests/conftest.py` - Add fixtures for common test data

## Pytest Best Practices Requirements

### Mandatory Implementation Standards
1. **Fixtures** - Use `@pytest.fixture` for shared test data and setup
   - Create reusable fixtures in conftest.py for common configurations
   - Use fixture composition for complex test scenarios
   - Implement proper fixture scoping (function, class, module, session)

2. **Parametrization** - Use `@pytest.mark.parametrize` for data-driven tests
   - Test multiple input combinations with single test methods
   - Include edge cases and boundary values in parameters
   - Use descriptive parameter IDs for clear test output

3. **Markers** - Use `@pytest.mark` for test categorization
   - `@pytest.mark.slow` for long-running tests
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests
   - Define markers in conftest.py with descriptions

4. **Assertions** - Use pytest assertion features
   - `pytest.approx()` for floating-point comparisons
   - Descriptive assertion messages with f-strings
   - Multiple assertions with clear failure context

5. **Test Organization**
   - Group related tests in classes
   - Follow naming convention: `test_<what>_<condition>_<expected>`
   - Use docstrings with Given/When/Then format
   - Keep tests isolated and independent

6. **Mock Usage** - Proper mocking with unittest.mock
   - Mock external dependencies
   - Use `MagicMock` for complex objects
   - Verify mock calls with assertions

7. **Error Testing** - Test error conditions
   - Use `pytest.raises` for exception testing
   - Test error messages and error types
   - Validate error recovery paths
- `src/processing/quality_scorer.py:445-450` - Fix duplicate method definitions

## Acceptance Criteria

- [ ] 100% line coverage of QualityScorer class
- [ ] All component scores tested independently
- [ ] Integration tests cover real-world scenarios
- [ ] Safety-critical thresholds verified
- [ ] Feature toggle behavior validated
- [ ] Edge cases handled without crashes
- [ ] Performance benchmarks met (<10ms/score)
- [ ] Memory usage stable under load

## Risks & Mitigations

**Main Risk**: False negatives allowing bad data through
**Mitigation**: Exhaustive boundary testing on safety thresholds, property-based testing for invariants

**Secondary Risk**: False positives rejecting valid measurements
**Mitigation**: Test with real-world data patterns, validate against medical literature

## Out of Scope

- Database integration tests (covered in test_processor.py)
- Visualization of quality scores (covered in test_visualizer.py)
- API endpoint testing (no API layer yet)
- Multi-threaded access patterns (scorer is stateless)