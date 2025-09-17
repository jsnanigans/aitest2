# Plan: PhysiologicalValidator Comprehensive Unit Tests

## Decision
**Approach**: Create exhaustive safety-critical test suite with boundary, gradient, and fault injection testing
**Why**: PhysiologicalValidator is a safety-critical component that prevents harmful data from entering the system
**Risk Level**: Low (testing only, no production changes)

## Implementation Steps

1. **Create test file** - Add `tests/test_physiological_validator.py` with comprehensive test suite
2. **Implement boundary tests** - Test exact limits at 20.0, 30.0, 400.0, 700.0 kg with precision edge cases
3. **Add gradient tests** - Verify safety score calculations near boundaries
4. **Add fault injection** - Test with NaN, infinity, negative, None values
5. **Add integration tests** - Test with quality scorer integration
6. **Add performance tests** - Ensure validation remains fast under load

## Files to Change
- `tests/test_physiological_validator.py` - New comprehensive test file
- `tests/conftest.py` - Add fixtures for test data generation

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

## Test Categories & Cases

### 1. Absolute Limit Validation (Safety Critical)
```python
# Hard rejection boundaries (lines 58-61 in validation.py)
test_absolute_min_exact()           # 30.0kg - should pass
test_absolute_min_below()           # 29.999kg - should reject
test_absolute_min_epsilon_below()   # 29.99999999 - should reject
test_absolute_max_exact()           # 400.0kg - should pass
test_absolute_max_above()           # 400.001kg - should reject
test_absolute_max_epsilon_above()   # 400.00000001 - should reject

# Legacy compatibility (constants show old limits)
test_legacy_min_20kg()              # Test if 20kg still rejected
test_legacy_max_700kg()             # Test if 700kg still rejected
```

### 2. Suspicious Range Detection
```python
# Suspicious boundaries (lines 65-71)
test_suspicious_min_exact()         # 40.0kg - edge of suspicious
test_suspicious_min_below()         # 39.9kg - suspicious flag
test_suspicious_max_exact()         # 300.0kg - edge of suspicious
test_suspicious_max_above()         # 300.1kg - suspicious flag
test_suspicious_range_warnings()    # Verify warnings generated
```

### 3. Rate of Change Validation
```python
# Daily change limits (lines 73-99)
test_max_daily_change_exact()       # 6.44kg/day - should pass
test_max_daily_change_exceeded()    # 6.45kg/day - should reject
test_hourly_change_limits()         # 4.22kg/1h limit
test_six_hour_limits()              # 6.23kg/6h limit
test_zero_time_diff()               # Handle time_diff = 0
test_negative_time_diff()           # Handle time_diff < 0
test_source_specific_limits()       # Different limits per source
```

### 4. Safety Score Calculations
```python
# Score gradients (quality_scorer lines 170-201)
test_safety_score_perfect_range()   # 50-250kg = 1.0 score
test_safety_score_gradient_low()    # 30-40kg gradient
test_safety_score_gradient_high()   # 300-400kg gradient
test_safety_score_bmi_penalty()     # BMI < 15 or > 60 = 0.5x
test_safety_score_clamping()        # Always [0.0, 1.0]
test_safety_score_with_height()     # Height-aware scoring
```

### 5. Comprehensive Validation
```python
# Full validation pipeline (lines 185-243)
test_comprehensive_all_checks()     # All validations pass
test_comprehensive_absolute_fail()  # Absolute limit failure
test_comprehensive_rate_fail()      # Rate limit failure
test_comprehensive_warnings_only()  # Pass with warnings
test_comprehensive_missing_data()   # Handle None inputs
test_feature_toggle_disable()       # Feature manager integration
```

### 6. Input Validation & Edge Cases
```python
# Malicious/invalid inputs
test_nan_weight()                   # float('nan')
test_infinity_weight()              # float('inf'), float('-inf')
test_negative_weight()              # -50.0kg
test_zero_weight()                  # 0.0kg
test_none_weight()                  # None input
test_string_weight()                # "100" (type error)
test_extreme_large()                # 1e10 kg
test_extreme_small()                # 1e-10 kg
```

### 7. Floating Point Precision
```python
# Precision edge cases
test_float_precision_boundaries()   # 30.0 vs 29.999999999999996
test_rounding_at_limits()          # Round-trip precision
test_decimal_comparison()          # Decimal vs float
test_epsilon_comparisons()         # Use math.isclose()
```

### 8. Pattern Detection
```python
# Measurement patterns (lines 102-147)
test_pattern_insufficient_data()    # < 2 measurements
test_pattern_oscillation_detect()   # Detect weight bouncing
test_pattern_high_variance()        # cv > threshold
test_pattern_suspicious_flag()      # std > 2 * typical
test_pattern_time_windowing()       # 24h window filtering
```

### 9. Source-Specific Behavior
```python
# Source profiles affect validation
test_care_team_thresholds()        # Most reliable source
test_iglucose_extra_scrutiny()     # Least reliable source
test_unknown_source_defaults()     # DEFAULT_PROFILE usage
test_source_noise_multipliers()    # Verify multipliers applied
```

### 10. Performance & Stress Tests
```python
# Performance requirements
test_validation_speed()             # < 1ms per validation
test_bulk_validation()              # 10000 validations
test_concurrent_validation()       # Thread safety
test_memory_usage()                # No memory leaks
```

## Acceptance Criteria
- [ ] 100% code coverage for PhysiologicalValidator class
- [ ] All boundary values tested with exact and epsilon variations
- [ ] All safety-critical paths have explicit tests
- [ ] Malicious input handling verified
- [ ] Performance < 1ms per validation
- [ ] Integration with QualityScorer tested
- [ ] Source-specific behavior verified

## Test Data Fixtures
```python
@pytest.fixture
def weight_generator():
    """Generate test weights at boundaries"""
    return {
        'absolute_min': 30.0,
        'absolute_max': 400.0,
        'suspicious_min': 40.0,
        'suspicious_max': 300.0,
        'normal': 75.0,
        'boundaries': [29.999, 30.0, 30.001, 399.999, 400.0, 400.001]
    }

@pytest.fixture
def malicious_inputs():
    """Malicious/edge case inputs"""
    return [
        float('nan'), float('inf'), float('-inf'),
        -100, 0, 1e10, 1e-10, None, "weight"
    ]
```

## Risks & Mitigations
**Main Risk**: Missing edge case could allow harmful data through
**Mitigation**: Use property-based testing (hypothesis) for exhaustive boundary exploration

**Secondary Risk**: Performance regression under load
**Mitigation**: Add benchmark tests with performance assertions

## Out of Scope
- Integration with database
- End-to-end pipeline testing
- UI validation testing
- Multi-user concurrent testing