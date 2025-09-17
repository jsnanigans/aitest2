# Plan: OutlierDetector Unit Tests

## Decision
**Approach**: Comprehensive pytest test suite covering all detection methods, edge cases, and quality overrides
**Why**: OutlierDetector is critical for data quality - false negatives could allow dangerous data through, false positives could reject legitimate medical interventions
**Risk Level**: High - Incorrect outlier detection impacts patient safety

## Implementation Steps

1. **Test Structure Setup** - Create `tests/test_outlier_detection.py` with fixtures
2. **Statistical Methods** - Test IQR, MAD, temporal consistency algorithms
3. **Quality Override Logic** - Verify high-quality data bypasses detection
4. **Kalman Deviation** - Test 15% threshold with mock database states
5. **Batch Processing** - Test minimum measurement requirements and batch behavior
6. **Feature Toggle Integration** - Verify feature flags control detection methods
7. **Edge Cases** - Handle empty data, identical values, extreme outliers

## Files to Change
- `tests/test_outlier_detection.py` - New comprehensive test suite
- `tests/fixtures/outlier_test_data.py` - Reusable test data generators

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

## Test Categories

### 1. Statistical Method Tests
```python
# test_iqr_detection()
- Normal distribution: detect outliers at 1.5*IQR bounds
- Small dataset (<4 points): returns empty set
- All identical values: no outliers
- Single extreme outlier: correctly identified
- Multiple outliers: all identified

# test_mad_detection()
- Modified z-score calculation with median/MAD
- Zero MAD case (identical values): returns empty
- Symmetric outliers: both detected
- Z-score threshold validation (default 3.0)
- Small dataset (<3 points): returns empty set

# test_temporal_consistency()
- Rapid weight change (>30% in <24h): flagged
- Gradual change over time: not flagged
- Measurements <1hr apart: skipped
- Missing timestamps: handled gracefully
- Edge case: first measurement has no previous
```

### 2. Quality Override Tests
```python
# test_quality_score_override()
- High quality (>0.7): never marked as outlier
- Low quality (<0.7): subject to detection
- Accepted flag: protects from outlier marking
- Missing quality scores: normal detection applies

# test_quality_feature_toggle()
- quality_override disabled: no protection
- quality_override enabled: protection active
```

### 3. Kalman Deviation Tests
```python
# test_kalman_outlier_detection()
- Mock database with user states
- 15% deviation threshold validation
- No user state: returns empty set
- State history navigation: finds closest snapshot
- Numpy array handling in states
- Datetime comparison edge cases
```

### 4. Batch Processing Tests
```python
# test_minimum_measurements()
- <5 measurements: returns empty (no analysis)
- Exactly 5: analysis proceeds
- Large batch (1000+): performance acceptable

# test_detect_outliers_integration()
- AND logic: statistical AND Kalman failures required
- Protected indices never in final output
- Empty input: returns empty set
- Mixed quality scores: correct filtering
```

### 5. Feature Toggle Tests
```python
# test_feature_manager_integration()
- outlier_detection disabled: always returns empty
- Individual method toggles (iqr, mad, temporal)
- Kalman deviation toggle independent
- Method combination with feature flags
```

### 6. Edge Cases
```python
# test_edge_cases()
- Empty measurements list: no crash, empty result
- Single measurement: no outliers
- All measurements identical: no outliers
- Extreme outlier (10x median): detected
- NaN/None values: handled gracefully
- Negative weights: processed correctly
- Zero weights: handled appropriately
```

### 7. Analysis and Utility Tests
```python
# test_analyze_outliers()
- Correct statistics calculation
- Outlier details populated correctly
- Percentage calculation accurate
- Context from neighbors included

# test_get_clean_measurements()
- Outliers removed from output
- Original list unchanged
- Chronological order preserved
- Indices mapping correct

# test_config_updates()
- Dynamic threshold updates
- Config persistence
- Default values when missing
```

## Test Data Fixtures
```python
@pytest.fixture
def normal_measurements():
    """10 measurements with normal variation"""

@pytest.fixture
def measurements_with_outliers():
    """Dataset with known statistical outliers"""

@pytest.fixture
def high_quality_measurements():
    """Measurements with quality_score > 0.7"""

@pytest.fixture
def mock_kalman_state():
    """Mock database with user Kalman states"""

@pytest.fixture
def temporal_test_data():
    """Time-series with rapid changes"""
```

## Acceptance Criteria
- [ ] All statistical methods tested independently
- [ ] Quality override logic verified with feature toggles
- [ ] Kalman deviation threshold (15%) validated
- [ ] Batch processing minimum (5) enforced
- [ ] AND logic for final outlier determination tested
- [ ] Edge cases handle gracefully without crashes
- [ ] Feature toggles control method activation
- [ ] Performance <100ms for 1000 measurements

## Risks & Mitigations
**Main Risk**: False negatives allowing dangerous weight changes through
**Mitigation**: Test extreme cases explicitly, verify 15% Kalman threshold catches medical emergencies

**Secondary Risk**: False positives rejecting legitimate medical interventions
**Mitigation**: Quality score override tested thoroughly, high-quality data always passes

## Out of Scope
- Performance optimization testing
- Database integration tests (use mocks)
- Visualization of outlier detection
- Real patient data testing