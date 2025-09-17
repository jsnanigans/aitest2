# Unit Test Priority Report - Weight Processing System

## Critical Components Requiring Unit Tests (Highest Priority)

### 1. **KalmanFilterManager** (src/processing/kalman.py:17)
**Priority: CRITICAL**
- Core of the entire processing pipeline
- **Test Cases:**
  - `test_initialize_immediate`: Verify correct initialization with first measurement
  - `test_predict_state`: Test state prediction with time delta calculations
  - `test_update_state`: Validate Kalman update equations
  - `test_adaptive_parameters`: Test noise parameter adaptation after resets
  - `test_observation_covariance_calculation`: Verify source-based noise multipliers
  - `test_edge_cases`: Handle extreme weights, negative time deltas, NaN values

### 2. **ResetManager** (src/processing/reset_manager.py:27)
**Priority: CRITICAL**
- Handles state transitions that affect entire processing chain
- **Test Cases:**
  - `test_initial_reset_detection`: First measurement triggers INITIAL reset
  - `test_hard_reset_30_day_gap`: Verify 30+ day gap detection
  - `test_soft_reset_manual_entry`: Test manual data entry reset logic
  - `test_reset_priority_order`: Ensure correct precedence (INITIAL > HARD > SOFT)
  - `test_adaptation_decay`: Verify adaptation parameters decay over time
  - `test_reset_timestamp_tracking`: Ensure reset timestamps are properly stored

### 3. **QualityScorer** (src/processing/quality_scorer.py:77)
**Priority: HIGH**
- Determines measurement acceptance/rejection
- **Test Cases:**
  - `test_safety_score_calculation`: Verify physiological limits scoring
  - `test_plausibility_score`: Test BMI and change rate validation
  - `test_consistency_score`: Validate temporal consistency checks
  - `test_reliability_score`: Test source-based reliability scoring
  - `test_overall_score_weighting`: Verify component weighting logic
  - `test_quality_override_threshold`: Test high-quality override behavior
  - `test_score_with_no_history`: Handle first measurement scoring

### 4. **OutlierDetector** (src/processing/outlier_detection.py:20)
**Priority: HIGH**
- Critical for data quality
- **Test Cases:**
  - `test_iqr_detection`: Test interquartile range outlier detection
  - `test_mad_detection`: Verify median absolute deviation method
  - `test_temporal_consistency`: Test rate-of-change detection
  - `test_batch_analysis`: Verify batch processing behavior
  - `test_quality_score_override`: High quality scores bypass outlier detection
  - `test_minimum_measurements`: Handle cases with < 5 measurements
  - `test_kalman_deviation_check`: Test 15% deviation threshold

### 5. **ProcessorStateDB** (src/database/database.py:26)
**Priority: HIGH**
- State persistence is critical for continuity
- **Test Cases:**
  - `test_save_and_retrieve_state`: Basic CRUD operations
  - `test_numpy_array_serialization`: Verify array conversion to/from JSON
  - `test_transaction_atomicity`: Test transaction commit/rollback
  - `test_state_migration`: Handle schema changes
  - `test_concurrent_access`: Multiple users accessing states
  - `test_measurement_history_buffer`: Verify 30-measurement limit
  - `test_backup_restore`: Test state backup/restoration

## Secondary Components (Medium Priority)

### 6. **PhysiologicalValidator** (src/processing/validation.py:43)
**Priority: MEDIUM**
- Safety validation layer
- **Test Cases:**
  - `test_absolute_limits`: Reject weights < 20kg or > 700kg
  - `test_suspicious_ranges`: Flag weights in suspicious ranges
  - `test_height_validation`: Validate height constraints
  - `test_safety_score_calculation`: Verify scoring gradients

### 7. **BMIValidator** (src/processing/validation.py:145)
**Priority: MEDIUM**
- Data quality check
- **Test Cases:**
  - `test_bmi_calculation`: Verify BMI formula
  - `test_bmi_limits`: Check < 10 and > 90 BMI rejection
  - `test_missing_height`: Handle missing height gracefully
  - `test_bmi_plausibility_score`: Test scoring logic

### 8. **FeatureManager** (src/feature_manager.py:11)
**Priority: MEDIUM**
- Controls system behavior
- **Test Cases:**
  - `test_feature_dependencies`: Verify dependency resolution
  - `test_mandatory_features`: Ensure mandatory features can't be disabled
  - `test_feature_toggle`: Test enable/disable behavior
  - `test_invalid_feature_names`: Handle unknown features

## Lower Priority Components

### 9. **DataQualityPreprocessor** (src/processing/validation.py:446)
**Priority: LOW**
- Data cleanup utility
- **Test Cases:**
  - `test_duplicate_removal`: Verify exact timestamp duplicate handling
  - `test_source_normalization`: Test source name standardization
  - `test_data_sorting`: Ensure chronological ordering

### 10. **Visualization** (src/viz/visualization.py)
**Priority: LOW**
- Output only, not critical for processing
- **Test Cases:**
  - `test_plot_generation`: Verify basic plot creation
  - `test_data_aggregation`: Test grouping and stats

## Integration Test Requirements

### Critical Integration Tests:
1. **End-to-end processing pipeline**: CSV → Processor → Database → Output
2. **Reset scenario testing**: Test all reset types with real data flows
3. **Quality override scenarios**: High-quality measurements overriding outlier detection
4. **Kalman adaptation flow**: Reset → Adaptation → Decay cycle
5. **Feature toggle combinations**: Test different feature combinations

## Test Data Requirements

### Essential Test Fixtures:
- Normal weight progression (60-80kg over time)
- Extreme values (15kg, 800kg)
- Rapid changes (>10% in one day)
- Long gaps (>30 days)
- Mixed source reliability data
- BMI edge cases (height missing, extreme BMIs)
- Duplicate timestamps
- Out-of-order measurements

## Coverage Goals

- **Critical Components**: >95% coverage
- **Secondary Components**: >80% coverage
- **Edge Cases**: 100% coverage for safety-critical paths
- **Integration Tests**: Cover all major data flows

## Pytest Best Practices Requirements

### Mandatory Implementation Standards for All Tests

1. **Fixtures** (`@pytest.fixture`)
   - Create reusable fixtures in conftest.py for test data and configurations
   - Use fixture composition for complex test scenarios
   - Implement proper fixture scoping (function, class, module, session)

2. **Parametrization** (`@pytest.mark.parametrize`)
   - Test multiple input combinations with single test methods
   - Include edge cases and boundary values in parameters
   - Use descriptive parameter IDs for clear test output

3. **Markers** (`@pytest.mark`)
   - `@pytest.mark.slow` for long-running tests (>1s)
   - `@pytest.mark.unit` for pure unit tests
   - `@pytest.mark.integration` for integration tests
   - `@pytest.mark.critical` for safety-critical tests

4. **Assertions**
   - Use `pytest.approx()` for all floating-point comparisons
   - Include descriptive assertion messages with f-strings
   - Never use bare `assert` without message

5. **Test Organization**
   - Group related tests in classes
   - Follow naming: `test_<component>_<scenario>_<expected>`
   - Use docstrings with Given/When/Then format

6. **Mock Usage**
   - Use `unittest.mock.MagicMock` for complex objects
   - Mock all external dependencies
   - Verify mock calls with assertions

7. **Error Testing**
   - Use `pytest.raises` for exception testing
   - Test specific error messages and types
   - Validate error recovery paths

## Recommended Testing Framework

```python
# conftest.py - Shared fixtures and configuration
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
from datetime import datetime, timedelta

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "critical: marks tests as safety-critical")

@pytest.fixture
def kalman_config():
    """Standard Kalman configuration for testing."""
    return {
        'initial_variance': 100,
        'observation_covariance': 1.0,
        'transition_covariance_weight': 0.1,
        'transition_covariance_trend': 0.001
    }

@pytest.fixture
def sample_state():
    """Sample Kalman filter state."""
    return {
        'kalman_params': {
            'initial_state_mean': [70.0, 0.0],
            'initial_state_covariance': [[1.0, 0.0], [0.0, 0.001]],
            'transition_covariance': [[0.1, 0.0], [0.0, 0.001]],
            'observation_covariance': [[1.0]]
        },
        'last_state': np.array([[70.0, 0.0]]),
        'last_covariance': np.array([[[1, 0], [0, 0.001]]]),
        'last_timestamp': datetime(2024, 1, 1, 10, 0),
        'last_raw_weight': 70.0
    }

@pytest.fixture
def base_timestamp():
    """Consistent base timestamp for tests."""
    return datetime(2024, 1, 1, 10, 0)

@pytest.fixture
def sample_weights():
    """Realistic weight sequence."""
    return [70.0, 70.5, 69.8, 70.2, 70.1]

@pytest.fixture
def numpy_random_seed():
    """Set reproducible random seed."""
    np.random.seed(42)
    yield
    np.random.seed()  # Reset after test
```

## Next Steps

1. Start with KalmanFilterManager tests (highest risk if broken)
2. Add ResetManager tests (complex state logic)
3. Implement QualityScorer tests (acceptance/rejection critical)
4. Add integration tests for complete workflows
5. Set up continuous testing with coverage reports