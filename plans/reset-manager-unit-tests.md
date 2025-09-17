# Plan: ResetManager Unit Tests

## Decision
**Approach**: Comprehensive test suite with 100% coverage of reset detection, priority, and adaptation logic
**Why**: ResetManager is critical for state transitions affecting entire processing chain - bugs cascade downstream
**Risk Level**: High (incorrect resets corrupt Kalman state and measurement processing)

## Implementation Steps

1. **Create test file** - Add `tests/test_reset_manager.py` with pytest framework
2. **Test reset detection** - Validate all three reset types trigger correctly
3. **Test priority ordering** - Verify INITIAL > HARD > SOFT when multiple conditions exist
4. **Test adaptation parameters** - Confirm correct parameters loaded per reset type
5. **Test decay calculations** - Validate adaptive factor calculation and timing
6. **Test state transitions** - Ensure state updates maintain invariants
7. **Test edge cases** - Handle missing state, corrupted timestamps, boundary conditions

## Files to Change
- `tests/test_reset_manager.py` - New comprehensive test suite (~600 lines)
- No modifications to `src/processing/reset_manager.py` unless bugs found

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

## Test Structure

### Test Classes & Methods

#### TestResetDetection
- `test_initial_reset_no_state()` - Empty state triggers INITIAL
- `test_initial_reset_no_kalman_params()` - Missing kalman_params triggers INITIAL
- `test_hard_reset_gap_detection()` - 30+ day gap triggers HARD
- `test_hard_reset_threshold_boundary()` - Test exactly 30 days vs 29.99 days
- `test_soft_reset_manual_sources()` - Each manual source triggers SOFT with 5kg+ change
- `test_soft_reset_weight_change_threshold()` - Test exactly 5kg vs 4.99kg change
- `test_soft_reset_cooldown()` - Verify cooldown prevents repeated soft resets
- `test_no_reset_normal_conditions()` - Verify no false positives

#### TestResetPriority
- `test_initial_beats_hard()` - No kalman_params + 30 day gap = INITIAL
- `test_initial_beats_soft()` - No kalman_params + manual source = INITIAL
- `test_hard_beats_soft()` - 30 day gap + manual source = HARD
- `test_all_conditions_met()` - All three conditions = INITIAL

#### TestResetParameters
- `test_initial_parameters_defaults()` - Verify default INITIAL parameters
- `test_hard_parameters_defaults()` - Verify default HARD parameters
- `test_soft_parameters_defaults()` - Verify default SOFT parameters
- `test_parameters_from_config()` - Config overrides defaults correctly
- `test_partial_config_override()` - Missing config keys use defaults

#### TestAdaptationBehavior
- `test_is_in_adaptive_period_measurements()` - True until measurement threshold
- `test_is_in_adaptive_period_days()` - True until day threshold
- `test_adaptive_factor_decay_curve()` - Factor increases 0->1 with decay rate
- `test_adaptive_factor_boundary_values()` - Factor clamped to [0,1]
- `test_adaptive_factor_no_reset()` - Returns 1.0 without reset state

#### TestStateTransitions
- `test_perform_reset_initial()` - Initial reset creates clean state
- `test_perform_reset_preserves_history()` - Reset events accumulate
- `test_reset_event_structure()` - Event contains all required fields
- `test_reset_timestamp_tracking()` - Timestamps properly converted and stored
- `test_reset_reason_generation()` - Human-readable reasons for each type

#### TestEdgeCases
- `test_string_timestamp_conversion()` - ISO string timestamps handled
- `test_missing_state_fields()` - Graceful handling of incomplete state
- `test_corrupted_reset_events()` - Handle malformed reset_events list
- `test_config_missing_sections()` - Missing config sections use defaults
- `test_simultaneous_reset_conditions()` - Multiple triggers = highest priority
- `test_manual_source_variations()` - All MANUAL_DATA_SOURCES recognized
- `test_custom_trigger_sources()` - Config trigger_sources extend defaults
- `test_timezone_aware_timestamps()` - Handle timezone-aware datetime objects
- `test_negative_gap_days()` - Future timestamps don't crash
- `test_extreme_weight_changes()` - Handle 100kg+ changes gracefully

### Test Fixtures

```python
@pytest.fixture
def base_config():
    """Standard config with all reset types enabled"""
    return {
        'kalman': {
            'reset': {
                'initial': {'enabled': True},
                'hard': {'enabled': True, 'gap_threshold_days': 30},
                'soft': {'enabled': True, 'min_weight_change_kg': 5}
            }
        }
    }

@pytest.fixture
def valid_state():
    """State with existing Kalman parameters"""
    return {
        'kalman_params': {'weight': [75.0, 0.0]},
        'last_timestamp': datetime(2024, 1, 1),
        'last_raw_weight': 75.0,
        'measurements_since_reset': 5
    }

@pytest.fixture
def reset_state():
    """State immediately after reset"""
    return {
        'reset_timestamp': datetime(2024, 1, 15),
        'reset_type': 'soft',
        'reset_parameters': {...},
        'measurements_since_reset': 0
    }
```

## Acceptance Criteria
- [ ] All three reset types detect correctly with proper thresholds
- [ ] Priority ordering enforced when multiple conditions exist
- [ ] Adaptation parameters match configuration or defaults
- [ ] Decay calculations produce smooth 0->1 curve
- [ ] State transitions preserve critical fields and history
- [ ] Edge cases handled without exceptions
- [ ] 100% line coverage of ResetManager class
- [ ] Tests run in <2 seconds total

## Risks & Mitigations
**Main Risk**: Incorrect reset detection causing cascade failures in Kalman filter
**Mitigation**: Parameterized tests covering boundary conditions, property-based testing for invariants

**Secondary Risk**: State corruption during reset transitions
**Mitigation**: Assert state invariants after each reset, verify history preservation

## Out of Scope
- Integration with KalmanFilter class (separate integration tests)
- Database persistence of reset events
- Performance testing of reset detection
- Testing actual Kalman filter behavior post-reset