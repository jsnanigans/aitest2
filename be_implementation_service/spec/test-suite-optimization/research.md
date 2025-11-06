# Test Suite Optimization Research

## Executive Summary

**Current State**: 35 tests (594 LOC) covering only buffered replay functionality
**Critical Gap**: 0% coverage of core processing logic (processor, Kalman, quality scoring, reset, validation)
**Recommendation**: Keep most existing tests, add 40-60 focused tests for critical paths

---

## 1. Current Test Suite Analysis

### 1.1 What Exists (tests/unit/services/test_weight_processor_service.py)

| Test Class | Tests | Focus | Value Assessment |
|------------|-------|-------|------------------|
| `TestShould TriggerReplay` | 13 | Replay trigger conditions | **HIGH** - Critical business logic |
| `TestExecuteBufferedReplay` | 5 | Replay execution | **HIGH** - Integration point |
| `TestMergeReplayResults` | 8 | Result merging | **MEDIUM** - Complex but stable |
| `TestSnapshotCreation` | 3 | Snapshot timing | **HIGH** - Data integrity critical |

### 1.2 Test Quality Assessment

**Strengths:**
- ✅ Tests document behavior well (clear docstrings with test IDs)
- ✅ Good fixture design (reusable `service`, `base_timestamp`, helpers)
- ✅ Comprehensive coverage of replay trigger conditions
- ✅ Tests edge cases (empty buffer, exactly at threshold, multiple conditions)

**Weaknesses:**
- ❌ Test names could be more descriptive (e.g., `test_empty_buffer_returns_false` → `test_replay_not_triggered_when_buffer_is_empty`)
- ❌ Some tests verify implementation details (checking exact call_args structure)
- ❌ Missing tests for error scenarios in merge logic
- ❌ No tests for replay failure recovery

**Redundancy Analysis:**
- Minor: `test_multiple_conditions_priority` (line 125) overlaps with other trigger tests
- All other tests provide unique value

**Verdict**: Keep 32/35 tests, improve 10 test names, add error scenario coverage

---

## 2. Critical Code Analysis

### 2.1 Core Processing Pipeline (`src/core/processing/processor.py`)

**Lines of Code**: 720
**Current Test Coverage**: 0%
**Criticality**: 🔴 CRITICAL - Every measurement flows through this

**Key Functions (untested)**:
| Function | Lines | Criticality | Common Edge Cases |
|----------|-------|-------------|-------------------|
| `process_measurement()` | 121-575 | 🔴 CRITICAL | Null state, first measurement, reset scenarios |
| `_handle_reset_with_transaction()` | 603-647 | 🔴 CRITICAL | Circuit breaker open, transaction rollback |
| `_perform_transactional_reset()` | 649-720 | 🔴 CRITICAL | Validation failures, partial state |
| `_maybe_create_periodic_snapshot()` | 48-119 | 🟡 IMPORTANT | No snapshot exists, time calculations |

**Critical Paths**:
1. **Happy path**: Clean data → Kalman init/update → Quality scoring → State save
2. **Reset path**: Gap detected → Transaction → Kalman reset → Adaptive params
3. **Rejection path**: Bad data → Preprocessing fails → Return rejection
4. **Snapshot path**: Accepted → Check interval → Create snapshot

**Recommended Tests**: 12-15 tests
- Process first measurement (Kalman initialization)
- Process subsequent measurement (Kalman update)
- Process measurement triggering hard reset (30+ day gap)
- Process measurement triggering soft reset (manual entry with 5kg change)
- Preprocessing rejection (invalid weight/unit)
- Quality scoring rejection (low score)
- State persistence after accepted measurement
- Snapshot creation after reset
- Periodic snapshot creation (24h interval)
- Transaction rollback on reset failure
- Circuit breaker behavior (3 failures → open)
- Measurement with user_height_m parameter

---

### 2.2 Kalman Filter (`src/core/processing/kalman.py`)

**Lines of Code**: ~300 (excluding reset manager code)
**Current Test Coverage**: 0%
**Criticality**: 🔴 CRITICAL - Core algorithm for weight estimation

**Key Functions**:
| Function | Lines | Criticality | Edge Cases |
|----------|-------|-------------|------------|
| `initialize_immediate()` | 36-85 | 🔴 CRITICAL | Custom obs_covariance, config params |
| `update_state()` | 86-176 | 🔴 CRITICAL | Time delta edge cases (0.1 - 30 days) |
| `predict_next_state()` | 294-366 | 🔴 CRITICAL | No prior state, long time gaps |
| `get_adaptive_kalman_params()` | 426-508 | 🟡 IMPORTANT | Reset params, decay calculations |

**What Makes This Critical**:
- Maintains state across measurements
- Handles time gaps (0.1 - 30+ days)
- Adaptive parameters after resets
- Trend limiting (±5kg/week max)

**Recommended Tests**: 10-12 tests
- Initialize with first measurement
- Update state with normal time delta (1 day)
- Update state with extreme time deltas (0.1 day, 30 days)
- Predict next state (used by quality scorer)
- Adaptive params calculation (within 7 days of reset)
- Trend limiting (clamp to ±0.714 kg/day)
- State shape handling (1D vs 2D arrays from DB)
- Decimal to float conversion (DynamoDB types)
- Get current state values
- Calculate confidence from normalized innovation

---

### 2.3 Quality Scoring (`src/core/processing/unified_quality_scorer.py`)

**Lines of Code**: 1051
**Current Test Coverage**: 0%
**Criticality**: 🔴 CRITICAL - Determines accept/reject

**Key Components**:
| Component | Weight | Criticality | Edge Cases |
|-----------|--------|-------------|------------|
| `calculate_kalman_fit()` | 40% | 🔴 CRITICAL | No prediction, adaptive period, time decay |
| `calculate_temporal_consistency()` | 30% | 🔴 CRITICAL | No previous weight, large time gaps |
| `calculate_anomaly_detection()` | 20% | 🔴 CRITICAL | Absolute limits, rapid measurements, burst patterns |
| `calculate_source_reliability()` | 5% | 🟢 LOW | Simple lookup |
| `calculate_trend_alignment()` | 5% | 🟢 LOW | < 5 measurements |

**Critical Thresholds** (nail these down with tests):
- Overall acceptance threshold: 0.46
- Duplicate detection: < 5 seconds
- Rapid change: < 5 minutes
- Burst pattern: 5+ measurements in 30 minutes
- Absolute limits: 20-300 kg
- Suspicious limits: 35-250 kg

**Recommended Tests**: 15-18 tests
- Overall quality score calculation (weighted geometric mean)
- Kalman fit: perfect prediction (score ~1.0)
- Kalman fit: 3σ deviation (score ~0.2)
- Kalman fit: time decay for 30-day gap
- Kalman fit: adaptive period (relaxed thresholds)
- Temporal consistency: acceptable change (1kg in 1 day)
- Temporal consistency: excessive change (5kg in 1 hour)
- Anomaly: absolute min violated (< 20kg)
- Anomaly: absolute max violated (> 300kg)
- Anomaly: duplicate detection (same weight < 5 sec)
- Anomaly: rapid impossible change (2kg in 1 min)
- Anomaly: burst pattern detection (6 measurements in 30 min)
- Anomaly: physiological change limits by time
- Source reliability scoring
- Trend alignment (sufficient/insufficient data)
- Rejection reason generation
- Update temporal baseline

---

### 2.4 Reset Logic (`src/core/processing/reset_manager.py`)

**Lines of Code**: 333
**Current Test Coverage**: 0%
**Criticality**: 🔴 CRITICAL - Wrong reset = corrupted state for days

**Key Functions**:
| Function | Criticality | Edge Cases |
|----------|-------------|------------|
| `should_trigger_reset()` | 🔴 CRITICAL | Priority order (INITIAL → HARD → SOFT) |
| `perform_reset()` | 🔴 CRITICAL | State preservation, parameter calculation |
| `get_reset_parameters()` | 🟡 IMPORTANT | Config merging, defaults |
| `is_in_adaptive_period()` | 🟡 IMPORTANT | Time vs measurement-based |

**Reset Types & Thresholds**:
1. **INITIAL**: No Kalman params → Most aggressive adaptation
2. **HARD**: 30+ day gap → Aggressive adaptation
3. **SOFT**: Manual source + 5kg change + 3-day cooldown → Gentle adaptation

**Critical Parameters**:
```
INITIAL: initial_variance_multiplier=10, weight_noise=50x, trend_noise=500x
HARD:    initial_variance_multiplier=5,  weight_noise=20x, trend_noise=200x
SOFT:    initial_variance_multiplier=2,  weight_noise=20x, trend_noise=200x
```

**Recommended Tests**: 8-10 tests
- Should trigger INITIAL reset (no kalman_params)
- Should trigger HARD reset (31 day gap)
- Should NOT trigger HARD reset (29 day gap)
- Should trigger SOFT reset (manual source, 6kg change, no recent reset)
- Should NOT trigger SOFT reset (4kg change - below threshold)
- Should NOT trigger SOFT reset (within 3-day cooldown)
- Reset priority order (INITIAL > HARD > SOFT)
- Perform reset: state cleared correctly
- Perform reset: parameters applied correctly
- Is in adaptive period (within 7 days AND < 10 measurements)

---

### 2.5 Data Validation (`src/core/processing/validation.py`)

**Lines of Code**: 729
**Current Test Coverage**: 0%
**Criticality**: 🟡 IMPORTANT - Safety gate for bad data

**Key Functions**:
| Function | Criticality | Edge Cases |
|----------|-------------|------------|
| `DataQualityPreprocessor.preprocess()` | 🔴 CRITICAL | Unit conversion, BMI validation |
| `PhysiologicalValidator.validate_absolute_limits()` | 🔴 CRITICAL | Min/max bounds |
| `BMIValidator.detect_and_convert()` | 🟡 IMPORTANT | BMI confusion detection |

**Critical Validations**:
- Absolute limits: 20-300 kg
- BMI limits: 10-80 (impossible), 12-65 (suspicious)
- Supported units: kg, lb/lbs, g, st (stone)
- Unit conversion accuracy

**Recommended Tests**: 8-10 tests
- Preprocess: valid kg input
- Preprocess: lb to kg conversion
- Preprocess: stone to kg conversion
- Preprocess: reject missing unit
- Preprocess: reject unsupported unit (e.g., "bmi")
- Preprocess: reject impossible BMI (5.0 with height 1.75m)
- Validate absolute limits: min (19kg)
- Validate absolute limits: max (301kg)
- BMI detection: value in BMI range (15-50)
- Rate of change validation

---

### 2.6 API Models (`src/aws/api/models.py`)

**Lines of Code**: 310
**Current Test Coverage**: 0%
**Criticality**: 🟡 IMPORTANT - API contract with clients

**Key Models**:
| Model | Criticality | Validation Rules |
|-------|-------------|------------------|
| `Measurement` | 🔴 CRITICAL | weight > 0, unit validation, timestamp |
| `MeasurementResult` | 🟡 IMPORTANT | quality_score ∈ [0,1] |
| `ProcessResponseData` | 🟡 IMPORTANT | counts consistency |
| `HistoricalConflictResponse` | 🟡 IMPORTANT | Conflict detection logic |

**Recommended Tests**: 6-8 tests
- Measurement: valid input with aliases
- Measurement: unit normalization (lb/lbs/pound → lbs)
- Measurement: reject invalid unit
- Measurement: reject negative weight
- Measurement: weight range validation (convert to kg, check 10-500 kg)
- MeasurementResult: quality_score bounds
- ProcessResponseData: counts match results length
- HistoricalConflictResponse: serialization

---

### 2.7 Replay Service (`src/aws/services/replay_service.py`)

**Lines of Code**: 98
**Current Test Coverage**: ~40% (indirectly via service tests)
**Criticality**: 🟡 IMPORTANT - Historical data correctness

**Current Coverage via Service Tests**:
- ✅ Replay measurements called with correct params (test_correct_parameters_passed_to_replay_measurements:219)
- ✅ Replay success handling (test_successful_replay_returns_expected_dict:169)
- ✅ Replay failure handling (test_replay_service_exception_is_propagated:202, test_replay_failure_status_raises_exception:253)

**Gaps**:
- ❌ Snapshot restoration logic
- ❌ Measurement filtering (>= replay_from)
- ❌ State deletion when no snapshot
- ❌ Result structure consistency

**Recommended Tests**: 3-4 tests
- Replay with snapshot available (restore from snapshot)
- Replay without snapshot (delete state, start fresh)
- Replay filters measurements by timestamp
- Replay creates snapshot after completion

---

## 3. Test Coverage Gaps Summary

| Module | LOC | Current Tests | Recommended Tests | Priority |
|--------|-----|---------------|-------------------|----------|
| processor.py | 720 | 0 | 12-15 | 🔴 CRITICAL |
| kalman.py | 300 | 0 | 10-12 | 🔴 CRITICAL |
| unified_quality_scorer.py | 1051 | 0 | 15-18 | 🔴 CRITICAL |
| reset_manager.py | 333 | 0 | 8-10 | 🔴 CRITICAL |
| validation.py | 729 | 0 | 8-10 | 🟡 IMPORTANT |
| models.py | 310 | 0 | 6-8 | 🟡 IMPORTANT |
| replay_service.py | 98 | ~14 (indirect) | 3-4 | 🟢 LOW |
| weight_processor_service.py | ~500 | 35 | 32 (keep existing) | ✅ GOOD |

**Total Recommended**: 32 (keep) + 62-77 (new) = **94-109 tests**

---

## 4. Common Edge Cases Across Modules

### 4.1 Time Handling
- First measurement (no last_timestamp)
- Extremely short gaps (< 1 minute)
- Extremely long gaps (> 30 days)
- Exact boundary conditions (exactly 30 days, exactly 24 hours)
- Timezone handling (all should be UTC-aware)

### 4.2 State Handling
- No state (first ever measurement)
- State with no Kalman params (after reset)
- State from DynamoDB (Decimal types)
- State array shapes (1D vs 2D from DB serialization)
- Missing fields in state

### 4.3 Data Type Edge Cases
- Decimal from DynamoDB → float conversion
- None/null values
- String timestamps → datetime conversion
- Array shapes from numpy

### 4.4 Configuration Edge Cases
- Missing config keys (use defaults)
- Feature flags disabled
- Custom reset parameters
- Adaptive period settings

---

## 5. Testing Strategy Recommendations

### 5.1 Test Organization
```
tests/
├── unit/
│   ├── processing/
│   │   ├── test_processor.py           # 12-15 tests
│   │   ├── test_kalman.py              # 10-12 tests
│   │   ├── test_quality_scorer.py      # 15-18 tests
│   │   ├── test_reset_manager.py       # 8-10 tests
│   │   └── test_validation.py          # 8-10 tests
│   ├── api/
│   │   └── test_models.py              # 6-8 tests
│   └── services/
│       ├── test_weight_processor_service.py  # 32 tests (existing, improved)
│       └── test_replay_service.py      # 3-4 tests
```

### 5.2 Fixture Strategy
```python
# conftest.py - shared fixtures
@pytest.fixture
def base_config():
    """Standard config for tests."""
    return ConfigManager.load_config()

@pytest.fixture
def base_timestamp():
    """Standard timestamp for tests."""
    return datetime(2025, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

@pytest.fixture
def mock_state_store():
    """Mock state store with common methods."""
    return Mock(spec=StateStore)

@pytest.fixture
def clean_state():
    """Fresh state for first measurement."""
    return {
        "kalman_params": None,
        "measurement_history": [],
        "measurements_since_reset": 0,
    }

@pytest.fixture
def initialized_state(base_timestamp):
    """State after first measurement processed."""
    return {
        "kalman_params": {...},
        "last_state": np.array([[70.0, 0.0]]),
        "last_covariance": np.array([[[0.361, 0], [0, 0.001]]]),
        "last_timestamp": base_timestamp,
        "last_raw_weight": 70.0,
        "measurements_since_reset": 1,
        "measurement_history": [],
    }
```

### 5.3 Test Naming Convention
```python
# CURRENT (not ideal):
def test_empty_buffer_returns_false(service, base_timestamp):

# BETTER:
def test_replay_not_triggered_when_buffer_is_empty(service, base_timestamp):

# EVEN BETTER (for critical logic):
def test_replay_not_triggered_when_buffer_empty_because_minimum_2_measurements_required(service, base_timestamp):
```

### 5.4 Mocking Strategy
- **Mock external dependencies**: state_store, config
- **Don't mock units under test**: Kalman, quality scorer, reset manager
- **Use real data structures**: numpy arrays, datetimes
- **Patch sparingly**: Only for non-deterministic operations (time.time, datetime.now)

### 5.5 Test Data Strategy
```python
# Good: Use realistic values
weight = 70.0  # kg, typical adult
height = 1.75  # m, typical adult
bmi = 22.9  # normal range

# Good: Use boundary values for edge cases
min_weight = 20.0  # absolute minimum
max_weight = 300.0  # absolute maximum
reset_gap_threshold = 30.0  # days

# Good: Use named constants
from src.core.constants import PHYSIOLOGICAL_LIMITS
assert weight < PHYSIOLOGICAL_LIMITS["ABSOLUTE_MAX_WEIGHT"]
```

---

## 6. Risks & Mitigation

### 6.1 Risk: Breaking Existing Behavior
**Mitigation**:
- Keep all existing tests (32/35)
- Run full test suite after each new test addition
- Add tests that document current behavior first, then optimize

### 6.2 Risk: Over-Testing Implementation Details
**Mitigation**:
- Test behavior, not implementation (don't check internal state unless critical)
- Focus on inputs → outputs
- Only test private methods if they're complex and critical

### 6.3 Risk: Slow Test Suite
**Mitigation**:
- All tests should be unit tests (no DB, no external APIs)
- Use mocks for state_store
- Target: < 10 seconds for full suite

### 6.4 Risk: False Confidence from Tests
**Mitigation**:
- Test realistic scenarios, not just happy paths
- Include tests for common production bugs
- Test boundary conditions, not just middle-range values

---

## 7. Implementation Priority

### Phase 1: Critical Safety Tests (Week 1)
**Goal**: Prevent data corruption and incorrect medical decisions

1. `test_processor.py`: Basic process flow (5 tests)
   - First measurement initialization
   - Subsequent measurement update
   - Preprocessing rejection
   - Quality rejection
   - State persistence

2. `test_reset_manager.py`: Reset detection (5 tests)
   - INITIAL, HARD, SOFT trigger conditions
   - Priority order
   - Parameter application

3. `test_validation.py`: Data safety (5 tests)
   - Absolute limits
   - Unit validation
   - BMI validation

**Total**: 15 tests, ~300 LOC

### Phase 2: Algorithm Correctness (Week 2)
**Goal**: Ensure Kalman filter and quality scoring work correctly

1. `test_kalman.py`: Filter operations (8 tests)
   - Initialize, update, predict
   - Time delta handling
   - Adaptive parameters

2. `test_quality_scorer.py`: Scoring logic (10 tests)
   - Overall score calculation
   - Kalman fit, temporal consistency, anomaly detection
   - Critical thresholds

**Total**: 18 tests, ~400 LOC

### Phase 3: Edge Cases & Polish (Week 3)
**Goal**: Handle edge cases and improve existing tests

1. Complete remaining tests for all modules
2. Improve existing test names (10 tests)
3. Add error scenario tests for replay
4. API model validation tests

**Total**: 35-40 tests, ~600 LOC

### Phase 4: Documentation & Cleanup
**Goal**: Make tests serve as documentation

1. Add comprehensive docstrings to all tests
2. Create test README explaining organization
3. Add examples of how to run specific test groups

---

## 8. Success Metrics

### Quantitative
- ✅ 100% of critical functions have >= 1 test
- ✅ All common edge cases covered (per section 4)
- ✅ Test suite runs in < 15 seconds
- ✅ 94-109 total tests

### Qualitative
- ✅ Tests immediately catch if behavior changes
- ✅ Test names document expected behavior
- ✅ New developers can understand system by reading tests
- ✅ Confidence to refactor critical code

---

## 9. References

**Code Locations**:
- Current tests: `tests/unit/services/test_weight_processor_service.py`
- Processor: `src/core/processing/processor.py:121-575`
- Kalman: `src/core/processing/kalman.py:32-366`
- Quality: `src/core/processing/unified_quality_scorer.py:126-223`
- Reset: `src/core/processing/reset_manager.py:33-333`
- Validation: `src/core/processing/validation.py:578-729`
- API: `src/aws/api/models.py:42-100`
- Replay: `src/aws/services/replay_service.py:13-98`

**Key Constants** (from `src/core/constants.py`):
- PHYSIOLOGICAL_LIMITS: {ABSOLUTE_MIN_WEIGHT: 20, ABSOLUTE_MAX_WEIGHT: 300, ...}
- BMI_LIMITS: {IMPOSSIBLE_LOW: 10, IMPOSSIBLE_HIGH: 80, ...}
- KALMAN_DEFAULTS: {observation_covariance: 3.49, ...}
