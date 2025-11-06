# Test Suite Optimization Specifications

## Goal
Optimize the weight_processor unit test suite to provide **100% coverage of critical code** with **minimal, high-value tests** that serve as behavioral documentation and immediately detect regressions.

## Philosophy
- **Less is more**: Minimal tests that cover critical paths
- **Behavior documentation**: Tests should document expected behavior
- **Regression detection**: Immediately catch if code works differently
- **Common edge cases only**: Skip obscure edge cases if not obviously critical
- **No integration tests**: Focus on unit tests only (no external APIs/live DB)

## Critical Code Areas (All Must Be Tested)

### 1. Replay Feature
**Location**: `src/aws/services/weight_processor_service.py`, `src/aws/services/replay_service.py`
**Why Critical**: Ensures historical data reprocessing maintains data integrity
**Current Coverage**: ~35 tests in `test_weight_processor_service.py` (594 lines)
**Assessment Needed**: Are current tests focused on most important behaviors?

### 2. Main Processing Pipeline
**Location**: `src/core/processing/processor.py` (720 lines)
**Why Critical**: Core business logic that processes every weight measurement
**Current Coverage**: 0 tests ❌
**Key Functions**:
- `process_measurement()` - Main entry point
- `_handle_reset_with_transaction()` - Reset logic
- `_perform_transactional_reset()` - Transaction safety
- `_maybe_create_periodic_snapshot()` - Snapshot creation

### 3. Data Integrity & Validation
**Location**: `src/core/processing/validation.py`, `src/core/processing/persistence_validator.py`
**Why Critical**: Prevents corrupt data from entering the system
**Current Coverage**: 0 tests ❌
**Key Components**:
- `DataQualityPreprocessor` - Input cleaning
- `PersistenceValidator` - State validation before save

### 4. Kalman Filter
**Location**: `src/core/processing/kalman.py`, `src/core/processing/kalman_filter.py`
**Why Critical**: Core algorithm for weight estimation and smoothing
**Current Coverage**: 0 tests ❌
**Key Components**:
- `KalmanFilterManager` - State management
- State initialization, prediction, update cycle
- Adaptive parameter adjustment

### 5. State Management & Database Logic
**Location**: `src/core/database/dynamodb_store.py`, state operations in processor
**Why Critical**: State corruption would poison all future measurements
**Current Coverage**: Partial (tested indirectly through replay tests)
**Key Operations**:
- State save/load
- Snapshot creation/retrieval
- State validation

### 6. Quality Scoring
**Location**: `src/core/processing/unified_quality_scorer.py`
**Why Critical**: Determines which measurements to accept/reject
**Current Coverage**: 0 tests ❌
**Key Components**:
- `UnifiedQualityScorer.calculate_quality_score()` - Main scoring logic
- Quality thresholds and rejection criteria

### 7. Reset Logic
**Location**: `src/core/processing/reset_manager.py`, `src/core/processing/reset_transaction.py`
**Why Critical**: Prevents bad state from poisoning measurements; transaction safety
**Current Coverage**: 0 tests ❌
**Key Components**:
- `ResetManager.should_trigger_reset()` - Reset detection
- `ResetManager.perform_reset()` - Reset execution
- `ResetTransaction` - Transaction safety

### 8. API Parameter & Response Formatting
**Location**: `src/aws/api/models.py`, `src/aws/lambda_handler.py`
**Why Critical**: Contract between service and clients
**Current Coverage**: 0 tests ❌
**Key Models**:
- `Measurement` - Input model
- `MeasurementResult` - Output model
- `ProcessResponseData` - Batch response
- API validation and serialization

## Current Test Suite Assessment

### Existing Tests
**File**: `tests/unit/services/test_weight_processor_service.py` (594 lines, ~35 tests)

**Test Classes**:
1. `TestShouldTriggerReplay` (13 tests)
   - Tests replay trigger conditions
   - Tests: empty buffer, time window, buffer size, is_last flag

2. `TestExecuteBufferedReplay` (5 tests)
   - Tests replay execution
   - Tests: success, exceptions, parameter passing

3. `TestMergeReplayResults` (8 tests)
   - Tests result merging logic
   - Tests: buffered vs non-buffered measurements, field updates

4. `TestSnapshotCreation` (3 tests)
   - Tests snapshot creation timing
   - Tests: before first measurement, rejected measurements, once per window

### Questions for Current Tests
1. **Are they focused on most critical behaviors?**
   - Need to verify these aren't testing implementation details
   - Check if test names clearly document expected behavior

2. **Are there redundant tests?**
   - Multiple tests for similar conditions
   - Over-specification of edge cases that rarely occur

3. **Could test names be improved?**
   - Do they clearly document "what" not "how"?

## Success Criteria

### Coverage Requirements
- **100% of critical functions** have at least one test
- **Common edge cases** are tested (e.g., null values, first measurement, reset scenarios)
- **Obscure edge cases** are NOT tested if unlikely

### Quality Requirements
- Test names clearly document expected behavior
- Each test has a single, clear purpose
- Tests serve as documentation of system behavior
- Tests immediately catch behavioral changes

### Quantitative Goals
- **Total tests**: Aim for 50-100 focused tests (vs current 35 unfocused)
- **Lines of test code**: Aim for 800-1500 lines (vs current 594)
- **Test-to-code ratio**: ~1:5 to 1:10 (currently ~1:20 for weight_processor_service)

## Out of Scope
- Integration tests with external APIs
- Integration tests with live DynamoDB
- Performance/load testing
- Testing every possible edge case
- Testing trivial getters/setters
- Testing framework code (boto3, pydantic, etc.)

## Deliverables
1. **Updated test suite** with:
   - Renamed tests for clarity
   - Removed redundant tests
   - Added tests for untested critical code

2. **Test coverage report** showing:
   - Which critical functions are tested
   - Which common edge cases are covered
   - Justification for untested code

3. **Documentation** in test docstrings:
   - Clear description of what behavior is being tested
   - Reference to specification/requirement if applicable
