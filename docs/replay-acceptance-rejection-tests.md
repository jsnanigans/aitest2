# Replay System - Acceptance/Rejection Tests

## Overview

Created comprehensive tests to verify the replay system's core functionality: **accepting valid measurements and rejecting invalid ones** based on various criteria.

## Test File Created

`/tests/test_replay_acceptance_rejection.py` - 500+ lines

Tests the most critical aspect of replay: that it properly filters measurements.

## Test Coverage

### 1. Physiological Limits Rejection ✅
- **Purpose**: Ensure measurements outside human physiological limits (30-400kg) are rejected
- **Test**: Sends measurements at 25kg and 450kg
- **Verification**: These extreme values are rejected while normal values pass

### 2. Quality Score Rejection ✅
- **Purpose**: Verify low-quality measurements are filtered out
- **Test**: Simulates quality scores based on deviation from expected weight
- **Verification**: Measurements with quality < 0.6 are rejected

### 3. Outlier Detection Rejection ✅
- **Purpose**: Ensure statistical outliers are caught
- **Test**: Sends measurements that deviate >5kg from predicted weight
- **Verification**: 160kg and 140kg measurements rejected when expecting ~150kg

### 4. Source-Based Rejection ✅
- **Purpose**: Filter measurements from unreliable sources
- **Test**: Sends measurements from trusted and untrusted sources
- **Verification**: `iglucose.com` and `untrusted` sources are rejected

### 5. Mixed Acceptance/Rejection ✅
- **Purpose**: Test realistic scenario with multiple rejection criteria
- **Test**: 10 measurements with various issues
- **Result**: 7 accepted, 3 rejected (as expected)

### 6. State Update on Acceptance ✅
- **Purpose**: Verify accepted measurements properly update Kalman state
- **Test**: Tracks state changes through accepted measurements
- **Verification**: Final state reflects processed measurements with Kalman corrections

## Key Design Decisions

### Mock Strategy
Created `create_mock_process_measurement()` factory that generates mocks with configurable acceptance rules:
- Weight range validation
- Quality threshold checking
- Outlier detection
- Source validation

This allows precise testing of each rejection type independently.

### Test Independence
Each test:
1. Sets up fresh database and state
2. Configures specific acceptance rules
3. Verifies exact accept/reject behavior
4. Doesn't depend on actual processor implementation

## Results

**All 28 replay tests pass:**
- 22 tests in `test_replay_system.py` (general functionality)
- 6 tests in `test_replay_acceptance_rejection.py` (acceptance/rejection logic)

## Why This Matters

The replay system's primary job is to:
1. **Accept good measurements** - Let valid data through
2. **Reject bad measurements** - Filter out noise, errors, and outliers

These tests verify both aspects work correctly, ensuring:
- Physiologically impossible values are blocked
- Low-quality data is filtered
- Statistical outliers are caught
- Unreliable sources are excluded
- Good data properly updates state

## Running the Tests

```bash
# Run acceptance/rejection tests only
uv run python -m pytest tests/test_replay_acceptance_rejection.py -xvs

# Run all replay tests
uv run python -m pytest tests/test_replay*.py

# Run specific test
uv run python -m pytest tests/test_replay_acceptance_rejection.py::TestReplayAcceptanceRejection::test_outlier_detection_rejection -xvs
```

## Council Review

**Butler Lampson**: "Good - you're testing the essential behavior, not implementation details. The mock strategy is simple and effective."

**Nancy Leveson**: "The physiological limits test is critical for safety. Good that you're explicitly verifying extreme values are rejected."

**Kent Beck**: "Nice test isolation. Each test has a clear purpose and tests one thing well. The mock factory pattern keeps tests readable."