# End-to-End API Validation Test Plan

## Overview

Create a comprehensive end-to-end test that validates the SAM Local API produces identical filtering results to the reference filtered dataset (`2025-09-29_all_filtered_e.csv`).

**Test User**: `c51ef96b-5618-4295-a910-233faed5ab60`
- **Source measurements**: 49
- **Expected accepted**: 9
- **Acceptance rate**: 18.4%

## Revision History

### V3: Continuous Replay with Check/Execute Endpoints (Current)

**Key Changes from V2**:
- **New Endpoints**: `/api/v1/replay/{userId}/check` and `/api/v1/replay/{userId}/execute`
- **Processing**: Measurements processed one at a time with automatic replay checking
- **Replay Triggers**: After each measurement, check if replay should trigger
- **Acceptance Tracking**: Replay execution returns NEW acceptance results to update tracking
- **External Control**: Caller decides when to execute replay (service provides advisory)

**Rationale**: This approach provides better separation of concerns:
1. Service layer handles replay logic (outlier detection, state restoration)
2. Caller controls when to trigger replay (external trigger model)
3. Clear contract: check → execute → update acceptance
4. Same logic available to both API and local processing

### V2: Pragmatic 72-Hour Replay Simulation (Deprecated)

**Key Changes from V1**:
- **Processing**: Measurements processed one at a time (mimics real-time ingestion)
- **Replay Triggers**: Detected at natural 72-hour boundaries in test data
- **Final Replay**: Always executed at end (simulates pending timeout)
- **Acceptance Tracking**: Only final state matters (replay replaces status)

**Rationale**: This approach better simulates production behavior where:
1. Measurements arrive one at a time
2. Replay timeout scheduled when >1 measurement in last 72h
3. Timeout fires after 72h, reprocessing recent measurements
4. Replay updates acceptance status based on improved Kalman state

### V1: Single Middle-Point Replay (Deprecated)

**Old Approach**:
- Processed all measurements individually
- Executed single replay from middle point
- Combined individual + replay acceptance sets

**Why Changed**: Didn't reflect production replay behavior (timeout-based, rolling windows)

## Test Architecture

### Location
`tests/api/test_e2e_validation.py` - New test file for end-to-end validation tests

### Dependencies
- SAM Local API running on port 3080
- DynamoDB local running on port 8000
- Source CSV: `./data/2025-09-29_weights_all.csv`
- Reference CSV: `./data/2025-09-29_all_filtered_e.csv`

### Key Components

1. **TestDataExtractor**: Extract single-user data from CSVs
2. **APIProcessor**: Orchestrate API calls (mimics api_main.py logic)
3. **AcceptanceValidator**: Compare API results with reference data
4. **ReplayCoordinator**: Manage replay sequence

## Detailed Test Steps

### Phase 1: Test Data Preparation

```python
def extract_user_measurements(csv_path: str, user_id: str) -> List[Dict]:
    """Extract all measurements for a specific user, sorted by timestamp."""
    # Load CSV
    # Filter by user_id
    # Sort by effective_date_time
    # Return measurement list
```

```python
def extract_expected_accepted(filtered_csv_path: str, user_id: str) -> Set[str]:
    """Extract the measurement IDs that should be accepted."""
    # Load filtered CSV
    # Filter by user_id
    # Return set of measurement IDs (id column)
```

### Phase 2: API Processing (Individual Measurements)

**Key Insight from api_main.py**: Individual processing happens first, then replay.

```python
def process_individual_measurements(
    api_client,
    user_id: str,
    measurements: List[Dict],
    batch_size: int = 1
) -> Set[str]:
    """
    Process measurements individually, tracking accepted IDs.

    Returns: Set of accepted measurement IDs
    """
    # Sort measurements by effectiveDateTime
    # For each measurement (or batch):
    #   - Call POST /api/v1/process/{user_id}
    #   - Parse response.data.results
    #   - Track which measurements have "accepted": true
    # Return set of accepted IDs
```

**API Response Format** (from api_main.py:323-334):
```json
{
  "success": true,
  "data": {
    "measurements_processed": 1,
    "measurements_accepted": 1,
    "results": [
      {
        "accepted": true,
        "quality_score": 0.95,
        "timestamp": "2020-04-17T00:00:00Z"
      }
    ]
  }
}
```

### Phase 3: Continuous Replay with Check/Execute (V3 - Current)

**New API Endpoints:**

#### Check if Replay Should Trigger
```http
POST /api/v1/replay/{userId}/check
Content-Type: application/json

{
  "user_id": "user_123",
  "current_timestamp": "2025-09-30T12:00:00Z",
  "buffer_hours": 72  // optional, defaults to 72
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "should_trigger": true,
    "window_info": {
      "window_start": "2025-09-27T12:00:00Z",
      "window_end": "2025-09-30T12:00:00Z",
      "measurements_in_window": 15,
      "measurement_ids": ["id1", "id2", ...]
    }
  }
}
```

#### Execute Replay
```http
POST /api/v1/replay/{userId}/execute
Content-Type: application/json

{
  "user_id": "user_123",
  "window_info": {
    "window_start": "2025-09-27T12:00:00Z",
    "window_end": "2025-09-30T12:00:00Z",
    "measurements_in_window": 15,
    "measurement_ids": ["id1", "id2", ...]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user_id": "user_123",
    "success": true,
    "window_start": "2025-09-27T12:00:00Z",
    "window_end": "2025-09-30T12:00:00Z",
    "measurement_results": [
      {
        "measurement_id": "id1",
        "accepted": false,  // NEW status
        "quality_score": 0.45,
        "rejection_reason": "outlier"
      },
      {
        "measurement_id": "id2",
        "accepted": true,  // NEW status
        "quality_score": 0.92
      }
    ],
    "outliers_detected": ["id1"],
    "outliers_count": 1,
    "corrections_made": 3
  }
}
```

**Implementation:**
```python
def process_with_continuous_replay(
    api_client,
    user_id: str,
    measurements: List[Dict]
) -> Set[str]:
    """
    Process measurements with continuous replay checking.

    Uses the new check/execute endpoints for replay.

    Returns: Final set of accepted measurement IDs
    """
    accepted_ids = set()
    sorted_measurements = sorted(
        measurements,
        key=lambda m: parse_timestamp(m["effectiveDateTime"])
    )

    # Process measurements one at a time
    for measurement in sorted_measurements:
        # 1. Process single measurement
        response = api_client.process_measurements(user_id, [measurement])
        if response["accepted"]:
            accepted_ids.add(measurement["uuid"])

        # 2. Check if replay should trigger
        measurement_timestamp = parse_timestamp(measurement["effectiveDateTime"])
        check_response = api_client.check_replay(user_id, measurement_timestamp)

        if check_response["should_trigger"]:
            # 3. Execute replay
            window_info = check_response["window_info"]
            replay_response = api_client.execute_replay(user_id, window_info)

            # 4. Update acceptance tracking with NEW results
            if replay_response["success"]:
                # Clear previous acceptances for this window
                accepted_ids = {
                    mid for mid in accepted_ids
                    if mid not in window_info["measurement_ids"]
                }

                # Add NEW acceptances from replay results
                for result in replay_response["measurement_results"]:
                    if result["accepted"]:
                        accepted_ids.add(result["measurement_id"])

    return accepted_ids
```

**Key Differences from V2:**
- ✅ **External trigger control**: Caller decides when to execute replay
- ✅ **Service handles complexity**: Outlier detection, state restoration in service layer
- ✅ **Clear contract**: Check → Execute → Update acceptance
- ✅ **NEW results**: Replay returns definitive acceptance statuses for all measurements in window

### Phase 3 (Legacy): Replay Processing (Pragmatic 72-Hour Window Simulation)

**Production Behavior**:
- After processing each measurement, if >1 measurement exists in last 72 hours, schedule a replay timeout
- When 72 hours elapse, replay all measurements within that window
- Replay may change acceptance status (reject previously accepted, accept previously rejected)

**Test Simulation Strategy (Pragmatic)**:
Since we cannot simulate real-time 72-hour timeouts in a test, we use natural time boundaries:
1. Process measurements one at a time in chronological order
2. After each measurement, detect if next measurement is >72 hours away (natural boundary)
3. Execute replay for measurements within the 72-hour window before that boundary
4. After all measurements processed, execute final replay for remaining 72-hour window

```python
def process_with_72h_replay_simulation(
    api_client,
    user_id: str,
    measurements: List[Dict]
) -> Set[str]:
    """
    Simulate production 72-hour replay windows using natural time boundaries.

    Returns: Final set of accepted measurement IDs (after all replays)
    """
    from datetime import timedelta

    accepted_ids = set()
    sorted_measurements = sorted(
        measurements,
        key=lambda m: parse_timestamp(m["effectiveDateTime"])
    )

    # Phase 1: Process individually with conditional replays at boundaries
    for i, measurement in enumerate(sorted_measurements):
        # Process single measurement
        response = process_single_measurement(api_client, user_id, measurement)
        if response["accepted"]:
            accepted_ids.add(measurement["uuid"])

        # Check if next measurement crosses 72-hour boundary
        if i + 1 < len(sorted_measurements):
            current_time = parse_timestamp(measurement["effectiveDateTime"])
            next_time = parse_timestamp(sorted_measurements[i+1]["effectiveDateTime"])

            if (next_time - current_time) > timedelta(hours=72):
                # Natural 72-hour boundary detected - execute replay
                logger.info(f"72-hour gap detected, executing replay")

                # Replay window: last 72 hours up to current measurement
                replay_from = current_time - timedelta(hours=72)
                window_measurements = [
                    m for m in sorted_measurements[:i+1]
                    if parse_timestamp(m["effectiveDateTime"]) >= replay_from
                ]

                if len(window_measurements) > 1:
                    replay_results = execute_replay(
                        api_client, user_id, replay_from, window_measurements
                    )
                    # Replay updates acceptance status
                    accepted_ids = replay_results

    # Phase 2: Final replay for remaining window
    # Simulates the "timeout still scheduled" scenario
    last_measurement_time = parse_timestamp(sorted_measurements[-1]["effectiveDateTime"])
    replay_from = last_measurement_time - timedelta(hours=72)
    final_window = [
        m for m in sorted_measurements
        if parse_timestamp(m["effectiveDateTime"]) >= replay_from
    ]

    if len(final_window) > 1:
        logger.info("Executing final replay (end of processing)")
        replay_results = execute_replay(
            api_client, user_id, replay_from, final_window
        )
        # Final replay determines acceptance status
        accepted_ids = replay_results

    return accepted_ids
```

**Critical Question**: Does replay replace or augment the acceptance tracking?
- Based on production behavior, replay **replaces** acceptance tracking for the replayed window
- Replay recalculates quality scores with updated Kalman state
- Measurements previously rejected may become accepted (and vice versa)
- **We must track the FINAL state after all replays**

### Phase 4: Validation

```python
def validate_acceptance(
    actual_accepted: Set[str],
    expected_accepted: Set[str]
) -> ValidationResult:
    """
    Compare actual vs expected acceptance sets.

    Checks:
    1. Exact match (ideal case)
    2. Missing measurements (expected but not accepted)
    3. Extra measurements (accepted but not expected)

    Returns detailed diff report
    """
```

## Implementation Strategy

### Test Structure (V3 - Continuous Replay)

```python
@pytest.mark.e2e
def test_single_user_end_to_end_validation_v3(api_client, ensure_clean_state):
    """
    E2E test: Process a real user using continuous replay (V3).

    This test uses the new check/execute replay endpoints:
    1. Extracts user data from source CSV
    2. Processes measurements one at a time
    3. After each measurement, checks if replay should trigger
    4. If replay triggers, executes replay and updates acceptance tracking
    5. Validates accepted measurements match filtered dataset

    Processing Logic:
    - Each measurement processed individually (mimics real-time ingestion)
    - After processing, check if replay should trigger (POST /replay/{userId}/check)
    - If should_trigger=true, execute replay (POST /replay/{userId}/execute)
    - Update acceptance tracking with NEW results from replay
    - Replay results REPLACE acceptance status for the replayed window
    """
    # Arrange
    user_id = "c51ef96b-5618-4295-a910-233faed5ab60"

    # Reset state for clean test
    cleanup_user(api_client, user_id)

    # Load test data
    measurements = extract_user_measurements(SOURCE_CSV, user_id)
    expected_accepted = extract_expected_accepted(FILTERED_CSV, user_id)

    # Act: Process with continuous replay (V3)
    final_accepted = process_with_continuous_replay(
        api_client, user_id, measurements
    )

    # Assert
    validation = validate_acceptance(final_accepted, expected_accepted)

    assert validation.is_exact_match, (
        f"Acceptance mismatch:\n"
        f"  Expected: {len(expected_accepted)} measurements\n"
        f"  Actual: {len(final_accepted)} measurements\n"
        f"  Missing: {validation.missing_ids}\n"
        f"  Extra: {validation.extra_ids}"
    )
```

### Test Structure (Legacy V2)

```python
@pytest.mark.e2e
def test_single_user_end_to_end_validation_v2(api_client, ensure_clean_state):
    """
    E2E test: Process a real user and validate against reference dataset (V2 legacy).

    This test simulates production behavior:
    1. Extracts user data from source CSV
    2. Processes measurements one at a time
    3. Executes replays at natural 72-hour boundaries
    4. Executes final replay after last measurement
    5. Validates accepted measurements match filtered dataset

    Processing Logic:
    - Each measurement processed individually (mimics real-time ingestion)
    - When >72h gap detected, execute replay for previous window
    - After all measurements, execute final replay (simulates timeout expiration)
    - Replay results REPLACE acceptance status for the replayed window
    """
    # Arrange
    user_id = "c51ef96b-5618-4295-a910-233faed5ab60"

    # Reset state for clean test
    cleanup_user(api_client, user_id)

    # Load test data
    measurements = extract_user_measurements(SOURCE_CSV, user_id)
    expected_accepted = extract_expected_accepted(FILTERED_CSV, user_id)

    # Act: Process with 72-hour replay simulation
    final_accepted = process_with_72h_replay_simulation(
        api_client, user_id, measurements
    )

    # Assert
    validation = validate_acceptance(final_accepted, expected_accepted)

    assert validation.is_exact_match, (
        f"Acceptance mismatch:\n"
        f"  Expected: {len(expected_accepted)} measurements\n"
        f"  Actual: {len(final_accepted)} measurements\n"
        f"  Missing: {validation.missing_ids}\n"
        f"  Extra: {validation.extra_ids}"
    )
```

### Helper Fixtures

```python
@pytest.fixture
def source_csv_path():
    """Path to source CSV file."""
    return Path("./data/2025-09-29_weights_all.csv")

@pytest.fixture
def filtered_csv_path():
    """Path to reference filtered CSV file."""
    return Path("./data/2025-09-29_all_filtered_e.csv")

@pytest.fixture
def test_user_id():
    """User ID for E2E testing."""
    return "c51ef96b-5618-4295-a910-233faed5ab60"

@pytest.fixture
def ensure_clean_state(api_client, test_user_id):
    """Ensure clean state before and after test."""
    # Reset before test
    cleanup_user(api_client, test_user_id, cleanup_type="full_reset")
    yield
    # Reset after test (cleanup)
    cleanup_user(api_client, test_user_id, cleanup_type="full_reset")
```

## Key Design Decision: 72-Hour Replay Window Simulation

### Why Natural Time Boundaries?

**Production Behavior**:
- In production, a replay timeout is scheduled after processing any measurement if >1 measurement exists in the last 72 hours
- The timeout fires 72 hours after being scheduled
- This ensures recent measurements are reprocessed with updated Kalman state

**Test Simulation Challenge**:
- Cannot wait 72 real hours in a test
- Cannot manipulate system time without affecting API behavior
- Need deterministic, repeatable test results

**Pragmatic Solution**:
- Detect natural 72-hour gaps in test data (timestamps >72h apart)
- Execute replay at these boundaries (simulates timeout firing)
- Execute final replay at end (simulates pending timeout)
- This approximates production behavior while remaining deterministic

**Trade-offs**:
- ✅ Deterministic and repeatable
- ✅ Exercises replay logic multiple times
- ✅ Tests final acceptance state (what matters for validation)
- ⚠️ May not trigger replays at exact same points as production
- ⚠️ Assumes test data has natural 72-hour boundaries

## Council Review: Key Concerns

### 🎯 Kent Beck (Testing)
**"How can we make this test deterministic and debuggable?"**

**Concerns**:
- Replay timing: Using natural time boundaries makes test deterministic
- Test isolation: Must reset user state completely before test
- Debugging: Need detailed logging for each replay trigger and acceptance changes

**Recommendations**:
1. Log each replay trigger point with timestamp
2. Log acceptance status changes after each replay
3. Track how many replays were executed (should align with data characteristics)
4. Add dry-run mode that logs what would be compared without asserting

### 🔍 Martin Kleppmann (Data Consistency)
**"What happens during replay? Can measurements change acceptance status?"**

**Concerns**:
- Replay recalculates quality scores with updated Kalman state (more measurements = better state estimation)
- Replay **replaces** acceptance status for the replayed window (doesn't augment)
- Multiple replays may occur: need to track final state only
- State persistence: API must maintain Kalman state in DynamoDB between calls

**Recommendations**:
1. Verify Kalman state persists in DynamoDB between individual measurements
2. Track only FINAL acceptance state (after all replays complete)
3. Log when replay changes acceptance (measurement was accepted → rejected or vice versa)
4. Document that intermediate acceptance states are not meaningful (only final state matters)

### 🏗️ Barbara Liskov (Invariants)
**"What assumptions does this test make about the system?"**

**Concerns**:
- Assumes production processing order: individual measurements → conditional replays → final replay
- Assumes test data has natural 72-hour boundaries for replay triggering
- Assumes the filtered CSV represents ground truth (expected behavior)
- Assumes replay replaces (not augments) acceptance status

**Recommendations**:
1. Document that this test validates "compatibility with reference dataset"
2. Verify test data characteristics (check for 72-hour gaps)
3. Add comments explaining replay trigger logic
4. Consider adding test to verify replay behavior in isolation

## Edge Cases to Consider

### Measurement Ordering
- **Issue**: CSV may not be strictly ordered by timestamp
- **Solution**: Always sort measurements by `effective_date_time` before processing

### Timestamp Formats
- **Issue**: Timestamps must be parsed consistently for 72-hour window calculations
- **Solution**: Use same `parse_timestamp()` logic throughout (convert to timezone-aware datetimes)

### 72-Hour Boundary Detection
- **Issue**: What if test data has no natural 72-hour gaps?
- **Solution**: Final replay still executes (covers all measurements in last 72h window)

### Replay Window Calculation
- **Issue**: Replay window is "last 72 hours from current measurement"
- **Solution**: `replay_from = current_time - timedelta(hours=72)`
- **Edge case**: If measurements span <72 hours, all measurements included in replay

### Multiple Measurements at Same Timestamp
- **Issue**: Multiple measurements with identical timestamps
- **Solution**: Process in CSV order, treat as chronologically sequential

### Empty Result Sets
- **Issue**: What if NO measurements are accepted?
- **Solution**: Test should still pass (expect empty set vs empty set)

## Success Criteria

✅ **Test passes**: Accepted measurement IDs match filtered CSV exactly
✅ **Debuggable**: Clear error messages show which measurements differ
✅ **Isolated**: Test can run repeatedly with same results
✅ **Fast**: Completes in < 30 seconds
✅ **Documented**: Comments explain the processing flow

## Next Steps

**Recommended: Implement V3 (Continuous Replay)**

1. **Update api_main.py** ✅ COMPLETED
   - Add `check_replay()` method to APIClient
   - Add `execute_replay()` method to APIClient
   - Add `update_from_replay_results()` to AcceptanceTracker
   - Add `process_measurements_with_continuous_replay()` function
   - Add CLI flags: `--enable-continuous-replay` and `--disable-replay`

2. **Test api_main.py with continuous replay**
   ```bash
   # Start SAM API
   make sam-local

   # Test continuous replay mode
   python api_main.py --csv-file data/test_weights.csv \
     --max-users 3 \
     --enable-continuous-replay

   # Test with replay disabled
   python api_main.py --csv-file data/test_weights.csv \
     --max-users 3 \
     --enable-continuous-replay \
     --disable-replay
   ```

3. **Update test implementation** (in `tests/api/test_e2e_validation.py`)
   - Add `test_single_user_end_to_end_validation_v3()` using continuous replay
   - Add helper function `process_with_continuous_replay()`
   - Keep V2 test as `test_single_user_end_to_end_validation_v2()` for regression

4. **Run tests and verify**
   ```bash
   # Run V3 test
   uv run pytest tests/api/test_e2e_validation.py::test_single_user_end_to_end_validation_v3 -xvs

   # Run V2 test for regression
   uv run pytest tests/api/test_e2e_validation.py::test_single_user_end_to_end_validation_v2 -xvs
   ```

**Legacy: V2 Implementation (Optional)**

1. **Analyze test data characteristics**
   - Check timestamps for natural 72-hour gaps
   - Verify expected replay trigger points

2. **Add helper function** `process_with_72h_replay_simulation()`
   - Implements legacy 72-hour boundary detection
   - Uses old `/replay/{userId}` endpoint

3. **Run test and verify**
   - `uv run pytest tests/api/test_e2e_validation.py::test_single_user_end_to_end_validation_v2 -xvs --log-cli-level=INFO`

## Implementation Notes

### Critical Implementation Details

1. **Replay Replaces Acceptance Status**
   ```python
   # WRONG: Augmenting acceptance
   final_accepted = accepted_after_individual | accepted_from_replay

   # CORRECT: Replay replaces status for replayed window
   final_accepted = accepted_from_replay  # or track per-measurement acceptance
   ```

2. **72-Hour Window Calculation**
   ```python
   # Calculate replay window: last 72 hours from current point
   replay_from = current_time - timedelta(hours=72)
   window_measurements = [
       m for m in measurements
       if parse_timestamp(m["effectiveDateTime"]) >= replay_from
   ]
   ```

3. **Replay Trigger Logic**
   ```python
   # Trigger replay at natural 72-hour boundaries
   if (next_time - current_time) > timedelta(hours=72):
       execute_replay(replay_from, window_measurements)
   ```

4. **Final Replay Always Executes**
   ```python
   # After all measurements processed, execute final replay
   # This simulates the "pending timeout" scenario
   if len(final_window) > 1:
       final_accepted = execute_replay(replay_from, final_window)
   ```

### Logging Strategy

Log the following for debugging:
- Each measurement processed (timestamp, weight, ID)
- Replay trigger point (timestamp, number of measurements in window)
- Acceptance status changes after replay (accepted → rejected, rejected → accepted)
- Final statistics (total measurements, replays executed, final accepted count)

## Open Questions

1. **Should we test multiple users?**
   - Pro: Better coverage, find edge cases
   - Con: Slower test execution
   - **Recommendation**: Start with one user, add parametrization later

2. **What if test data has no 72-hour gaps?**
   - Final replay will still execute (covers entire dataset if <72h span)
   - Test may not exercise multiple replay triggers
   - **Recommendation**: Document data characteristics in test docstring

3. **What if the filtered CSV is wrong?**
   - This test validates consistency, not correctness
   - If the test fails, investigate both API and reference dataset
   - Consider adding a "known differences" list if justified

4. **Should we verify Kalman state between calls?**
   - Pro: Ensures state persistence working correctly
   - Con: Requires introspecting internal state
   - **Recommendation**: Rely on acceptance outcomes (state persistence is implied)

## File References

- API main script: `api_main.py:347-432` (individual processing)
- API main script: `api_main.py:435-517` (replay processing)
- Acceptance tracking: `api_main.py:302-344` (AcceptanceTracker class)
- API response parsing: `api_main.py:45-71` (APIResponse handling)