# Replay Service Layer Simplification Plan

## Executive Summary

**Goal:** Push replay complexity from `local_main.py` into the `weight_values/src` service layer, making it available to both local processing AND the Lambda API.

**Key Insight:** Replay should be a **first-class feature of the service layer** with **external triggering control**. The service provides helper methods, but the caller (local_main.py or API client) decides when to trigger and how to handle results.

**Architectural Constraint:**
- ✅ Replay is **triggered externally** by the caller
- ✅ Service provides `should_trigger_replay()` (advisory) and `execute_replay()` (execution)
- ✅ Caller maintains control over acceptance tracking
- ✅ Service returns NEW acceptance results; caller updates tracking

**Benefits:**
1. **Simplifies local_main.py** - Service handles outlier detection, state restoration, replay execution (~60% code reduction)
2. **Makes replay available to the API** - Lambda can use same service methods with external triggering
3. **Better separation of concerns** - Service does processing, caller does orchestration
4. **DRY principle** - Replay logic lives in service layer, not duplicated in callers
5. **Caller control** - Caller decides when to trigger and how to update acceptance tracking

---

## Council Architectural Review

```
-- COUNCIL REVIEW --
Task: Refactor replay from external logic in local_main.py to service layer methods

Council's Key Insights:

• **Butler Lampson**: "The simplest design is to have the service do what it's supposed to do. If replay is essential to correct processing, it shouldn't be optional external logic - it should be built in."

• **Barbara Liskov**: "WeightProcessorService currently violates the invariant that 'processed measurements are correct.' By making replay external, we allow incorrect states to exist. The service should guarantee correctness internally."

• **Alan Kay**: "What's the real problem? Users need correct weight tracking. The API currently doesn't solve this - it needs replay too. Push it down to the service layer."

• **Leslie Lamport**: "The causal ordering is clearer when the service manages it. The service knows when to replay; the caller shouldn't."

Recommendation: **Proceed with service layer integration.** This is the right architectural pattern.

Key design principles:
1. Service layer exposes `process_with_replay()` method
2. Replay trigger logic lives in ReplayManager
3. State store provides measurement history queries
4. API and local processing use same code path
-- END COUNCIL --
```

---

## Proposed New Methods

**CRITICAL CONSTRAINT:** Replay must be **triggered externally** by the caller (local_main.py or API client). The service provides helper methods but does NOT automatically trigger replay. The caller maintains control over acceptance tracking.

### 1. WeightProcessorService: Query and Execution Methods

#### 1.1 `should_trigger_replay()` - Check if replay needed (caller decides)
```python
def should_trigger_replay(
    self,
    user_id: str,
    current_timestamp: datetime,
    buffer_hours: int = 72
) -> Tuple[bool, Optional[ReplayWindowInfo]]:
    """
    Check if replay should trigger after processing a measurement.

    This method provides information to help the CALLER decide whether
    to trigger replay. It does NOT execute replay automatically.

    Replay should trigger when there are measurements in the buffer window
    before the current timestamp.

    Args:
        user_id: User identifier
        current_timestamp: Timestamp of measurement just processed
        buffer_hours: Size of replay window in hours (default: 72)

    Returns:
        Tuple of (should_trigger, window_info)
        - should_trigger: True if replay is recommended
        - window_info: ReplayWindowInfo with window details (if should trigger)

    Example:
        >>> service = WeightProcessorService()
        >>> should_trigger, window_info = service.should_trigger_replay(user_id, now)
        >>> if should_trigger:
        >>>     # Caller decides to execute replay
        >>>     replay_result = service.execute_replay(user_id, window_info)
    """
    pass


@dataclass
class ReplayWindowInfo:
    """Information about a replay window."""
    window_start: datetime
    window_end: datetime
    measurements_in_window: int
    measurement_ids: List[str]  # For caller to track which measurements to re-evaluate
```

#### 1.2 `execute_replay()` - Execute replay (caller provides measurements)
```python
def execute_replay(
    self,
    user_id: str,
    window_info: ReplayWindowInfo,
    measurements_to_replay: Optional[List[Measurement]] = None
) -> ReplayResultData:
    """
    Execute replay for a measurement window.

    The CALLER triggers this method and must handle the results by updating
    acceptance tracking. This method:
    1. Restores state to before window
    2. Detects outliers using pre-window state
    3. Replays clean measurements chronologically
    4. Returns NEW acceptance results for caller to process

    Args:
        user_id: User identifier
        window_info: Window information from should_trigger_replay()
        measurements_to_replay: Optional list of measurements (if None, queries from DB)

    Returns:
        ReplayResultData containing:
        - List of measurements with NEW acceptance statuses
        - Outlier information
        - State changes

    IMPORTANT: Caller must update acceptance tracking based on results!

    Example:
        >>> # After processing a measurement
        >>> should_trigger, window_info = service.should_trigger_replay(user_id, now)
        >>> if should_trigger:
        >>>     # Execute replay
        >>>     replay_result = service.execute_replay(user_id, window_info)
        >>>
        >>>     # Caller updates acceptance tracking
        >>>     for result in replay_result.measurement_results:
        >>>         acceptance_tracker.update(
        >>>             result.measurement_id,
        >>>             accepted=result.accepted
        >>>         )
    """
    pass


@dataclass
class ReplayResultData:
    """Results from replay execution that caller must process."""

    user_id: str
    success: bool
    window_start: datetime
    window_end: datetime

    # NEW acceptance results - caller must update tracking
    measurement_results: List[MeasurementResult]  # One per measurement in window

    # Metadata
    outliers_detected: List[str]  # measurement_ids marked as outliers
    outliers_count: int
    corrections_made: int  # Number of acceptance changes
    state_restored_to: datetime

    error: Optional[str] = None
```

### 2. ReplayManager: Trigger and Execution Logic

#### 2.1 `check_should_trigger()` - Decision logic
```python
def check_should_trigger(
    self,
    user_id: str,
    current_timestamp: datetime,
) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    """
    Check if replay should trigger after processing a measurement.

    Replay triggers when there are measurements in the 72-hour window
    before the current timestamp.

    Args:
        user_id: User identifier
        current_timestamp: Timestamp of measurement just processed

    Returns:
        Tuple of (should_trigger, window_measurements)
        - should_trigger: True if replay should execute
        - window_measurements: List of measurements in window (if should trigger)

    Example:
        >>> manager = ReplayManager(db, config)
        >>> should_trigger, window = manager.check_should_trigger(user_id, now)
        >>> if should_trigger:
        >>>     print(f"Found {len(window)} measurements in window")
    """
    buffer_hours = self.config.get("replay", {}).get("buffer_hours", 72)
    window_start = current_timestamp - timedelta(hours=buffer_hours)

    # Query measurements in window from state store
    window_measurements = self.db.get_measurements_in_window(
        user_id, window_start, current_timestamp
    )

    # Trigger if there are measurements in window
    should_trigger = len(window_measurements) > 0

    return should_trigger, window_measurements if should_trigger else None
```

#### 2.2 `execute_inline_replay()` - Execution logic
```python
def execute_inline_replay(
    self,
    user_id: str,
    current_measurement: Dict[str, Any],
    window_measurements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute replay for a measurement window inline during processing.

    This is the core replay workflow:
    1. Snapshot current state (for rollback)
    2. Get pre-window Kalman state for outlier baseline
    3. Detect outliers using pre-window state
    4. Restore to pre-window state
    5. Replay clean measurements chronologically (including current)

    Args:
        user_id: User identifier
        current_measurement: The measurement that triggered replay
        window_measurements: Measurements in the replay window

    Returns:
        Replay result dict with:
        - success: bool
        - outliers_found: int
        - corrections_made: int
        - clean_measurements_processed: int
        - error: Optional[str]

    Example:
        >>> result = manager.execute_inline_replay(user_id, current, window)
        >>> if result["success"]:
        >>>     print(f"Corrected {result['corrections_made']} measurements")
    """
    try:
        # 1. Backup current state
        self._create_state_backup(user_id)

        # 2. Get pre-window state for outlier detection baseline
        window_start = min(m["timestamp"] for m in window_measurements)
        pre_window_state = self.db.get_state_snapshot_before(user_id, window_start)

        # 3. Detect outliers using pre-window state
        outlier_detector = OutlierDetector(self.config.get("outlier_detection", {}), db=self.db)
        clean_measurements, outlier_indices = outlier_detector.get_clean_measurements(
            window_measurements,
            user_id=user_id,
            reference_state=pre_window_state
        )

        if len(outlier_indices) == 0:
            self._clear_state_backup(user_id)
            return {
                "success": True,
                "outliers_found": 0,
                "corrections_made": 0,
                "skipped": "No outliers found"
            }

        # 4. Restore to pre-window state
        self.db.restore_state_snapshot(user_id, window_start)

        # 5. Replay clean measurements + current measurement chronologically
        all_clean = clean_measurements + [current_measurement]
        all_clean.sort(key=lambda m: m["timestamp"])

        replay_result = self._replay_measurements_chronologically(
            user_id, all_clean, time.time()
        )

        if not replay_result["success"]:
            self._restore_state_from_backup(user_id)
            return replay_result

        self._clear_state_backup(user_id)

        return {
            "success": True,
            "outliers_found": len(outlier_indices),
            "corrections_made": len(outlier_indices),
            "clean_measurements_processed": len(all_clean)
        }

    except Exception as e:
        self._restore_state_from_backup(user_id)
        return {
            "success": False,
            "error": str(e)
        }
```

### 3. StateStore/Database: Measurement History Queries

#### 3.1 `get_measurements_in_window()` - Query measurements
```python
def get_measurements_in_window(
    self,
    user_id: str,
    start_time: datetime,
    end_time: datetime
) -> List[Dict[str, Any]]:
    """
    Get measurements for a user within a time window.

    Used by replay trigger logic to find measurements in the 72-hour window.

    Args:
        user_id: User identifier
        start_time: Window start time (inclusive)
        end_time: Window end time (exclusive)

    Returns:
        List of measurement dicts with keys:
        - timestamp: datetime
        - weight: float
        - source: str
        - unit: str
        - metadata: dict

    Example:
        >>> window_start = now - timedelta(hours=72)
        >>> measurements = db.get_measurements_in_window(user_id, window_start, now)
    """
    state = self.get_state(user_id)
    if not state or "measurement_history" not in state:
        return []

    measurements = []
    for m in state["measurement_history"]:
        timestamp = m.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        if start_time <= timestamp < end_time:
            measurements.append({
                "timestamp": timestamp,
                "weight": m.get("weight"),
                "source": m.get("source", "unknown"),
                "unit": m.get("unit", "kg"),
                "metadata": m.get("metadata", {})
            })

    return measurements
```

#### 3.2 `get_state_snapshot_before()` - Get pre-window state
```python
def get_state_snapshot_before(
    self,
    user_id: str,
    timestamp: datetime
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent state snapshot before a given timestamp.

    Used by replay to get the Kalman state from before the window
    for outlier detection baseline.

    Args:
        user_id: User identifier
        timestamp: Find snapshot before this time

    Returns:
        State dict or None if no snapshot exists

    Example:
        >>> window_start = now - timedelta(hours=72)
        >>> pre_window_state = db.get_state_snapshot_before(user_id, window_start)
    """
    # Query snapshots for user
    snapshots = self._get_all_snapshots(user_id)

    # Find most recent snapshot before timestamp
    valid_snapshots = [
        s for s in snapshots
        if s.get("snapshot_time") < timestamp
    ]

    if not valid_snapshots:
        return None

    # Return most recent
    return max(valid_snapshots, key=lambda s: s["snapshot_time"])
```

---

## How This Simplifies local_main.py

### Before (Current): ~200 lines of replay logic
```python
# Current local_main.py has:
# - process_individual_measurements() - batch processing
# - process_replay_with_outlier_detection() - separate replay phase
# - Complex two-phase architecture
# - Manual outlier detection and state restoration

def main():
    # Phase 1: Process all measurements
    individual_results = process_individual_measurements(...)

    # Phase 2: Run replay for eligible users (separate, after all processing)
    replay_results = process_replay_with_outlier_detection(...)

    # Write filtered CSV
    write_filtered_csv(...)
```

### After (Proposed): ~80 lines, continuous processing with external triggering
```python
# Simplified local_main.py with caller-controlled replay:

def process_measurements_with_continuous_replay(
    service: WeightProcessorService,
    user_measurements: Dict[str, List[Measurement]],
    acceptance_tracker: AcceptanceTracker,
    enable_replay: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements one at a time with external replay triggering.

    The caller maintains control over acceptance tracking and decides
    when to trigger replay based on service recommendations.
    """
    results = {}

    for user_id, measurements in user_measurements.items():
        user_results = {
            "measurements_processed": 0,
            "replays_triggered": 0,
            "total_corrections": 0
        }

        # Sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

        # Process one at a time
        for measurement in sorted_measurements:
            # 1. Process measurement
            result = service.process_batch(user_id, [measurement])
            user_results["measurements_processed"] += 1

            # 2. Track initial acceptance
            acceptance_tracker.mark_batch_results(user_id, [measurement], result)

            # 3. Check if replay should trigger
            if enable_replay:
                should_trigger, window_info = service.should_trigger_replay(
                    user_id, measurement.measured_at
                )

                if should_trigger:
                    # 4. Execute replay (service handles outlier detection + replay)
                    replay_result = service.execute_replay(user_id, window_info)

                    if replay_result.success:
                        user_results["replays_triggered"] += 1
                        user_results["total_corrections"] += replay_result.corrections_made

                        # 5. Update acceptance tracking based on NEW results
                        acceptance_tracker.update_from_replay_results(
                            user_id, replay_result.measurement_results
                        )

                        print(f"  Replay: {replay_result.outliers_count} outliers, "
                              f"{replay_result.corrections_made} corrections")

        results[user_id] = user_results

    return results


def main():
    # Initialize service
    service = WeightProcessorService(state_store=state_store, config=config)

    # Single phase: Process with continuous replay (caller controls triggering)
    results = process_measurements_with_continuous_replay(
        service, user_measurements, acceptance_tracker,
        enable_replay=not args.disable_replay
    )

    # Write filtered CSV (acceptance_tracker has final acceptances)
    write_filtered_csv(original_rows, acceptance_tracker, output_path)
```

**Key Differences from "Automatic" Approach:**
- ✅ **Caller controls triggering:** Decides when to call `execute_replay()`
- ✅ **Caller updates acceptances:** Service returns results, caller updates tracking
- ✅ **Service provides helpers:** `should_trigger_replay()` advises, doesn't execute
- ✅ **Separation of concerns:** Service does processing, caller does orchestration

**Lines of code reduction:** ~60% reduction (from ~200 to ~80 lines)
**Complexity reduction:** Service handles outlier detection, state restoration, replay execution

---

## How This Benefits the API/Lambda

### Current API (No Replay)
```python
# lambda_handler.py
@app.post("/process")
def process_measurements(request: ProcessRequest):
    service = WeightProcessorService()
    result = service.process_batch(request.user_id, request.measurements)
    return result

# Problem: API doesn't benefit from replay logic!
# Early poor measurements still pollute state
```

### New API (With External Replay Triggering)
```python
# lambda_handler.py

@app.post("/process")
def process_measurements(request: ProcessRequest):
    """
    Process measurements with optional replay check.

    Returns replay recommendations if measurements trigger replay window.
    Client decides whether to call /replay endpoint.
    """
    service = WeightProcessorService()

    # Process measurement
    result = service.process_batch(request.user_id, request.measurements)

    # Check if replay should trigger (advisory only)
    replay_recommendation = None
    if request.check_replay:
        last_timestamp = request.measurements[-1].measured_at
        should_trigger, window_info = service.should_trigger_replay(
            request.user_id, last_timestamp
        )

        if should_trigger:
            replay_recommendation = {
                "replay_recommended": True,
                "window_start": window_info.window_start,
                "window_end": window_info.window_end,
                "measurements_in_window": window_info.measurements_in_window,
            }

    return ProcessResponse(
        user_id=request.user_id,
        result=result,
        replay_recommendation=replay_recommendation
    )


@app.post("/replay")
def execute_replay(request: ReplayRequest):
    """
    Execute replay for a user's measurement window.

    Client calls this endpoint when they decide to trigger replay.
    Returns NEW acceptance results that client must use to update their state.
    """
    service = WeightProcessorService()

    # Execute replay
    replay_result = service.execute_replay(
        user_id=request.user_id,
        window_info=request.window_info
    )

    return ReplayResponse(
        user_id=request.user_id,
        success=replay_result.success,
        measurement_results=replay_result.measurement_results,  # NEW acceptances
        outliers_detected=replay_result.outliers_detected,
        corrections_made=replay_result.corrections_made,
        error=replay_result.error
    )

# Benefit: API has same replay logic as local processing!
# Client controls when to trigger and how to handle results
```

**Key API Benefits:**
1. **Client control:** Client decides when to trigger replay
2. **Consistent behavior:** API and local processing use same service methods
3. **Better UX:** Early poor measurements can be corrected
4. **Backwards compatible:** Existing `/process` endpoint unchanged, `/replay` is new
5. **Separation of concerns:** Processing and replay are separate operations

**API Flow:**
```
Client                          API
  |                              |
  |--POST /process-------------->|
  |  {measurement: {...}}        | (1) Process measurement
  |                              | (2) Check if replay recommended
  |<----Response-----------------|
  |  {result: {...},             |
  |   replay_recommendation: {   |
  |     replay_recommended: true,|
  |     window_info: {...}       |
  |   }}                         |
  |                              |
  |--POST /replay--------------->|
  |  {window_info: {...}}        | (3) Execute replay
  |                              | (4) Return NEW acceptances
  |<----Response-----------------|
  |  {measurement_results: [     |
  |    {id: X, accepted: false}, | <- Client updates: X rejected
  |    {id: Y, accepted: true}   | <- Client updates: Y accepted
  |  ]}                          |
  |                              |
```

---

## Implementation Plan

### Phase 1: Add Database Query Methods (Low Risk)
**Time:** 1-2 hours
**Files:** `weight_values/src/core/database/database.py`, `dynamodb_store.py`

1. Add `get_measurements_in_window()`
2. Add `get_state_snapshot_before()`
3. Unit tests for both methods

**Risk:** Low - pure query methods, no state modification

### Phase 2: Extend ReplayManager (Medium Risk)
**Time:** 2-3 hours
**Files:** `weight_values/src/core/replay/replay_manager.py`

1. Add `check_should_trigger()`
2. Add `execute_inline_replay()`
3. Update `OutlierDetector.get_clean_measurements()` to accept `reference_state` parameter
4. Unit tests for new methods

**Risk:** Medium - modifies replay logic but doesn't change existing methods

### Phase 3: Add Service Layer Method (Medium Risk)
**Time:** 2-3 hours
**Files:** `weight_values/src/aws/services/weight_processor_service.py`, `api/models.py`

1. Add `ProcessWithReplayResponseData` model
2. Add `process_with_replay()` method
3. Unit tests
4. Integration tests with full workflow

**Risk:** Medium - new code path, existing `process_batch()` unchanged

### Phase 4: Simplify local_main.py (Low Risk)
**Time:** 2-3 hours
**Files:** `local_main.py`

1. Remove `process_replay_with_outlier_detection()`
2. Simplify `process_individual_measurements()` to use `process_with_replay()`
3. Simplify `AcceptanceTracker` (no longer needs removal logic)
4. Update main() to single-phase processing
5. Characterization tests to verify behavior

**Risk:** Low - we have characterization tests from previous plan

### Phase 5: Update API (Optional, Low Risk)
**Time:** 1-2 hours
**Files:** `weight_values/src/aws/lambda_handler.py`

1. Add `/process-with-replay` endpoint (or update existing `/process`)
2. Add `enable_replay` parameter to request model
3. API tests

**Risk:** Low - additive change, doesn't break existing API

**Total Time:** 8-13 hours

---

## API Design Decisions

### Option A: New Endpoint (Safer, Backwards Compatible)
```python
# Add new endpoint, keep old one
@app.post("/process")  # Existing, no replay
def process_measurements(request: ProcessRequest):
    return service.process_batch(...)

@app.post("/process-with-replay")  # New, with replay
def process_measurements_with_replay(request: ProcessWithReplayRequest):
    return service.process_with_replay(...)
```

**Pros:** No breaking changes, gradual migration
**Cons:** Two endpoints doing similar things

### Option B: Update Existing Endpoint (Cleaner, Requires Migration)
```python
# Update existing endpoint with replay flag
@app.post("/process")
def process_measurements(request: ProcessRequest):
    if request.enable_replay:  # Default: True
        return service.process_with_replay(...)
    else:
        return service.process_batch(...)
```

**Pros:** Single endpoint, cleaner API
**Cons:** Behavior change for existing clients (mitigated by default=True)

### Option C: Auto-detect (Smartest, Most Complex)
```python
# Automatically use replay for single measurements
@app.post("/process")
def process_measurements(request: ProcessRequest):
    if len(request.measurements) == 1:
        # Single measurement: use replay
        return service.process_with_replay(request.user_id, request.measurements[0])
    else:
        # Batch: use existing logic
        return service.process_batch(request.user_id, request.measurements)
```

**Pros:** Best UX, automatic optimization
**Cons:** More complex routing logic

**Recommendation:** Start with **Option A** (new endpoint), migrate to **Option B** after validation.

---

## Testing Strategy

### Unit Tests (Phase 1-3)
- `test_get_measurements_in_window_returns_measurements_in_range()`
- `test_get_measurements_in_window_empty_when_no_measurements()`
- `test_get_state_snapshot_before_returns_most_recent()`
- `test_check_should_trigger_true_when_measurements_in_window()`
- `test_check_should_trigger_false_when_no_measurements_in_window()`
- `test_execute_inline_replay_corrects_outliers()`
- `test_execute_inline_replay_rollback_on_failure()`
- `test_process_with_replay_triggers_when_window_exists()`
- `test_process_with_replay_no_trigger_when_no_window()`

### Integration Tests (Phase 3-4)
- `test_early_poor_measurement_corrected_by_replay()`
  - Process measurement A (poor fit, low quality, but accepted)
  - Process measurement B (better fit)
  - Verify replay triggered, A discarded, B accepted
  - Verify final state reflects B, not A

- `test_continuous_replay_with_multiple_users()`
  - Process 100 users with measurements over time
  - Verify replay triggers appropriately
  - Verify no state corruption

### API Tests (Phase 5)
- `test_api_process_with_replay_endpoint()`
- `test_api_replay_triggered_metadata_in_response()`

---

## Migration Path

### Week 1: Build Core Methods
- Implement database queries
- Implement replay manager methods
- Unit tests

### Week 2: Service Layer Integration
- Implement `process_with_replay()`
- Integration tests
- Validate behavior with test data

### Week 3: Refactor local_main.py
- Simplify using new service methods
- Characterization tests
- Regression validation

### Week 4 (Optional): Update API
- Add new endpoint
- API tests
- Documentation

---

## Success Criteria

1. ✅ **Simplicity:** local_main.py reduced by >50% lines of code
2. ✅ **Correctness:** Characterization tests pass, replay correctly fixes "early poor measurement" problem
3. ✅ **API Availability:** Same replay logic available to Lambda API
4. ✅ **Backwards Compatible:** Existing `process_batch()` still works, new method is additive
5. ✅ **Well-tested:** >90% coverage for new methods
6. ✅ **No regressions:** All existing tests pass

---

## Council Final Approval

```
-- COUNCIL REVIEW --
This approach addresses all architectural concerns:

• **Butler Lampson (Simplicity)**: ✅ 60% code reduction in local_main.py, cleaner separation of concerns
• **Barbara Liskov (Invariants)**: ✅ Service layer now guarantees correctness internally
• **Alan Kay (Real Problem)**: ✅ Solves the problem for both local AND API users
• **Leslie Lamport (Ordering)**: ✅ Service manages causal ordering, caller doesn't need to know
• **Martin Kleppmann (Consistency)**: ✅ Clear transaction boundaries in execute_inline_replay()
• **Nancy Leveson (Safety)**: ✅ Rollback on failure built into service layer
• **Michael Feathers (Testing)**: ✅ Comprehensive testing strategy with characterization tests

Recommendation: **STRONGLY APPROVE**

This is the right architectural pattern. Proceed with implementation.

Key advantages:
1. DRY principle - replay logic in one place
2. API benefits from same logic
3. Service layer maintains its own correctness
4. Simpler caller code
5. Better testability

Proceed with Phase 1 (database queries) first, then iterate.
-- END COUNCIL --
```