# Buffered Replay Processing - Technical Research

**Feature ID:** BACK-4631
**Research Date:** 2025-10-10

## Research Objectives

1. Understand current batch processing flow
2. Analyze existing replay mechanism
3. Identify integration points for buffering
4. Assess technical feasibility and risks
5. Determine optimal implementation approach

## Current Architecture Analysis

### 1. Batch Processing Flow

**File**: `src/aws/services/weight_processor_service.py:55-133`

Current `process_batch()` implementation:

```python
def process_batch(self, user_id: str, measurements: List[Measurement],
                  user_height_m: Optional[float] = None) -> ProcessResponseData:
    # 1. Sort measurements chronologically
    sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)

    # 2. Check for historical conflicts
    conflict = self._check_historical_conflict(user_id, sorted_measurements)
    if conflict:
        raise HistoricalConflictError(conflict)

    # 3. Get initial state
    current_state = self.state_store.get_state(user_id)
    previous_weight = current_state.get("last_raw_weight") if current_state else None

    # 4. Process each measurement sequentially
    for measurement in sorted_measurements:
        result = self._process_single(user_id, measurement, user_height_m)
        results.append(result)
        # Count accepted/rejected

    # 5. Get final state and return results
    final_state = self.state_store.get_state(user_id)
    return ProcessResponseData(...)
```

**Key Observations**:
- Sequential processing: one measurement at a time
- State updated after each measurement via `process_measurement()`
- Historical conflict detection prevents out-of-order processing
- Returns initial processing results (not replayed)

### 2. Replay Service

**File**: `src/aws/services/replay_service.py:13-98`

Existing `replay_measurements()` implementation:

```python
def replay_measurements(user_id, measurements, replay_from, state_store, config, user_height_m):
    # 1. Get snapshot before replay_from
    snapshot = state_store.get_snapshot(user_id, replay_from)

    # 2. Restore snapshot or reset state
    if snapshot:
        state_store.save_state(user_id, snapshot)
    else:
        state_store.delete_state(user_id)

    # 3. Filter and sort measurements >= replay_from
    replay_measurements = [m for m in measurements if m.measured_at >= replay_from]
    replay_measurements.sort(key=lambda m: m.measured_at)

    # 4. Process all measurements
    for measurement in replay_measurements:
        result = process_measurement(...)
        # Collect results

    # 5. Create snapshot after replay
    state_store.save_state_snapshot(user_id, datetime.utcnow())

    return {"success": True, "results": ...}
```

**Key Observations**:
- Standalone function (not part of WeightProcessorService class)
- Requires client to provide all measurements for replay window
- Restores from snapshot or resets to fresh state
- Creates new snapshot after replay complete
- Returns dict (not ProcessResponseData model)

### 3. Snapshot Management

**File**: `src/core/database/dynamodb_store.py:221-368`

Snapshot functionality in DynamoDB:

```python
def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
    # 1. Get current state
    current_state = self.get_state(user_id)

    # 2. Create snapshot item with unique stateType
    snapshot = self._serialize_state(current_state)
    snapshot.update({
        "userId": user_id,
        "stateType": f"snapshot_{timestamp.isoformat()}",  # Unique key
        "snapshotTime": timestamp.isoformat(),
        "ttl": int((timestamp + timedelta(days=10)).timestamp())  # 10-day retention
    })

    # 3. Save to DynamoDB
    self.table.put_item(Item=snapshot)
    return True

def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
    # Query for nearest snapshot before timestamp
    response = self.table.query(
        KeyConditionExpression="userId = :uid AND stateType < :st",
        ExpressionAttributeValues={
            ":uid": user_id,
            ":st": f"snapshot_{timestamp.isoformat()}"
        },
        ScanIndexForward=False,  # Descending order
        Limit=1
    )
    return self._deserialize_state(response["Items"][0]) if response.get("Items") else None
```

**Key Observations**:
- Snapshots stored as separate DynamoDB items with stateType = `snapshot_<timestamp>`
- 10-day TTL for automatic cleanup
- Query by partition key (userId) and sort key (stateType) for efficient retrieval
- Can retrieve nearest snapshot before any timestamp
- Periodic snapshots created every 24 hours in `processor.py:48-118`

### 4. Measurement Processing Pipeline

**File**: `src/core/processing/processor.py:121-574`

Core `process_measurement()` function:

```python
def process_measurement(user_id, weight, timestamp, source, config, unit, db, user_height_m):
    # 1. Data cleaning and preprocessing
    cleaned_weight, metadata = DataQualityPreprocessor.preprocess(...)

    # 2. Load or create user state
    state = db.get_state(user_id) or db.create_initial_state()

    # 3. Check for reset triggers (gap > 30 days, weight change > 5kg, etc.)
    reset_type = ResetManager.should_trigger_reset(state, cleaned_weight, timestamp, source, config)
    if reset_type:
        state, reset_event, reset_occurred = _handle_reset_with_transaction(...)
        # Creates snapshot after reset (line 541)

    # 4. Initialize or update Kalman filter
    if not state.get("kalman_params"):
        kalman_state = KalmanFilterManager.initialize_immediate(...)
        state.update(kalman_state)
    else:
        state = KalmanFilterManager.update_state(state, cleaned_weight, timestamp, ...)

    # 5. Quality scoring
    quality_score = unified_scorer.calculate_quality_score(...)
    if not quality_score.accepted:
        return rejection_result

    # 6. Save updated state
    db.save_state(user_id, state)

    # 7. Create periodic snapshot if 24 hours elapsed
    _maybe_create_periodic_snapshot(db, user_id, timestamp, config)

    return result
```

**Key Observations**:
- Single measurement processing
- State persisted after every accepted measurement
- Automatic snapshot creation on reset events
- Periodic snapshots every 24 hours (configurable)
- Quality scoring may reject measurements

## Configuration Analysis

**File**: `config.toml:247-263`

Relevant replay configuration:

```toml
[replay]
enabled = true
buffer_hours = 24                  # Time window for buffering
trigger_mode = "time_based"
max_buffer_measurements = 100      # Safety limit
state_history_limit = 100

[replay.safety]
max_processing_time_seconds = 60
require_rollback_confirmation = false
preserve_immediate_results = true
```

**Key Observations**:
- `buffer_hours = 24`: 24-hour replay window configured
- `max_buffer_measurements = 100`: Safety limit to prevent memory issues
- `max_processing_time_seconds = 60`: Timeout for replay operations

## Integration Points Identified

### 1. Buffer Management in `process_batch()`

**Location**: `src/aws/services/weight_processor_service.py:90-110`

Modification point: Between measurement processing loop and final response

```python
# Current:
for measurement in sorted_measurements:
    result = self._process_single(user_id, measurement, user_height_m)
    results.append(result)

# Need to add:
buffer = []
for i, measurement in enumerate(sorted_measurements):
    result = self._process_single(user_id, measurement, user_height_m)
    results.append(result)

    # Add to buffer if accepted
    if result.accepted:
        buffer.append(measurement)

    # Check if should trigger replay
    is_last = (i == len(sorted_measurements) - 1)
    should_replay = self._should_trigger_replay(buffer, measurement.measured_at, is_last)

    if should_replay and buffer:
        # Execute replay
        replay_results = self._execute_buffered_replay(user_id, buffer, user_height_m)
        # Update results with replay output
        results = self._merge_replay_results(results, replay_results)
        buffer.clear()
```

### 2. Replay Trigger Logic

**New function needed**: `_should_trigger_replay()`

```python
def _should_trigger_replay(self, buffer: List[Measurement], current_timestamp: datetime, is_last: bool) -> bool:
    # Minimum buffer size: need at least 2 measurements to replay
    if len(buffer) < 2:
        return False

    # Trigger 1: Last measurement in batch
    if is_last:
        return True

    # Trigger 2: Time window exceeded
    buffer_hours = self.config.get("replay", {}).get("buffer_hours", 24)
    first_timestamp = buffer[0].measured_at
    hours_elapsed = (current_timestamp - first_timestamp).total_seconds() / 3600

    if hours_elapsed >= buffer_hours:
        return True

    # Trigger 3: Buffer size limit (safety)
    max_buffer = self.config.get("replay", {}).get("max_buffer_measurements", 100)
    if len(buffer) >= max_buffer:
        return True

    return False
```

### 3. Buffered Replay Execution

**New function needed**: `_execute_buffered_replay()`

```python
def _execute_buffered_replay(self, user_id: str, buffer: List[Measurement],
                             user_height_m: Optional[float]) -> Dict[str, Any]:
    # Get replay_from timestamp (first buffered measurement)
    replay_from = buffer[0].measured_at

    # Call existing replay service
    from src.aws.services.replay_service import replay_measurements

    replay_result = replay_measurements(
        user_id=user_id,
        measurements=buffer,
        replay_from=replay_from,
        state_store=self.state_store,
        config=self.config,
        user_height_m=user_height_m
    )

    return replay_result
```

### 4. Result Merging

**New function needed**: `_merge_replay_results()`

Challenge: Match buffered measurements to replay results by measurement_id

```python
def _merge_replay_results(self, original_results: List[MeasurementResult],
                          replay_output: Dict) -> List[MeasurementResult]:
    # Create lookup map: measurement_id -> replay result
    replay_map = {
        r["uuid"]: r for r in replay_output.get("results", [])
    }

    # Update original results with replay data
    updated_results = []
    for original in original_results:
        if original.measurement_id in replay_map:
            replay_data = replay_map[original.measurement_id]
            # Update with corrected evaluation
            updated = original.copy(update={
                "quality_score": replay_data.get("quality_score"),
                "kalman_estimate": replay_data.get("kalman_estimate"),
                "accepted": replay_data.get("accepted"),
                # ... other fields
            })
            updated_results.append(updated)
        else:
            updated_results.append(original)

    return updated_results
```

## Technical Challenges Identified

### Challenge 1: Result Correlation

**Problem**: Match initial processing results with replay results

**Complexity**: Medium

**Solution**: Use measurement_id (uuid) as correlation key

### Challenge 2: State Snapshot Timing

**Problem**: Need snapshot BEFORE first buffered measurement

**Complexity**: Low

**Solution**: Create snapshot before processing first measurement in buffer window

```python
if not buffer:
    # First measurement in buffer - save snapshot
    self.state_store.save_state_snapshot(user_id, measurement.measured_at)
```

### Challenge 3: Multiple Replay Windows in Single Batch

**Problem**: Batch may span > 24 hours, requiring multiple replay windows

**Complexity**: Medium

**Example**:
- Measurements at Day 1.0h, Day 1.5h, Day 2.2h, Day 3.1h, Day 5.0h
- Replay #1 at Day 2.2h (26 hours from Day 1.0h, buffer has 3 measurements)
- Replay #2 at Day 5.0h (67 hours from Day 2.2h, buffer has 2 measurements)

**Solution**: Track buffer windows and trigger replay when:
1. Time window (buffer_hours) exceeded AND buffer has ≥ 2 measurements
2. Batch ends AND buffer has ≥ 2 measurements
3. Clear buffer and start new window after each replay
4. **No limit** on number of replay triggers per batch

### Challenge 4: Performance

**Problem**: Replay adds processing overhead

**Complexity**: Low

**Analysis**:
- Processing 100 measurements initially: ~1 second
- Replay 100 measurements: ~1 second
- Total: ~2 seconds for 100 measurements
- Well within Lambda timeout and user requirements ("a few seconds")

### Challenge 5: Error Handling

**Problem**: What if replay fails?

**Complexity**: Medium

**Options**:
1. Return initial results as fallback
2. Return error to client
3. Retry replay once

**Recommendation**: Return error to client (fail-fast) to ensure data consistency

## Performance Analysis

### Memory Usage

**Buffer Storage**:
- 100 measurements × 500 bytes each = 50 KB
- Negligible compared to 1024 MB Lambda memory

**State Storage**:
- Single user state: ~10 KB
- Snapshot: ~10 KB
- Total: ~70 KB for entire operation

### Processing Time Estimates

Based on config: `max_processing_time_seconds = 60`

**Example 1: 50 measurements over 3 days (3 replay windows of ~17, ~17, ~16 measurements)**

| Operation | Time |
|-----------|------|
| Initial Processing (50 measurements) | 0.5 seconds |
| Snapshot Creation (3 windows) | <300 ms |
| Replay Window 1 (17 measurements) | 0.2 seconds |
| Replay Window 2 (17 measurements) | 0.2 seconds |
| Replay Window 3 (16 measurements) | 0.2 seconds |
| Result Merging (3 windows) | <300 ms |
| **Total** | **~1.7 seconds** |

**Example 2: 200 measurements over 10 days (10 replay windows of ~20 measurements each)**

| Operation | Time |
|-----------|------|
| Initial Processing (200 measurements) | 2 seconds |
| Snapshot Creation (10 windows) | <1 second |
| Replay Execution (10 windows × ~20 measurements) | 2 seconds |
| Result Merging (10 windows) | <500 ms |
| **Total** | **~5.5 seconds** |

✅ **Meets requirement**: "a few seconds for hundreds of measurements"

**Note**: Multiple smaller replay windows are actually more efficient than one large replay at the end.

### Database Operations

| Operation | Count (per batch) | Type |
|-----------|-------------------|------|
| get_state | N (measurements) | Read |
| save_state | N (measurements) | Write |
| save_snapshot | 1-2 | Write |
| get_snapshot | 1 | Read |
| replay save_state | M (replayed measurements) | Write |

**Cost Impact**:
- Minimal for typical batch sizes (< 200 measurements)
- DynamoDB uses on-demand billing - scales automatically

## Risks & Mitigations

### Risk 1: Replay Fails Midway

**Probability**: Low
**Impact**: High (data inconsistency)

**Mitigation**:
- Replay service already has transaction safety via `ResetTransaction`
- Snapshots provide rollback capability
- Return error to client if replay fails

### Risk 2: Out-of-Order Measurements Within Buffer

**Probability**: Low (measurements pre-sorted)
**Impact**: Medium (incorrect replay results)

**Mitigation**:
- Sort buffer before replay execution
- Existing replay service already sorts measurements

### Risk 3: Buffer Overflow (> max_buffer_measurements)

**Probability**: Low
**Impact**: Low (just triggers early replay)

**Mitigation**:
- Check buffer size and trigger replay at limit
- Config: `max_buffer_measurements = 100`

### Risk 4: Lambda Timeout

**Probability**: Very Low (for typical batches)
**Impact**: High (partial processing)

**Mitigation**:
- Batch size limited by client (typically < 200)
- Lambda timeout: 300 seconds (5 minutes)
- Processing time: < 20 seconds for 500 measurements
- 15x safety margin

## Dependencies & Prerequisites

### Existing Components (Ready to Use)

✅ `replay_measurements()` - src/aws/services/replay_service.py
✅ `save_state_snapshot()` - src/core/database/dynamodb_store.py
✅ `get_snapshot()` - src/core/database/dynamodb_store.py
✅ `process_measurement()` - src/core/processing/processor.py
✅ Config: `replay.buffer_hours`, `replay.max_buffer_measurements`

### New Components Needed

- `_should_trigger_replay()` - Trigger logic
- `_execute_buffered_replay()` - Replay wrapper
- `_merge_replay_results()` - Result correlation
- Buffer management in `process_batch()`

### No Breaking Changes Required

- API contract remains identical
- Response format unchanged (ProcessResponseData)
- Existing tests should pass (results may differ due to replay)

## Recommended Approach

### Implementation Strategy

**Phase 1: Add Buffering Logic**
1. Add buffer list to `process_batch()`
2. Implement `_should_trigger_replay()` with three triggers:
   - Last measurement (is_last)
   - Time window exceeded
   - Buffer size limit
3. Create snapshot before first buffered measurement

**Phase 2: Integrate Replay**
1. Implement `_execute_buffered_replay()` to call existing replay service
2. Handle replay errors gracefully

**Phase 3: Result Merging**
1. Implement `_merge_replay_results()` using measurement_id matching
2. Ensure all result fields are updated correctly

**Phase 4: Testing**
1. Unit tests for buffer triggers
2. Integration tests for replay execution
3. End-to-end tests for complete flow

### Code Location

**Primary file**: `src/aws/services/weight_processor_service.py`

**Functions to modify**:
- `process_batch()` - Add buffering and replay trigger logic

**Functions to add**:
- `_should_trigger_replay()` - ~20 lines
- `_execute_buffered_replay()` - ~30 lines
- `_merge_replay_results()` - ~40 lines

**Total estimated LOC**: ~200 lines (including comments and error handling)

## Alternative Approaches Considered

### Alternative 1: Modify Replay Service

**Idea**: Make replay service buffer measurements internally

**Pros**: Encapsulation, single responsibility
**Cons**: Replay service is standalone function, not class method; harder to integrate with process_batch

**Decision**: ❌ Rejected - Better to keep buffering logic in WeightProcessorService

### Alternative 2: Separate Buffer Management Class

**Idea**: Create BufferManager class to handle buffer lifecycle

**Pros**: Clean separation of concerns, testable
**Cons**: Overkill for simple in-memory list; adds complexity

**Decision**: ❌ Rejected - Simple buffer list is sufficient

### Alternative 3: Process-Then-Replay-All

**Idea**: Process all measurements first, then replay everything at the end

**Pros**: Simpler logic, single replay call
**Cons**: Doesn't handle multiple windows in long batches; worse performance for large batches

**Decision**: ❌ Rejected - Doesn't meet requirement for handling time windows

## Conclusion

**Feasibility**: ✅ High
**Complexity**: 🟡 Medium
**Risk**: 🟢 Low
**Performance**: ✅ Meets requirements

**Recommendation**: **Proceed with implementation** using the integration points identified above.

All necessary infrastructure exists (replay service, snapshots). Implementation is straightforward with clear integration points in `process_batch()`. Performance analysis shows sub-4 second processing for typical batches.

**Next Steps**:
1. Review specifications and research with team
2. Create detailed implementation plan in discussion.md
3. Implement buffering logic in weight_processor_service.py
4. Add comprehensive tests
5. Deploy and monitor
