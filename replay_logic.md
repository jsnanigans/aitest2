# Buffered Replay System

## Overview

The buffered replay system allows the weight processor to reconsider measurements with better temporal context. When measurements arrive close together in time, they are buffered and later replayed together from a snapshot, allowing the Kalman filter and quality scoring system to make better decisions with full context.

## How It Works

### 1. Buffering Strategy

Measurements are buffered based on a **time window** (default: 24 hours):

```
Timeline:
M1 (Jan 1)  → accepted → buffer=[M1]
M2 (Jan 20) → 456h gap > 24h → buffer cleared → buffer=[M2]
M3 (Jan 20) → 20min gap < 24h → buffer=[M2, M3]
M4 (Jan 25) → 116h gap > 24h → REPLAY TRIGGERED → buffer cleared
```

**Key behaviors:**
- **ALL measurements** are added to the buffer (both accepted and rejected)
- This allows rejected measurements to be reconsidered with better context during replay
- Buffer is cleared when time gap exceeds the window or after replay completes

### 2. Replay Triggers

Replay is triggered in three scenarios:

#### A. Time Gap Trigger
When the next measurement is **outside the buffer time window** from the last buffered measurement:

```python
time_gap = current_measurement.time - last_buffered_measurement.time
if time_gap >= buffer_hours (24h default) and buffer_size >= 2:
    trigger_replay()
```

**Important:** This happens **BEFORE** processing the current measurement.

#### B. Batch End Trigger
When processing the last measurement in a batch:

```python
if is_last_measurement and buffer_size >= 2:
    trigger_replay()
```

#### C. Buffer Overflow Trigger
When buffer reaches maximum size (default: 100 measurements):

```python
if buffer_size >= max_buffer_measurements:
    trigger_replay()
```

### 3. Replay Process

When replay is triggered:

1. **Restore snapshot:** State is restored to the point before the first buffered measurement
2. **Reprocess measurements:** All buffered measurements are processed sequentially with restored state
3. **Merge results:** Replay results replace original results for buffered measurements
4. **Clear buffer:** Buffer is cleared for the next window

```
Before Replay:
M2: accepted=True  (initial processing)
M3: accepted=False (rejected as anomaly)

During Replay:
- Restore state to before M2
- Reprocess M2 and M3 together
- M2: might be rejected as outlier (7kg jump)
- M3: might be accepted (more reasonable)

After Merge:
M2: accepted=False (replay result)
M3: accepted=True  (replay result)
```

### 4. Result Merging

When replay completes, results are merged to ensure consistency:

```python
# All processing-related fields are updated from replay
updated_result = {
    "accepted": replay_result.accepted,
    "quality_score": replay_result.quality_score,
    "kalman_estimate": replay_result.kalman_estimate,
    "rejection_reason": replay_result.rejection_reason,    # ← Updated
    "processing_stage": replay_result.processing_stage,    # ← Updated
    # Original measurement data preserved
    "value": original.value,
    "unit": original.unit,
    "effective_date_time": original.effective_date_time,
}
```

This ensures no contradictory data like:
```json
{
  "accepted": false,
  "processing_stage": "accepted",  // ← Would be inconsistent!
  "rejection_reason": null         // ← Would be inconsistent!
}
```

## Configuration

Replay behavior is controlled via configuration:

```python
config = {
    "replay": {
        "enabled": True,                    # Enable/disable replay
        "buffer_hours": 24,                 # Time window for buffering (hours)
        "max_buffer_measurements": 100,     # Maximum buffer size
    }
}
```

## API Response

When replay occurs, metadata is included in the response:

```json
{
  "success": true,
  "data": {
    "measurements_processed": 4,
    "measurements_accepted": 3,
    "measurements_rejected": 1,
    "results": [...],
    "replay_metadata": [
      {
        "trigger": "time_gap",
        "buffer_size": 2,
        "replay_from": "2024-01-20T15:00:00",
        "replay_to": "2024-01-20T15:20:00",
        "measurements_replayed": 2,
        "duration_seconds": 0.06,
        "timestamp": "2025-11-06T12:44:58.046463+00:00"
      }
    ]
  }
}
```

## Use Cases

### 1. Correction of False Rejections

A measurement might be rejected due to lack of context, but accepted during replay:

```
Initial: 106kg rejected (seems like outlier)
Replay:  106kg → 100kg processed together
         100kg provides context, 106kg accepted
```

### 2. Detection of Outliers

A measurement might be accepted initially, but rejected during replay with better context:

```
Initial: 150kg accepted (first measurement)
Replay:  100kg → 150kg → 102kg processed together
         150kg clearly an outlier, rejected
```

### 3. Better Kalman State

Processing measurements together allows the Kalman filter to build better state:

```
Sequential:  M1 → M2 (each processed in isolation)
Replay:      M1 + M2 (processed with shared state evolution)
Result:      Better trend estimation, smoother filtering
```

## Implementation Details

### State Snapshots

Before each buffer window, a snapshot is created:

```python
if not buffer:
    buffer_start_time = measurement.measured_at
    state_store.save_state_snapshot(user_id, buffer_start_time)
```

Snapshots include:
- Kalman filter state
- Last processed timestamp
- Measurement history
- Adaptation parameters

### Buffer Management

The buffer is a simple list that accumulates measurements:

```python
buffer: List[Measurement] = []

# Add measurement (both accepted and rejected)
buffer.append(measurement)

# Clear on replay or time gap
buffer.clear()
```

### Replay Execution

```python
def _execute_buffered_replay(user_id, buffer, buffer_start_time, user_height):
    # 1. Restore snapshot
    snapshot = state_store.get_snapshot(user_id, buffer_start_time)
    state_store.save_state(user_id, snapshot)

    # 2. Reprocess measurements
    for measurement in buffer:
        result = process_measurement(...)
        results.append(result)

    # 3. Create new snapshot after replay
    state_store.save_state_snapshot(user_id, datetime.utcnow())

    return results
```

## Testing

Test scenario demonstrating replay trigger:

```json
{
  "measurements": [
    {"uuid": "test-001", "weight": 99.0, "effectiveDateTime": "2024-01-01T10:00:00"},
    {"uuid": "test-002", "weight": 106.0, "effectiveDateTime": "2024-01-20T15:00:00"},
    {"uuid": "test-003", "weight": 100.0, "effectiveDateTime": "2024-01-20T15:20:00"},
    {"uuid": "test-004", "weight": 98.0, "effectiveDateTime": "2024-01-25T12:00:00"}
  ]
}
```

Expected behavior:
1. M1 processed → buffer=[M1]
2. M2 processed → 456h gap → buffer cleared → buffer=[M2]
3. M3 processed → 20min gap → buffer=[M2, M3]
4. M4 arrival → **116h gap → REPLAY TRIGGERED** with buffer=[M2, M3]
5. M4 processed separately

## Performance Considerations

### Memory

- Buffer size limited to `max_buffer_measurements` (default: 100)
- Snapshots stored in DynamoDB with TTL

### Latency

- Replay adds processing time proportional to buffer size
- Typical replay of 2-10 measurements: ~50-200ms
- Batch end replays happen after all measurements processed

### Database Operations

Each replay involves:
- 1 snapshot retrieval (DynamoDB GetItem)
- 1 state restoration (DynamoDB PutItem)
- 1 snapshot creation after replay (DynamoDB PutItem)

## Troubleshooting

### Replay Not Triggering

Check:
1. Buffer has >= 2 measurements
2. Time gap >= `buffer_hours` configuration
3. Replay is enabled in configuration
4. Measurements are being added to buffer (both accepted and rejected)

### Inconsistent Results

If results show contradictions:
1. Verify merge logic updates all fields from replay
2. Check replay service returns `rejection_reason` and `processing_stage`
3. Ensure `model_dump(mode='json')` for datetime serialization

### Unexpected Acceptance/Rejection Changes

Replay can legitimately change results:
- This is the intended behavior
- Measurements are reconsidered with better context
- Check `replay_metadata` to understand what was replayed
- Review quality scores and quality components

## Future Enhancements

Potential improvements:
- Adaptive buffer window based on measurement frequency
- Parallel replay processing for large buffers
- Replay preview mode (show what would change without committing)
- Buffer persistence across Lambda invocations
- Configurable minimum buffer size per trigger type
