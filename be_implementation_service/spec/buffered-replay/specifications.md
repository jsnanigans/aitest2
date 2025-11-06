# Buffered Replay Processing - Feature Specification

**Feature ID:** BACK-4631
**Created:** 2025-10-10
**Status:** Specification Phase

## Executive Summary

Add automatic buffered replay processing to the `process` endpoint to ensure measurement evaluations are corrected within a rolling time window. This feature will buffer measurements as they are processed, then trigger replay when the window completes to provide corrected results in the response.

## Problem Statement

Currently, when the `process` endpoint receives multiple measurements:
1. Measurements are processed sequentially, one at a time
2. Each measurement affects the Kalman filter state immediately
3. Early measurements may be evaluated differently than they would be with full context
4. The response contains the initial evaluation results, not the corrected results after replay
5. Users must manually call the separate `replay` endpoint to get corrected evaluations

### Example Issue

When processing measurements [M1, M2, M3, M4, M5]:
- M1 is processed with limited context
- M2 is processed with only M1 as history
- M3 is processed with M1, M2 as history
- ...
- Response shows initial evaluation results
- To get correct results, user must call replay endpoint separately

## Goals

1. **Automatic Correction**: Automatically apply replay processing within the buffer window
2. **Correct Results**: Return final corrected evaluation results in the process endpoint response
3. **No Manual Replay**: Eliminate need for separate replay endpoint calls
4. **Synchronous**: Complete response only after all processing and replay is finished
5. **Performance**: Complete processing in a few seconds for hundreds of measurements

## Requirements

### Functional Requirements

**FR1: In-Memory Buffer Management**
- Buffer measurements as they are processed
- Keep measurements within the configured time window (default: 24 hours from first buffered measurement)
- Clear buffer after replay is triggered

**FR2: Window Completion Triggers**
- Trigger replay when next measurement timestamp > buffer_hours after first buffered measurement
  - Only trigger if buffer contains 2 or more measurements
  - Clear buffer and start new window after replay
- Trigger replay when last measurement in batch is processed
  - Only trigger if buffer contains 2 or more measurements
- **Recurring triggers**: Replay can trigger multiple times per batch as measurements span multiple time windows
- **Minimum buffer size**: Only replay if 2+ measurements in buffer (single measurements don't need replay)

**FR3: Processing Flow**
- Process measurements sequentially through normal pipeline
- Update Kalman state and database as normal
- Simultaneously add measurements to replay buffer
- When window completes, replay all buffered measurements
- Return final results after replay

**FR4: Response Format**
- Response contains final evaluation results after replay
- No provisional, pending, or intermediate status indicators
- Response is identical to current process endpoint format
- Include replay metadata indicating replay was performed

**FR5: Configuration**
- Use existing `replay.buffer_hours` config (default: 24)
- Use existing `replay.max_buffer_measurements` config (default: 100)
- No new configuration required

### Non-Functional Requirements

**NFR1: Performance**
- Process hundreds of measurements in a few seconds
- In-memory buffer only (no persistence between invocations)
- Single Lambda invocation completes all processing and replay

**NFR2: Reliability**
- Handle measurements arriving out of chronological order
- Gracefully handle buffer overflow (> max_buffer_measurements)
- Maintain state consistency if replay fails

**NFR3: Observability**
- Log when replay is triggered
- Log buffer statistics (size, time range)
- Include replay metadata in response

**NFR4: Backward Compatibility**
- No breaking changes to process endpoint API
- Existing response format unchanged
- Feature can be toggled via configuration

## User Scenarios

### Scenario 1: Batch Upload with Multiple Replay Windows

**Context**: User uploads 50 weight measurements spread evenly over 3 days (72 hours)

**Distribution**: ~17 measurements per day, one measurement every ~1.5 hours

**Flow**:
1. Lambda receives batch of 50 measurements
2. Measurements are sorted chronologically
3. **Window 1 (Hours 0-24)**:
   - Process M1-M17 (Day 1 measurements)
   - Each processed normally and added to buffer
   - Buffer starts at M1 timestamp
   - Snapshot created before M1
4. **Process M18** (timestamp at ~25 hours):
   - Time since buffer start: 25 hours > 24 hours
   - Buffer has 17 measurements (≥ 2)
   - **Trigger replay #1** on M1-M17
   - Restore snapshot, replay M1-M17 with full context
   - Clear buffer
   - Add M18 to new buffer (new window starts)
5. **Window 2 (Hours 24-48)**:
   - Process M18-M34 (Day 2 measurements)
   - Each added to buffer
   - Snapshot created before M18
6. **Process M35** (timestamp at ~49 hours):
   - Time since buffer start: 25 hours > 24 hours
   - Buffer has 17 measurements (≥ 2)
   - **Trigger replay #2** on M18-M34
   - Clear buffer
   - Add M35 to new buffer
7. **Window 3 (Hours 48-72)**:
   - Process M35-M50 (Day 3 measurements)
   - Each added to buffer
   - Snapshot created before M35
8. **End of batch reached**:
   - Buffer has 16 measurements (≥ 2)
   - **Trigger replay #3** on M35-M50 (final replay)
9. Return final response with corrected results from all 3 replays

**Result**: Single API call, 3 replay windows automatically triggered, all results corrected, no manual replay needed

### Scenario 2: Out-of-Order Measurements

**Context**: Measurements arrive slightly out of chronological order

**Flow**:
1. Measurements sorted before processing
2. Buffer manages measurements by timestamp
3. Replay processes in correct chronological order
4. Final results reflect correct evaluation

### Scenario 3: Multiple Time Windows (Recurring Replay)

**Context**: Measurements span 5 days with multiple 24-hour windows

**Example**: Measurements at Day 1.0h, Day 1.5h, Day 1.8h, Day 2.2h, Day 3.1h, Day 3.5h, Day 5.0h

**Flow**:
1. Process measurements Day 1.0h, 1.5h, 1.8h
   - All added to buffer (window starts at Day 1.0h)
   - Buffer: [M1, M2, M3]
2. Process Day 2.2h measurement
   - Time since first buffered: 26 hours > 24 hours
   - Buffer has 3 measurements (≥ 2)
   - **Trigger replay #1** on buffer [M1, M2, M3]
   - Clear buffer after replay
   - Start new window, add Day 2.2h to buffer
3. Process Day 3.1h, 3.5h measurements
   - Added to buffer (window started at Day 2.2h)
   - Buffer: [M4, M5, M6]
4. Process Day 5.0h measurement
   - Time since first buffered: 67 hours > 24 hours
   - Buffer has 3 measurements (≥ 2)
   - **Trigger replay #2** on buffer [M4, M5, M6]
   - Clear buffer
   - Add Day 5.0h to buffer
5. End of batch reached
   - Buffer has 1 measurement (< 2)
   - **No replay** (only 1 measurement, doesn't need replay)
6. Return final response with results from both replays

**Result**: Two replay windows triggered automatically, corrected results for all measurements

## Constraints & Assumptions

### Constraints

1. **Lambda Timeout**: Must complete within Lambda timeout (max 15 minutes, typical 5 minutes)
2. **Memory Limits**: Buffer must fit in Lambda memory (typical 1024MB configured)
3. **Config Limits**: `replay.max_buffer_measurements = 100` enforced
4. **Single Invocation**: All processing and replay occurs in one Lambda execution

### Assumptions

1. Measurements are provided in a single batch per API call
2. Batch size is reasonable (< 500 measurements typical)
3. User does not need intermediate results during processing
4. Database supports snapshot functionality (already implemented)
5. Replay service exists and is functional (already implemented)

## Technical Constraints

1. **No State Persistence**: Buffer exists only in memory during Lambda invocation
2. **Synchronous Processing**: Client waits for complete response
3. **Database Consistency**: State must remain consistent through processing and replay
4. **Snapshot Management**: Must create/restore snapshots efficiently

## Success Criteria

1. ✅ Process endpoint returns corrected results after replay
2. ✅ No manual replay endpoint calls required
3. ✅ Performance: 200 measurements processed in < 5 seconds
4. ✅ Zero breaking changes to API contract
5. ✅ Feature can be disabled via configuration
6. ✅ All existing tests continue to pass

## Out of Scope

1. **Cross-Invocation Buffering**: Buffering measurements across multiple API calls
2. **Asynchronous Processing**: Background replay processing
3. **Persistent Buffer**: Storing buffer state in DynamoDB
4. **Multiple Users**: Buffering measurements for multiple users in one call
5. **Streaming Responses**: Partial results during processing

## Dependencies

1. Existing `replay_measurements` service (src/aws/services/replay_service.py)
2. Existing snapshot functionality in state store
3. Existing `process_measurement` function
4. Existing `config.toml` with replay configuration

## Open Questions

None - all clarified by user.

## References

- DEPLOYMENT_USAGE.md - Existing replay endpoint documentation
- example_implementation_local_calls_borken_deps.py - Reference implementation pattern
- config.toml - Replay configuration (buffer_hours, max_buffer_measurements)
- src/aws/services/replay_service.py - Existing replay service
- src/core/processing/processor.py - Measurement processing pipeline
