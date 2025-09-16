# Plan: Retrospective Data Quality Processor

## Decision
**Approach**: Add 72-hour buffering system with batch outlier detection and replay mechanism alongside existing immediate processing
**Why**: Current system latches onto early outliers in measurement sequences, causing downstream quality degradation. Retrospective analysis can identify and exclude outliers for better Kalman filter performance.
**Risk Level**: High (complex state management, potential data corruption)

## Implementation Steps
1. **Create RetroBuffer Component** - Add `src/retro_buffer.py` with 72-hour measurement buffering
2. **Extend Database Schema** - Modify `database.py` to store up to 100 historical states per user
3. **Add Batch Outlier Detection** - Create `src/outlier_detection.py` with IQR, modified Z-score, temporal consistency
4. **Create Replay Manager** - Add `src/replay_manager.py` to restore state and reprocess non-outliers
5. **Integrate with Main Pipeline** - Modify `main.py:stream_process()` to trigger retrospective review
6. **Add Configuration Controls** - Update `config.toml` with retrospective processing settings
7. **Comprehensive Testing** - Unit tests for each component plus integration tests
8. **Monitoring** - Add logging and monitoring hooks for retrospective processing and add them to the output debug files
9. **Visualization** - Update the dashboard to show a simple indication of retrospective processing status and results

## Files to Change
- `main.py:43` - Add buffer initialization and 72-hour trigger logic
- `src/database.py:83` - Add state history tracking methods
- `src/processor.py:90` - Optional integration hook for retrospective mode
- `config.toml` - Add `[retrospective]` configuration section
- `src/retro_buffer.py` - New buffering component (main logic)
- `src/outlier_detection.py` - New batch analysis component
- `src/replay_manager.py` - New state restore and replay component
- `tests/test_retrospective.py` - New comprehensive test suite

## Architecture & Integration

### Core Components
1. **RetroBuffer**: Thread-safe in-memory storage for 72-hour windows
2. **StateHistory**: Database extension tracking last 100 states per user
3. **OutlierDetector**: Statistical analysis for batch outlier identification
4. **ReplayManager**: State restoration and chronological reprocessing

### Processing Flow
```
Immediate Processing (existing):
Measurement → Validation → Kalman → DB Update → Result

Retrospective Processing (new):
72h Trigger → Batch Analysis → Outlier Detection → State Restore → Replay Non-outliers → DB Update
```

### Integration Points
- **main.py**: Buffer management and trigger logic
- **database.py**: State history persistence
- **processor.py**: Replay-aware processing mode (optional flag)

## Data Structures & State Management

### RetroBuffer Schema
```python
{
  "user_id": {
    "measurements": [
      {
        "weight": float,
        "timestamp": datetime,
        "source": str,
        "unit": str,
        "metadata": dict
      }
    ],
    "first_timestamp": datetime,
    "last_timestamp": datetime
  }
}
```

### State History Schema (Database Extension)
```python
{
  "state_history": [
    {
      "timestamp": datetime,
      "kalman_state": array,
      "kalman_covariance": array,
      "kalman_params": dict,
      "measurements_count": int
    }
    # ... up to 100 entries
  ]
}
```

## Core Components

### 1. RetroBuffer (`src/retro_buffer.py`)
- Thread-safe measurement storage
- Automatic 72-hour window management
- Memory-efficient (max 72h * typical measurement frequency)
- Configurable buffer limits

### 2. OutlierDetector (`src/outlier_detection.py`)
- **IQR Method**: Identify values outside Q1-1.5*IQR to Q3+1.5*IQR
- **Modified Z-Score**: Use median absolute deviation for robust outlier detection
- **Temporal Consistency**: Flag measurements with impossible rate-of-change
- **Configurable thresholds** per detection method

### 3. ReplayManager (`src/replay_manager.py`)
- State restoration from history
- Chronological reprocessing of clean measurements
- Rollback capability on failure
- Integration with existing `process_measurement()`

### 4. StateHistory (Database Extension)
- Efficient state snapshots (circular buffer)
- Timestamp-based state retrieval
- Automatic cleanup (keep last 100 states)

## Configuration & Controls

### New Config Section (`config.toml`)
```toml
[retrospective]
enabled = true
buffer_hours = 72
trigger_mode = "time_based"  # "time_based" or "measurement_count"
max_buffer_measurements = 100
state_history_limit = 100

[retrospective.outlier_detection]
iqr_multiplier = 1.5
z_score_threshold = 3.0
temporal_max_change_percent = 0.30
min_measurements_for_analysis = 5

[retrospective.safety]
max_processing_time_seconds = 60
require_rollback_confirmation = false
preserve_immediate_results = true
```

## Error Handling & Safety

### Failure Modes & Mitigations
1. **State Corruption**: Atomic operations with rollback capability
2. **Memory Overflow**: Buffer size limits and automatic cleanup
3. **Processing Timeout**: Configurable time limits with graceful degradation
4. **Outlier Detection Failure**: Fallback to immediate processing results

### Safety Mechanisms
- **Isolated Processing**: Retrospective analysis never affects immediate results until commit
- **Atomic State Updates**: All-or-nothing state transitions
- **Rollback Capability**: Restore previous state on any failure
- **Processing Limits**: Memory and time bounds to prevent resource exhaustion

### Error Recovery
```python
try:
    retrospective_result = process_retrospectively(buffer)
    if retrospective_result.success:
        commit_retrospective_changes(retrospective_result)
    else:
        log_error("Retrospective processing failed", retrospective_result.error)
        # Keep immediate processing results
except Exception as e:
    log_error("Retrospective processing exception", e)
    rollback_to_previous_state()
```

## Performance Considerations

### Memory Usage
- **Per User**: ~72 measurements * 200 bytes = ~14KB per user per buffer
- **Total System**: With 10K active users = ~140MB maximum buffer memory
- **History Storage**: ~100 states * 1KB = ~100KB per user in database

### Processing Overhead
- **Trigger Frequency**: Once per 72-hour window per user (minimal)
- **Batch Processing**: ~50-200ms per retrospective analysis
- **I/O Impact**: Minimal additional database operations (state history append-only)

### Optimization Strategies
- **Lazy Loading**: Load state history only when needed
- **Memory Pool**: Reuse buffer objects to minimize GC pressure

## Testing Strategy

### Unit Tests
- `test_retro_buffer.py`: Buffer management, thread safety, memory limits
- `test_outlier_detection.py`: Statistical methods, edge cases, configuration
- `test_replay_manager.py`: State restoration, replay accuracy, error handling
- `test_state_history.py`: Database operations, serialization, cleanup

### Integration Tests
- `test_end_to_end.py`: Full pipeline with known outlier patterns
- `test_user_0040872d.py`: Specific test case from problem description
- `test_failure_scenarios.py`: Error injection and recovery validation
- `test_performance.py`: Memory usage and processing time validation

### Test Data Requirements
- Synthetic datasets with known outliers
- Real measurement sequences from problematic users
- Edge cases: single measurements, processing failures, concurrent access

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- Create `RetroBuffer` with basic 72-hour windowing
- Extend `database.py` with state history storage
- Update `main.py` with buffer initialization
- Basic configuration support

### Phase 2: Outlier Detection (Week 2)
- Implement statistical outlier detection methods
- Add configurable thresholds and methods
- Unit tests for detection algorithms
- Integration with buffer component

### Phase 3: Replay System (Week 3)
- Create `ReplayManager` with state restoration
- Implement chronological reprocessing
- Add error handling and rollback mechanisms
- Integration testing with existing pipeline

### Phase 4: Safety & Performance (Week 4)
- Add comprehensive error handling
- Implement performance optimizations
- Add monitoring and logging
- End-to-end testing with real data

### Phase 5: Production Readiness (Week 5)
- Configuration validation and documentation
- Performance benchmarking
- Production deployment testing
- Monitoring dashboard integration

## Acceptance Criteria
- [ ] System processes user `0040872d-333a-4ace-8c5a-b2fcd056e65a` correctly, rejecting early outliers
- [ ] Memory usage stays under 200MB for 10K active users
- [ ] Retrospective processing completes within 60 seconds per user
- [ ] Zero data corruption during normal operation and failure scenarios
- [ ] Immediate processing performance unaffected (< 5% overhead)
- [ ] Configuration allows complete disable of retrospective processing
- [ ] Comprehensive test suite with >95% coverage of new components

## Risks & Mitigations

**Main Risk**: Complex state management could introduce data corruption or system instability
**Mitigation**: Phased rollout with extensive testing, atomic operations, comprehensive rollback mechanisms

**Secondary Risk**: Memory usage growth with large numbers of active users
**Mitigation**: Configurable buffer limits, automatic cleanup, memory monitoring

**Performance Risk**: Retrospective processing could impact system responsiveness
**Mitigation**: Processing time limits, background execution, performance benchmarking

## Out of Scope
- Backwards compatibility with existing stored measurements
- Real-time streaming of retrospective results
- Machine learning-based outlier detection (use statistical methods only)
- Integration with external data quality systems
- Historical reprocessing of measurements older than current buffer window
