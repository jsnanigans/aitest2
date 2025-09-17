# Detailed Replay Test Scenarios

## 1. ReplayBuffer Isolated Tests

### Buffer Lifecycle Tests

#### test_buffer_creation_with_config
```python
def test_buffer_creation_with_config():
    """Test buffer creation with various configurations."""
    # Given: Different configuration scenarios
    configs = [
        {'buffer_hours': 1, 'max_buffer_measurements': 10},
        {'buffer_hours': 72, 'max_buffer_measurements': 100},
        {'buffer_hours': 0.5, 'max_buffer_measurements': 5}  # 30 minutes
    ]

    # When: Creating buffers with each config
    # Then: Verify correct initialization and limits
```

#### test_buffer_cleanup_removes_all_data
```python
def test_buffer_cleanup_removes_all_data():
    """Test that cleanup() properly releases all resources."""
    # Given: Buffer with multiple users and measurements
    # When: Calling cleanup()
    # Then: All internal data structures are cleared
    # And: Memory is released (check with sys.getsizeof)
```

### Buffer Operation Tests

#### test_add_measurement_boundary_conditions
```python
def test_add_measurement_boundary_conditions():
    """Test adding measurements at exact boundary conditions."""
    # Test cases:
    # 1. Add at exactly buffer_hours limit
    # 2. Add 1 nanosecond before window expires
    # 3. Add exactly at max_buffer_measurements
    # 4. Add when buffer is empty
    # 5. Add duplicate timestamps
```

#### test_buffer_window_sliding
```python
def test_buffer_window_sliding():
    """Test that buffer window slides correctly over time."""
    # Given: Buffer with 1-hour window
    # When: Adding measurements over 3 hours
    # Then: Old measurements are automatically removed
    # And: Only last hour of measurements remain
```

### Trigger Condition Tests

#### test_time_based_trigger_accuracy
```python
def test_time_based_trigger_accuracy(freezer):
    """Test time-based triggers fire at correct times."""
    # Use freezegun to control time
    # Given: Buffer with 1-hour trigger
    # When: Time advances by 59 minutes -> No trigger
    # When: Time advances by 61 minutes -> Trigger fires
```

#### test_count_based_trigger_edge_cases
```python
def test_count_based_trigger_edge_cases():
    """Test count-based triggers with edge cases."""
    # Test: Exactly at limit
    # Test: One below limit
    # Test: Rapid additions
    # Test: Trigger reset after processing
```

### Thread Safety Tests

#### test_concurrent_add_from_multiple_threads
```python
def test_concurrent_add_from_multiple_threads():
    """Test thread-safe concurrent additions."""
    # Given: 10 threads, each adding 100 measurements
    # When: All threads run simultaneously
    # Then: All 1000 measurements are captured
    # And: No data corruption occurs
    # And: No deadlocks occur
```

#### test_concurrent_trigger_and_clear
```python
def test_concurrent_trigger_and_clear():
    """Test triggering and clearing happen atomically."""
    # Thread 1: Continuously adds measurements
    # Thread 2: Checks for triggers
    # Thread 3: Clears triggered buffers
    # Verify: No measurements lost, no double processing
```

## 2. ReplayManager Isolated Tests

### State Management Tests

#### test_backup_state_immutability
```python
def test_backup_state_immutability():
    """Test that backed up states are immutable."""
    # Given: State with numpy arrays
    # When: Creating backup and modifying original
    # Then: Backup remains unchanged (deep copy verified)
```

#### test_restore_with_partial_state
```python
def test_restore_with_partial_state():
    """Test restoration with incomplete state data."""
    # Test missing fields
    # Test None values
    # Test corrupted numpy arrays
    # Test incompatible data types
```

### Replay Chronology Tests

#### test_measurements_replayed_in_order
```python
def test_measurements_replayed_in_order():
    """Test measurements are replayed chronologically."""
    # Given: Measurements with random insertion order
    # When: Replaying
    # Then: Processing order matches timestamp order
```

#### test_replay_with_time_gaps
```python
def test_replay_with_time_gaps():
    """Test replay handles measurement gaps correctly."""
    # Scenarios:
    # - 1-hour gap between measurements
    # - 1-week gap (triggers hard reset)
    # - Microsecond differences
    # - Duplicate timestamps
```

### Rollback Tests

#### test_rollback_on_processor_exception
```python
def test_rollback_on_processor_exception():
    """Test state rollback when processor fails."""
    # Given: Mock processor that fails on 3rd measurement
    # When: Replaying 5 measurements
    # Then: State rolled back to original
    # And: Error details logged
```

#### test_nested_rollback_prevention
```python
def test_nested_rollback_prevention():
    """Test that nested operations don't cause double rollback."""
    # Simulate nested replay attempts
    # Verify only one rollback occurs
```

## 3. Integration Tests (Still Isolated)

### Buffer-to-Manager Flow

#### test_buffer_trigger_to_replay_flow
```python
def test_buffer_trigger_to_replay_flow():
    """Test complete flow from buffer trigger to replay completion."""
    # Given: Mock components
    # When: Buffer triggers with outliers
    # Then: Manager processes correctly
    # And: Clean measurements replayed
    # And: Buffer cleared
```

### Memory Management Tests

#### test_memory_limit_enforcement
```python
def test_memory_limit_enforcement():
    """Test that memory limits are enforced."""
    # Given: Global memory limit of 100MB
    # When: Adding measurements approaching limit
    # Then: LRU eviction occurs
    # And: Active users preserved
```

#### test_memory_leak_detection
```python
def test_memory_leak_detection():
    """Test for memory leaks in long-running scenarios."""
    # Run 1000 cycles of:
    # - Add measurements
    # - Trigger processing
    # - Clear buffers
    # Verify: Memory usage remains stable
```

## 4. Failure Scenario Tests

### Database Failure Tests

#### test_snapshot_not_found_handling
```python
def test_snapshot_not_found_handling():
    """Test graceful handling when no snapshot exists."""
    # Mock database returns None
    # Verify appropriate error response
    # Verify no state corruption
```

#### test_database_timeout_during_restore
```python
def test_database_timeout_during_restore():
    """Test timeout handling during state restoration."""
    # Mock database with artificial delay
    # Verify timeout triggers
    # Verify rollback occurs
```

### Extreme Data Tests

#### test_extreme_weight_values
```python
def test_extreme_weight_values():
    """Test handling of extreme weight values."""
    # Test: 0 kg
    # Test: 999 kg
    # Test: Negative values
    # Test: NaN/Inf values
    # Test: Very small changes (0.001 kg)
```

#### test_massive_buffer_sizes
```python
def test_massive_buffer_sizes():
    """Test with unusually large buffers."""
    # 10,000 measurements in one buffer
    # Verify performance remains acceptable
    # Verify no integer overflows
```

## 5. Performance Benchmark Tests

### Throughput Tests

#### test_buffer_addition_throughput
```python
def test_buffer_addition_throughput(benchmark):
    """Benchmark buffer addition rate."""
    # Measure: Additions per second
    # Target: >10,000 additions/second
    # Use pytest-benchmark
```

#### test_replay_processing_speed
```python
def test_replay_processing_speed(benchmark):
    """Benchmark replay processing rate."""
    # Measure: Measurements replayed per second
    # Target: >1,000 measurements/second
    # Profile with different buffer sizes
```

### Latency Tests

#### test_trigger_detection_latency
```python
def test_trigger_detection_latency():
    """Measure trigger detection latency."""
    # Measure time from condition met to trigger fired
    # Target: <10ms
```

#### test_state_restoration_latency
```python
def test_state_restoration_latency():
    """Measure state restoration time."""
    # With various snapshot sizes
    # Target: <50ms for typical state
```

## 6. Mock Utilities

### TimeMocker
```python
class TimeMocker:
    """Control time for testing."""
    def __init__(self, start_time):
        self.current_time = start_time

    def advance(self, seconds):
        self.current_time += timedelta(seconds=seconds)

    def now(self):
        return self.current_time
```

### FailureInjector
```python
class FailureInjector:
    """Inject failures at specific points."""
    def __init__(self):
        self.failure_points = {}

    def inject_at(self, call_number, exception):
        self.failure_points[call_number] = exception

    def check_and_fail(self):
        # Raise exception if at failure point
```

### MetricsCollector
```python
class MetricsCollector:
    """Collect metrics during tests."""
    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, metric_name, value):
        self.metrics[metric_name].append(value)

    def get_stats(self, metric_name):
        # Return min, max, avg, p95, p99
```

## Test Data Patterns

### Realistic Weight Patterns
- Gradual weight loss: -0.2 kg/day with daily variations
- Weight gain: +0.1 kg/day with weekend spikes
- Maintenance: ±0.5 kg variation around baseline
- Rapid change: -2 kg/week (medication effect)
- Measurement noise: ±0.3 kg random variation

### Outlier Patterns
- Single spike: One measurement +5kg from trend
- Double measurement: Same weight reported twice
- Scale malfunction: Series of identical readings
- Data entry error: 850 lbs instead of 85 lbs
- Network duplicate: Same measurement multiple timestamps

## Success Metrics

1. **Coverage**: 100% line coverage for replay components
2. **Performance**: All tests complete in <10 seconds
3. **Reliability**: Tests pass 100 times in a row
4. **Isolation**: Zero dependencies on external systems
5. **Clarity**: Each test has single clear purpose

## Implementation Priority

1. **Phase 1** (Critical):
   - Basic buffer operations
   - State backup/restore
   - Thread safety tests

2. **Phase 2** (Important):
   - Failure scenarios
   - Rollback mechanisms
   - Memory management

3. **Phase 3** (Nice to have):
   - Performance benchmarks
   - Extreme edge cases
   - Stress tests