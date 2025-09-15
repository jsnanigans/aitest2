# Plan: Phase 4 - Batch Optimization

## Summary
Optimize batch processing performance for handling multiple measurements efficiently, reducing latency and DynamoDB operations while maintaining consistency and ordering guarantees.

## Context
- Prerequisites: Phases 0-3 complete
- Current state: Sequential processing, multiple DB calls per measurement
- Target state: Optimized batch processing with minimal DB operations

## Requirements
- Process batches up to 500 measurements efficiently
- Maintain per-user ordering guarantees
- Minimize DynamoDB read/write operations
- Support partial batch failures
- Keep response time under 5 seconds for max batch

## Alternatives

### Option A: Parallel User Processing
**Approach**: Process different users in parallel, sequential within user
- Pros: Good parallelism, maintains ordering
- Cons: Complex coordination, memory overhead
- Risks: Lambda memory limits

### Option B: Bulk DynamoDB Operations
**Approach**: Batch all DB operations, process in memory
- Pros: Minimal DB calls, fast processing
- Cons: Large memory footprint, complex rollback
- Risks: Partial failure handling

### Option C: Stream Processing
**Approach**: Use DynamoDB Streams + SQS for async
- Pros: Unlimited scale, resilient
- Cons: Async complexity, eventual consistency
- Risks: Increased architectural complexity

## Recommendation
**Option A: Parallel User Processing** with Option B techniques where applicable

## High-Level Design

### Processing Pipeline
```
1. Receive batch request
2. Group measurements by user_id
3. Load all user states (batch)
4. Process users in parallel:
   - Sort measurements by timestamp
   - Process sequentially per user
   - Collect results
5. Save all states (batch)
6. Return aggregated results
```

### Optimization Strategies
- Batch DynamoDB operations (BatchGetItem/BatchWriteItem)
- Parallel processing with thread pool
- Connection pooling for DynamoDB
- Result streaming for large responses
- Lazy loading of Kalman filters

## Implementation Plan (No Code)

### Step 1: Create Batch Coordinator
**File**: `src/batch/coordinator.py` (new)
- Batch coordination:
  - Receive batch request
  - Group by user_id
  - Distribute to workers
  - Collect results
  - Handle failures
- Concurrency control:
  - Thread pool (size = CPU cores)
  - Or asyncio for I/O bound
  - Semaphore for memory limits
  - Timeout per user group

### Step 2: Implement Batch State Loader
**File**: `src/batch/state_loader.py` (new)
- Bulk state loading:
  - Collect unique user_ids
  - BatchGetItem (max 100 per call)
  - Handle pagination
  - Cache loaded states
  - Create missing states
- Optimization:
  - Parallel batch requests
  - Compression for large states
  - Partial loading (only needed fields)
  - Bloom filter for existence check

### Step 3: Add Parallel Processing Engine
**File**: `src/batch/parallel_processor.py` (new)
- Parallel execution:
  - Worker pool/threads
  - Process user groups in parallel
  - Maintain result order
  - Error isolation
- Memory management:
  - Limit concurrent users
  - Stream large results
  - Clear processed data
  - Monitor memory usage
- CPU optimization:
  - Vectorized operations where possible
  - Reuse Kalman filter instances
  - Batch numpy operations

### Step 4: Optimize State Persistence
**File**: `src/batch/state_writer.py` (new)
- Bulk state writing:
  - Collect all state updates
  - BatchWriteItem (max 25 per call)
  - Handle unprocessed items
  - Retry with backoff
- Transaction support:
  - Group related writes
  - Use TransactWriteItems
  - Rollback on failure
  - Maintain consistency
- Write optimization:
  - Only write changed fields
  - Compress large histories
  - Batch similar updates

### Step 5: Implement Result Aggregator
**File**: `src/batch/result_aggregator.py` (new)
- Result collection:
  - Maintain insertion order
  - Aggregate statistics
  - Group errors by type
  - Generate summary
- Memory efficiency:
  - Stream large results
  - Paginate if needed
  - Compress responses
  - Use generators
- Statistics:
  - Success/failure counts
  - Average quality scores
  - Processing time per user
  - Rejection reasons

### Step 6: Add Batch Caching Layer
**File**: `src/batch/cache_manager.py` (new)
- Caching strategy:
  - Cache user states during batch
  - Warm cache for frequent users
  - TTL based on activity
  - Invalidate on write
- Implementation:
  - LRU cache with size limit
  - Write-through updates
  - Read-aside for misses
  - Batch cache warming
- Metrics:
  - Hit/miss rates
  - Cache size
  - Eviction rate
  - Memory usage

### Step 7: Create Performance Monitor
**File**: `src/batch/performance_monitor.py` (new)
- Performance tracking:
  - Processing time per user
  - DB operation latency
  - Memory usage
  - CPU utilization
- Bottleneck detection:
  - Identify slow users
  - Track DB throttling
  - Memory pressure
  - Thread contention
- Auto-scaling hints:
  - Recommend batch sizes
  - Suggest parallelism level
  - Memory allocation
  - Timeout adjustments

### Step 8: Implement Batch Retry Logic
**File**: `src/batch/retry_handler.py` (new)
- Retry strategies:
  - Exponential backoff
  - Jitter for thundering herd
  - Circuit breaker pattern
  - Dead letter queue
- Partial failure handling:
  - Track failed items
  - Retry only failures
  - Maintain attempt count
  - Final failure handling
- Idempotency:
  - Track retry attempts
  - Prevent double processing
  - Maintain request state

## Validation & Testing

### Test Strategy
- Load tests with varying batch sizes
- Concurrent batch processing
- Memory pressure tests
- Failure injection tests

### Test Cases
- [ ] Small batch (10 items)
- [ ] Medium batch (100 items)
- [ ] Large batch (500 items)
- [ ] Multiple users in batch
- [ ] Single user many measurements
- [ ] Concurrent batches
- [ ] Partial failures
- [ ] Memory limits
- [ ] Timeout scenarios

### Performance Benchmarks
- 10 measurements: < 200ms
- 100 measurements: < 1 second
- 500 measurements: < 5 seconds
- Memory usage: < 512MB for 500 items
- DB operations: < 10 per 100 measurements

## Risks & Mitigations

### Risk 1: Memory Exhaustion
- **Impact**: Lambda crashes
- **Mitigation**: Streaming, pagination, memory monitoring
- **Monitoring**: Memory usage alarms

### Risk 2: DynamoDB Throttling
- **Impact**: Batch failures
- **Mitigation**: Retry logic, request spreading
- **Monitoring**: Throttle metrics

### Risk 3: Timeout Issues
- **Impact**: Incomplete processing
- **Mitigation**: Chunking, async processing option
- **Monitoring**: Duration metrics

### Risk 4: Ordering Violations
- **Impact**: Incorrect state
- **Mitigation**: Sequential per user, testing
- **Monitoring**: Ordering check metrics

## Acceptance Criteria
- [ ] 500 item batch processes in < 5 seconds
- [ ] Memory usage stays under 512MB
- [ ] DB operations minimized (< 10%)
- [ ] Ordering maintained per user
- [ ] Partial failures handled
- [ ] Results maintain order
- [ ] No data loss or corruption
- [ ] Performance metrics tracked

## Configuration
```yaml
# Environment Variables
BATCH_MAX_SIZE: 500
BATCH_PARALLELISM: 10
BATCH_TIMEOUT_SECONDS: 30
BATCH_MEMORY_LIMIT_MB: 512
CACHE_BATCH_SIZE: 100
CACHE_WARMUP_SIZE: 20
DB_BATCH_GET_SIZE: 100
DB_BATCH_WRITE_SIZE: 25
ENABLE_BATCH_CACHE: true
ENABLE_STREAMING: true
```

## Performance Optimizations

### Database Optimizations
```python
# Before: 100 measurements = 200 DB calls
for measurement in measurements:
    state = get_state(user_id)  # 100 reads
    process(measurement, state)
    save_state(user_id, state)  # 100 writes

# After: 100 measurements = 4 DB calls
states = batch_get_states(user_ids)  # 1 read
for user_id, measurements in grouped:
    process_all(measurements, states[user_id])
batch_save_states(states)  # 3 writes (25 item limit)
```

### Memory Optimizations
- Use generators for large datasets
- Clear processed data immediately
- Compress state histories
- Stream responses over 1MB
- Pool Kalman filter instances

### CPU Optimizations
- Vectorize numpy operations
- Parallel user processing
- Reuse compiled regex
- Cache computed values
- Profile hot paths

## Monitoring Dashboard

### Key Metrics
- Batch size distribution
- Processing time by batch size
- Memory usage percentiles
- DB operation counts
- Cache hit rates
- Error rates by type
- Timeout occurrences

### Alerts
- p99 latency > 10 seconds
- Memory usage > 80%
- DB throttling detected
- Error rate > 5%
- Timeout rate > 1%

## Cost Optimization
- Batch operations reduce DB costs by 80%
- Caching reduces reads by 60%
- Optimal batch size: 50-100 items
- Memory/speed tradeoff analysis
- Consider Step Functions for > 500 items

## Dependencies
- Phases 0-3 complete
- No new external dependencies
- Optional: Redis for distributed cache

## Migration Path
1. Deploy with feature flag
2. Test with small batches
3. Gradually increase batch size
4. Monitor performance metrics
5. Tune parameters
6. Full rollout

## Next Steps
- Phase 5: Monitoring & Operations
- Future: Async processing for large batches
- Future: Distributed processing with Step Functions