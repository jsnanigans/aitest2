# Plan: Multi-Threading Performance Improvement

## Decision
**Approach**: Hybrid threading model with user-level parallelism for I/O-bound operations
**Why**: Preserves temporal integrity per user while maximizing parallelism for independent operations
**Risk Level**: Medium

## Implementation Steps

### Phase 1: Thread-Safe Database Layer
1. **Add SQLite connection pooling** - Modify `src/database/database.py:28-39` to implement thread-local storage
2. **Add write queue** - Create `src/database/write_queue.py` with single writer thread for serialized commits
3. **Add read connection pool** - Update `database.py:41-51` with read-only connection pool (5-10 connections)

### Phase 2: Parallel User Processing
1. **Create work queue manager** - Add `src/processing/parallel_manager.py` with ThreadPoolExecutor
2. **Refactor main loop** - Modify `main.py:217-357` to distribute users across worker threads
3. **Add user batch accumulator** - Buffer measurements by user in `main.py:165-215` first pass
4. **Process users in parallel** - Each thread processes one complete user timeline sequentially

### Phase 3: Parallel Visualization
1. **Create viz thread pool** - Add dedicated ThreadPoolExecutor in `main.py:425-493`
2. **Batch visualization tasks** - Process `create_weight_timeline()` calls in parallel (10-20 threads)
3. **Add progress aggregator** - Thread-safe progress reporting in `main.py:449-452`

### Phase 4: Async Retrospective Processing
1. **Create retro task queue** - Add `src/retro/async_processor.py` with background worker pool
2. **Decouple buffer processing** - Modify `main.py:337-356` to queue tasks instead of inline processing
3. **Add result collector** - Aggregate retro results asynchronously in `main.py:358-377`

## Files to Change
- `src/database/database.py` - Thread-local storage, connection pooling
- `src/database/write_queue.py` - NEW: Serialized write queue
- `src/processing/parallel_manager.py` - NEW: Thread pool management
- `main.py:165-215` - User batch accumulation
- `main.py:217-357` - Parallel user processing loop
- `main.py:425-493` - Parallel visualization generation
- `src/retro/async_processor.py` - NEW: Async retro processing
- `config.toml` - Thread pool size configurations

## Threading Strategy

### User Processing (CPU-bound)
- **Pool Size**: `min(cpu_count(), max_users // 10, 8)`
- **Work Distribution**: Queue-based with user batches
- **Synchronization**: User-level isolation, no shared state

### Database Operations (I/O-bound)
- **Write Queue**: Single writer thread with batch commits
- **Read Pool**: 5-10 read connections for parallel queries
- **WAL Mode**: Enable SQLite WAL for read concurrency

### Visualization (I/O + CPU)
- **Pool Size**: 10-20 threads (I/O heavy with Plotly)
- **Work Distribution**: One thread per user visualization
- **Memory Guard**: Limit concurrent visualizations to prevent memory explosion

### Retrospective Processing (CPU-bound)
- **Pool Size**: 2-4 background workers
- **Priority**: Lower than main processing
- **Buffering**: Process in batches when idle

## Performance Targets
- **Current**: ~500-1000 rows/sec single-threaded
- **Target**: 2000-4000 rows/sec with 4-8 threads
- **Visualization**: 10x speedup for multi-user batches
- **Memory**: < 2x current usage with pooling

## Acceptance Criteria
- [ ] SQLite database operations remain ACID compliant
- [ ] User measurement ordering preserved (temporal integrity)
- [ ] Progress reporting remains accurate with aggregation
- [ ] Memory usage under 2GB for 10,000 user dataset
- [ ] No data corruption under concurrent access
- [ ] Graceful degradation if thread pools exhausted

## Risks & Mitigations

**Main Risk**: SQLite concurrency limitations causing lock contention
**Mitigation**: WAL mode + write queue + connection pooling + retry logic

**Risk**: Memory explosion from parallel visualization
**Mitigation**: Semaphore limiting concurrent visualizations + lazy loading

**Risk**: Race conditions in Kalman state updates
**Mitigation**: User-level isolation - each user processed by single thread

## Configuration Schema
```toml
[threading]
enabled = true
user_pool_size = 8  # 0 for auto-detect
viz_pool_size = 16
retro_pool_size = 4
db_read_connections = 10
db_write_batch_size = 100

[threading.memory]
max_concurrent_viz = 20
viz_memory_limit_mb = 100
```

## Implementation Priority
1. **Database thread safety** (Critical - blocks everything else)
2. **Parallel visualization** (Highest ROI - 10x speedup, low risk)
3. **User batch processing** (Medium ROI - 2-4x speedup)
4. **Async retrospective** (Low priority - nice-to-have)

## Out of Scope
- Intra-user parallelism (measurements must stay sequential)
- Distributed processing (single-machine only)
- GPU acceleration
- Caching layer (separate concern)