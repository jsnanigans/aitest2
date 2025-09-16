# Plan: Global Memory Limits with LRU Eviction

## Decision
**Approach**: Implement global memory cap with hybrid LRU eviction using memory estimation and priority scoring
**Why**: Nancy Leveson identified unbounded memory growth as critical safety issue. With 1000+ users at 100 measurements each, memory exhaustion is guaranteed without global limits.
**Risk Level**: High (system failure without fix)

## Implementation Steps

1. **Create Memory Manager** - Add `src/processing/memory_manager.py` with global memory tracking
   - Singleton pattern for thread-safe global state
   - Track total memory usage across all buffers
   - Implement memory estimation for measurements
   - Provide memory pressure metrics

2. **Add LRU Cache Structure** - Modify `src/processing/replay_buffer.py:46-50`
   - Add OrderedDict for LRU tracking of user access
   - Track last_accessed timestamp per user buffer
   - Add buffer_priority scores (active vs idle)
   - Implement thread-safe access tracking

3. **Implement Memory Estimation** - Add to `memory_manager.py`
   - Per-measurement size calculation (~500 bytes per measurement)
   - Include Python object overhead
   - Account for buffer metadata
   - Provide get_buffer_memory_size() method

4. **Add Eviction Policy** - Extend `replay_buffer.py:285-305`
   - Create _evict_lru_buffers() method
   - Priority scoring: activity_score * 0.6 + data_age_score * 0.3 + size_score * 0.1
   - Evict lowest priority buffers first
   - Emergency eviction for critical memory pressure (>90%)

5. **Integrate Memory Checks** - Modify `replay_buffer.py:71-94`
   - Check global memory before adding measurements
   - Trigger eviction if approaching limits
   - Update LRU tracking on every access
   - Log memory pressure events

6. **Add Configuration** - Update `config.toml:119-135`
   - global_memory_limit_mb (default: 500)
   - eviction_trigger_percent (default: 80)
   - emergency_eviction_percent (default: 90)
   - min_buffers_to_keep (default: 10)

7. **Add Monitoring Metrics** - Extend `replay_buffer.py:248-267`
   - Current memory usage (MB)
   - Memory pressure percentage
   - Evictions per hour
   - Active vs idle buffer ratio
   - Average buffer size

8. **Create Emergency Release** - Add to `memory_manager.py`
   - Force evict all idle buffers (>1 hour inactive)
   - Clear buffers over age threshold
   - Reduce per-user limits temporarily
   - Log emergency state to monitoring

## Files to Change

- `src/processing/memory_manager.py` - NEW: Global memory management singleton
- `src/processing/replay_buffer.py:46` - Add LRU tracking structures
- `src/processing/replay_buffer.py:71` - Add memory checks before add
- `src/processing/replay_buffer.py:285` - Add eviction methods
- `src/processing/replay_buffer.py:248` - Extend stats with memory metrics
- `config.toml:119` - Add memory limit configuration
- `tests/test_memory_limits.py` - NEW: Test memory limits and eviction

## Acceptance Criteria

- [ ] Global memory never exceeds configured limit (500MB default)
- [ ] LRU eviction triggers at 80% memory usage
- [ ] Active users prioritized over idle (>1 hour) users
- [ ] Thread-safe memory tracking with no race conditions
- [ ] Memory pressure metrics available in stats
- [ ] Emergency eviction at 90% prevents OOM
- [ ] Performance: <10ms overhead per measurement add

## Risks & Mitigations

**Main Risk**: Data loss from aggressive eviction
**Mitigation**: Persist critical buffers to disk before eviction, maintain minimum buffer count

**Secondary Risk**: Thread contention on global memory lock
**Mitigation**: Use RLock, batch memory updates, optimize critical sections

**Performance Risk**: Memory calculation overhead
**Mitigation**: Cache size estimates, update periodically not per-operation

## Out of Scope

- Disk persistence of evicted buffers
- Compression of in-memory data
- Distributed memory across processes
- User-specific memory quotas