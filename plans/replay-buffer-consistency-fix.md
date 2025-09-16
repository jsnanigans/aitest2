# Plan: Fix Lost Update Problem in Replay Buffer Processing

## Decision
**Approach**: Implement two-phase snapshot with buffer versioning and atomic exchange
**Why**: Ensures causal consistency by capturing all measurements up to snapshot point while handling concurrent arrivals
**Risk Level**: Medium

## Implementation Steps

### Phase 1: Buffer Versioning System
1. **Add version tracking to ReplayBuffer** - Modify `src/processing/replay_buffer.py:30-50`
   - Add `buffer_version` counter and `snapshot_in_progress` flag per user
   - Add `pending_measurements` queue for arrivals during snapshot
   - Implement version-aware add_measurement method

2. **Create snapshot coordination** - Add new method in `src/processing/replay_buffer.py`
   - `begin_snapshot(user_id, timestamp)` - Mark snapshot start, increment version
   - `finalize_snapshot(user_id)` - Merge pending queue, return frozen buffer
   - `abort_snapshot(user_id)` - Rollback on failure

### Phase 2: Atomic State Capture
3. **Implement coordinated snapshot** - Modify `main.py:400-402`
   - Call `replay_buffer.begin_snapshot(user_id, timestamp)` BEFORE db snapshot
   - Get frozen buffer state with `finalize_snapshot()`
   - Pass frozen buffer to `_process_replay_buffer()`

4. **Update process_replay_buffer** - Modify `main.py:604-650`
   - Accept frozen_buffer parameter instead of live buffer reference
   - Remove direct buffer access, use frozen copy
   - Clear buffer only after successful replay completion

### Phase 3: Concurrent Measurement Handling
5. **Queue management during snapshot** - Modify `src/processing/replay_buffer.py:60-118`
   - Route new measurements to pending queue when snapshot active
   - Atomic swap of queues on finalize
   - Preserve FIFO ordering with logical timestamps

6. **Add replay completion callback** - Modify `src/replay/replay_manager.py:61-149`
   - Add `on_replay_complete` callback parameter
   - Call `replay_buffer.clear_buffer(user_id)` only after successful replay
   - Handle failure cases with buffer restoration

## Files to Change
- `src/processing/replay_buffer.py:30` - Add versioning fields to __init__
- `src/processing/replay_buffer.py:60` - Modify add_measurement for version awareness
- `src/processing/replay_buffer.py:NEW` - Add snapshot coordination methods (lines ~380)
- `main.py:400-416` - Implement two-phase snapshot protocol
- `main.py:604` - Update _process_replay_buffer signature for frozen buffer
- `src/replay/replay_manager.py:61` - Add completion callback support

## Acceptance Criteria
- [ ] No measurements lost between snapshot and replay processing
- [ ] Measurements arriving during replay are queued and preserved
- [ ] Buffer state remains consistent even with concurrent operations
- [ ] Failed replays don't corrupt buffer state
- [ ] Performance impact < 5% for normal processing
- [ ] All existing tests pass with new implementation

## Risks & Mitigations
**Main Risk**: Deadlock if snapshot not properly finalized
**Mitigation**: Add timeout (30s) and automatic abort with cleanup

**Secondary Risk**: Memory growth from pending queues
**Mitigation**: Limit pending queue to 2x max_buffer_measurements, oldest-first eviction

## Alternative Approaches Considered
1. **Full buffer locking** - Simpler but blocks all writes during replay (10-60s)
2. **Copy-on-write buffers** - Complex memory management, Python limitations
3. **Event sourcing** - Over-engineered for this use case

## Implementation Notes
- Use `threading.RLock()` for reentrant locking in buffer operations
- Maintain backward compatibility with existing buffer interface
- Add metrics for snapshot duration and pending queue sizes
- Consider using `collections.deque` for efficient FIFO queues

## Testing Strategy
1. Unit test concurrent add_measurement during snapshot
2. Integration test with simulated delayed replay processing
3. Stress test with 1000 measurements/second during replay
4. Failure injection test for snapshot abort scenarios

## Out of Scope
- Distributed system coordination (single-node only)
- Persistent queue storage (memory only)
- Multi-version concurrency control (single version sufficient)