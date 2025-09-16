# Plan: Write-Ahead Log (WAL) for Crash Recovery

## Decision
**Approach**: Implement append-only WAL with checksums, async writes, and automatic checkpointing
**Why**: Provides durability without blocking main processing, follows proven database patterns
**Risk Level**: Medium

## Implementation Steps

### Phase 1: WAL Infrastructure (2 hours)
1. **Create WAL Manager** - Add `src/replay/wal_manager.py`
   - Append-only log writer with checksums
   - Async write queue with batch flushing
   - File rotation and naming scheme

2. **Define WAL Entry Format** - Update `src/replay/wal_manager.py`
   - Binary format: [header|checksum|timestamp|user_id|operation|payload|footer]
   - Support for START_BUFFER, ADD_MEASUREMENT, END_BUFFER operations
   - CRC32 checksums for corruption detection

3. **Configure Storage** - Update `config.toml`
   - Add `[wal]` section with paths, rotation policy, retention
   - Default: `data/wal/` directory with 7-day retention
   - Max file size: 100MB before rotation

### Phase 2: Buffer Integration (1.5 hours)
1. **Hook ReplayBuffer** - Modify `src/processing/replay_buffer.py:60-117`
   - Log START_BUFFER when creating user buffer
   - Log ADD_MEASUREMENT for each addition
   - Log END_BUFFER when triggering processing

2. **Add Recovery Interface** - Extend `src/processing/replay_buffer.py`
   - Add `recover_from_wal()` method
   - Rebuild buffer state from WAL entries
   - Handle partial/corrupted entries gracefully

3. **State Snapshots** - Modify `src/replay/replay_manager.py:61-149`
   - Log CHECKPOINT before replay operations
   - Log COMMIT after successful replay
   - Log ROLLBACK on failures

### Phase 3: Recovery Process (2 hours)
1. **WAL Reader** - Add to `src/replay/wal_manager.py`
   - Sequential log reader with checksum validation
   - Skip corrupted entries with logging
   - Maintain recovery statistics

2. **Recovery Orchestrator** - Add `src/replay/recovery.py`
   - Scan WAL directory on startup
   - Identify incomplete transactions (START without END)
   - Replay uncommitted buffer operations

3. **Startup Hook** - Modify `main.py`
   - Check for WAL files before processing
   - Run recovery if WAL exists
   - Clean up after successful recovery

### Phase 4: Performance Optimization (1 hour)
1. **Async Write Queue** - In `src/replay/wal_manager.py`
   - Thread-safe queue for log entries
   - Batch writes every 100ms or 1MB
   - fsync() after each batch for durability

2. **Memory-Mapped Files** - Optional enhancement
   - Use mmap for faster reads during recovery
   - Pre-allocate WAL files to reduce fragmentation

3. **Compression** - Add to `src/replay/wal_manager.py`
   - Compress completed WAL files with gzip
   - Transparent decompression during recovery

### Phase 5: Maintenance & Monitoring (1 hour)
1. **Automatic Cleanup** - Add to `src/replay/wal_manager.py`
   - Delete WAL files older than retention period
   - Archive important checkpoints
   - Monitor disk usage

2. **Health Checks** - Add monitoring
   - WAL write latency metrics
   - Recovery time tracking
   - Corruption detection rate

3. **Manual Tools** - Add `scripts/wal_tools.py`
   - `--inspect`: View WAL contents
   - `--verify`: Check integrity
   - `--compact`: Manual compaction

## Files to Change
- `src/replay/wal_manager.py` - [NEW: Core WAL implementation]
- `src/replay/recovery.py` - [NEW: Recovery orchestrator]
- `src/processing/replay_buffer.py:60-117` - [Add WAL logging hooks]
- `src/replay/replay_manager.py:61-149` - [Add transaction markers]
- `main.py:~50` - [Add recovery check on startup]
- `config.toml` - [Add [wal] configuration section]
- `scripts/wal_tools.py` - [NEW: Maintenance utilities]

## Acceptance Criteria
- [ ] System recovers buffer state after crash without data loss
- [ ] WAL writes add <5ms latency to buffer operations
- [ ] Recovery completes in <30s for 10K buffered measurements
- [ ] Corrupted entries are detected and skipped
- [ ] Old WAL files auto-delete after 7 days
- [ ] No blocking on main processing thread

## Risks & Mitigations
**Main Risk**: Disk I/O bottleneck during high throughput
**Mitigation**: Async write queue with batching, monitor write latency, add backpressure if needed

**Secondary Risk**: WAL corruption during system failure
**Mitigation**: CRC32 checksums, skip corrupted entries, maintain backup checkpoint

**Storage Risk**: Disk space exhaustion from WAL growth
**Mitigation**: Automatic rotation at 100MB, 7-day retention, monitoring alerts

## Out of Scope
- Distributed WAL replication
- Point-in-time recovery beyond current buffer
- WAL-based audit logging
- Encryption of WAL files