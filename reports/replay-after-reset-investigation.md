# Investigation: Daily Replay Not Correcting After Reset

## Bottom Line
**Root Cause**: Snapshots are not saved after resets, preventing replay from restoring to pre-reset state
**Fix Location**: `src/processing/processor.py:186` (after reset transaction)
**Confidence**: High

## What's Happening
After a reset occurs (e.g., on 100.1kg value), the replay system cannot restore to the correct pre-reset state because no snapshot was saved immediately after the reset. When replay triggers 1 hour later, it tries to restore state but finds no valid snapshot, causing it to skip replay processing entirely.

## Why It Happens
**Primary Cause**: Missing snapshot save after reset operations
**Trigger**: `main.py:445` - Snapshots only saved when buffer is ready (1 hour later)
**Decision Point**: `src/processing/processor.py:184-186` - Reset performed but no snapshot saved

### Evidence Chain:
1. **Reset occurs**: `processor.py:184` performs reset via `_handle_reset_with_transaction`
2. **No snapshot saved**: After line 186, processing continues without saving snapshot
3. **Buffer fills**: Measurements added to replay buffer over next hour
4. **Replay triggers**: `main.py:443` buffer becomes ready after 1 hour
5. **Snapshot saved too late**: `main.py:445` saves snapshot AFTER reset already applied
6. **Replay fails**: `replay_manager.py:233` cannot find snapshot before buffer start time

## Evidence
- **Key File**: `src/processing/processor.py:186` - Reset completes without snapshot
- **Search Used**: `rg "save_state_snapshot"` - Only found saves at buffer trigger time
- **Config**: `config.toml:141` - `buffer_hours = 1` confirms 1-hour delay
- **Replay Restore**: `replay_manager.py:256-257` - "No snapshot found" causes abort

## Next Steps
1. Add `db.save_state_snapshot(user_id, timestamp)` immediately after line 186 in processor.py when `reset_occurred == True`
2. Ensure snapshot is saved BEFORE any new measurements are processed post-reset
3. Test with user `e751ebe4-3e13-423d-bf50-88a9dd13f132` to verify replay now corrects the 100.1kg value

## Risks
- Without fix, all post-reset measurements cannot be replayed/corrected
- Users with resets will have permanently incorrect trajectories if outliers accepted during adaptation period