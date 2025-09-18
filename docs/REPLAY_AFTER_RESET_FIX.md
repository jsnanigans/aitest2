# Fix: Replay After Reset Issue

## Problem
Daily replay was not working properly after resets occurred. When a reset happened (e.g., accepting a 100.1kg value), the replay system couldn't restore to the correct pre-reset state when it triggered later, preventing it from re-evaluating whether that reset value should have been accepted.

## Root Cause
Snapshots were not being saved immediately after resets occurred. The replay system requires snapshots to restore state, but:
1. For initial resets: Snapshots were attempted before state was saved to DB
2. For soft/hard resets: No snapshots were saved at all
3. When replay triggered (1 hour later), it couldn't find appropriate snapshots

## Solution
Added snapshot saves immediately AFTER reset states are persisted to the database:

### Changes Made

1. **src/processing/processor.py** (lines 260-266, 643-649):
   - Added snapshot save after state persistence when `reset_occurred == True`
   - Ensures snapshot captures the post-reset state
   - Handles both initialization path and main processing path

2. **src/processing/persistence_validator.py** (line 274):
   - Fixed numpy array comparison issue that was causing errors
   - Changed from `if field == 'last_state' and current_val and prev_val:`
   - To: `if field == 'last_state' and current_val is not None and prev_val is not None:`

3. **tests/test_replay_after_reset.py** (new file):
   - Added comprehensive tests for replay after reset functionality
   - Tests verify snapshots are saved for all reset types
   - Tests verify replay can restore to correct state

## How It Works Now

1. **Reset Occurs**: When a measurement triggers a reset (initial, soft, or hard)
2. **State Updated**: Reset updates the Kalman state and parameters
3. **State Persisted**: Updated state is saved to database
4. **Snapshot Saved**: Immediately after persistence, a snapshot is saved
5. **Replay Triggers**: When buffer fills (1 hour later)
6. **State Restored**: Replay finds and restores from the post-reset snapshot
7. **Re-evaluation**: Measurements are re-processed with correct context

## Verification

Run the test suite to verify the fix:
```bash
uv run python -m pytest tests/test_replay_after_reset.py -xvs
```

All three tests should pass:
- `test_snapshot_saved_after_reset`: Verifies snapshots are saved
- `test_replay_can_restore_after_reset`: Verifies replay can restore
- `test_replay_fails_gracefully_without_snapshot`: Verifies graceful failure

## Impact

This fix ensures that:
- Replay can properly re-evaluate measurements after resets
- Outliers accepted during the adaptation period can be corrected
- User weight trajectories remain accurate even after resets
- The system maintains consistency between immediate and replay processing

## Configuration

No configuration changes required. The fix works with existing replay settings:
- `replay.buffer_hours`: Controls when replay triggers (default: 1 hour)
- `replay.enabled`: Must be true for replay to work
- `features.state_persistence`: Must be true for snapshots to be saved