# Investigation: Replay System Not Triggered for Outlier Rejection

## Bottom Line
**Root Cause**: Pipeline calls non-existent ReplayManager methods (`validate_replay` and `process_replay`)
**Fix Location**: `src/processing/pipeline.py:376-380`
**Confidence**: High

## What's Happening
User e751ebe4-3e13-423d-bf50-88a9dd13f132 has a BMI value (34.565) misinterpreted as weight on 2025-04-10 at 19:27:05, creating a 100.1kg spike. Three correct measurements (~79kg) follow within the hour but aren't used to correct the error because the replay system never executes.

## Why It Happens
**Primary Cause**: Method name mismatch between pipeline and ReplayManager
**Trigger**: `src/processing/pipeline.py:376` - Calls `replay_manager.validate_replay()`
**Decision Point**: `src/processing/pipeline.py:380` - Calls `replay_manager.process_replay()`

The pipeline attempts to call:
- `replay_manager.validate_replay(user_id, measurements, outliers)` 
- `replay_manager.process_replay(user_id, measurements, outliers)`

But ReplayManager only has:
- `replay_clean_measurements(user_id, clean_measurements, buffer_start_time)`

These calls silently fail (likely caught by exception handling), so replay never runs.

## Evidence
- **Key File**: `src/processing/pipeline.py:376-380` - Invalid method calls
- **Search Used**: `rg "def " src/replay/replay_manager.py` - Shows actual methods
- **Data**: 2025-04-10 shows BMI→weight conversion accepted, subsequent correct values rejected

## Next Steps
1. Fix method calls in `pipeline.py:_perform_replay()` to use `replay_clean_measurements`
2. Add proper outlier filtering before calling replay
3. Increase `buffer_hours` from 1 to at least 24 for better outlier detection window

## Risks
- Silent failures continue allowing bad data through
- Other users likely affected by same issue
- Replay system has never actually run in production