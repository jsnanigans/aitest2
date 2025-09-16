# Investigation: Kalman Filter Trajectory Discontinuities

## Bottom Line
**Root Cause**: Retrospective buffer processing is restoring Kalman state to timestamps that are too far back, causing trajectory "time travel" jumps
**Fix Location**: `src/retro_buffer.py:340` and `src/replay_manager.py:196`
**Confidence**: High

## What's Happening
The Kalman trajectory shows dramatic non-physical jumps (10-20kg) because retrospective processing restores the filter state to much earlier timestamps than the problematic measurements, effectively causing the trajectory to "jump back in time" and then forward again.

## Why It Happens
**Primary Cause**: Time-based buffer triggering with poor state restoration anchoring
**Trigger**: `src/retro_buffer.py:340` - 72-hour buffer triggers when gaps exceed 3 days
**Decision Point**: `src/replay_manager.py:196` - State restoration finds snapshots too far back in history

## Evidence
- **Gap Analysis**: 134.5-hour (5.6-day) gap between `2025-07-05 05:53:55` (58.8kg) and `2025-07-10 20:22:41` (79.7kg) triggers retrospective processing
- **Buffer Config**: `config.toml:155` - 72-hour buffer means measurements from July 7th onward trigger replay of measurements from July 4th
- **State Restoration**: `database.py:271` - `get_state_snapshot_before()` finds state from July 4th (~75kg), not July 5th (~58kg)
- **Trajectory Jumps**: Three major discontinuities: 89.7→75.0kg (-14.7kg), 74.7→58.9kg (-15.8kg), 58.8→79.7kg (+20.9kg)

## Next Steps
1. **Reduce buffer window** - Change `buffer_hours = 24` in config.toml (from 72) to limit retrospective scope
2. **Improve state anchor selection** - Modify `get_state_snapshot_before()` to find states closer to buffer start time
3. **Add trajectory continuity checks** - Implement max jump detection in replay validation

## Risks
- Current system allows 20kg weight "teleportation" which breaks user trust
- Large gaps in data cause retrospective processing to restore from completely different weight regimes
