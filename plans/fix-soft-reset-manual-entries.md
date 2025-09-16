# Plan: Fix Soft Reset for Manual/Questionnaire Entries

## Decision
**Approach**: Remove weight change threshold for manual data sources to ensure soft resets always trigger
**Why**: Manual entries represent human corrections that should always recalibrate the Kalman filter, regardless of weight change magnitude
**Risk Level**: Low

## Implementation Steps

1. **Update Reset Manager** - Modify `src/reset_manager.py:66-76` to bypass weight change check for manual sources
   - Keep the existing logic structure
   - Add conditional check: if manual source, always trigger soft reset
   - Preserve cooldown period to prevent reset spam

2. **Add Source-Specific Configuration** - Update `config.toml:80-86` to support per-source thresholds
   - Add `override_sources` list for sources that always trigger resets
   - Keep `min_weight_change_kg` as default for non-manual sources
   - Document behavior in comments

3. **Ensure Retrospective Compatibility** - Verify `src/replay_manager.py` respects soft resets
   - Check that replayed manual entries trigger resets correctly
   - Ensure state restoration doesn't skip reset logic
   - Add logging for reset decisions during replay

4. **Add Test Coverage** - Create `tests/test_soft_reset_manual.py` with specific scenarios
   - Test questionnaire entry with <5kg change triggers reset
   - Test care-team-upload always triggers reset
   - Test cooldown period still applies
   - Test retrospective processing preserves reset behavior

5. **Update Documentation** - Add behavior description to `CLAUDE.md`
   - Document manual source reset policy
   - Explain rationale for always-reset approach

## Files to Change

- `src/reset_manager.py:69-72` - Add bypass logic for manual sources
  ```python
  # After line 69, before weight change check:
  if source in MANUAL_DATA_SOURCES or source in soft_config.get('override_sources', []):
      # Manual entries always trigger soft reset (skip weight change check)
      cooldown_days = soft_config.get('cooldown_days', 3)
      last_reset = ResetManager.get_last_reset_timestamp(state)
      if not last_reset or (timestamp - last_reset).total_seconds() / 86400.0 > cooldown_days:
          return ResetType.SOFT
  ```

- `config.toml:84` - Add override_sources configuration
  ```toml
  override_sources = ["questionnaire", "internal-questionnaire", "care-team-upload", "patient-upload"]
  # Sources in override_sources always trigger soft reset regardless of weight change
  ```

- `tests/test_soft_reset_manual.py` - New test file for manual entry reset behavior

## Acceptance Criteria

- [ ] Questionnaire entries with <5kg change trigger soft reset
- [ ] Care-team-upload entries always trigger soft reset
- [ ] Patient-upload entries always trigger soft reset
- [ ] Cooldown period (3 days) still prevents excessive resets
- [ ] Retrospective processing respects manual entry resets
- [ ] Non-manual sources still require 5kg change for soft reset
- [ ] All existing tests pass
- [ ] No performance regression in processing speed

## Risks & Mitigations

**Main Risk**: Excessive resets from frequent manual entries could destabilize the filter
**Mitigation**: Keep 3-day cooldown period to limit reset frequency; monitor reset events in production

**Secondary Risk**: Backward compatibility with existing processed data
**Mitigation**: Changes only affect future processing; existing state snapshots remain valid

## Out of Scope

- Changing hard reset behavior (30+ day gaps)
- Modifying initial reset logic
- Altering Kalman filter adaptation parameters
- Changing quality scoring thresholds
- Fixing trajectory discontinuity issues (separate concern)