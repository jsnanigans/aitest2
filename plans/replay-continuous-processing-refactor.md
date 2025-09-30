# Replay Continuous Processing Refactor Plan

## Executive Summary

**Goal:** Refactor `local_main.py` from two-phase batch processing to continuous streaming processing where replay runs immediately after each measurement within a sliding 72-hour window.

**Impact:** Critical fix - prevents "early poor measurement pollutes state" problem where initial accepted measurements in a window block better subsequent measurements.

**Risk Level:** HIGH - Major architectural change affecting core processing logic and acceptance tracking.

---

## Problem Statement

### Current Flaws
1. **Two-phase architecture:** All measurements processed first, replay runs second
2. **Acceptance pollution:** Phase 1 acceptances are locked in, not fixed by Phase 2 replay
3. **Non-causal ordering:** Replay happens outside the measurement processing timeline
4. **Arbitrary windows:** Uses middle-point selection instead of sliding windows
5. **No per-measurement triggers:** Replay doesn't check "are there recent measurements?" after each add

### Desired Behavior
- Process measurements **one at a time** per user in chronological order
- After processing each measurement, check if replay should trigger
- **Trigger condition:** If there are measurements in the 72-hour window before current timestamp
- **Replay logic:**
  - Get Kalman state from before the window
  - Evaluate all windowed measurements as if each were "the only one"
  - Discard outliers
  - Re-process clean measurements chronologically
  - Update acceptance tracker with corrected results

---

## Council Review: Architecture Decision

```
-- COUNCIL REVIEW --
Task: Refactor local_main.py to continuous processing with inline replay

Council's Key Concerns:

• **Nancy Leveson**: "What happens if replay fails mid-processing? We could corrupt state for remaining measurements. Need rollback strategy."

• **Martin Kleppmann**: "We're mixing two consistency models: streaming (one-at-a-time) and batch (replay window). What's the transaction boundary? Can we guarantee atomicity?"

• **Michael Feathers**: "We have NO tests for the current behavior. We must add characterization tests before refactoring or we won't know what we broke."

• **Barbara Liskov**: "The acceptance tracker's invariant changes. Currently it's append-only. Now it needs to support 'clear window and re-track'. This affects downstream consumers."

• **Butler Lampson**: "Do we need full continuous processing for local batch scripts? Could we simplify by just fixing the acceptance tracker update in Phase 2?"

Recommendation: Proceed with full refactor, but MUST address:
1. Add characterization tests first
2. Define clear transaction boundaries for replay windows
3. Implement rollback on replay failure
4. Version the output format (flag in JSON results)
-- END COUNCIL --
```

---

## Refactoring Strategy

### Phase 0: Safety Net (Pre-Refactor)
**Goal:** Lock down current behavior before making changes

1. **Create characterization tests**
   - Save current output for a small test dataset
   - Test both with and without replay enabled
   - Verify acceptance counts, filtered CSV row counts, Kalman states

2. **Add replay unit tests**
   - Test `_should_trigger_replay()` logic
   - Test window boundary calculations
   - Test state restoration and rollback

3. **Document current behavior**
   - Trace through current code path with example data
   - Document what acceptance_tracker currently tracks

### Phase 1: Refactor Core Processing Loop
**Goal:** Change from batch-then-replay to continuous-with-inline-replay

#### 1.1 Extract Replay Trigger Logic
```python
def _should_trigger_replay(
    user_id: str,
    current_measurement: Measurement,
    processed_measurements: List[Measurement],
    config: Dict[str, Any]
) -> Tuple[bool, Optional[List[Measurement]]]:
    """
    Determine if replay should trigger after processing current measurement.

    Returns:
        (should_trigger, window_measurements)
    """
    buffer_hours = config.get("replay", {}).get("buffer_hours", 72)
    window_start = current_measurement.measured_at - timedelta(hours=buffer_hours)

    # Get measurements in window (excluding current)
    window_measurements = [
        m for m in processed_measurements
        if window_start <= m.measured_at < current_measurement.measured_at
    ]

    # Trigger if there are measurements in window
    return (len(window_measurements) > 0, window_measurements)
```

#### 1.2 Create Inline Replay Function
```python
def _replay_window_inline(
    service: WeightProcessorService,
    state_store: ProcessorStateDB,
    user_id: str,
    window_measurements: List[Measurement],
    current_measurement: Measurement,
    acceptance_tracker: AcceptanceTracker,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute replay for a measurement window inline during processing.

    Steps:
    1. Snapshot state from before window
    2. Detect outliers using pre-window Kalman state
    3. Restore to pre-window state
    4. Clear acceptances for windowed measurements
    5. Re-process clean measurements chronologically
    6. Update acceptance tracker with corrected results

    Returns:
        Replay result dict with success status, outliers found, corrections made
    """
    window_start = window_measurements[0].measured_at

    try:
        # 1. Save state snapshot before window
        state_store.save_state_snapshot(user_id, window_start)

        # 2. Get pre-window Kalman state for outlier detection
        pre_window_state = state_store.get_state_at_time(user_id, window_start)

        # 3. Detect outliers using pre-window state as baseline
        outlier_detector = OutlierDetector(config.get("outlier_detection", {}), db=state_store)

        # Convert to dict format for detector
        window_dicts = [_measurement_to_dict(m) for m in window_measurements]
        clean_measurements, outlier_indices = outlier_detector.get_clean_measurements(
            window_dicts,
            user_id=user_id,
            reference_state=pre_window_state  # NEW: use pre-window state
        )

        outliers_found = len(outlier_indices)

        if outliers_found == 0:
            return {
                "triggered": True,
                "outliers_found": 0,
                "corrections_made": 0,
                "replay_executed": False,
                "reason": "No outliers detected"
            }

        # 4. Restore state to before window
        state_store.restore_state_snapshot(user_id, window_start)

        # 5. Clear acceptance tracker for window measurements
        for m in window_measurements:
            acceptance_tracker.remove_acceptance(user_id, m.measured_at.isoformat())

        # 6. Re-process clean measurements chronologically
        clean_measurement_objects = [window_measurements[i] for i in range(len(window_measurements)) if i not in outlier_indices]
        clean_measurement_objects.append(current_measurement)  # Include current
        clean_measurement_objects.sort(key=lambda m: m.measured_at)

        for clean_m in clean_measurement_objects:
            response = service.process_batch(user_id, [clean_m])
            acceptance_tracker.mark_batch_results(user_id, [clean_m], response)

        return {
            "triggered": True,
            "outliers_found": outliers_found,
            "corrections_made": outliers_found,
            "replay_executed": True,
            "clean_measurements_processed": len(clean_measurement_objects)
        }

    except Exception as e:
        # Rollback on error
        state_store.restore_state_snapshot(user_id, window_start)
        return {
            "triggered": True,
            "replay_executed": False,
            "error": str(e)
        }
```

#### 1.3 Refactor `process_individual_measurements()`
```python
def process_individual_measurements(
    service: WeightProcessorService,
    state_store: ProcessorStateDB,  # NEW: need state store for replay
    user_measurements: Dict[str, List[Measurement]],
    acceptance_tracker: AcceptanceTracker,
    config: Dict[str, Any],  # NEW: need config for replay settings
    enable_replay: bool = True  # NEW: toggle for replay
) -> Dict[str, Dict[str, Any]]:
    """
    Process measurements one at a time with inline replay triggers.
    """
    results = {}

    for i, (user_id, measurements) in enumerate(user_measurements.items(), 1):
        print(f"[{i}/{len(user_measurements)}] Processing user {user_id[:12]}...")

        user_results = {
            "measurements_processed": 0,
            "measurements_accepted": 0,
            "measurements_rejected": 0,
            "replays_triggered": 0,
            "total_outliers_found": 0,
            "total_corrections": 0,
            "errors": []
        }

        # Sort measurements by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.measured_at)
        processed_so_far = []

        # Process ONE AT A TIME
        for j, measurement in enumerate(sorted_measurements):
            try:
                # Process current measurement
                response = service.process_batch(user_id, [measurement])
                user_results["measurements_processed"] += 1
                user_results["measurements_accepted"] += response.measurements_accepted
                user_results["measurements_rejected"] += response.measurements_rejected

                # Track acceptance
                acceptance_tracker.mark_batch_results(user_id, [measurement], response)
                processed_so_far.append(measurement)

                # Check if replay should trigger
                if enable_replay:
                    should_trigger, window_measurements = _should_trigger_replay(
                        user_id, measurement, processed_so_far, config
                    )

                    if should_trigger:
                        replay_result = _replay_window_inline(
                            service, state_store, user_id,
                            window_measurements, measurement,
                            acceptance_tracker, config
                        )

                        if replay_result.get("replay_executed"):
                            user_results["replays_triggered"] += 1
                            user_results["total_outliers_found"] += replay_result.get("outliers_found", 0)
                            user_results["total_corrections"] += replay_result.get("corrections_made", 0)
                            print(f"  └─ Replay triggered: {replay_result.get('outliers_found', 0)} outliers, {replay_result.get('corrections_made', 0)} corrections")

            except Exception as e:
                error_msg = str(e)
                user_results["errors"].append(f"Measurement {j+1}: {error_msg}")
                print(f"  Error processing measurement {j+1}: {error_msg}")

        results[user_id] = user_results

    return results
```

### Phase 2: Update AcceptanceTracker
**Goal:** Support removing/updating acceptances during replay

#### 2.1 Add removal methods
```python
class AcceptanceTracker:
    """Tracks which measurements were accepted during processing."""

    def remove_acceptance(self, user_id: str, timestamp: str):
        """Remove a previously accepted measurement (for replay)."""
        self.accepted_measurements.discard((user_id, timestamp))

        if user_id in self.user_acceptance_details:
            self.user_acceptance_details[user_id] = [
                info for info in self.user_acceptance_details[user_id]
                if info.get("timestamp") != timestamp
            ]

    def clear_window(self, user_id: str, window_start: datetime, window_end: datetime):
        """Clear all acceptances in a time window for replay."""
        if user_id not in self.user_acceptance_details:
            return

        # Remove from set
        to_remove = [
            (uid, ts) for uid, ts in self.accepted_measurements
            if uid == user_id and window_start <= parse_timestamp(ts) <= window_end
        ]
        for item in to_remove:
            self.accepted_measurements.discard(item)

        # Remove from details
        self.user_acceptance_details[user_id] = [
            info for info in self.user_acceptance_details[user_id]
            if not (window_start <= parse_timestamp(info.get("timestamp", "")) <= window_end)
        ]
```

### Phase 3: Update Main Function
**Goal:** Remove two-phase architecture, use single continuous phase

#### 3.1 Simplify main()
```python
def main():
    # ... argument parsing, config loading, CSV loading ...

    # Single phase: Process with inline replay
    print("\n=== Processing with Continuous Replay ===\")
    print(f"Replay: {'ENABLED' if args.enable_replay else 'DISABLED'}")

    processing_results = process_individual_measurements(
        service=service,
        state_store=state_store,  # NEW
        user_measurements=user_measurements,
        acceptance_tracker=acceptance_tracker,
        config=config,  # NEW
        enable_replay=not args.disable_replay  # NEW
    )

    overall_results = {
        "version": "2.0-continuous-replay",  # NEW: version flag
        "start_time": start_time.isoformat(),
        "processing_results": processing_results,
        "replay_mode": "continuous" if not args.disable_replay else "disabled",
    }

    # Remove Phase 2 replay call entirely
    # (lines 796-811 deleted)

    # Write filtered CSV (unchanged)
    # ... rest of main ...
```

### Phase 4: Migrate Replay Components
**Goal:** Ensure replay_manager and outlier_detector work with continuous model

#### 4.1 Add `reference_state` parameter to OutlierDetector
- Modify `get_clean_measurements()` to accept optional `reference_state`
- Use reference state instead of current state for outlier scoring

#### 4.2 Verify ReplayManager.replay_clean_measurements()
- Ensure it properly restores state before replaying
- Verify chronological ordering of replayed measurements

---

## Testing Strategy

### Pre-Refactor Tests (Phase 0)
1. **Characterization test:** Run current code with 100-user sample, save outputs
2. **Regression baseline:** Acceptance counts, CSV row counts, Kalman state snapshots

### Post-Refactor Tests
1. **Unit tests:**
   - `test_should_trigger_replay_with_window_measurements()`
   - `test_should_trigger_replay_no_window_measurements()`
   - `test_replay_window_inline_removes_outliers()`
   - `test_replay_window_inline_updates_acceptance_tracker()`
   - `test_acceptance_tracker_remove_acceptance()`
   - `test_acceptance_tracker_clear_window()`

2. **Integration test:**
   - Create synthetic user with "early poor measurement" scenario
   - Verify replay correctly discards early measurement and accepts later one
   - Compare Phase 1 acceptances vs final acceptances

3. **Regression test:**
   - Run refactored code with same 100-user sample
   - Compare acceptance counts (expect changes - this is the fix!)
   - Verify no crashes, all users processed

---

## Migration Path & Backward Compatibility

### Flags & Versioning
- Keep `--disable-replay` flag (now disables inline replay)
- Remove `--enable-replay` flag (inline replay is default)
- Add version field to JSON output: `"version": "2.0-continuous-replay"`

### Output Format Changes
- `processing_results` now includes replay stats per user:
  - `replays_triggered`
  - `total_outliers_found`
  - `total_corrections`
- Remove top-level `replay_processing` section (no longer separate phase)

### Rollback Plan
- Keep `local_main.py.backup` with two-phase logic
- If critical issues, can revert with: `git checkout HEAD~1 local_main.py`

---

## Implementation Order

1. ✅ **Review & approve this plan**
2. **Phase 0:** Create characterization tests (1-2 hours)
3. **Phase 2:** Update AcceptanceTracker with removal methods (30 min)
4. **Phase 1.1:** Extract replay trigger logic (30 min)
5. **Phase 1.2:** Create inline replay function (2 hours)
6. **Phase 1.3:** Refactor process_individual_measurements (1 hour)
7. **Phase 3:** Update main() (30 min)
8. **Phase 4:** Migrate replay components (1 hour)
9. **Testing:** Run full test suite (1 hour)
10. **Validation:** Compare outputs, verify fix works (1 hour)

**Total estimated time:** 8-10 hours

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Replay fails mid-processing, corrupts state | Medium | HIGH | Implement try/catch with rollback in `_replay_window_inline()` |
| Acceptance tracker removal breaks downstream | Low | Medium | Version output format, document change |
| Performance degrades with frequent replays | Medium | Medium | Add replay frequency limits, profiling |
| Outlier detector doesn't support reference_state | Low | HIGH | Test with current codebase first, may need weight_values changes |
| Characterization tests miss edge cases | High | Medium | Run on large dataset (1000+ users) |

---

## Dependencies & External Changes

### Changes to `weight_values/` codebase (if needed)
- `OutlierDetector.get_clean_measurements()`: Add optional `reference_state` parameter
- May need to expose `state_store.get_state_at_time()` method
- May need to expose `state_store.save_state_snapshot()` method

### No changes needed to
- API endpoints (this is local processing only)
- DynamoDB schema
- Core Kalman filter logic

---

## Success Criteria

1. ✅ All characterization tests pass (no regressions in disabled-replay mode)
2. ✅ Synthetic "early poor measurement" test shows correct behavior
3. ✅ Filtered CSV has different (corrected) acceptances compared to old two-phase model
4. ✅ No crashes or state corruption on 1000+ user dataset
5. ✅ Replay triggers appropriately (not on every measurement, only when window exists)
6. ✅ Code is cleaner (less code overall, single processing phase)

---

## Council Final Approval

```
-- COUNCIL REVIEW --
This plan addresses the core concerns:

• **Michael Feathers**: ✓ Characterization tests before refactoring
• **Martin Kleppmann**: ✓ Clear transaction boundaries (replay window with rollback)
• **Nancy Leveson**: ✓ Explicit error handling and rollback strategy
• **Barbara Liskov**: ✓ AcceptanceTracker invariant change is explicit and versioned
• **Butler Lampson**: ✓ Simplifies to single-phase processing (less code overall)

Recommendation: PROCEED with implementation in the order specified.

Key watch-outs during implementation:
1. Verify state_store has snapshot/restore methods
2. Test outlier detector with reference_state parameter
3. Monitor replay frequency on real data (may need throttling)
-- END COUNCIL --
```