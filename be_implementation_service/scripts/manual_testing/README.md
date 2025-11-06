# Manual Testing Scripts

This directory contains manual test scripts for development and debugging of the weight processor.

## Scripts

### `test_replay_trigger.py`

Manual test script that verifies the buffered replay trigger logic.

**Purpose:**
- Tests time_gap replay trigger
- Validates buffer management
- Confirms replay metadata generation
- Verifies acceptance/rejection decisions after replay

**Usage:**
```bash
python scripts/manual_testing/test_replay_trigger.py
```

**What it tests:**
1. Processes 4 measurements with specific time gaps:
   - M1 at Day 1 (baseline)
   - M2 at Day 20 (456h gap, clears buffer)
   - M3 at Day 20 + 20min (within 24h window, buffered with M2)
   - M4 at Day 25 (116h gap, triggers replay of M2+M3)
2. Verifies that replay is triggered exactly once
3. Verifies buffer contains 2 measurements (M2, M3)
4. Verifies trigger type is 'time_gap'
5. Checks that M2 (106kg outlier) is correctly rejected after replay

**Related:**
- Test data: `../../tests/fixtures/replay_scenario.json`
- Design doc: `../../BUFFERED_REPLAY.md`
- Integration tests: `../../tests/integration/test_buffered_replay.py`

## Adding New Scripts

When adding manual test scripts:
1. Add executable permissions: `chmod +x scripts/manual_testing/your_script.py`
2. Include proper docstring with usage instructions
3. Use fixtures from `tests/fixtures/` when possible
4. Document the script in this README
