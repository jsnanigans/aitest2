# Investigation: Kalman Filter State Divergence

## Bottom Line

**Root Cause**: TypeScript replay mechanism processes additional measurements Python doesn't, causing cumulative 0.156 kg divergence
**Fix Location**: Replay trigger logic consistency between implementations
**Confidence**: High

## What's Happening

TypeScript and Python Kalman filters diverge by 0.156 kg (Python: 57.974 kg, TypeScript: 58.130 kg) at measurement timestamp 2025-09-12T21:02:04. The divergence isn't from the Kalman math itself but from different measurement sequences being processed.

## Why It Happens

**Primary Cause**: TypeScript processes extra measurements via replay mechanism
**Trigger**: `logs_ts.txt:1432` - "Triggering replay for user... trigger=time_window"  
**Decision Point**: `logs_ts.txt:1354,1426` - Extra Kalman updates before state convergence

The TypeScript implementation processes two additional state updates early in sequence:
- State [58.989507, -0.059543] at line 1354
- State [57.721367, -0.081277] at line 1426

These don't appear in Python logs. After replay, both reach [57.023784, 0.000060] but TypeScript has accumulated error from the extra processing steps. This 0.156 kg offset then propagates through all subsequent measurements.

## Evidence

- **Key File**: `logs_ts.txt:1354-1487` - Extra state updates before convergence
- **Search Used**: `grep "State after update: \[5[78]"` - Found divergent state sequences
- **Replay Trigger**: Both use identical `_should_trigger_replay()` logic (minimum 2 measurements, time window, buffer overflow)

## Next Steps

1. Verify replay mechanism is triggered at identical points in both implementations
2. Ensure measurement ordering is deterministic after replay events
3. Check if replay snapshots/restoration differs between implementations

## Risks

- 0.156 kg error compounds over time with each replay cycle
- Different weight trends reported between platforms for same user data
