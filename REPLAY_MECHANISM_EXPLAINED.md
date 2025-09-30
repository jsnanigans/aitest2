# Replay Mechanism - Corrected Implementation

## Summary of Changes

The replay feature in `local_main.py` was initially implemented incorrectly. This document explains the correct behavior and what was fixed.

---

## ❌ What Was WRONG (Initial Implementation)

### Naive Full Reprocessing

```python
def process_replay_from_beginning():
    # WRONG: Just reset everything and reprocess
    1. Clear all user states
    2. Reprocess ALL measurements from beginning
    3. Clear acceptance tracker
    4. Repopulate with new results
```

**Problems:**
- No outlier detection
- No selective replay
- No state snapshots
- Changes acceptance history (wrong!)
- Just brute-force reprocessing

---

## ✅ What Is CORRECT (Fixed Implementation)

### Windowed Replay with Outlier Detection

```python
def process_replay_with_outlier_detection():
    # CORRECT: Sophisticated outlier detection + selective replay
    1. Choose buffer window (e.g., last 72 hours)
    2. Save state snapshot BEFORE buffer
    3. Detect outliers by comparing to snapshot state
    4. Restore to snapshot
    5. Replay ONLY clean measurements
```

**Based on:** `local_old.py:_process_replay_buffer()` (lines 802-946)

---

## The Real-World Problem Replay Solves

### Scenario: Order-Dependent Acceptance

**Timeline of measurements arriving:**

```
10:00 AM - 82.0 kg (first in window, slightly off but passes threshold)
           → Accepted, Kalman state updates to 82 kg

10:15 AM - 75.0 kg (actually fits trend better at 75-76 kg)
           → REJECTED! Now 7kg away from Kalman state of 82kg

10:30 AM - 83.0 kg
           → Accepted, reinforces the 82kg anchor

10:45 AM - 76.0 kg
           → REJECTED! Still far from 82-83kg state
```

**Result:** The system accepted the outlier (82kg) and rejected good measurements (75kg, 76kg) just because of timing!

### How Replay Fixes This

**Step 1: Buffer Analysis**
```
Buffer window: [82kg, 75kg, 83kg, 76kg, 81kg]
Snapshot state (before window): 75kg trend
```

**Step 2: Outlier Detection**
Each measurement compared against **snapshot state** (75kg), not current state:
- 82kg: 9.3% deviation → **OUTLIER**
- 75kg: 0% deviation → clean
- 83kg: 10.7% deviation → **OUTLIER**
- 76kg: 1.3% deviation → clean
- 81kg: 8% deviation → borderline, kept

**Step 3: Selective Replay**
```
Restore to snapshot state (75kg)
Replay: [75kg, 76kg, 81kg]  <- Only clean measurements
New Kalman state: 77kg (correct trend!)
```

**Result:** Kalman state now reflects the actual trend, not distorted by outliers!

---

## Key Concepts

### 1. Buffer Windows

**Not:** Process all data from beginning
**Instead:** Analyze recent windows (default 72 hours)

**Why?**
- Efficient (don't reprocess everything)
- Realistic (mimics real-time processing)
- Focused (recent data most likely to have issues)

### 2. State Snapshots

**Critical:** Save Kalman state BEFORE buffer window

**Why?**
- Compare against "what we knew before these measurements"
- Detect measurements that don't fit established trend
- Enable restoration for replay

### 3. Outlier Detection Methods

**IQR (Interquartile Range):**
```python
Q1, Q3 = percentiles(weights, [25, 75])
IQR = Q3 - Q1
outlier_threshold = Q3 + (1.5 * IQR)
```

**MAD (Median Absolute Deviation):**
```python
median = np.median(weights)
MAD = np.median(abs(weights - median))
outlier_threshold = median + (3.0 * MAD)
```

**Kalman Prediction Deviation:**
```python
predicted_weight = kalman_predict_from_snapshot()
deviation_percent = abs(measurement - predicted) / predicted
if deviation_percent > 0.10:  # 10% threshold
    mark_as_outlier()
```

**Quality Score Override:**
```python
if quality_score > 0.7:
    # Never mark as outlier - too high quality
    keep_measurement()
```

### 4. Selective Replay

**Not:** Replay all measurements
**Instead:** Replay ONLY measurements that aren't outliers

**Process:**
```python
1. outliers = detect_outliers(buffer, snapshot_state)
2. clean = buffer - outliers
3. restore_state(snapshot)
4. for measurement in clean:
       process_chronologically(measurement)
```

---

## What Replay Does and Doesn't Do

### ✅ Replay DOES:

1. **Detect outliers** in measurement windows
2. **Correct Kalman filter state** by removing bad data
3. **Improve prediction accuracy** for future measurements
4. **Fix order-dependent problems** (like the scenario above)
5. **Report outlier statistics** for data quality analysis

### ❌ Replay DOES NOT:

1. **Change acceptance decisions** - filtered CSV unchanged
2. **Modify acceptance tracker** - original decisions preserved
3. **Update quality scores** - historical scores unchanged
4. **Rewrite measurement metadata** - original metadata kept
5. **Affect API responses** - only internal Kalman state

**Why?** The filtered CSV represents "what happened" during live processing. Replay is an internal correction for Kalman state, not a retrospective re-evaluation of acceptance decisions.

---

## Configuration

From `config/lambda.env.template` and default config:

```python
"replay": {
    "enabled": True,                     # Enable replay processing
    "buffer_hours": 72,                  # Window size for analysis (3 days)
    "min_measurements": 10,              # Minimum data for replay
    "trigger_mode": "time_based",        # When to trigger replay
    "outlier_methods": ["iqr", "mad"],   # Detection methods
    "iqr_multiplier": 1.5,              # IQR sensitivity
    "mad_threshold": 3.0,                # MAD sensitivity
    "rollback_on_error": True,           # Safety: rollback if replay fails
}

"outlier_detection": {
    "enabled": True,
    "iqr_multiplier": 1.5,              # Standard 1.5x IQR rule
    "mad_threshold": 3.0,                # 3 sigma rule
    "quality_score_threshold": 0.7,      # Never outlier if > 0.7 quality
    "kalman_deviation_threshold": 0.10,  # 10% prediction deviation
}
```

---

## Implementation Details

### File Structure

**Main Components:**
- `weight_values/src/core/replay/replay_manager.py` - State restoration & replay execution
- `weight_values/src/core/processing/outlier_detection.py` - Statistical outlier detection
- `weight_values/src/core/processing/buffer_factory.py` - Measurement buffering
- `local_main.py:process_replay_with_outlier_detection()` - Integration

### Flow in `local_main.py`

```python
Phase 1: Individual Processing
  → Process measurements one-by-one
  → Build Kalman state incrementally
  → Track acceptance decisions

Phase 2: Replay (if --enable-replay)
  → For each user with ≥10 measurements:
      1. Choose buffer window (middle point → end)
      2. Save state snapshot before buffer
      3. Convert measurements to dict format
      4. Detect outliers (OutlierDetector)
      5. If outliers found:
         → Replay clean measurements (ReplayManager)
         → Update Kalman state
      6. Report outlier statistics

  → Acceptance tracker UNCHANGED
  → Filtered CSV based on Phase 1 results
```

---

## Testing & Validation

### Test Without Replay (Baseline)
```bash
./local_main.py --max-users 100 --batch-size 50
# Matches reference dataset creation
# No outlier detection or correction
```

### Test With Replay (Production Mode)
```bash
./local_main.py --max-users 100 --enable-replay
# Detects outliers
# Corrects Kalman state
# Reports corrections made
```

### Expected Output

**Without Replay:**
```
Phase 1 (Individual): 1,234 processed, 987 accepted
NOTE: Filtered CSV contains INDIVIDUAL results (no replay)
```

**With Replay:**
```
Phase 1 (Individual): 1,234 processed, 987 accepted
Phase 2 (Replay with Outlier Detection):
  Measurements analyzed: 456
  Outliers detected: 23
  Corrections made: 23
  Successful replays: 12/15 users

NOTE: Replay corrects Kalman state but does NOT change acceptance decisions
      Filtered CSV still contains Phase 1 (individual) acceptance results
```

---

## Comparison: Old vs New

| Aspect | ❌ Old (Wrong) | ✅ New (Correct) |
|--------|--------------|----------------|
| **Approach** | Full reprocessing | Windowed analysis |
| **State Management** | Reset everything | State snapshots |
| **Outlier Detection** | None | IQR, MAD, Kalman deviation |
| **Replay Scope** | All measurements | Only clean measurements |
| **Acceptance Impact** | Clears & repopulates | No change (correct!) |
| **Performance** | Slow (reprocess all) | Fast (window only) |
| **Accuracy** | Same as original | Improved (outliers removed) |
| **Matches Production** | No | Yes (`local_old.py`) |

---

## References

- **Original Implementation:** `local_old.py:_process_replay_buffer()` (lines 802-946)
- **Outlier Detection:** `weight_values/src/core/processing/outlier_detection.py`
- **Replay Manager:** `weight_values/src/core/replay/replay_manager.py`
- **E2E Test:** `tests/api/test_e2e_validation.py`
- **Requirements Doc:** `E.md`

---

## Key Takeaways

1. **Replay is NOT reprocessing** - it's sophisticated outlier detection + selective correction
2. **Windows, not everything** - analyze recent buffers, not full history
3. **State snapshots are critical** - compare against pre-window knowledge
4. **Selective, not exhaustive** - replay only clean measurements
5. **Correction, not re-evaluation** - fix Kalman state, don't change acceptance history

The corrected implementation now matches production behavior and solves the real-world problem of order-dependent acceptance issues!