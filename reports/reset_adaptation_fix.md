# Investigation: Reset Adaptation Not Working

## Summary
The adaptation phase after resets was not working because measurements were being rejected by the extreme deviation check before the adaptation parameters could be applied.

## The Complete Story

### 1. Trigger Point
**Location**: `src/processor.py:149-161`
**What happens**: Reset triggers correctly and creates state with adaptation parameters
```python
if reset_type:
    state, reset_event = ResetManager.perform_reset(
        state, reset_type, timestamp, cleaned_weight, source, config
    )
    reset_occurred = True
```

### 2. State Initialization Issue #1
**Location**: `src/processor.py:183`
**What happens**: Kalman initialization was overwriting the state
**Why it matters**: Lost reset_type and reset_parameters
```python
# BEFORE (BUG):
state = KalmanFilterManager.initialize_immediate(...)

# AFTER (FIXED):
kalman_state = KalmanFilterManager.initialize_immediate(...)
state.update(kalman_state)
```

### 3. Counter Increment Issue
**Location**: `src/processor.py:213 and 452`
**What happens**: measurements_since_reset only incremented during initialization
**Why it matters**: Counter stuck at 1 for subsequent measurements
```python
# ADDED to main acceptance path:
state['measurements_since_reset'] = state.get('measurements_since_reset', 0) + 1
```

### 4. Extreme Deviation Check Issue
**Location**: `src/processor.py:355-372`
**What happens**: Strict 20% deviation threshold applied before adaptation check
**Why it matters**: Rejected valid measurements during adaptation phase
```python
# BEFORE (BUG):
extreme_threshold = processing_config.get('extreme_threshold', 0.20)

# AFTER (FIXED):
if measurements_since_reset < adaptation_measurements:
    if quality_acceptance_threshold == 0:
        extreme_threshold = float('inf')  # Disable check
    else:
        extreme_threshold = 0.5  # 50% during adaptation
```

### 5. Final Outcome
**Location**: Multiple fixes combined
**Result**: Adaptation phase now works correctly
**Root Cause**: Multiple issues preventing adaptation parameters from being applied

## Key Insights

1. **Primary Cause**: Extreme deviation check happened before adaptation parameters were considered
2. **Contributing Factors**: 
   - State being overwritten during Kalman initialization
   - Counter not incrementing properly
   - No adaptation logic in deviation check
3. **Design Intent**: Allow lenient acceptance during initial 30 measurements after reset

## Evidence Trail

### Files Examined
- `src/processor.py`: Main processing logic with multiple issues
- `src/reset_manager.py`: Correctly created reset parameters
- `src/kalman.py`: Initialize function was overwriting state
- `config.toml`: Correct configuration with adaptation_measurements=30

### Search Commands Used
```bash
rg -n "measurements_since_reset" src/
rg -n "reset_parameters" src/
rg -n "extreme_threshold" src/
rg -n "kalman_deviation" src/
```

### Test Results
- Before fixes: 56.7% acceptance rate (34/60 accepted)
- After fixes: 95.0% acceptance rate (57/60 accepted)
- Remaining rejections are genuine outliers (~95kg vs normal ~115kg)

## Confidence Assessment
**Overall Confidence**: High
**Reasoning**: 
- Found multiple concrete bugs in the code
- Fixes directly address the root causes
- Test results show dramatic improvement
- Remaining rejections are appropriate

## Alternative Explanations
None - the bugs were clear and the fixes directly addressed them.
