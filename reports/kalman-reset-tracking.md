# Investigation: Kalman Filter Reset Tracking in Visualization

## Bottom Line

**Root Cause**: Reset events are captured but not propagated to visualization results
**Fix Location**: `src/processing/processor.py:lines 228-233, 584-588`
**Confidence**: High

## What's Happening

The system detects and performs Kalman filter resets correctly, storing reset events in the state. However, the reset information is only added to results as a `reset_event` dict, while the visualization expects `was_reset` and `reset_reason` fields at the top level of each result.

## Why It Happens

**Primary Cause**: Mismatch between data structure expectations
**Trigger**: `src/processing/processor.py:229-233` - Reset info added as nested dict
**Decision Point**: `src/viz/visualization.py:108` - Visualization expects top-level fields

## Evidence

- **Key File**: `src/processing/processor.py:228-233` - Shows reset_event added as nested dict
- **Key File**: `src/viz/visualization.py:108-113` - Expects `was_reset` boolean at top level
- **Search Used**: `rg "was_reset|reset_event"` - Found mismatch in field structure
- **Key File**: `src/processing/reset_manager.py:232-252` - Reset reasons generated correctly

## Next Steps

1. Modify processor.py to add `was_reset`, `reset_reason`, and `gap_days` at result's top level
2. Store reset information in measurement results for persistence across sessions
3. Enhance visualization to show reset type (INITIAL/HARD/SOFT) with specific markers

## Risks

- Missing reset indicators makes it hard to understand weight trajectory changes
- Users can't distinguish between natural variation and post-reset adaptation

## Implementation Recommendations

### 1. Capture Reset Information (Immediate Fix)

In `src/processing/processor.py`, modify lines 228-233 and 584-588:

```python
# Instead of:
if reset_occurred:
    result['reset_event'] = {...}

# Use:
if reset_occurred:
    result['was_reset'] = True
    result['reset_reason'] = reset_event.get('reason', 'unknown')
    result['gap_days'] = reset_event.get('gap_days', 0)
    result['reset_type'] = reset_event.get('type', 'unknown')
```

### 2. Store Reset History (Enhancement)

Add reset tracking to database state:
- Track last N reset events per user
- Include reset type, timestamp, reason, and trigger weight
- Allows visualization of reset patterns over time

### 3. Enhance Visualization (User Experience)

Modify `src/viz/visualization.py` to:
- Show different markers/colors for each reset type
- Display reset reason in hover text
- Add legend entry for reset events
- Consider showing adaptation period as shaded region

## Reset Types Summary

**INITIAL**: First measurement for user
- Most aggressive adaptation (20 measurements/21 days)
- Triggered when no Kalman params exist

**HARD**: 30+ day gaps
- Medium adaptation (10 measurements/7 days)  
- Triggered by extended absence

**SOFT**: Manual entry with 5+ kg change
- Gentle adaptation (15 measurements/10 days)
- Triggered by questionnaire/manual sources

## Technical Details

Reset flow:
1. `ResetManager.should_trigger_reset()` checks conditions
2. `ResetManager.perform_reset()` creates new state and event
3. `processor.py` adds event to result dict
4. `visualization.py` looks for fields that aren't set

The fix requires bridging this gap by flattening the reset information into the result structure that visualization expects.
