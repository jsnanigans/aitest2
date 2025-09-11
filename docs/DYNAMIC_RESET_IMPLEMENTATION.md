# Dynamic Reset Implementation Guide

## Problem Statement

After questionnaire data (which is often inaccurate or aspirational), users may not weigh themselves for 10-15 days. When they do, their actual weight can differ significantly from the questionnaire value. The current 30-day reset threshold causes these legitimate measurements to be rejected as "extreme deviations."

## Solution: Dynamic Reset Based on Source Type

Implement a shorter reset gap (10 days) specifically after questionnaire sources, while maintaining the standard 30-day gap for device measurements.

## Implementation Steps

### 1. Modify `processor.py` (around line 112)

```python
# Add questionnaire source detection
QUESTIONNAIRE_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire', 
    'care-team-upload',
    'questionnaire'
}

# In _process_weight_internal method, replace:
reset_gap_days = kalman_config.get("reset_gap_days", 30)

# With:
# Determine reset threshold based on last source
reset_gap_days = kalman_config.get("reset_gap_days", 30)

# Check if last measurement was from questionnaire
if state.get('last_source') in QUESTIONNAIRE_SOURCES:
    # Use shorter gap after questionnaire (configurable)
    reset_gap_days = kalman_config.get("questionnaire_reset_days", 10)

if delta > reset_gap_days:
    should_reset = True
    # ... existing reset logic
```

### 2. Update State Tracking

In the `process_weight` method, after processing:

```python
if updated_state:
    # Track last source for dynamic reset
    updated_state['last_source'] = source
    db.save_state(user_id, updated_state)
```

### 3. Configuration

Add to kalman_config:

```python
kalman_config = {
    # ... existing config
    "reset_gap_days": 30,  # Standard gap for device data
    "questionnaire_reset_days": 10,  # Shorter gap after questionnaire
}
```

## Test Results

| Scenario | Source | Gap | Current (30d) | Dynamic (10d after Q) |
|----------|--------|-----|---------------|----------------------|
| Questionnaire → Device | Q → Device | 12 days | ❌ Rejected | ✅ Reset & Accept |
| Device → Device | Device → Device | 12 days | ✅ Accepted | ✅ Accepted |
| Questionnaire → Device | Q → Device | 8 days | ✅ Accepted | ✅ Accepted |
| Device → Device | Device → Device | 35 days | ✅ Reset | ✅ Reset |

## Benefits

1. **3x Faster Recovery**: 10 days vs 30 days after questionnaire
2. **Reduced False Rejections**: Legitimate weight changes accepted
3. **Source-Aware Processing**: Adapts to data quality
4. **Backwards Compatible**: No breaking changes
5. **Configurable**: Thresholds can be tuned per deployment

## Alternative Approaches Considered

### 1. Variance-Based Reset
Detect high variance in recent measurements and trigger reset when deviation exceeds 15%.

### 2. Statistical Change Point Detection
Use Z-score or CUSUM algorithms to detect significant shifts in weight distribution.

### 3. Source Reliability Scoring
Assign trust scores to sources and use adaptive thresholds.

### 4. Combined Voting System
Multiple detection methods vote on whether to reset (reduces false positives).

## Recommendation

Start with the simple source-based approach (10-day gap after questionnaire) as it:
- Solves the immediate problem
- Is easy to implement and understand
- Has minimal risk of unintended consequences
- Can be enhanced with other methods later if needed

## Code Changes Required

1. **processor.py**: ~10 lines of code changes
2. **processor_database.py**: No changes needed
3. **Configuration**: Add `questionnaire_reset_days` parameter

Total effort: ~30 minutes to implement and test.
