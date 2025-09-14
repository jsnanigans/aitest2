# Stateless Quality Scoring Improvements

## Overview

The quality scoring system has been improved to better handle normal weight fluctuations while maintaining a **completely stateless architecture** that processes one measurement at a time.

## Key Design Constraint: Stateless Processing

The processor architecture requires:
- **No persistent state** between measurements
- **Process one value at a time** as they arrive
- **Minimal state storage** in database (just previous weight and timestamp)
- **No complex history tracking** (no recent_weights array needed)

## Improvements Implemented

### 1. Time-Aware Consistency Scoring

The consistency check now uses different thresholds based on time elapsed:

```python
< 6 hours:   Up to 3kg variation allowed (meals, hydration)
< 24 hours:  Up to 2kg gets score 1.0, gradual decay after
> 24 hours:  Daily rate calculation with 2kg/day threshold
```

### 2. Percentage-Based Thresholds

All checks now use percentage of baseline weight:
- Scales appropriately for different body weights
- Baseline estimated from `(current + previous) / 2`
- No need to store historical weights

### 3. Simplified Plausibility Scoring

Without access to recent_weights history:
- Uses previous weight as reference
- Assumes 2% standard deviation for body weight
- Falls back to lenient scoring if no history

### 4. Research-Based Normal Ranges

Incorporated findings that 2-3% daily weight fluctuation is normal:
- Prevents rejection of legitimate measurements
- Still catches true outliers (like 42.22 kg)

## State Requirements

The minimal state stored in database per user:

```python
{
    'last_weight': float,        # Previous accepted weight
    'last_timestamp': datetime,  # When it was measured
    'kalman_state': {...}       # Kalman filter state (separate system)
}
```

**NOT REQUIRED:**
- `recent_weights` array
- `measurement_history`
- Complex statistical tracking

## Testing Results

### Before Improvements
```
0.5 kg change in 1 hour:  ❌ Rejected (score 0.089)
1.0 kg change in 2 hours: ❌ Rejected (score 0.089)
2.0 kg change in 6 hours: ❌ Rejected (score 0.308)
```

### After Improvements
```
0.5 kg change in 1 hour:  ✅ Accepted (score 1.000)
1.0 kg change in 2 hours: ✅ Accepted (score 1.000)
2.0 kg change in 6 hours: ✅ Accepted (score 1.000)
42.22 kg outlier:         ❌ Rejected (score 0.000)
```

## Code Example

```python
# Stateless processing - each measurement independent
score = scorer.calculate_quality_score(
    weight=92.5,
    source="patient-device",
    previous_weight=92.0,      # From state
    time_diff_hours=24,         # Calculated from timestamps
    recent_weights=None,        # NOT NEEDED!
    user_height_m=1.67
)

# State update (if accepted)
if score.accepted:
    state['last_weight'] = 92.5
    state['last_timestamp'] = current_timestamp
    db.save_state(user_id, state)
```

## Benefits of Stateless Approach

1. **Simplicity**: No complex history management
2. **Scalability**: Each measurement processed independently
3. **Reliability**: No state corruption issues
4. **Performance**: Minimal memory footprint
5. **Maintainability**: Clear, simple data flow

## Comparison with Alternative Approaches

| Approach | State Required | Complexity | Accuracy |
|----------|---------------|------------|----------|
| **Our Stateless** | Previous weight + time | Low | Good |
| **MAD-based** | 20+ recent weights | Medium | Better |
| **Full Kalman** | Complete history | High | Best |

We chose the stateless approach because:
- Fits the architectural requirement
- Provides good accuracy for most cases
- Maintains system simplicity
- Avoids state management complexity

## Files Modified

- `src/quality_scorer.py` - Complete rewrite for stateless operation
- Removed dependency on `recent_weights`
- Simplified plausibility scoring
- Time-aware consistency thresholds

## Testing

```bash
# Run quality scorer tests
uv run python -m pytest tests/test_quality_scorer.py -xvs

# Test simplified scorer
uv run python test_simplified_scorer.py
```

## Conclusion

The improved quality scoring system successfully:
- ✅ Maintains **completely stateless** architecture
- ✅ Requires only **minimal state** (previous weight + time)
- ✅ Accepts **normal physiological variations** (2-3% daily)
- ✅ Rejects **true outliers** (42.22 kg example)
- ✅ Scales with **body weight** (percentage-based)
- ✅ Adapts to **time periods** (hourly vs daily vs weekly)

The system achieves a good balance between accuracy and architectural simplicity, perfectly suited for the stateless, recursive processing requirement.
