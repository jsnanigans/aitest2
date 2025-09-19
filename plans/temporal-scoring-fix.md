# Fix Plan: Temporal Scoring Step Functions

## Problem
Temporal scoring creates artificial cycles through discrete step functions at 6h, 24h boundaries, causing rejection count to determine acceptance rather than actual quality.

## Solution: Replace with Continuous Exponential Temporal Scoring

### 1. Replace Step Functions with Smooth Decay

**Current approach (problematic):**
```python
if time_diff_hours <= 6:
    threshold = 0.5kg
elif time_diff_hours <= 24:
    threshold = 2.0kg
else:
    threshold = sustained_rate
```

**New approach (continuous) - REPLACE EXISTING METHOD:**
```python
def calculate_temporal_consistency(self, time_diff_hours, weight_change):
    # Exponential growth of acceptable change over time
    # Starts at 0.5kg for immediate, grows to ~5kg at 7 days
    max_acceptable_change = 0.5 + 4.5 * (1 - exp(-time_diff_hours / 48))

    # Smooth scoring based on deviation from acceptable
    if weight_change <= max_acceptable_change:
        # Within acceptable range: high score with smooth decay
        score = 0.8 + 0.2 * exp(-weight_change / max_acceptable_change)
    else:
        # Beyond acceptable: exponential penalty
        excess_ratio = (weight_change - max_acceptable_change) / max_acceptable_change
        score = 0.8 * exp(-excess_ratio)

    return max(0.2, min(1.0, score))  # Clamp between 0.2 and 1.0
```

### 2. Maintain Temporal State Continuity

**Problem:** Scores reset after acceptance, creating discontinuity

**Solution:** Track rolling temporal baseline
```python
# Add to state
state['temporal_baseline'] = {
    'last_weight': weight,
    'last_timestamp': timestamp,
    'rolling_avg_change_rate': 0.0,  # Exponentially weighted
}

# Update rolling average instead of resetting
def update_temporal_baseline(self, state, new_weight, new_timestamp):
    baseline = state.get('temporal_baseline', {})
    if baseline.get('last_weight'):
        time_diff = (new_timestamp - baseline['last_timestamp']).total_seconds() / 3600
        weight_change = abs(new_weight - baseline['last_weight'])
        daily_rate = weight_change / max(time_diff / 24, 0.1)

        # Exponential moving average with α=0.3
        baseline['rolling_avg_change_rate'] = (
            0.3 * daily_rate +
            0.7 * baseline.get('rolling_avg_change_rate', daily_rate)
        )

    baseline['last_weight'] = new_weight
    baseline['last_timestamp'] = new_timestamp
    return baseline
```

### 3. Adaptive Scoring Based on User Patterns

**Enhancement:** Adjust expectations based on user's typical variability
```python
def get_user_adjusted_threshold(self, base_threshold, state):
    # Look at recent measurement variability
    recent_history = state.get('measurement_history', [])[-20:]
    if len(recent_history) >= 5:
        weights = [m['weight'] for m in recent_history]
        std_dev = np.std(weights)
        # Users with higher variability get more lenient thresholds
        adjustment_factor = 1.0 + min(0.5, std_dev / 5.0)
        return base_threshold * adjustment_factor
    return base_threshold
```

## Implementation Steps

### Phase 1: Core Function Changes
1. **Replace existing method** `calculate_temporal_consistency` in `unified_quality_scorer.py` with continuous version
2. **Remove step function logic** completely - no backward compatibility needed
3. **Import math.exp** for exponential calculations

### Phase 2: State Management
1. **Extend state structure** to include `temporal_baseline`
2. **Update processor.py** to maintain baseline after both accepts and rejects
3. **Ensure persistence** of temporal baseline in database

### Phase 3: Testing & Validation
1. **Unit tests** for continuous function with various time/weight combinations
2. **Integration tests** with historical data to verify cycle elimination
3. **Verify** improved behavior on problematic users (bc1c9d20, 9575e299)

## Configuration Changes

Add to `config.toml` (optional parameters for tuning):
```toml
[quality_scoring.temporal]
min_score = 0.2
max_score = 1.0
initial_threshold_kg = 0.5
max_threshold_kg = 5.0
time_constant_hours = 48
variability_adjustment = true
```

## No Backward Compatibility

- Completely replace the step function implementation
- All users will use the new continuous scoring
- Simpler codebase without conditional logic
- No feature flags needed

## Success Metrics

1. **Eliminate cycles**: No more predictable 9-10 rejection patterns
2. **Consistent acceptance**: Similar weights get similar scores regardless of timing
3. **Smooth transitions**: Quality scores change gradually, not in steps
4. **User satisfaction**: Fewer "inexplicable" rejections

## Risks & Mitigations

**Risk**: Too lenient, accepts bad data
**Mitigation**: Keep min score at 0.2, other components still validate

**Risk**: Too strict for users with natural variability
**Mitigation**: User-specific adaptation based on historical patterns

**Risk**: Complex to debug
**Mitigation**: Extensive logging of score components and thresholds