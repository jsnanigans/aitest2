# Implementation Steps: Physiological Limits

## Step-by-Step Implementation Guide

### Step 1: Add Configuration (config.toml)
```toml
[physiological]
# Maximum weight changes by time period (kg)
max_change_1h = 2.0      # Hydration/bathroom
max_change_6h = 3.0      # Meals + hydration
max_change_24h = 5.0     # Extreme dehydration
max_sustained_daily = 0.5 # Long-term change per day

# Session detection
session_timeout_minutes = 5
session_variance_threshold = 5.0  # kg - reject high variance sessions
```

### Step 2: Update processor.py - Add Helper Methods

Add these as @staticmethod methods in WeightProcessor class:

```python
@staticmethod
def _calculate_time_delta_hours(
    current_timestamp: datetime,
    last_timestamp: Optional[datetime]
) -> float:
    """Calculate hours between measurements."""
    if last_timestamp is None:
        return float('inf')  # First measurement

    if isinstance(last_timestamp, str):
        last_timestamp = datetime.fromisoformat(last_timestamp)

    delta = (current_timestamp - last_timestamp).total_seconds() / 3600
    return max(0.0, delta)  # Ensure non-negative

@staticmethod
def _get_physiological_limit(time_delta_hours: float, config: dict) -> tuple[float, str]:
    """Get maximum allowed weight change based on time elapsed."""
    phys_config = config.get('physiological', {})

    if time_delta_hours < 1:
        limit = phys_config.get('max_change_1h', 2.0)
        reason = "hydration/bathroom"
    elif time_delta_hours < 6:
        limit = phys_config.get('max_change_6h', 3.0)
        reason = "meals+hydration"
    elif time_delta_hours < 24:
        limit = phys_config.get('max_change_24h', 5.0)
        reason = "daily maximum"
    else:
        # Long-term sustained change
        daily_rate = phys_config.get('max_sustained_daily', 0.5)
        limit = min(5.0, time_delta_hours / 24 * daily_rate)
        reason = f"sustained ({daily_rate}kg/day)"

    return limit, reason
```

### Step 3: Replace _validate_weight() Method

Replace the current validation (lines 195-211) with:

```python
@staticmethod
def _validate_weight(
    weight: float,
    state: Optional[Dict[str, Any]],
    timestamp: datetime,
    processing_config: dict
) -> tuple[bool, Optional[str]]:
    """
    Validate weight with physiological limits.
    Returns (is_valid, rejection_reason).
    """
    # Basic bounds check
    min_weight = processing_config.get('min_weight', 30)
    max_weight = processing_config.get('max_weight', 400)

    if weight < min_weight or weight > max_weight:
        return False, f"Weight {weight}kg outside bounds [{min_weight}, {max_weight}]"

    # Get last weight from state
    if state is None or not state.get('last_state'):
        return True, None  # First measurement always valid

    last_state = state.get('last_state')
    if isinstance(last_state, np.ndarray):
        last_weight = last_state[-1][0] if len(last_state.shape) > 1 else last_state[0]
    else:
        last_weight = last_state[0] if isinstance(last_state, (list, tuple)) else last_state

    # Calculate time delta
    last_timestamp = state.get('last_timestamp')
    time_delta_hours = WeightProcessor._calculate_time_delta_hours(
        timestamp, last_timestamp
    )

    # Apply physiological limits
    change = abs(weight - last_weight)
    max_change, reason = WeightProcessor._get_physiological_limit(
        time_delta_hours, processing_config
    )

    if change > max_change:
        return False, (f"Change of {change:.1f}kg in {time_delta_hours:.1f}h "
                      f"exceeds {reason} limit of {max_change:.1f}kg")

    # Optional: Check for session variance (multi-user detection)
    phys_config = processing_config.get('physiological', {})
    session_timeout = phys_config.get('session_timeout_minutes', 5) / 60  # Convert to hours

    if time_delta_hours < session_timeout:
        # This is part of the same measurement session
        session_variance_threshold = phys_config.get('session_variance_threshold', 5.0)
        if change > session_variance_threshold:
            return False, (f"Session variance {change:.1f}kg exceeds threshold "
                          f"{session_variance_threshold}kg (likely different user)")

    return True, None
```

### Step 4: Update process_weight() Method

Modify the validation call in process_weight() (around line 150):

```python
# Validate weight with physiological limits
is_valid, rejection_reason = cls._validate_weight(
    weight, state, timestamp, processing_config
)

if not is_valid:
    # Log the rejection reason for debugging
    result = {
        'user_id': user_id,
        'timestamp': timestamp,
        'weight': weight,
        'source': source,
        'result': 'rejected',
        'filtered_weight': None,
        'confidence': None,
        'trend': None,
        'rejection_reason': rejection_reason  # NEW: Include reason
    }
    return result
```

### Step 5: Add Session Detection (Optional Enhancement)

Add to state structure (backward compatible):

```python
@staticmethod
def _update_session_info(
    state: Dict[str, Any],
    timestamp: datetime,
    weight: float,
    processing_config: dict
) -> Dict[str, Any]:
    """Track session information for multi-user detection."""
    phys_config = processing_config.get('physiological', {})
    session_timeout = phys_config.get('session_timeout_minutes', 5) / 60

    last_session_time = state.get('last_session_time')
    time_delta = WeightProcessor._calculate_time_delta_hours(
        timestamp, last_session_time
    )

    if time_delta < session_timeout:
        # Same session - update stats
        session_stats = state.get('session_stats', {
            'count': 0,
            'weights': [],
            'start_time': timestamp
        })
        session_stats['count'] += 1
        session_stats['weights'].append(weight)

        # Keep only last 10 weights to limit memory
        if len(session_stats['weights']) > 10:
            session_stats['weights'] = session_stats['weights'][-10:]
    else:
        # New session
        session_stats = {
            'count': 1,
            'weights': [weight],
            'start_time': timestamp
        }

    return {
        'last_session_time': timestamp,
        'session_stats': session_stats
    }
```

### Step 6: Testing Strategy

1. **Update existing test files** to handle new validation
2. **Add test_physiological_limits.py** (already created)
3. **Test backward compatibility** with existing states
4. **Verify Kalman stability** with new validation

### Step 7: Migration Path

1. **Deploy with conservative limits** first
2. **Monitor rejection reasons** in logs
3. **Tune thresholds** based on real data
4. **Enable session detection** as optional feature

## Key Principles Maintained

✅ **Stateless Architecture**: All methods remain @staticmethod
✅ **Kalman Integrity**: Validation happens before filter update
✅ **Backward Compatible**: Old states work with new code
✅ **Separation of Concerns**: Validation separate from filtering
✅ **Performance**: O(1) operations, minimal state growth
✅ **Configurability**: All thresholds in config file

## Testing Checklist

- [ ] Test with user `0040872d-333a-4ace-8c5a-b2fcd056e65a`
- [ ] Verify 40kg→75kg jumps rejected
- [ ] Confirm 2kg changes accepted
- [ ] Test long gaps (>30 days)
- [ ] Verify state compatibility
- [ ] Check Kalman convergence
- [ ] Test single-user data (no regression)
- [ ] Verify rejection reasons logged

## Rollback Plan

If issues arise:
1. Revert to percentage-based validation (one line change)
2. Keep physiological config for future
3. Analyze rejection patterns
4. Adjust thresholds and redeploy
