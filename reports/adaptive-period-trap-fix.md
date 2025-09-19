# Proposed Fix: Adaptive Period Measurement Trap

## Solution Overview

Increment `measurements_since_reset` for ALL measurements during adaptive period, not just accepted ones. This ensures the system can exit the adaptive period even if initial measurements are rejected.

## Code Changes

### Option 1: Simple Counter Fix (Recommended)

**File**: `src/processing/processor.py`

**Current Code (lines 396-408)**:
```python
if not quality_score.accepted:
    return {
        'accepted': False,
        # ... rejection details
    }
```

**Proposed Change**:
```python
if not quality_score.accepted:
    # During adaptive period, still increment counter to prevent infinite loop
    if in_adaptive_period and state:
        state['measurements_since_reset'] = state.get('measurements_since_reset', 0) + 1
        # Persist the updated counter
        if feature_manager.is_enabled('state_persistence'):
            db.save_state(user_id, state)
    
    return {
        'accepted': False,
        # ... rejection details
    }
```

Apply same logic to rejection at line 465-477.

### Option 2: Rejection Limit Safety Valve

Add a maximum consecutive rejection limit during adaptive period:

**File**: `src/processing/processor.py` 

**Add after line 328**:
```python
# Safety valve: exit adaptive period after too many rejections
consecutive_rejections = state.get('consecutive_rejections_in_adaptive', 0)
if in_adaptive_period and consecutive_rejections > 20:
    # Force exit from adaptive period
    state['measurements_since_reset'] = adaptation_measurements
    in_adaptive_period = False
    logger.warning(f"Forced exit from adaptive period after {consecutive_rejections} rejections")
```

### Option 3: Smart Reset Detection

**File**: `src/processing/reset_manager.py`

Add validation before accepting reset-triggering measurements:

```python
@staticmethod
def validate_reset_trigger(
    weight: float, 
    last_weight: float,
    source: str,
    user_height_m: float
) -> bool:
    """Validate if a measurement should trigger a reset."""
    
    # Check BMI plausibility
    bmi = weight / (user_height_m ** 2)
    if bmi > 50 or bmi < 15:
        return False  # Likely data error
    
    # Manual entries need extra scrutiny for large changes
    if source == 'internal-questionnaire':
        change_pct = abs(weight - last_weight) / last_weight
        if change_pct > 0.3:  # 30% change
            # Could require confirmation or additional checks
            return False
    
    return True
```

## Implementation Priority

1. **Immediate**: Implement Option 1 - fixes the trap with minimal risk
2. **Short-term**: Add Option 2 - provides safety valve 
3. **Long-term**: Implement Option 3 - prevents bad resets

## Testing Requirements

1. Test with erroneous manual entry followed by correct measurements
2. Verify adaptive period exits after configured number of measurements
3. Ensure legitimate large weight changes still work
4. Check state persistence updates correctly on rejections

## Risk Mitigation

- Log all forced adaptive period exits for monitoring
- Add metrics to track average time in adaptive period
- Consider notification when unusual reset patterns detected
