# Parameterized Reset System - Implementation Complete

## Summary
Successfully implemented a modular, parameterized reset system that handles three different reset scenarios with appropriate adaptation strategies.

## What Was Implemented

### 1. ResetManager Module (`src/reset_manager.py`)
- Centralized reset logic in one place
- Three reset types: HARD, INITIAL, SOFT
- Configurable parameters per reset type
- Intelligent trigger detection

### 2. Reset Types

#### Hard Reset (30+ day gaps)
- **Trigger**: Gap ≥ 30 days
- **Weight boost**: 10x
- **Trend boost**: 100x
- **Decay rate**: 3 (fast)
- **Adaptation**: 7 days
- **Use case**: Long absence

#### Initial Reset (first measurement)
- **Trigger**: No Kalman params
- **Weight boost**: 10x
- **Trend boost**: 100x
- **Decay rate**: 3 (fast)
- **Adaptation**: 7 days
- **Use case**: New user

#### Soft Reset (manual data entry)
- **Trigger**: Manual source + ≥5kg change
- **Weight boost**: 3x (gentler)
- **Trend boost**: 10x (gentler)
- **Decay rate**: 5 (slower)
- **Adaptation**: 10 days
- **Use case**: User corrections

### 3. Manual Data Sources
```python
MANUAL_DATA_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'questionnaire',
    'patient-upload',
    'user-upload',
    'care-team-upload',
    'care-team-entry'
}
```

### 4. Configuration Structure
```toml
[kalman.reset.hard]
gap_threshold_days = 30
weight_boost_factor = 10
trend_boost_factor = 100
...

[kalman.reset.soft]
min_change_kg = 5
weight_boost_factor = 3
trend_boost_factor = 10
...
```

## Key Features

### Soft Reset Logic
1. Detects manual data sources (questionnaires, uploads)
2. Checks for significant weight change (≥5kg)
3. Enforces cooldown period (3 days) to prevent loops
4. Applies gentler adaptation parameters
5. Slower return to normal operation

### Benefits
- **Trust manual data more**: Gentler adaptation for user/care team input
- **Prevent over-correction**: Soft reset doesn't swing wildly
- **Better UX**: System gracefully accepts user corrections
- **Maintainable**: All reset logic in one module
- **Configurable**: Easy to tune parameters

## Testing Results

### Test Scenario
```
Initial: 85kg (device) → establishes baseline
Manual: 86kg (patient-upload) → no reset (only 1kg change)
Manual: 92kg (care-team) → SOFT RESET (7kg change)
```

### Verification
```python
# Direct test confirms soft reset triggers correctly:
First: 85kg - Accepted=True
State has last_raw_weight: 85.0
Manual: 92kg (7kg change) - Reset=True
  Reset type: soft
```

## Files Modified
1. **Created** `src/reset_manager.py` - Core reset logic
2. **Updated** `src/processor.py` - Use ResetManager
3. **Updated** `src/kalman_adaptive.py` - Use reset parameters
4. **Updated** `config.toml` - Added reset type configurations

## Usage

The system automatically detects and applies the appropriate reset type:

```python
# Automatic detection in processor
reset_type = ResetManager.should_trigger_reset(
    state, weight, timestamp, source, config
)

if reset_type:
    state, reset_event = ResetManager.perform_reset(
        state, reset_type, timestamp, weight, source, config
    )
```

## Impact

### For Users
- Manual weight corrections are handled gracefully
- Less frustration with rejected measurements
- System adapts appropriately to different scenarios

### For Developers
- Clean, modular reset system
- Easy to add new reset types
- Clear configuration structure
- Comprehensive state tracking

## Future Enhancements
- Add confidence levels for reset decisions
- Learn optimal parameters per user
- Add reset reasons to visualization
- Support custom reset triggers

## Conclusion
The parameterized reset system successfully provides different adaptation strategies for different scenarios. Manual data entries from users or care teams now trigger a gentler "soft reset" that trusts the data more while maintaining quality control. The system is modular, maintainable, and easily extensible.