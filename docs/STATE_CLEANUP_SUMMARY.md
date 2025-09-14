# State Management Cleanup Summary

## Investigation Findings

### Unused/Phantom State Fields Removed

1. **`last_attempt_timestamp`** - Was set but never read (redundant with `last_timestamp`)
2. **`rejection_count_since_accept`** - Was set to 0 but never incremented or used
3. **`state_reset_count`** - CSV export expected it but it was never set in code
4. **`last_reset_timestamp`** - CSV export expected it but it was never set in code
5. **`adapted_params`** - CSV export expected it but it was never set in code

### State Fields Now in Use

The cleaned state now contains exactly these fields:

```python
{
    'last_state': None,           # Kalman state vector [weight, trend]
    'last_covariance': None,      # Kalman covariance matrix
    'last_timestamp': None,       # Last measurement timestamp
    'kalman_params': None,        # Kalman filter parameters
    'last_source': None,          # Source of last measurement (scale/manual/questionnaire)
    'last_raw_weight': None,      # Raw weight before Kalman filtering
    'measurement_history': [],    # Last 30 measurements for quality scoring
}
```

## Database CSV Export

The CSV export now reflects exactly what a database would store:

### CSV Columns
- `user_id` - User identifier
- `last_timestamp` - Last measurement timestamp
- `last_weight` - Filtered weight from Kalman state
- `last_trend` - Weight trend from Kalman state
- `last_source` - Source of last measurement
- `last_raw_weight` - Raw weight before filtering
- `has_kalman_params` - Whether Kalman is initialized
- `process_noise` - Kalman process noise parameter
- `measurement_noise` - Kalman measurement noise parameter
- `initial_uncertainty` - Initial state uncertainty

### Removed from CSV
- ❌ `has_adapted_params` - Never set in code
- ❌ `adapted_process_noise` - Never set in code
- ❌ `adapted_measurement_noise` - Never set in code
- ❌ `state_reset_count` - Never tracked
- ❌ `last_reset_timestamp` - Never tracked

### Added to CSV
- ✅ `last_source` - Useful for debugging and analysis
- ✅ `last_raw_weight` - Important for understanding filtering effects

## Files Modified

1. **`src/database.py`**
   - Fixed `create_initial_state()` to include all used fields
   - Updated CSV export to match actual state fields
   - Added state schema documentation

2. **`src/kalman.py`**
   - Removed `last_attempt_timestamp` assignments
   - Removed `rejection_count_since_accept` assignments

## Benefits

1. **Consistency** - State initialization matches actual usage
2. **Clarity** - No phantom fields that confuse developers
3. **Accuracy** - CSV export reflects true database state
4. **Maintainability** - Clear documentation of state schema

## Testing

Created comprehensive tests to verify:
- Initial state has correct fields
- State persists correctly through processing
- CSV export contains accurate data
- Measurement history is properly tracked

## Example Clean DB Dump

```csv
user_id,last_timestamp,last_weight,last_trend,last_source,last_raw_weight,has_kalman_params,process_noise,measurement_noise,initial_uncertainty
user001,2025-09-14T09:52:01,69.90,0.0032,scale,70.14,true,0.016,5.235,0.361
user002,2025-09-14T09:52:01,85.29,-0.0208,manual,83.72,true,0.016,5.235,0.361
```

This represents exactly what would be stored in a real database, with no unused or phantom fields.
