# Developer Quick Reference - Weight Stream Processor

## 🚨 MOST IMPORTANT: The Processor is STATELESS!

```python
# The ONLY way to use the processor:
result = WeightProcessor.process_weight(
    user_id='user_001',        # Key for state load/save
    weight=70.5,                # Measurement
    timestamp=datetime.now(),   # When measured
    source='scale',             # Data source
    processing_config=config,   # Processing settings
    kalman_config=kalman         # Kalman settings
)
```

## Architecture in 3 Sentences

1. **Processor** (`processor.py`) has NO state - all methods are static
2. **Database** (`processor_database.py`) stores/retrieves state per user
3. Every measurement: Load state → Process → Save state

## State Lifecycle

```
Measurement 1-9:   Buffer in init_buffer → Return None
Measurement 10:    Initialize Kalman → Process all 10 → Return result
Measurement 11+:   Load state → Update → Save → Return result
After 30+ day gap: Reset state with new baseline
```

## What's in State (Don't Touch Directly!)

```python
state = {
    'initialized': False,        # Becomes True after 10 measurements
    'init_buffer': [...],        # Cleared after initialization
    'kalman_params': {...},      # Parameters, NOT KalmanFilter object!
    'last_state': [70.5, -0.01], # [weight, trend_kg_per_day]
    'last_covariance': [...],    # Uncertainty matrix
    'last_timestamp': datetime,  # For time delta calculation
    'adapted_params': {...}      # User-specific (set once at init)
}
```

## The 10 Commandments

### ✅ DO's
1. Keep methods static
2. Copy state before modifying
3. Handle None results (buffering)
4. Pass user_id always
5. Return (result, new_state)

### ❌ DON'Ts
1. Add instance variables
2. Store KalmanFilter objects
3. Modify state in-place
4. Accumulate history
5. Break stateless pattern

## Common Operations

### Process a weight
```python
result = WeightProcessor.process_weight(user_id, weight, timestamp, source, proc_config, kalman_config)
if result:
    print(f"Filtered: {result['filtered_weight']:.1f}kg")
else:
    print("Still buffering...")
```

### Check user state (debugging)
```python
state = WeightProcessor.get_user_state('user_001')
print(f"Initialized: {state.get('initialized', False)}")
```

### Reset user (fresh start)
```python
WeightProcessor.reset_user('user_001')
```

### Process multiple users
```python
for user_id in user_list:
    # Each user is completely independent
    result = WeightProcessor.process_weight(user_id, ...)
```

## Quick Debugging

```python
# See what's happening
import json
state = WeightProcessor.get_user_state('user_001')
print(json.dumps(state, indent=2, default=str))

# Check database
from processor_database import get_state_db
db = get_state_db()
print(f"Total users: {len(db.states)}")
```

## Testing Pattern

```python
def test_processing():
    # Always reset first
    WeightProcessor.reset_user('test_user')
    
    # Process measurements
    for i in range(15):
        result = WeightProcessor.process_weight(
            user_id='test_user',
            weight=70.0 + random(),
            timestamp=base_date + timedelta(days=i),
            source='test',
            processing_config=config,
            kalman_config=kalman
        )
        
        if result:
            assert result['accepted'] in [True, False]
        else:
            assert i < 10  # Should only buffer for first 10
```

## Files You Can Modify

✅ **Can Modify:**
- `config.toml` - Tuning parameters
- `tests/*.py` - Add new tests
- `visualization.py` - Improve charts

⚠️ **Modify Carefully:**
- `processor.py` - Keep stateless!
- `main.py` - Keep streaming!

❌ **Don't Modify:**
- `processor_database.py` - Unless changing backend

## Performance Limits

- **State size**: ~1KB per user
- **Memory**: O(1) per user
- **Speed**: ~10,000 rows/sec
- **Users**: Unlimited (database dependent)
- **History**: Only last 2 states kept

## Need Help?

1. Check if state is initialized: `state.get('initialized')`
2. Check if buffering: `result is None`
3. Check state isolation: Different user_ids
4. Check numpy arrays: Database handles serialization
5. Check the tests: `tests/test_stateless_processor.py`

## Configuration Changes (v2.0.0)

### Migration Required
If you have custom config.toml files, see `CONFIG_MIGRATION_GUIDE.md` for details.

**Key changes:**
- Removed ~50% of unused settings
- Connected verbosity system to config
- Connected adaptive noise to config
- Added config validation on startup

**Quick migration:**
```bash
# Backup old config
cp config.toml config.toml.old

# Use new cleaned config
# The system will validate and report any issues
```

## Remember: STATELESS = SCALABLE! 🚀