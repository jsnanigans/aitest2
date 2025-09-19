# Investigation: last_accepted_timestamp vs last_timestamp

## Summary
Both fields track timestamps but serve different purposes:
- **`last_timestamp`**: Legacy field, kept for backward compatibility
- **`last_accepted_timestamp`**: Primary field for tracking the last accepted measurement

## Key Differences

### 1. Primary Usage
- `last_accepted_timestamp` is the authoritative field for determining gaps between measurements
- `last_timestamp` is maintained alongside for backward compatibility
- Both are updated simultaneously when a measurement is accepted (src/processing/processor.py:710-711)

### 2. Gap Detection Logic
The system checks both fields with fallback logic (src/processing/processor.py:48):
```python
last_timestamp = state.get("last_accepted_timestamp") or state.get("last_timestamp")
```
This pattern appears in:
- processor.py:48 (gap detection)
- kalman.py:480, 625 (reset detection)
- reset_manager.py:53, 194 (hard reset triggers)

### 3. State Updates
When a measurement is accepted (processor.py:710-712):
```python
state["last_timestamp"] = timestamp  # Keep for backward compatibility
state["last_accepted_timestamp"] = timestamp
```

### 4. Reset Behavior
During resets, both fields are preserved in the new state (kalman.py:653-656):
```python
'last_timestamp': state.get('last_timestamp'),
'last_accepted_timestamp': state.get('last_accepted_timestamp'),
```

## Implications
1. **No functional difference** - Both fields contain the same value when measurements are accepted
2. **Redundancy by design** - Maintains compatibility with older state data
3. **Consistent fallback pattern** - Code always checks `last_accepted_timestamp` first, falls back to `last_timestamp`

## Recommendation
The dual-field approach is intentional for backward compatibility. New code should:
1. Always write to both fields when updating timestamps
2. Read using the fallback pattern: `state.get("last_accepted_timestamp") or state.get("last_timestamp")`
3. Eventually deprecate `last_timestamp` once all legacy states are migrated