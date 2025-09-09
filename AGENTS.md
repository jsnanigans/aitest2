# Agent Instructions for Weight Stream Processor

## CRITICAL: Architecture Overview

**The processor is COMPLETELY STATELESS with separate state database:**
- `processor.py` - Pure functional computation (ALL methods are @staticmethod)
- `processor_database.py` - State persistence layer (in-memory, replaceable)
- State is loaded → processed → saved for EVERY measurement
- NO instance variables, NO class state, ONLY static methods

## Project Structure
```
.
├── main.py                 # Entry point - NEVER split this
├── processor.py            # STATELESS processor - all static methods
├── processor_database.py   # State persistence layer
├── visualization.py        # Visualization - NEVER split this
├── config.toml            # Configuration file
├── tests/                 # ALL test files go here
│   └── test_*.py         # Test files prefixed with test_
├── data/                  # Input CSV files
└── output/                # Generated results and visualizations
```

## Build & Test Commands
```bash
# Install dependencies (using uv package manager)
uv pip install -r requirements.txt

# Run main processor
uv run python main.py [csv_file]

# Run tests (all test files in tests/ directory)
uv run python tests/test_stateless_processor.py  # Main processor test
uv run python tests/test_viz_improvements.py

# Type checking (if needed)
pyright
```

## Code Style Guidelines
- **Python 3.11+**, type hints optional (pyright typeCheckingMode: off)
- **Imports**: Standard library first, then third-party (numpy, pykalman, matplotlib), then local
- **Line length**: 100 chars max (per pyproject.toml)
- **Docstrings**: Brief module docstrings, class/method descriptions for complex logic
- **Architecture**: Keep it simple - 4 core files (main, processor, processor_database, visualization)
- **NO COMMENTS** unless absolutely critical for understanding complex math
- **Error handling**: Basic validation only (weight 30-400kg, physiological limits)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **State management**: Stateless processor + database - O(1) memory per user
- **Test files**: ALWAYS place new test files in `tests/` directory with `test_` prefix
- **Never split core files**: Keep each file focused on single responsibility

## CRITICAL DO's ✅

1. **DO keep ALL methods in processor.py as @staticmethod**
2. **DO use WeightProcessor.process_weight() interface** - Single entry point
3. **DO pass user_id to identify which state to load/save**
4. **DO copy state before modifying** - Never modify in-place
5. **DO return (result, new_state) tuple** - new_state=None if unchanged
6. **DO handle None results** - Buffering phase returns None
7. **DO preserve numpy array types** - Database handles serialization
8. **DO validate inputs early** - Check bounds before processing
9. **DO test with multiple users** - Ensure state isolation
10. **DO keep state minimal** - Only essential Kalman variables

## CRITICAL DON'Ts ❌

1. **DON'T add instance variables to WeightProcessor** - Must stay stateless
2. **DON'T store KalmanFilter objects** - Store parameters only
3. **DON'T accumulate history** - Keep only last 2 states maximum
4. **DON'T modify passed configurations** - Treat as immutable
5. **DON'T access db.states directly** - Use database methods
6. **DON'T assume initialization** - Always check 'initialized' flag
7. **DON'T process out of order** - Assumes chronological
8. **DON'T forget to save state** - Only save when actually changed
9. **DON'T mix computation and persistence** - Keep them separate
10. **DON'T break the stateless pattern** - It's essential for scaling

## State Structure (What Gets Persisted)
```python
{
    'initialized': bool,           # Is Kalman ready?
    'init_buffer': [],             # Temporary during init
    'kalman_params': {             # Filter parameters (NOT object!)
        'initial_state_mean': [...],
        'initial_state_covariance': [...],
        'transition_covariance': [...],
        'observation_covariance': [...]
    },
    'last_state': np.array([weight, trend]),
    'last_covariance': np.array([...]),
    'last_timestamp': datetime,
    'adapted_params': {            # User-specific, computed once
        'observation_covariance': float,
        'extreme_threshold': float,
        ...
    }
}
```

## Common Pitfalls & Solutions

### Pitfall: Trying to Create Processor Instance
```python
# ❌ WRONG
processor = WeightProcessor(user_id, config, kalman_config)
result = processor.process_weight(weight, timestamp, source)

# ✅ CORRECT
result = WeightProcessor.process_weight(
    user_id=user_id,
    weight=weight,
    timestamp=timestamp,
    source=source,
    processing_config=config,
    kalman_config=kalman_config
)
```

### Pitfall: Modifying State In-Place
```python
# ❌ WRONG
state['last_timestamp'] = timestamp
return result, state  # Original state modified!

# ✅ CORRECT
new_state = state.copy()
new_state['last_timestamp'] = timestamp
return result, new_state
```

### Pitfall: Not Handling Buffering Phase
```python
# ❌ WRONG
result = WeightProcessor.process_weight(...)
print(result['filtered_weight'])  # Crashes if None!

# ✅ CORRECT
result = WeightProcessor.process_weight(...)
if result:
    print(result['filtered_weight'])
else:
    print("Still buffering...")
```

## Testing Checklist
- [ ] Test initialization with exactly 10 measurements
- [ ] Test buffering returns None for < 10 measurements
- [ ] Test state persistence across calls
- [ ] Test multiple users don't interfere
- [ ] Test 30+ day gap triggers reset
- [ ] Test extreme outliers are rejected
- [ ] Test adapted parameters for clean vs noisy data
- [ ] Test database can be cleared and rebuilt

## Performance Notes
- State size: ~1KB per user
- Database ops: 1 load + 1 save per measurement
- Computation: O(1) Kalman update
- Memory: Constant per user (no accumulation)
- Ready for parallel processing of different users