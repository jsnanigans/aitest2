# Weight Stream Processor

A clean, minimal implementation of weight data processing using Kalman filtering.

## Architecture

Three core files:

- `main.py` - Streams CSV data and orchestrates processing
- `processor.py` - Stateless Kalman filter processor per user
- `visualization.py` - Creates dashboard visualizations

## Project Structure

```
.
├── main.py                 # Entry point for processing CSV files
├── processor.py            # Core Kalman filter weight processor
├── visualization.py        # Dashboard generation for analysis
├── config.toml            # Configuration file
├── tests/                 # All test files go here
│   ├── test_viz_improvements.py
│   └── test_progress.py
├── data/                  # Input CSV files
└── output/                # Generated results and visualizations
```

## Features

- **True streaming**: Processes line-by-line, maintains only current state
- **Mathematical Kalman**: Uses `pykalman` for correct implementation
- **Simple validation**: Basic physiological limits (30-400kg)
- **Adaptive processing**: Handles gaps, resets state when needed
- **Enhanced visualization**: Kalman processing evaluation dashboard

## Installation

```bash
uv pip install -r requirements.txt
```

Only 3 dependencies:

- `numpy` - Numerical operations
- `pykalman` - Kalman filtering
- `matplotlib` - Visualization

## Usage

```bash
# Process weight data
uv run python main.py your_data.csv

# Or use default file
uv run python main.py
```

## Testing

All test files are located in the `tests/` directory:

```bash
# Run visualization test with synthetic data
uv run python tests/test_viz_improvements.py

# Run progress tracking test
uv run python tests/test_progress.py
```

## Output

Creates timestamped output directory with:

- `results_TIMESTAMP.json` - All processed data
- `viz_TIMESTAMP/` - User dashboards (top 10 users)

## Processing Flow

1. **Stream Processing**
   - Read CSV line by line
   - Route to appropriate user processor
   - No full dataset in memory

2. **Per-User Processing**
   - Buffer first 5 readings for initialization
   - Establish baseline using median
   - Apply Kalman filter with adaptive parameters
   - Reset on 30+ day gaps

3. **Validation**
   - Weight range: 30-400 kg
   - Max daily change: 3% (normal) or 50% (extreme)
   - Deviation threshold: 30% from prediction

4. **Visualization**
   - Weight trajectory with filtered overlay
   - Distribution histogram
   - Confidence scores over time
   - Trend analysis (kg/week)
   - 7-day moving average
   - Statistics summary

## Performance

- Memory: O(1) per user (true streaming)
- Speed: ~10,000-20,000 rows/second
- Scales to millions of rows

## Key Improvements

- **300 lines vs 331,000 lines** (99.9% reduction)
- **3 files vs 50+ files**
- **3 dependencies vs 10+**
- **Clear single responsibility per file**
- **Mathematically correct Kalman (via pykalman)**
- **No over-engineering or unnecessary abstractions**

## Developer Guide: Working with the Processor

### Architecture Overview

The processor is now **completely stateless** with a separate state database:
- `processor.py` - Pure functional computation (all static methods)
- `processor_database.py` - State persistence layer (in-memory, can be replaced with Redis/PostgreSQL)

### Critical Concepts to Understand

#### 1. **Stateless Processing Pattern**
```python
# State is loaded → processed → saved for EVERY measurement
result = WeightProcessor.process_weight(
    user_id='user_001',  # Key to load/save state
    weight=70.5,
    timestamp=datetime.now(),
    source='scale',
    processing_config=config,
    kalman_config=kalman_config
)
```

#### 2. **State Structure**
Each user's state contains:
- `initialized`: Boolean flag for Kalman readiness
- `init_buffer`: Temporary buffer during initialization (cleared after)
- `kalman_params`: Filter parameters (NOT the filter object itself!)
- `last_state`: Numpy array of [weight, trend]
- `last_covariance`: Uncertainty matrix
- `last_timestamp`: For time delta calculation
- `adapted_params`: User-specific adaptations (computed once, then frozen)

#### 3. **Kalman Filter Lifecycle**
1. **Buffering Phase**: First 10 measurements collected
2. **Initialization**: Baseline established, parameters adapted
3. **Processing**: Each measurement updates state
4. **Reset**: After 30+ day gaps, state resets with new baseline

### DO's ✅

1. **DO keep methods static** - No instance variables in processor
2. **DO use the database abstraction** - Don't access `db.states` directly
3. **DO preserve numpy array types** - The serialization handles them
4. **DO maintain O(1) memory** - Keep only last 2 states maximum
5. **DO validate inputs early** - Check weight bounds before processing
6. **DO handle None results** - Buffering returns None, not empty dict
7. **DO use MAD for variance** - More robust than standard deviation
8. **DO test with multiple users** - Ensure state isolation
9. **DO keep configurations immutable** - Never modify passed configs
10. **DO compute adaptations once** - During initialization only

### DON'Ts ❌

1. **DON'T store the KalmanFilter object** - Store parameters only
2. **DON'T modify state in-place** - Always copy then modify
3. **DON'T assume initialization** - Check `initialized` flag
4. **DON'T accumulate history** - Keep only essential state
5. **DON'T trust raw timestamps** - Handle string/datetime conversion
6. **DON'T skip validation** - Even for "trusted" sources
7. **DON'T modify the database structure** - Keep it simple
8. **DON'T add instance variables** - Processor must stay stateless
9. **DON'T process out of order** - Assumes chronological processing
10. **DON'T forget to save state** - Only save when state actually changes

### Common Pitfalls & Solutions

#### Pitfall 1: State Not Persisting
```python
# ❌ WRONG - Modifying state without saving
state['last_timestamp'] = timestamp

# ✅ CORRECT - Return modified state for saving
new_state = state.copy()
new_state['last_timestamp'] = timestamp
return result, new_state  # new_state will be saved
```

#### Pitfall 2: Numpy Array Serialization
```python
# The database handles this automatically via custom encoder
# Arrays are converted to lists in JSON, restored on load
# Don't worry about it, just use numpy arrays normally
```

#### Pitfall 3: Multiple Users Interfering
```python
# Each user has completely isolated state
# The database ensures no cross-contamination
# user_id is the only connection between calls
```

### Testing Checklist

- [ ] Test with < 10 measurements (buffering)
- [ ] Test with exactly 10 measurements (initialization)
- [ ] Test with > 10 measurements (normal processing)
- [ ] Test with 30+ day gap (reset)
- [ ] Test with extreme outliers (rejection)
- [ ] Test with multiple users in parallel
- [ ] Test state persistence across sessions
- [ ] Test with very noisy data (adaptation)
- [ ] Test with very clean data (adaptation)
- [ ] Test with weight loss trend (adaptation)

### Performance Considerations

1. **State Size**: Keep state minimal (~1KB per user)
2. **Database Calls**: One load + one save per measurement
3. **Computation**: Kalman update is O(1) for 2D state
4. **Memory**: No accumulation, constant per user
5. **Parallelization**: Ready for concurrent processing

### Future Extension Points

1. **Database Backend**: Replace `ProcessorStateDB` with Redis/PostgreSQL
2. **Batch Processing**: Add method for processing multiple measurements
3. **State Migration**: Version state structure for upgrades
4. **Monitoring**: Add metrics collection hooks
5. **Caching**: Cache frequently accessed states in memory

### Mathematical Notes

- **State Vector**: [weight, trend_kg_per_day]
- **Observation**: Single weight measurement
- **Process Noise**: How much weight naturally varies
- **Observation Noise**: Scale measurement uncertainty
- **Innovation**: Difference between predicted and measured
- **Confidence**: exp(-0.5 * normalized_innovation²)

### Debugging Tips

```python
# Check user state
state = WeightProcessor.get_user_state('user_001')
print(json.dumps(state, indent=2, default=str))

# Reset user for fresh start
WeightProcessor.reset_user('user_001')

# Check database contents
db = get_state_db()
print(f"Total users: {len(db.states)}")
```

