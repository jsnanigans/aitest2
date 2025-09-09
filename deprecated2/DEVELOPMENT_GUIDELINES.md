# Development Guidelines for Weight Stream Processing Framework

## Core Architectural Principles

### 1. Strict Layer Separation
- **Layer 1 (Heuristic Filters)**: Fast, stateless checks (O(1) complexity)
  - Physiological limits (30-400 kg)
  - Rate of change limits (±3% daily)
  - Moving MAD filter
- **Layer 2 (Time-Series Modeling)**: Contextual analysis (O(n) for window)
  - ARIMA-based outlier detection
  - Outlier classification (AO, IO, LS, TC)
- **Layer 3 (State Estimation)**: Pure Kalman filtering (O(1) updates)
  - State prediction and update only
  - NO outlier detection logic
  - NO physiological validation

### 2. Mathematical Correctness
- State transition matrix MUST remain: F = [[1, Δt], [0, 1]]
- Single, consistent process noise covariance matrix Q
- Proper Kalman gain calculation: K = P_pred @ H.T @ inv(S)
- State vector: [weight, trend] - 2D only
- Measurement model: H = [1, 0]

### 3. Separation of Concerns
- **Kalman Filter**: State estimation ONLY
- **Outlier Detection**: Dedicated layers ONLY
- **Baseline Establishment**: Separate component
- **Pipeline Orchestrator**: Manages flow between components
- **Validation Gate**: Between predict and update steps ONLY

## Implementation Standards

### Code Organization
```
src/
├── filters/           # Each filter is independent
│   ├── layer1_heuristic.py
│   ├── layer2_arima.py
│   └── layer3_kalman.py
├── processing/        # Orchestration and utilities
│   ├── weight_pipeline.py
│   ├── robust_baseline.py
│   └── user_processor.py
└── visualization/     # Output generation only
```

### Performance Requirements
- Stream processing: Line-by-line, no full dataset loading
- Memory: O(1) per user (only current user in memory)
- Speed: 2-3 users/second minimum
- Kalman updates: O(1) constant time

### Data Flow Rules
1. **Input**: Raw CSV → optimize_csv.py → Sorted CSV
2. **Processing**: Line-by-line streaming through pipeline
3. **Validation**: ALWAYS through validation gate (γ threshold)
4. **Output**: JSON results + visualizations

## Critical DO NOTs

### Architecture Violations
- ❌ DO NOT mix outlier detection in Kalman filter
- ❌ DO NOT modify state transition matrix F
- ❌ DO NOT have multiple process noise calculations
- ❌ DO NOT bypass the validation gate
- ❌ DO NOT combine layers or merge responsibilities

### Implementation Anti-patterns
- ❌ DO NOT load entire dataset into memory
- ❌ DO NOT use pandas/numpy for core processing
- ❌ DO NOT create monolithic functions (>200 lines)
- ❌ DO NOT hardcode thresholds - use config.toml
- ❌ DO NOT skip the robust baseline establishment

## Testing Requirements

### Unit Tests
- Each layer tested independently
- Mock data for predictable outcomes
- Test edge cases (gaps, outliers, missing data)

### Integration Tests
- Full pipeline with synthetic data
- Verify 87%+ acceptance rate on valid data
- Confirm outlier classification accuracy

### Performance Tests
- Benchmark 2-3 users/second
- Memory usage stays constant during streaming
- Kalman convergence within expected iterations

## Configuration Management

### Required config.toml sections
```toml
[processing]
validation_gamma = 3.0  # Validation gate threshold

[processing.layer1]
mad_threshold = 3.0
mad_window_size = 15
max_daily_change_percent = 3.0

[processing.layer2]
arima_order = [1, 0, 1]
residual_threshold = 3.0

[processing.kalman]
process_noise_weight = 0.5
process_noise_trend = 0.01
```

## State Management

### User Processing State
- BASELINE_PENDING: Collecting initial data
- BASELINE_ESTABLISHED: Ready for Kalman
- PROCESSING: Active Kalman filtering
- GAP_DETECTED: Reinitializing after 30+ day gap

### Kalman State Persistence
- Save: state vector (x), covariance (P), last timestamp
- Restore: Full state for continuation
- Reset: On gap detection or regime change

## Real-Time Production Requirements

### CRITICAL: Stateless Processing Architecture
The system MUST operate in a completely recursive/stateless manner for production deployment:

#### Core Constraint
- **NO HISTORICAL DATA ACCESS**: Cannot query all previous measurements
- **SINGLE MEASUREMENT PROCESSING**: Each new weight is processed independently
- **STATE-BASED ONLY**: Must rely entirely on saved state from previous processing
- **DATABASE/CACHE BACKED**: State must persist between invocations

#### Implementation Requirements
1. **State Persistence**
   - Save complete Kalman state after each measurement
   - Store: weight, trend, covariance matrix, timestamp, count
   - Persist baseline parameters separately
   - Track last processed timestamp for gap detection

2. **Processing Flow**
   ```python
   # Production entry point
   def process_new_weight(user_id, weight, timestamp):
       # 1. Load saved state from DB/cache
       state = load_state(user_id)
       
       # 2. Check if initialization needed
       if not state:
           return {"needs_initialization": True}
       
       # 3. Check for gaps
       if gap_detected(state, timestamp):
           delete_state(user_id)
           return {"needs_reinitialization": True}
       
       # 4. Restore Kalman filter from state
       kalman = restore_kalman(state)
       
       # 5. Process single measurement
       result = kalman.process_measurement(weight, timestamp)
       
       # 6. Save updated state
       save_state(user_id, kalman.get_state())
       
       # 7. Return result
       return result
   ```

3. **State Storage Schema**
   ```json
   {
     "user_id": "string",
     "kalman": {
       "weight": 80.5,
       "trend": 0.01,
       "covariance": [[0.5, 0], [0, 0.001]],
       "measurement_count": 42,
       "last_timestamp": "2025-01-15T10:30:00Z"
     },
     "baseline": {
       "weight": 80.0,
       "variance": 0.5,
       "confidence": "high"
     },
     "last_processed": "2025-01-15T10:30:00Z"
   }
   ```

4. **Initialization Protocol**
   - Collect 3+ measurements over 7 days
   - Store temporarily until baseline can be established
   - Once established, switch to real-time processing
   - Delete temporary storage after initialization

5. **Gap Handling**
   - If gap > 30 days: Delete state and re-initialize
   - If gap > 7 days: Increase process noise
   - If gap < 1 hour: Possible duplicate, check carefully

### Production Deployment Checklist
- [ ] State manager implements database/Redis interface
- [ ] No queries for historical data during processing
- [ ] Single measurement processing tested
- [ ] State save/restore verified
- [ ] Gap detection working correctly
- [ ] Initialization queue implemented
- [ ] Performance: <100ms per measurement
- [ ] Concurrent user processing safe
- [ ] State versioning for migrations

## Production API Interface

### Single Measurement Processing Endpoint
```python
# POST /api/weight/process
{
    "user_id": "user_123",
    "weight": 80.5,
    "timestamp": "2025-01-15T10:30:00Z",
    "source": "care-team-upload"
}

# Response
{
    "accepted": true,
    "confidence": 0.95,
    "filtered_weight": 80.4,
    "trend_kg_per_week": 0.15,
    "prediction_error": 0.3,
    "needs_initialization": false
}
```

### User Initialization Endpoint
```python
# POST /api/weight/initialize
{
    "user_id": "user_123",
    "measurements": [
        {"weight": 80.0, "timestamp": "2025-01-01T08:00:00Z", "source": "care-team"},
        {"weight": 80.2, "timestamp": "2025-01-02T08:00:00Z", "source": "care-team"},
        {"weight": 79.8, "timestamp": "2025-01-03T08:00:00Z", "source": "patient"}
    ]
}

# Response
{
    "success": true,
    "baseline_weight": 80.0,
    "baseline_confidence": "high"
}
```

### State Management Endpoints
```python
# GET /api/weight/state/{user_id}
# DELETE /api/weight/state/{user_id}  # For re-initialization
```

### Database Schema (PostgreSQL/MongoDB)
```sql
-- PostgreSQL
CREATE TABLE kalman_states (
    user_id VARCHAR(50) PRIMARY KEY,
    weight FLOAT NOT NULL,
    trend FLOAT NOT NULL,
    covariance JSONB NOT NULL,
    measurement_count INTEGER NOT NULL,
    last_timestamp TIMESTAMP NOT NULL,
    baseline JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- MongoDB
{
    "_id": "user_123",
    "kalman": {
        "weight": 80.5,
        "trend": 0.01,
        "covariance": [[0.5, 0], [0, 0.001]],
        "measurement_count": 42,
        "last_timestamp": ISODate("2025-01-15T10:30:00Z")
    },
    "baseline": {
        "weight": 80.0,
        "variance": 0.5,
        "confidence": "high"
    },
    "updated_at": ISODate("2025-01-15T10:30:00Z")
}
```

### Redis Cache Structure
```redis
# Key pattern: kalman:state:{user_id}
SET kalman:state:user_123 '{"weight":80.5,"trend":0.01,...}'
EXPIRE kalman:state:user_123 86400  # 24-hour TTL

# Initialization queue
LPUSH kalman:init:user_123 '{"weight":80.0,"timestamp":"..."}'
```

## Validation Gate Protocol

### Real-time Validation
```python
# Prediction step
x_pred = F @ x_prev
P_pred = F @ P_prev @ F.T + Q

# Innovation
innovation = z_new - H @ x_pred
S = H @ P_pred @ H.T + R

# Validation gate
if abs(innovation) > gamma * sqrt(S):
    # Reject measurement
    return OUTLIER
else:
    # Accept and update
    K = P_pred @ H.T / S
    x_new = x_pred + K * innovation
    P_new = (I - K @ H) @ P_pred
```

## Baseline Establishment Protocol

### Required Steps
1. Collect 7-14 days of initial data
2. Apply IQR outlier removal
3. Calculate median for baseline weight
4. Calculate MAD for variance estimation
5. Initialize Kalman with baseline values

### Fallback Strategies
- Primary: 7-day window from signup
- Secondary: First N readings if sparse
- Tertiary: Re-establish after 30+ day gaps

## Error Handling

### Data Quality Issues
- Missing timestamps: Skip reading
- Invalid weights (<30kg or >400kg): Reject
- Duplicate readings: Keep latest
- Unit confusion: Flag for review

### System Failures
- Kalman divergence: Reset with higher Q
- ARIMA failure: Fallback to Layer 1 only
- Memory issues: Process in smaller batches

## Documentation Requirements

### Code Documentation
- Module-level docstrings explaining purpose
- Function signatures with type hints
- Complex algorithms: Reference equations

### Output Documentation
- results.json: Standard format
- baseline_results.json: Enhanced with viz data
- app.log: Structured logging (WARNING level)

## Version Control

### Branch Strategy
- main: Production-ready code
- develop: Integration branch
- feature/*: New features
- fix/*: Bug fixes

### Commit Standards
- Clear, descriptive messages
- Reference issue numbers
- Include test updates

## Monitoring & Metrics

### Key Performance Indicators
- Processing speed (users/second)
- Outlier detection rate (<15%)
- Baseline establishment rate (100%)
- Kalman coverage (100%)
- Memory usage (stable)

### Quality Metrics
- False positive rate for outliers
- Trend prediction accuracy (MAE)
- Gap handling success rate
- User data completeness

## Migration Path

### From Old Implementation
1. Move old code to deprecated/
2. Maintain data compatibility
3. Parallel run for validation
4. Gradual cutover with fallback

### Future Enhancements
- Layer 3 ML (Isolation Forest, LOF)
- Change point detection
- Kalman smoother for batch
- Adaptive noise parameters
- ARIMAX with contextual features

## Compliance Checklist

Before any PR:
- [ ] Maintains layer separation
- [ ] Preserves mathematical correctness
- [ ] Passes all unit tests
- [ ] Meets performance benchmarks
- [ ] Updates relevant documentation
- [ ] Follows coding standards
- [ ] Configuration via config.toml
- [ ] Handles edge cases gracefully