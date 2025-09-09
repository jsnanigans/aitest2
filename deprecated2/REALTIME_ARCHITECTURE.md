# Real-Time Production Architecture

## Overview
The weight processing system is designed for **completely stateless, recursive processing** suitable for production deployment where each weight measurement is processed independently without access to historical data.

## Key Architecture Principles

### 1. Complete State Independence
- **NO HISTORICAL QUERIES**: System never queries past measurements
- **SINGLE MEASUREMENT PROCESSING**: Each weight processed in isolation
- **STATE-BASED CONTINUATION**: Uses only saved state from previous processing
- **ZERO MEMORY BETWEEN CALLS**: Each API call is completely independent

### 2. State Persistence Model
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  New Weight     │────▶│  Load State     │────▶│  Process with   │
│  Measurement    │     │  from DB/Cache  │     │  Kalman Filter  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                        ┌─────────────────┐              ▼
                        │  Return Result  │◀────┌─────────────────┐
                        │  to Caller      │     │  Save Updated   │
                        └─────────────────┘     │  State to DB    │
                                                └─────────────────┘
```

## Implementation Components

### 1. RealTimeKalmanProcessor
Main processor class that handles single measurements:

```python
processor = RealTimeKalmanProcessor(config)

# Process single measurement (production endpoint)
result = processor.process_single_measurement(
    user_id="user_123",
    weight=80.5,
    timestamp=datetime.now(),
    source="care-team-upload"
)
```

### 2. StateManager
Handles persistence and restoration of Kalman state:

```python
# Save state after processing
state_manager.save_state(user_id, {
    'kalman_state': kalman.get_state(),
    'baseline': baseline_params,
    'last_processed': timestamp,
    'measurement_count': count
})

# Load state before processing
saved_state = state_manager.load_state(user_id)
```

### 3. State Schema
Complete state stored between measurements:

```json
{
    "user_id": "user_123",
    "kalman": {
        "weight": 80.5,              // Current weight estimate
        "trend": 0.01,                // Trend in kg/day
        "covariance": [               // 2x2 covariance matrix
            [0.5, 0],
            [0, 0.001]
        ],
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

## Production Deployment

### API Endpoints

#### Process Weight Measurement
```http
POST /api/weight/process
Content-Type: application/json

{
    "user_id": "user_123",
    "weight": 80.5,
    "timestamp": "2025-01-15T10:30:00Z",
    "source": "care-team-upload"
}

Response:
{
    "accepted": true,
    "confidence": 0.95,
    "filtered_weight": 80.4,
    "trend_kg_per_week": 0.15,
    "prediction_error": 0.3,
    "needs_initialization": false
}
```

#### Initialize User
```http
POST /api/weight/initialize
Content-Type: application/json

{
    "user_id": "user_123",
    "measurements": [
        {"weight": 80.0, "timestamp": "2025-01-01T08:00:00Z", "source": "care-team"},
        {"weight": 80.2, "timestamp": "2025-01-02T08:00:00Z", "source": "care-team"},
        {"weight": 79.8, "timestamp": "2025-01-03T08:00:00Z", "source": "patient"}
    ]
}
```

### Database Options

#### PostgreSQL
```sql
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

CREATE INDEX idx_last_timestamp ON kalman_states(last_timestamp);
```

#### MongoDB
```javascript
db.kalman_states.createIndex({ "user_id": 1 }, { unique: true })
db.kalman_states.createIndex({ "updated_at": 1 })

// Document structure
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

#### Redis Cache
```redis
# Primary state storage
SET kalman:state:user_123 '{"weight":80.5,"trend":0.01,...}'
EXPIRE kalman:state:user_123 604800  # 7-day TTL

# Initialization queue for new users
LPUSH kalman:init:user_123 '{"weight":80.0,"timestamp":"..."}'
EXPIRE kalman:init:user_123 86400  # 24-hour TTL
```

## Performance Characteristics

### Processing Time
- **Target**: < 100ms per measurement
- **Breakdown**:
  - State load: 10-20ms (Redis), 20-50ms (DB)
  - Kalman processing: 1-2ms
  - State save: 10-20ms (Redis), 20-50ms (DB)
  - Total: 30-100ms

### Scalability
- **Horizontal**: Completely stateless, infinitely scalable
- **Concurrent**: No locks needed, each user independent
- **Cache Strategy**: Redis with DB fallback
- **Load Pattern**: O(1) per measurement

## Special Scenarios

### 1. New User Flow
```
1. First measurement arrives
2. System returns "needs_initialization"
3. Measurements queued temporarily
4. After 3+ measurements: initialize_user called
5. Baseline established, state saved
6. Future measurements processed normally
```

### 2. Gap Detection
```
1. Measurement arrives after long gap (>30 days)
2. System detects gap from last_processed timestamp
3. State deleted, returns "needs_reinitialization"
4. User goes through initialization flow again
```

### 3. State Recovery
```
1. If state missing from cache, check database
2. If state corrupted, mark for re-initialization
3. If database unavailable, queue for retry
4. Log all state issues for monitoring
```

## Testing

### Unit Test
```python
# Test single measurement processing
def test_realtime_processing():
    processor = RealTimeKalmanProcessor(config)
    
    # Initialize user
    processor.initialize_user("test_user", initial_measurements)
    
    # Process single measurement
    result = processor.process_single_measurement(
        "test_user", 80.5, datetime.now(), "care-team"
    )
    
    assert result['accepted'] == True
    assert result['confidence'] > 0.8
    assert result['filtered_weight'] is not None
```

### Integration Test
```bash
# Run test simulation
python test_realtime_processing.py

# Start example API server
uvicorn example_api_server:app --reload

# Test with curl
curl -X POST http://localhost:8000/api/weight/process \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","weight":80.5,"timestamp":"2025-01-15T10:00:00Z","source":"care-team"}'
```

### Load Test
```python
# Simulate production load
import concurrent.futures
import time

def process_measurement(user_id, weight):
    return processor.process_single_measurement(
        user_id, weight, datetime.now(), "test"
    )

# Process 1000 measurements concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    start = time.time()
    futures = [
        executor.submit(process_measurement, f"user_{i}", 80.0 + i*0.1)
        for i in range(1000)
    ]
    results = [f.result() for f in futures]
    elapsed = time.time() - start
    
print(f"Processed 1000 measurements in {elapsed:.2f}s")
print(f"Average: {elapsed/1000*1000:.1f}ms per measurement")
```

## Monitoring

### Key Metrics
- **Processing Latency**: P50, P95, P99
- **Acceptance Rate**: % of measurements accepted
- **State Hit Rate**: Cache vs DB loads
- **Initialization Queue**: Length and age
- **Gap Detections**: Frequency of re-initializations

### Alerts
- Processing latency > 200ms (P95)
- Acceptance rate < 70%
- State load failures > 1%
- Initialization queue > 100 users
- Database connection errors

## Migration Path

### From Batch Processing
1. **Phase 1**: Run both systems in parallel
2. **Phase 2**: Compare results, tune parameters
3. **Phase 3**: Route % of traffic to real-time
4. **Phase 4**: Full migration with fallback
5. **Phase 5**: Decommission batch system

### State Version Management
```python
# Handle state schema changes
def migrate_state_v1_to_v2(old_state):
    return {
        **old_state,
        'version': 2,
        'new_field': default_value
    }

# In processor
state = load_state(user_id)
if state['version'] < CURRENT_VERSION:
    state = migrate_state(state)
```

## Security Considerations

### Data Protection
- Encrypt state at rest (database encryption)
- Use TLS for Redis connections
- Implement field-level encryption for PII
- Audit log all state access

### Access Control
- API key authentication
- Rate limiting per user
- Role-based access (read vs write)
- Separate endpoints for admin operations

## Cost Optimization

### Storage
- Compress covariance matrices
- TTL on Redis entries
- Archive old states to cold storage
- Batch database writes if possible

### Compute
- Cache Kalman matrices
- Use connection pooling
- Optimize JSON serialization
- Consider protocol buffers for state

## Conclusion

This architecture provides:
- ✅ **True real-time processing** without historical data access
- ✅ **Horizontal scalability** through stateless design
- ✅ **Production readiness** with proper state management
- ✅ **Fault tolerance** through persistent state
- ✅ **Easy deployment** with standard REST APIs

The system is ready for production deployment with appropriate database/cache backend configuration.