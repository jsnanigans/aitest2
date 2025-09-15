# Plan: Phase 2 - State Persistence with DynamoDB

## Summary
Implement DynamoDB adapter for persistent state storage, replacing in-memory storage while maintaining the same interface contract from Phase 0.

## Context
- Prerequisites: Phase 0 abstractions, Phase 1 API
- Current state: In-memory state storage only
- Target state: DynamoDB-backed persistence with consistency guarantees

## Requirements
- Implement AbstractStateStore interface for DynamoDB
- Maintain per-user Kalman state across Lambda invocations
- Support concurrent access with consistency
- Handle state versioning for conflict resolution
- Optimize for read-heavy workload

## Alternatives

### Option A: Single Table Design
**Approach**: One table with composite keys for all entities
- Pros: Single query for related data, cost-effective
- Cons: Complex key design, harder migrations
- Risks: Hot partitions with popular users

### Option B: Multi-Table Design
**Approach**: Separate tables for states and measurements
- Pros: Clear separation, easier to understand
- Cons: Multiple queries, higher cost
- Risks: Consistency across tables

### Option C: Single Table + S3
**Approach**: DynamoDB for metadata, S3 for large states
- Pros: Cost-effective for large data, unlimited size
- Cons: Higher latency, more complex
- Risks: Consistency between services

## Recommendation
**Option A: Single Table Design** - Best balance of performance and cost

## High-Level Design

### Table Structure
```
Table: weight-validation-service
Partition Key (PK): String
Sort Key (SK): String

Access Patterns:
1. Get user state: PK=USER#userId, SK=STATE
2. Get measurement: PK=USER#userId, SK=MEAS#timestamp#measurementId
3. List user measurements: PK=USER#userId, SK begins_with MEAS#
4. Batch get states: BatchGetItem with multiple PKs
```

### Item Types
```
User State:
  PK: USER#userId
  SK: STATE
  Attributes:
    - last_state: List[Number]
    - last_covariance: List[List[Number]]
    - last_timestamp: String (ISO-8601)
    - kalman_params: Map
    - measurement_history: List[Map] (last 30)
    - version: Number
    - updated_at: String (ISO-8601)
    - ttl: Number (optional)

Measurement Record:
  PK: USER#userId  
  SK: MEAS#timestamp#measurementId
  Attributes:
    - measurement_id: String
    - quality_score: Number
    - accepted: Boolean
    - result: Map
    - created_at: String (ISO-8601)
    - ttl: Number (optional)
```

## Implementation Plan (No Code)

### Step 1: Create DynamoDB Adapter
**File**: `src/adapters/dynamodb_store.py` (new)
- Implement `DynamoDBStateStore(AbstractStateStore)`:
  - Initialize with table name and client
  - Implement all abstract methods
  - Handle serialization/deserialization
  - Add retry logic with exponential backoff
- Key serialization:
  - Convert numpy arrays to lists
  - Convert datetime to ISO strings
  - Handle None values properly
  - Compress large histories

### Step 2: Implement State Operations
**Methods in DynamoDBStateStore**:
- `get_state(user_id)`:
  - Construct key: PK=USER#{user_id}, SK=STATE
  - Use GetItem with consistent read
  - Deserialize numpy arrays
  - Handle item not found
- `save_state(user_id, state)`:
  - Serialize state to DynamoDB format
  - Include version increment
  - Use PutItem with condition
  - Handle version conflicts
- `create_initial_state()`:
  - Return empty state structure
  - No database operation needed

### Step 3: Add Versioning and Optimistic Locking
**File**: `src/adapters/dynamodb_store.py`
- Version management:
  - Add version field to all states
  - Increment on each update
  - Use conditional expressions
- Conflict resolution:
  - Retry with fresh state on conflict
  - Maximum retry attempts (3)
  - Log conflicts for monitoring
- Atomic operations:
  - Use UpdateItem where possible
  - TransactWriteItems for multi-item

### Step 4: Implement Batch Operations
**Methods in DynamoDBStateStore**:
- `batch_get_states(user_ids)`:
  - Use BatchGetItem (max 100 items)
  - Handle pagination for large batches
  - Return dict of user_id → state
  - Cache results briefly
- `batch_save_states(states_dict)`:
  - Use BatchWriteItem (max 25 items)
  - Handle unprocessed items
  - Maintain version consistency
  - Return success/failure per user

### Step 5: Add Measurement History Management
**File**: `src/adapters/history_manager.py` (new)
- History operations:
  - Append new measurements
  - Maintain sliding window (30 items)
  - Compress old entries
  - Calculate statistics
- Optimization:
  - Store only essential fields
  - Use UpdateItem with list append
  - Periodic cleanup of old data
- Query support:
  - Get recent measurements
  - Filter by date range
  - Aggregate statistics

### Step 6: Create Migration Tools
**File**: `scripts/migrate_to_dynamodb.py` (new)
- Migration from in-memory:
  - Export current states to JSON
  - Transform to DynamoDB format
  - Batch import to table
  - Verify data integrity
- Rollback support:
  - Backup before migration
  - Restore from backup
  - Switch back to in-memory
- Progress tracking:
  - Log migration progress
  - Handle partial failures
  - Resume from checkpoint

### Step 7: Implement Caching Layer
**File**: `src/adapters/cached_store.py` (new)
- Cache strategy:
  - LRU cache for recent states
  - TTL-based expiration (5 minutes)
  - Write-through for updates
  - Invalidate on errors
- Implementation:
  - In-memory cache for Lambda
  - Optional Redis for persistent cache
  - Cache warming on cold start
  - Metrics for hit/miss rates

### Step 8: Add Monitoring and Metrics
**File**: `src/adapters/dynamodb_metrics.py` (new)
- CloudWatch metrics:
  - Read/write latency
  - Throttling events
  - Item sizes
  - Conflict rates
- Custom metrics:
  - Cache hit rates
  - Version conflicts
  - State sizes
  - History lengths
- Alarms:
  - High latency (>100ms)
  - Throttling detected
  - Large items (>400KB)
  - High conflict rate (>5%)

## Validation & Testing

### Test Strategy
- Unit tests with DynamoDB Local
- Integration tests with real table
- Load tests for concurrent access
- Chaos tests for failure scenarios

### Test Cases
- [ ] Single state read/write
- [ ] Concurrent updates (version conflict)
- [ ] Batch operations
- [ ] Large state handling
- [ ] Network failures and retries
- [ ] Throttling behavior
- [ ] Cache consistency
- [ ] Migration correctness

## Risks & Mitigations

### Risk 1: Hot Partitions
- **Impact**: Throttling for popular users
- **Mitigation**: Add jitter to keys, use on-demand billing
- **Monitoring**: Track per-partition metrics

### Risk 2: Large Item Sizes
- **Impact**: 400KB DynamoDB limit exceeded
- **Mitigation**: Compress data, offload to S3
- **Monitoring**: Track item sizes

### Risk 3: Consistency Issues
- **Impact**: Incorrect Kalman predictions
- **Mitigation**: Use consistent reads, versioning
- **Monitoring**: Track version conflicts

### Risk 4: Cost Overruns
- **Impact**: Unexpected AWS bills
- **Mitigation**: Use on-demand initially, optimize later
- **Monitoring**: Cost allocation tags

## Acceptance Criteria
- [ ] DynamoDB adapter passes all interface tests
- [ ] State persistence across Lambda invocations
- [ ] Concurrent access handled correctly
- [ ] Version conflicts resolved properly
- [ ] Batch operations performant
- [ ] Migration tools tested
- [ ] Monitoring in place
- [ ] Performance within targets (<100ms p99)

## Configuration
```yaml
# Environment Variables
DYNAMODB_TABLE: weight-validation-service
DYNAMODB_REGION: us-east-1
STATE_BACKEND: dynamodb  # or 'memory'
CACHE_ENABLED: true
CACHE_TTL_SECONDS: 300
MAX_BATCH_SIZE: 100
CONSISTENT_READS: true
MAX_RETRIES: 3
```

## DynamoDB Table Configuration
```yaml
TableName: weight-validation-service
BillingMode: PAY_PER_REQUEST  # Start with on-demand
PointInTimeRecovery: true
StreamSpecification:
  StreamEnabled: true
  StreamViewType: NEW_AND_OLD_IMAGES
Tags:
  - Service: weight-validation
  - Environment: production
  - Team: platform
```

## Performance Targets
- Read latency: p50 < 10ms, p99 < 100ms
- Write latency: p50 < 20ms, p99 < 200ms
- Batch read: 100 items < 500ms
- Cache hit rate: > 80%
- Version conflict rate: < 5%

## Cost Estimates
- On-demand pricing:
  - Reads: $0.25 per million
  - Writes: $1.25 per million
- Estimated monthly (100K users, 10 measurements/day):
  - Reads: 30M × $0.25/M = $7.50
  - Writes: 30M × $1.25/M = $37.50
  - Storage: 100GB × $0.25/GB = $25
  - Total: ~$70/month

## Dependencies
- Phase 0: Abstract interfaces defined
- Phase 1: API layer for testing
- AWS SDK (boto3) for DynamoDB
- DynamoDB Local for development

## Migration Plan
1. Deploy DynamoDB table
2. Test with small subset
3. Dual-write period (memory + DynamoDB)
4. Gradual migration by user cohort
5. Full cutover with fallback ready
6. Decommission in-memory storage

## Next Steps
- Phase 3: Idempotency & Ordering
- Phase 4: Batch Optimization
- Phase 5: Monitoring & Operations