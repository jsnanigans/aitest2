# Plan: Phase 3 - Idempotency & Ordering

## Summary
Implement idempotency guarantees and timestamp ordering enforcement to ensure consistent processing of weight measurements, preventing duplicates and maintaining chronological integrity.

## Context
- Prerequisites: Phases 0-2 complete (API + DynamoDB)
- Current state: No duplicate detection, no ordering enforcement
- Target state: Idempotent API with strict timestamp ordering per user

## Requirements
- Detect and handle duplicate measurements
- Enforce chronological ordering per user
- Support request-level idempotency
- Maintain audit trail of all attempts
- Handle out-of-order arrivals gracefully

## Alternatives

### Option A: Database-Level Idempotency
**Approach**: Use DynamoDB conditional writes with measurement ID
- Pros: Atomic guarantees, simple logic
- Cons: Extra storage, more DynamoDB calls
- Risks: Storage costs for deduplication data

### Option B: Application-Level Caching
**Approach**: In-memory cache of recent measurement IDs
- Pros: Fast lookups, no extra storage
- Cons: Cache misses, cold start issues
- Risks: Inconsistency across Lambda instances

### Option C: Hybrid Approach
**Approach**: Cache with database backing
- Pros: Fast common case, consistency guaranteed
- Cons: More complex implementation
- Risks: Cache coherency challenges

## Recommendation
**Option A: Database-Level Idempotency** - Strongest consistency guarantees

## High-Level Design

### Idempotency Strategy
```
1. Request arrives with measurement_id
2. Check if measurement exists in DynamoDB
3. If exists with same payload → return cached result
4. If exists with different payload → reject as duplicate
5. If not exists → process and store with result
```

### Ordering Strategy
```
1. Load user's last accepted timestamp
2. Compare with new measurement timestamp
3. If new < last → reject as out-of-order
4. If new >= last → process measurement
5. Update last timestamp atomically
```

## Implementation Plan (No Code)

### Step 1: Add Idempotency Storage
**DynamoDB Schema Addition**:
```
Idempotency Record:
  PK: USER#userId
  SK: IDEMP#measurementId
  Attributes:
    - payload_hash: String
    - result: Map
    - processed_at: String
    - ttl: Number (7 days)
```

**File**: `src/adapters/idempotency_store.py` (new)
- Methods:
  - `check_measurement(user_id, measurement_id, payload_hash)`
  - `store_measurement(user_id, measurement_id, payload_hash, result)`
  - `get_cached_result(user_id, measurement_id)`
- Behavior:
  - Return existing result if hash matches
  - Raise DuplicateError if hash differs
  - Store with TTL for cleanup

### Step 2: Implement Payload Hashing
**File**: `src/api/hashing.py` (new)
- Hash calculation:
  - Include: weight, timestamp, source, unit
  - Exclude: measurement_id, request metadata
  - Use SHA-256 for consistency
  - Handle floating point precision
- Normalization:
  - Convert units to standard (kg)
  - Round weights to 2 decimals
  - Normalize timestamp to UTC
  - Lowercase source strings

### Step 3: Add Timestamp Ordering
**File**: `src/adapters/ordering_manager.py` (new)
- Ordering enforcement:
  - Track last accepted timestamp per user
  - Compare with new measurement
  - Allow equal timestamps (different sources)
  - Reject older timestamps
- Grace period:
  - Allow small clock skew (5 minutes)
  - Configurable per environment
  - Log violations for monitoring
- Batch handling:
  - Sort measurements by timestamp
  - Process in chronological order
  - Stop on first rejection (optional)

### Step 4: Create Idempotency Middleware
**File**: `src/api/middleware/idempotency.py` (new)
- Request-level idempotency:
  - Extract idempotency key from header
  - Cache entire request/response
  - Return cached response if key matches
  - TTL of 24 hours
- Implementation:
  - Store in DynamoDB with request hash
  - Include in response headers
  - Support retry-after for in-progress

### Step 5: Implement Conflict Resolution
**File**: `src/api/conflict_handler.py` (new)
- Duplicate handling:
  - Return 409 Conflict status
  - Include original result in response
  - Provide diff of payloads
  - Suggest resolution steps
- Out-of-order handling:
  - Return 422 Unprocessable Entity
  - Include last accepted timestamp
  - Suggest resubmission time
  - Option to force override

### Step 6: Add Audit Trail
**DynamoDB Schema Addition**:
```
Audit Record:
  PK: USER#userId
  SK: AUDIT#timestamp#attemptId
  Attributes:
    - measurement_id: String
    - action: String (accepted/rejected/duplicate)
    - reason: String
    - payload: Map
    - correlation_id: String
    - ttl: Number (30 days)
```

**File**: `src/adapters/audit_logger.py` (new)
- Audit operations:
  - Log all validation attempts
  - Include success and failure
  - Track duplicate attempts
  - Record ordering violations
- Query support:
  - Get user's audit history
  - Filter by time range
  - Search by measurement_id
  - Export for compliance

### Step 7: Batch Processing Updates
**File**: `src/api/batch_processor.py` (update)
- Ordering within batch:
  - Group by user_id
  - Sort by timestamp
  - Process sequentially
  - Maintain transaction boundaries
- Failure handling:
  - Continue on duplicate
  - Stop on ordering violation (configurable)
  - Return detailed status per item
  - Include suggested retry order

### Step 8: Add Monitoring
**File**: `src/api/monitoring/idempotency_metrics.py` (new)
- Metrics to track:
  - Duplicate request rate
  - Out-of-order rate
  - Cache hit rate
  - Conflict resolution time
- Alerts:
  - High duplicate rate (>10%)
  - Ordering violations spike
  - Cache misses increase
  - Audit storage growth

## Validation & Testing

### Test Strategy
- Unit tests for each component
- Integration tests for workflows
- Concurrent access tests
- Time-based edge cases

### Test Cases
- [ ] Same measurement submitted twice
- [ ] Different payload, same ID
- [ ] Out-of-order timestamps
- [ ] Concurrent submissions
- [ ] Batch with mixed ordering
- [ ] Clock skew scenarios
- [ ] Idempotency key reuse
- [ ] TTL expiration

### Edge Cases
1. **Exactly same timestamp**: Allow if different measurement_id
2. **Clock adjustment**: Handle daylight saving changes
3. **Retroactive data**: Admin override capability
4. **Missing idempotency key**: Process but warn
5. **Batch partial failure**: Clear status per item

## Risks & Mitigations

### Risk 1: Storage Growth
- **Impact**: DynamoDB costs increase
- **Mitigation**: TTL on idempotency records
- **Monitoring**: Track table size

### Risk 2: Strict Ordering Too Restrictive
- **Impact**: Valid data rejected
- **Mitigation**: Configurable grace period
- **Monitoring**: Track rejection rates

### Risk 3: Cache Inconsistency
- **Impact**: Duplicate processing
- **Mitigation**: Database as source of truth
- **Monitoring**: Cache vs DB mismatches

## Acceptance Criteria
- [ ] Duplicate measurements detected
- [ ] Same request returns same response
- [ ] Out-of-order rejected correctly
- [ ] Audit trail complete
- [ ] Batch ordering maintained
- [ ] Conflicts handled gracefully
- [ ] Performance impact < 10%
- [ ] All edge cases handled

## Configuration
```yaml
# Environment Variables
ENABLE_IDEMPOTENCY: true
IDEMPOTENCY_TTL_DAYS: 7
ENABLE_ORDERING: true
ORDERING_GRACE_PERIOD_SECONDS: 300
AUDIT_TTL_DAYS: 30
BATCH_FAIL_FAST: false
ALLOW_TIMESTAMP_OVERRIDE: false
```

## API Changes

### Request Headers
```
X-Idempotency-Key: unique-request-id
X-Force-Override: true (admin only)
```

### Response Headers
```
X-Idempotency-Key: unique-request-id
X-Duplicate-Of: original-measurement-id
X-Retry-After: seconds (for rate limiting)
```

### Error Responses
```json
409 Conflict (Duplicate):
{
  "error": "DuplicateMeasurement",
  "message": "Measurement already exists with different payload",
  "measurement_id": "meas456",
  "original_result": {...},
  "payload_diff": {...}
}

422 Unprocessable (Ordering):
{
  "error": "OutOfOrderTimestamp",
  "message": "Timestamp older than last accepted",
  "last_accepted": "2025-09-15T10:00:00Z",
  "submitted": "2025-09-15T09:00:00Z",
  "user_id": "user123"
}
```

## Performance Impact
- Additional DynamoDB read per measurement
- ~20ms latency for idempotency check
- Minimal memory overhead
- Storage: ~1KB per measurement for 7 days

## Dependencies
- Phase 2: DynamoDB implementation
- No new external dependencies
- Optional: Redis for caching layer

## Migration Considerations
- Existing measurements won't have idempotency records
- Start enforcement for new measurements only
- Backfill option for historical data
- Gradual rollout with feature flag

## Next Steps
- Phase 4: Batch Optimization
- Phase 5: Monitoring & Operations
- Future: Event sourcing for full history