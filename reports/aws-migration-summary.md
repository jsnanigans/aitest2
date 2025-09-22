# AWS Weight Processing Service - Migration Summary & Quick Reference

## Overview

Migration of Python-based weight processing system to AWS microservice architecture, enabling real-time processing via REST API for Java backend integration.

## Core Features

### 1. **One-Time Cleanup** (`POST /api/v1/cleanup/{userId}`)
- Processes all historical data for a user
- Returns acceptance status for each measurement (identified by UUID)
- Resets Kalman state and starts fresh

### 2. **Process New Values** (`POST /api/v1/process/{userId}`)
- Processes multiple new measurements in a single request
- Automatically sorts measurements chronologically before processing
- All measurements must be after last processed timestamp (otherwise entire batch rejected)
- Returns acceptance status array matching all three endpoints
- Triggers replay requirement if any historical conflict detected

### 3. **Replay** (`POST /api/v1/replay/{userId}`)
- Handles historical data insertions
- Restores state from snapshot before replay point
- Reprocesses measurements chronologically
- Returns updated acceptance status for all replayed values

## Architecture Decision Summary

### Storage: DynamoDB
**Why DynamoDB over RDS PostgreSQL:**
- Key-value access pattern (userId -> state)
- Automatic scaling without connection pooling issues
- Built-in TTL for snapshot cleanup
- Lower operational overhead
- Cost-effective for this use case

**Schema Design:**
- Partition Key: `userId`
- Sort Key: `stateType` (current or snapshot_timestamp)
- Enables efficient state retrieval and snapshot queries

### Compute: AWS Lambda
**Why Lambda over ECS/Fargate:**
- Request-based workload (not continuous processing)
- Automatic scaling to zero
- No infrastructure management
- Pay-per-request pricing
- Sufficient for 60-second processing windows

**Configuration:**
- Runtime: Python 3.11
- Memory: 1024 MB
- Timeout: 60 seconds
- Reserved concurrency: 100

### State Management Strategy
**Snapshot System:**
- Daily snapshots per user (7-day retention)
- Enables replay from any point in last week
- Automatic cleanup via DynamoDB TTL
- Snapshot creation after significant operations

**Consistency Model:**
- Optimistic locking with version numbers
- Atomic state updates
- Idempotency through request deduplication

## API Quick Reference

### Request/Response Formats

#### Cleanup Request
```json
POST /api/v1/cleanup/{userId}
{
  "measurements": [{
    "uuid": "UUID",
    "weight": 75.5,
    "unit": "kg",
    "effectiveDateTime": "ISO8601",
    "source": "string"
  }],
  "options": {
    "resetState": true
  }
}
```

#### Process Request
```json
POST /api/v1/process/{userId}
{
  "measurements": [
    {
      "uuid": "UUID",
      "weight": 76.0,
      "unit": "kg",
      "effectiveDateTime": "ISO8601",
      "source": "string"
    },
    {
      "uuid": "UUID2",
      "weight": 75.8,
      "unit": "kg",
      "effectiveDateTime": "ISO8601",
      "source": "string"
    }
  ]
}
```

#### Replay Request
```json
POST /api/v1/replay/{userId}
{
  "replayFromTimestamp": "ISO8601",
  "measurements": [/* array of measurements */],
  "options": {
    "useSnapshot": true
  }
}
```

### Response Codes
- `200`: Successful processing
- `409`: Historical conflict (replay needed)
- `400`: Invalid request data
- `500`: Internal server error

## Key Implementation Files

```
src/
├── api/
│   ├── handlers.py         # Lambda entry points
│   └── schemas.py          # Pydantic models
├── services/
│   ├── state_service.py    # DynamoDB operations
│   └── replay_service.py   # Replay orchestration
├── core/
│   ├── processor.py        # Core logic (from current)
│   └── kalman.py          # Kalman filter (from current)
└── database/
    └── dynamodb_client.py  # DynamoDB client
```

## Java Integration Pattern

```java
// Recommended usage pattern
WeightProcessorClient client = new WeightProcessorClient(config);

// For new measurements (now supports batches)
try {
    List<Measurement> measurements = Arrays.asList(measurement1, measurement2);
    ProcessResponse response = client.process(userId, measurements);
    // Update local database with results for all measurements
} catch (HistoricalConflictException e) {
    // Fetch historical data from database
    List<Measurement> history = getHistorySince(e.getReplayFromTimestamp());
    history.addAll(measurements); // Include new measurements in replay
    ReplayResponse replay = client.replay(userId, history);
    // Update all affected measurements
}

// Single measurement convenience method still available
ProcessResponse response = client.processSingle(userId, measurement);
```

## Performance Characteristics

### Expected Latency
- Cleanup: 100-500ms per 100 measurements
- Process: 50-200ms for batch (up to 10 measurements)
- Replay: 200-1000ms depending on buffer size

### Scaling Limits
- Concurrent users: Unlimited (DynamoDB auto-scales)
- Requests/second: 1000 (API Gateway default)
- State size: 400KB per user (DynamoDB item limit)

## Cost Estimates (10K users, 100 req/day each)

### Monthly Costs
- API Gateway: ~$105
- Lambda: ~$20
- DynamoDB: ~$10
- CloudWatch: ~$10
- **Total: ~$145/month**

## Migration Checklist

### Week 1-2: Infrastructure
- [ ] Create AWS accounts
- [ ] Set up Terraform
- [ ] Deploy DynamoDB tables
- [ ] Configure Lambda functions

### Week 3-4: Implementation
- [ ] Port core processing logic
- [ ] Implement API handlers
- [ ] Create state management
- [ ] Add monitoring

### Week 5-6: Integration
- [ ] Java client library
- [ ] Integration tests
- [ ] Performance testing
- [ ] Documentation

### Week 7-8: Deployment
- [ ] Staging environment
- [ ] Parallel testing
- [ ] Data migration
- [ ] Production rollout

## Critical Success Factors

1. **State Consistency**: Ensure Kalman states remain consistent across operations
2. **Replay Accuracy**: Historical replay must produce identical results
3. **Performance**: Sub-200ms p50 latency for normal operations
4. **Reliability**: 99.9% availability with automatic recovery
5. **Cost Control**: Stay within 20% of estimated budget

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| State corruption | Snapshot system allows rollback |
| Lambda cold starts | Reserved concurrency |
| DynamoDB throttling | Auto-scaling and retry logic |
| Historical conflicts | Clear replay mechanism |
| Cost overruns | CloudWatch billing alerts |

## Support & Operations

### Monitoring
- CloudWatch dashboards for all metrics
- Alarms for error rates and latency
- X-Ray tracing for debugging

### Runbooks
1. High error rate: Check CloudWatch logs, verify DynamoDB
2. Slow performance: Check Lambda memory, DynamoDB throttling
3. State inconsistency: Use snapshot restore procedure

### Rollback Plan
1. Keep current system running in parallel
2. Dual-write during transition period
3. Switch traffic gradually
4. Maintain fallback capability for 30 days

## Contact Points

- **Technical Lead**: Define owner
- **AWS Support**: Premium support recommended
- **Escalation**: Define escalation path

## Next Immediate Steps

1. **Review & Approve**: Architecture and approach
2. **AWS Setup**: Create accounts and IAM roles
3. **Prototype**: Build minimal viable service
4. **Validate**: Test with sample data
5. **Plan**: Detailed project timeline