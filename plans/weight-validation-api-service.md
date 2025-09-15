# Plan: Weight Validation API Service

## Summary
Transform the existing weight stream processor into a standalone API service that validates weight measurements and returns quality scores. The service will maintain per-user state for Kalman filtering and provide a simple HTTP API for weight validation.

## Context
- Source: User request to restructure as validation API service
- Current state: CLI-based CSV processor with in-memory state
- Core requirement: Quality score as primary output
- Existing AWS Lambda plan reviewed but contains some inaccuracies

## Assumptions
- Service will be deployed as AWS Lambda or containerized service
- State persistence required for Kalman filter continuity
- Batch processing support needed for efficiency
- Quality score is the most critical output

## Requirements

### Functional
- Accept weight measurements via HTTP API (single or batch)
- Return quality score (0-1) for each measurement
- Return accepted/rejected status with reason
- Maintain per-user Kalman filter state
- Support idempotent operations
- Preserve measurement ordering per user

### Non-functional
- Sub-second response time for single measurements
- Support 100+ concurrent users
- 99.9% availability
- State consistency across restarts
- Audit trail for all decisions

## Current System Analysis

### Existing Components We Have
1. **Core Processing** (`src/processor.py`)
   - `process_measurement()` - Main entry point
   - Stateless processing with state passed in/out
   - Returns comprehensive result dict

2. **Quality Scoring** (`src/quality_scorer.py`)
   - Unified scoring system (0-1 scale)
   - Components: safety, plausibility, consistency, reliability
   - Configurable thresholds (default 0.61)
   - Stateless design

3. **Kalman Filtering** (`src/kalman.py`)
   - State tracking for weight predictions
   - Adaptive noise based on source
   - Reset logic for gaps (30 days default)

4. **Validation** (`src/validation.py`)
   - PhysiologicalValidator (30-400 kg limits)
   - BMIValidator (10-90 BMI range)
   - DataQualityPreprocessor
   - ThresholdCalculator

5. **State Management** (`src/database.py`)
   - In-memory ProcessorStateDB
   - Stores: Kalman state, covariance, last timestamp
   - Measurement history (last 30)

6. **Configuration** (`config.toml`)
   - Kalman parameters
   - Quality scoring weights
   - Thresholds and limits

### What We DON'T Have (Blockers)
1. **HTTP API layer** - Need to implement
2. **DynamoDB adapter** - Currently only in-memory
3. **Authentication/Authorization** - Not implemented
4. **Idempotency tracking** - Not built-in
5. **Batch processing optimization** - Sequential only

## Alternatives

### Option A: Minimal Lambda Function
**Approach**: Direct port to Lambda with DynamoDB state
- Pros: 
  - Fastest to implement
  - Leverages existing code structure
  - Simple deployment
- Cons:
  - Limited batch optimization
  - No caching layer
  - Cold start penalties with numpy/pykalman
- Risks: Performance issues with heavy dependencies

### Option B: FastAPI Service with Cache
**Approach**: Containerized FastAPI with Redis cache + DynamoDB
- Pros:
  - Better performance with caching
  - More control over runtime
  - Easier local development
- Cons:
  - More infrastructure to manage
  - Higher operational overhead
  - Requires container orchestration
- Risks: Complexity for small-scale deployment

### Option C: Lambda with Step Functions
**Approach**: Orchestrated processing with state machines
- Pros:
  - Better batch handling
  - Parallel user processing
  - Built-in retry/error handling
- Cons:
  - More complex architecture
  - Higher latency for single requests
  - Additional AWS services
- Risks: Over-engineering for current needs

## Recommendation
**Option A: Minimal Lambda Function** with incremental improvements

Rationale:
- Fastest path to production
- Existing code is already mostly stateless
- Can optimize later if needed
- Serverless scales automatically

## High-Level Design

### Architecture
```
API Gateway → Lambda → DynamoDB
                ↓
         Quality Scorer
                ↓
         Kalman Filter
```

### API Design
```
POST /validate
{
  "measurements": [
    {
      "user_id": "string",
      "measurement_id": "string",  
      "timestamp": "ISO-8601",
      "weight": number,
      "unit": "kg|lbs",
      "source": "string"
    }
  ]
}

Response:
{
  "results": [
    {
      "user_id": "string",
      "measurement_id": "string",
      "quality_score": 0.85,
      "accepted": true,
      "components": {
        "safety": 0.95,
        "plausibility": 0.88,
        "consistency": 0.82,
        "reliability": 0.75
      },
      "kalman_prediction": 75.2,
      "filtered_weight": 75.5,
      "reason": null
    }
  ]
}
```

### Data Model (DynamoDB)
```
Table: weight-validation-state
  PK: USER#{user_id}
  SK: STATE
  Attributes:
    - last_state: [weight, trend]
    - last_covariance: [[...]]
    - last_timestamp: ISO-8601
    - kalman_params: {...}
    - measurement_history: [...]
    - version: number

Table: weight-validation-measurements  
  PK: USER#{user_id}
  SK: MEAS#{timestamp}#{measurement_id}
  Attributes:
    - quality_score: number
    - accepted: boolean
    - processed_at: ISO-8601
    - result: {...}
```

## Implementation Plan (No Code)

### Phase 1: Core API Implementation
1. Create Lambda handler wrapping `process_measurement()`
2. Implement request/response mapping
3. Add input validation layer
4. Setup API Gateway integration

### Phase 2: State Persistence
1. Create DynamoDB tables
2. Implement DynamoDB adapter extending ProcessorStateDB
3. Add conditional writes for consistency
4. Implement state versioning

### Phase 3: Idempotency & Ordering
1. Add measurement deduplication check
2. Implement per-user ordering enforcement  
3. Add idempotency keys to API
4. Create measurement audit table

### Phase 4: Batch Optimization
1. Group measurements by user
2. Sort by timestamp within user
3. Process sequentially per user
4. Return aggregated results

### Phase 5: Monitoring & Operations
1. Add CloudWatch metrics
2. Implement structured logging
3. Create dashboards
4. Setup alarms

## Validation & Rollout

### Test Strategy
- Unit tests for Lambda handler
- Integration tests with LocalStack
- Load tests with expected traffic patterns
- Chaos testing for state consistency

### Manual QA Checklist
- [ ] Single measurement validation
- [ ] Batch processing with mixed users
- [ ] Duplicate measurement handling
- [ ] Out-of-order timestamp rejection
- [ ] State persistence across invocations
- [ ] Error handling and retries

### Rollout Plan
1. Deploy to dev environment
2. Run parallel validation with CSV processor
3. Compare results for consistency
4. Gradual traffic shift with feature flag
5. Monitor metrics and error rates

## Risks & Mitigations

### Risk 1: Cold Start Performance
- **Impact**: High latency for first requests
- **Mitigation**: Use Lambda SnapStart or provisioned concurrency
- **Monitoring**: Track cold start metrics

### Risk 2: State Consistency
- **Impact**: Incorrect Kalman predictions
- **Mitigation**: Use DynamoDB conditional writes
- **Monitoring**: Track version conflicts

### Risk 3: Dependency Size
- **Impact**: Deployment package too large
- **Mitigation**: Use Lambda layers for numpy/pykalman
- **Monitoring**: Track package size

## Acceptance Criteria
- [ ] API accepts weight measurements and returns quality scores
- [ ] Quality scores match existing processor output
- [ ] Per-user state persists across invocations
- [ ] Duplicate measurements return same result
- [ ] Out-of-order measurements are rejected
- [ ] Response time < 1 second for single measurement
- [ ] Service handles 100+ concurrent users

## Out of Scope
- User authentication (assume handled by API Gateway)
- Historical data migration
- Visualization features
- Real-time streaming
- Multi-region deployment
- Data export features

## Open Questions
1. Should we support bulk historical import?
2. What retention policy for measurements?
3. Need for async processing option?
4. Required SLA for availability?
5. Budget constraints for AWS services?

## Implementation Blockers

### Required Before Implementation
1. **DynamoDB Adapter**: Need to implement database.py adapter for DynamoDB
2. **Lambda Handler**: Need to create handler function and request mapping
3. **API Schema**: Need to finalize exact request/response format
4. **Error Codes**: Need to define standard error responses
5. **Configuration Management**: Need strategy for per-environment configs

### Nice-to-Have (Not Blocking)
1. Caching layer for frequently accessed states
2. Batch processing optimizations
3. Async processing option
4. WebSocket support for real-time updates

## Review Cycle
### Self-Review Notes
- ✓ Analyzed actual codebase capabilities
- ✓ Identified concrete blockers
- ✓ Provided realistic implementation phases
- ✓ Kept existing architecture patterns
- ✓ Focused on quality score as primary output

### Revisions Made
- Removed references to non-existent features from original plan
- Simplified data model based on actual state structure
- Aligned with existing stateless processor design
- Added specific implementation blockers section