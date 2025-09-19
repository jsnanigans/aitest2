# Plan: Lambda Migration Strategy for Weight Processing System

## Decision

**Approach**: Phased migration from monolithic stream processor to serverless Lambda architecture
**Why**: Maintain statistical integrity while gaining scalability, cost-efficiency, and operational simplicity
**Risk Level**: Medium

## Implementation Steps

### Phase 1: Core Processing Lambda (Week 1-2)

1. **Extract Processing Core** - Create `lambda/process_observations.py` from `src/processing/processor.py`
   - Remove file I/O and CSV processing logic
   - Keep Kalman filter, quality scoring, validation logic
   - Refactor to accept single/batch JSON payloads

2. **DynamoDB State Adapter** - Create `lambda/state_manager.py`
   - Map SQLite schema to DynamoDB structure
   - Implement optimistic locking with version numbers
   - Handle numpy array serialization/deserialization

3. **Configuration Management** - Migrate `config.toml` to Parameter Store
   - Create hierarchy: `/weight-processor/prod/kalman/*`
   - Map profiles to parameter sets
   - Implement config caching in Lambda

### Phase 2: State Migration (Week 2-3)

4. **DynamoDB Schema Design** - Create tables via CDK
   - Primary table: `weight-processor-states`
   - GSI for cleanup queries: `by-last-processed`
   - Snapshot table: `weight-processor-snapshots`

5. **State Serialization Layer** - Create `lambda/state_serializer.py`
   - Handle Kalman matrices (numpy → DynamoDB)
   - Preserve measurement history buffer
   - Maintain reset event tracking

6. **Migration Script** - Create `scripts/migrate_states.py`
   - Export from SQLite/in-memory store
   - Transform to DynamoDB format
   - Batch upload with error recovery

### Phase 3: API Gateway Integration (Week 3-4)

7. **Lambda Request Handlers** - Create `lambda/api_handlers.py`
   - Map API Gateway events to processor inputs
   - Handle batch vs single observation detection
   - Implement correlation ID tracking

8. **Error Mapping Layer** - Create `lambda/error_handler.py`
   - Map internal errors to HTTP status codes
   - Generate structured error responses
   - Implement retry-after headers

9. **Response Formatting** - Update result formatting
   - Add FHIR-compliant observation status
   - Include quality score components
   - Format for Spring Boot consumption

### Phase 4: Async Processing (Week 4-5)

10. **Cleanup Lambda** - Create `lambda/cleanup_handler.py`
    - Query historical observations from source
    - Reprocess through Kalman pipeline
    - Update state with new timeline

11. **SQS Integration** - Setup async processing
    - Configure dead letter queues
    - Implement batch processing (10 messages)
    - Add CloudWatch alarms for queue depth

12. **Replay Buffer Migration** - Adapt replay functionality
    - Move buffer to DynamoDB with TTL
    - Trigger via SQS for async replay
    - Maintain outlier detection logic

### Phase 5: Operational Readiness (Week 5-6)

13. **Monitoring Setup** - Configure CloudWatch/X-Ray
    - Custom metrics for quality scores
    - Distributed tracing with correlation IDs
    - Business metrics dashboards

14. **Testing Suite** - Create `tests/lambda_tests.py`
    - Unit tests for Lambda handlers
    - Integration tests with LocalStack
    - Load tests with synthetic data

15. **Deployment Pipeline** - Setup CI/CD
    - AWS CDK for infrastructure
    - GitHub Actions for deployment
    - Blue-green deployment pattern

## Files to Change

### New Files (Lambda Functions)
- `lambda/process_observations.py` - Main processing handler
- `lambda/state_manager.py` - DynamoDB state operations
- `lambda/api_handlers.py` - API Gateway request/response
- `lambda/cleanup_handler.py` - Async cleanup processing
- `lambda/error_handler.py` - Error mapping and responses
- `lambda/state_serializer.py` - State format conversion
- `lambda/config_manager.py` - Parameter Store integration

### Modified Files
- `src/processing/processor.py:98-663` - Extract core logic to Lambda
- `src/database/database.py:16-163` - Replace with DynamoDB adapter
- `src/processing/kalman.py` - Add serialization helpers
- `src/processing/validation.py` - Make stateless
- `src/processing/quality_scorer.py` - Remove file dependencies
- `config.toml` - Convert to Parameter Store format

### Infrastructure Files
- `cdk/app.py` - CDK application entry
- `cdk/stacks/lambda_stack.py` - Lambda functions
- `cdk/stacks/dynamodb_stack.py` - Database tables
- `cdk/stacks/api_stack.py` - API Gateway setup
- `openapi.yaml` - API specification

## Acceptance Criteria

### Functional Requirements
- [ ] Process single observation in <500ms P95
- [ ] Process batch (25 observations) in <5s
- [ ] Maintain Kalman filter mathematical precision
- [ ] Support all three reset types (INITIAL, HARD, SOFT)
- [ ] Preserve quality scoring accuracy
- [ ] Handle outdated observations with 409 response

### Non-Functional Requirements
- [ ] 99.9% availability SLA
- [ ] <3s cold start with dependencies
- [ ] Support 1000 req/s burst capacity
- [ ] Cost <$100/month for 10k users
- [ ] Zero data loss during migration
- [ ] Backward compatible API responses

### Migration Validation
- [ ] State migration script processes 10k users/hour
- [ ] Kalman states produce identical results
- [ ] Quality scores match within 0.01 tolerance
- [ ] Reset events preserved accurately
- [ ] Measurement history maintained

## Risks & Mitigations

**Main Risk**: State consistency during concurrent Lambda executions
**Mitigation**: DynamoDB optimistic locking with exponential backoff retry

**Secondary Risk**: Cold start latency with numpy/scipy dependencies
**Mitigation**: Use Lambda layers, provisioned concurrency for critical path

**Data Risk**: Loss of precision in Kalman matrix serialization
**Mitigation**: Use high-precision decimal encoding, validate roundtrip accuracy

## Out of Scope

- Visualization generation (remains separate service)
- Historical data backfill from source systems
- Real-time WebSocket updates
- Cross-region replication (phase 2)

## Technical Details

### DynamoDB State Schema
```python
{
    'user_id': 'uuid-string',  # Partition key
    'version': 123,  # Optimistic lock
    'last_processed': '2024-02-25T14:30:00Z',
    'kalman_state': {
        'state_vector': [75.5, 0.02],  # [weight, trend]
        'covariance_matrix': [[0.361, 0], [0, 0.0001]],
        'params': {
            'transition': [[1, 1], [0, 1]],
            'observation': [[1, 0]],
            'process_noise': [[0.016, 0], [0, 0.0001]],
            'observation_noise': 3.49
        }
    },
    'measurement_buffer': [
        {'timestamp': '...', 'weight': 75.5, 'quality': 0.85}
    ],
    'reset_events': [
        {'type': 'HARD', 'timestamp': '...', 'gap_days': 35}
    ],
    'metadata': {
        'measurements_since_reset': 45,
        'last_source': 'patient-device',
        'adaptation_state': 'completed'
    }
}
```

### Lambda Configuration Mapping
```python
# From config.toml to Parameter Store
/weight-processor/prod/kalman/initial_variance -> 0.361
/weight-processor/prod/kalman/observation_covariance -> 3.49
/weight-processor/prod/processing/extreme_threshold -> 0.15
/weight-processor/prod/quality/threshold -> 0.6
/weight-processor/prod/adaptive_noise/care-team-upload -> 0.5
```

### Replay Buffer Async Flow
```python
# SQS Message Format
{
    'user_id': 'uuid',
    'buffer_start': '2024-02-20T00:00:00Z',
    'buffer_end': '2024-02-25T00:00:00Z',
    'measurements': [...],
    'trigger': 'outlier_detected'
}
```

## Implementation Order

1. **Week 1**: Extract core processing, create DynamoDB adapter
2. **Week 2**: Implement state migration, test precision
3. **Week 3**: Build API handlers, error mapping
4. **Week 4**: Add async cleanup, replay processing
5. **Week 5**: Setup monitoring, testing suite
6. **Week 6**: Production deployment, validation

## Success Metrics

- Migration completes with 100% state preservation
- P95 latency meets <500ms target
- Cost projection validated at <$100/month
- Zero statistical regression in Kalman accuracy
- Spring Boot integration requires <100 lines of code