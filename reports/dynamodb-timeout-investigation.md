# DynamoDB Timeout and Validation Issues Investigation

## Date: 2025-09-26

## Summary
Investigation of test failures revealing DynamoDB connection timeouts and Kalman parameter validation errors during replay operations.

## Key Findings

### 1. DynamoDB Connection Timeouts
**Location**: `weight_values/src/core/database/dynamodb_store.py`

**Issue**: DynamoDB operations timing out after 9 retries (~26 seconds total)
- Error occurs during `GetItem` operations in replay scenarios
- No custom retry or timeout configuration in boto3 client initialization (lines 58-68)
- Using default boto3 retry settings which may be insufficient for local DynamoDB

**Root Cause**:
- Default boto3 retry configuration may be too aggressive for local DynamoDB
- Possible connection pool exhaustion during intensive test operations
- No connection pooling or session reuse implemented

### 2. Kalman Parameter Validation Failures
**Location**: `weight_values/src/core/processing/persistence_validator.py:157-190`

**Issue**: "Invalid Kalman parameters" errors during state persistence
- Validator expects `transition_covariance` as 2x2 matrix and `observation_covariance` as scalar/1x1 matrix
- Parameters are transformed during processing but validation may occur before transformation
- Mismatch between storage format and validation expectations

**Root Cause**:
- Inconsistent parameter structure between initialization and validation
- Parameters stored with individual fields (`transition_covariance_weight`, `transition_covariance_trend`) but validated as matrix

### 3. Replay Service Performance
**Location**: `weight_values/src/aws/services/replay_service.py`

**Issue**: Sequential processing without connection management
- Multiple state operations during replay without connection pooling
- No batch operations or transaction support
- Each measurement triggers individual DynamoDB operations

## Recommendations

### Immediate Fixes

1. **Add Retry Configuration to DynamoDB Client**
```python
from botocore.config import Config

# In dynamodb_store.py __init__ method
boto_config = Config(
    region_name=self.region,
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    },
    max_pool_connections=50,
    connect_timeout=5,
    read_timeout=10
)

self.dynamodb = boto3.resource(
    "dynamodb",
    config=boto_config,
    endpoint_url=endpoint_url,
    # ... other params
)
```

2. **Fix Kalman Parameter Validation**
- Ensure parameters are in correct format before validation
- Update validator to handle both formats (individual fields and matrix)
- Add transformation step before persistence

3. **Implement Connection Pooling**
- Reuse boto3 sessions across operations
- Implement connection pool management
- Add circuit breaker for failed connections

### Long-term Improvements

1. **Batch Operations**
- Implement batch write/read operations for replay scenarios
- Use DynamoDB transactions for atomic operations
- Reduce number of individual API calls

2. **Monitoring and Alerting**
- Add connection pool metrics
- Monitor retry counts and timeout rates
- Log detailed error context for debugging

3. **Test Environment Optimization**
- Consider using DynamoDB Local in-memory mode for tests
- Implement test data cleanup between test runs
- Add connection health checks before test execution

## Affected Files
- `weight_values/src/core/database/dynamodb_store.py` - Connection configuration
- `weight_values/src/core/processing/persistence_validator.py` - Validation logic
- `weight_values/src/aws/services/replay_service.py` - Replay processing
- `weight_values/src/core/processing/processor.py` - Kalman parameter handling

## Test Impact
- `test_replay_from_specific_timestamp` - Primary failure point
- Multiple tests showing validation warnings in logs
- Potential intermittent failures in other DynamoDB-dependent tests

## Next Steps
1. Implement retry configuration changes
2. Fix Kalman parameter validation logic
3. Add connection pooling
4. Re-run tests to verify fixes
5. Monitor for additional timeout issues