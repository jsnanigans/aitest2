# AWS MVP Infrastructure Review

## Executive Summary
The codebase is fundamentally sound for an MVP but contains significant unnecessary complexity. By removing visualization/analysis features and simplifying the API surface, we can reduce Lambda package size by ~60% and improve maintainability.

## Current State

### ✅ What's Working Well
- **Lambda Handler**: Clean routing, proper error handling
- **DynamoDB Integration**: State persistence with snapshots
- **API Gateway**: Rate limiting, API key auth, CORS configured
- **Monitoring**: CloudWatch alarms for errors and throttling
- **Core Processing**: Kalman filtering and quality scoring working

### ❌ Unnecessary for MVP
- **67KB** of visualization code (`src/viz/`)
- **344KB** of analysis/reporting code (`src/analysis/`)
- **Multiple redundant endpoints** (cleanup vs process with reset)
- **Unimplemented replay** (returns 501)
- **State management endpoints** (GET/DELETE state)

## Recommended Simplifications

### 1. Remove Non-Essential Code
```bash
# Delete these directories entirely for Lambda deployment:
rm -rf src/viz/
rm -rf src/analysis/
rm -rf presentation/
rm -rf scripts/
rm -rf integration-tests/
rm -rf reports/visualizations/
```

### 2. Consolidate API Endpoints

#### Current (5 endpoints):
- POST `/process/{userId}` - Process measurements
- POST `/cleanup/{userId}` - Process with state reset
- POST `/replay/{userId}` - Not implemented (501)
- GET `/state/{userId}` - Debug endpoint
- DELETE `/state/{userId}` - Debug endpoint

#### Recommended (2 endpoints):
```python
# 1. Process Readings (handles both incremental and full reset)
POST /api/v1/process/{userId}
{
  "measurements": [...],
  "options": {
    "reset_state": false  # true = cleanup behavior
  }
}

# 2. Run Replay (simple implementation added)
POST /api/v1/replay/{userId}
{
  "measurements": [...],
  "replay_from_timestamp": "2024-01-01T00:00:00Z"
}
```

### 3. Simplified Response Objects

#### Current (complex):
```json
{
  "status": "success",
  "processed_count": 10,
  "accepted_count": 8,
  "rejected_count": 2,
  "measurements": [
    {
      "uuid": "...",
      "accepted": true,
      "quality_score": 0.85,
      "kalman_estimate": 75.2,
      "kalman_uncertainty": 0.5,
      "rejection_reason": null,
      "stage": "accepted",
      "reset_triggered": false,
      "components": {...}
    }
  ],
  "state_update": {...}
}
```

#### Recommended (MVP):
```json
{
  "status": "success",
  "processed_count": 10,
  "accepted_count": 8,
  "rejected_count": 2,
  "measurements": [
    {
      "uuid": "...",
      "accepted": true,
      "quality_score": 0.85,
      "kalman_estimate": 75.2
    }
  ]
}
```

### 4. Configuration Simplification

#### Remove from Lambda environment variables:
- Visualization-related configs
- Analysis/reporting configs
- Feature toggles (always enabled for MVP)

#### Keep only essential configs:
```yaml
KALMAN_PROCESS_NOISE: '1.0'
KALMAN_OBS_NOISE: '4.0'
QS_OUTLIER_OVERRIDE: '0.8'
OUTLIER_IQR_MULTIPLIER: '1.5'
```

### 5. Logging Improvements

Add structured logging for better CloudWatch Insights:
```python
logger.info({
    "event": "measurement_processed",
    "user_id": user_id,
    "measurement_id": str(uuid),
    "accepted": accepted,
    "quality_score": quality_score,
    "source": source
})
```

### 6. Database Optimizations

#### Current DynamoDB schema:
- Partition key: userId
- Sort key: stateType (current/snapshot_*)
- TTL on snapshots (7 days)

#### Recommended additions:
- GSI for querying by timestamp range
- Reduced snapshot retention (3 days for MVP)
- Batch write for performance

## Implementation Checklist

### Immediate Actions
- [x] Implement simple replay service
- [x] Add replay route to SAM template
- [ ] Remove visualization/analysis directories
- [ ] Simplify API response models
- [ ] Update requirements-lambda.txt

### Testing Required
- [ ] Test process endpoint with various data
- [ ] Test replay endpoint with snapshots
- [ ] Verify DynamoDB state persistence
- [ ] Load test with expected volumes

### Deployment Steps
1. Clean up codebase (remove unnecessary files)
2. Update Lambda package
3. Deploy with SAM CLI
4. Test endpoints with sample data
5. Monitor CloudWatch metrics

## Performance Expectations

### Current Package Size
- Full codebase: ~2.5MB
- With visualization: ~1.8MB
- MVP version: ~800KB

### Lambda Performance
- Cold start: ~2-3 seconds
- Warm execution: <500ms for 100 measurements
- Memory usage: ~256MB typical, 512MB peak

### DynamoDB Capacity
- On-demand pricing (no provisioning needed)
- ~1 WCU per measurement processed
- ~1 RCU per state read
- Snapshot writes: 2-3 WCU

## Cost Estimates (Monthly)

### Low Volume (1000 users, 10 measurements/day)
- Lambda: $5-10
- DynamoDB: $10-15
- API Gateway: $3-5
- **Total: ~$20-30/month**

### Medium Volume (10K users, 50 measurements/day)
- Lambda: $50-100
- DynamoDB: $100-150
- API Gateway: $30-50
- **Total: ~$200-300/month**

## Security Considerations

### Current Implementation
- ✅ API key authentication
- ✅ Rate limiting (50 req/s, 10K/day)
- ✅ Input validation with Pydantic
- ⚠️ No user-level access control
- ⚠️ No encryption at rest for DynamoDB

### Recommended for Production
- Add Cognito or custom JWT authentication
- Enable DynamoDB encryption
- Add WAF rules for API Gateway
- Implement user-level access policies

## Next Steps

1. **Clean up codebase** - Remove unnecessary features
2. **Test replay implementation** - Verify snapshot restore works
3. **Update documentation** - API docs for the two endpoints
4. **Deploy to staging** - Test with real data
5. **Monitor and optimize** - Use CloudWatch Insights

## Questions to Resolve

1. **Snapshot retention period?** Currently 7 days, recommend 3 for MVP
2. **Batch size limits?** Currently unlimited, recommend 1000 measurements/request
3. **Historical data migration?** Need strategy for existing users
4. **Rate limiting per user?** Currently global, consider per-user limits