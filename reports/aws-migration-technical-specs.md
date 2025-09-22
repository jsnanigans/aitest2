# AWS Weight Processing Service - Technical Specifications

## Executive Summary

This document outlines the migration of the weight processing system from a batch CSV processor to an AWS-hosted microservice that interfaces with a Java backend. The service will maintain Kalman filter states for users, provide quality scoring, and support historical data replay capabilities.

## 1. Architecture Overview

### 1.1 Current System
- **Technology**: Python-based batch processor
- **Processing**: CSV file input with Kalman filtering
- **State Management**: In-memory database with SQLite persistence
- **Key Features**:
  - Adaptive Kalman filtering with reset detection
  - Quality scoring based on multiple factors
  - Outlier detection with quality override
  - Replay processing for historical corrections

### 1.2 Target Architecture
- **Service Type**: RESTful API microservice
- **Hosting**: AWS (Lambda + API Gateway or ECS/Fargate)
- **State Storage**: DynamoDB or RDS PostgreSQL
- **Language**: Python 3.11+ (maintaining core processing logic)
- **API Protocol**: REST with JSON payloads
- **Authentication**: API Key or JWT tokens from Java backend

## 2. API Specifications

### 2.1 Endpoints Overview

```
POST /api/v1/cleanup/{userId}
POST /api/v1/process/{userId}
POST /api/v1/replay/{userId}
GET  /api/v1/state/{userId}
DELETE /api/v1/state/{userId}
```

### 2.2 Detailed API Contracts

#### 2.2.1 One-Time Cleanup Endpoint
**Purpose**: Process all historical data for a user and return acceptance status for each measurement.

```http
POST /api/v1/cleanup/{userId}
```

**Request Body**:
```json
{
  "measurements": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "weight": 75.5,
      "unit": "kg",
      "effectiveDateTime": "2024-01-15T10:30:00Z",
      "source": "patient-device",
      "metadata": {
        "deviceId": "scale-001",
        "location": "home"
      }
    }
  ],
  "userProfile": {
    "height": 175,
    "heightUnit": "cm",
    "dateOfBirth": "1990-01-01",
    "gender": "M"
  },
  "options": {
    "resetState": true,
    "includeQualityScores": true,
    "includeDebugInfo": false
  }
}
```

**Response**:
```json
{
  "userId": "user-123",
  "processedCount": 50,
  "acceptedCount": 45,
  "rejectedCount": 5,
  "measurements": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "accepted": true,
      "qualityScore": 0.85,
      "kalmanEstimate": 75.3,
      "kalmanUncertainty": 0.2,
      "components": {
        "kalmanDeviation": 0.02,
        "temporalConsistency": 0.95,
        "sourceReliability": 0.8
      },
      "resetTriggered": false
    },
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440001",
      "accepted": false,
      "rejectionReason": "Extreme value - outside physiological bounds",
      "stage": "preprocessing",
      "details": "Weight 250kg exceeds maximum threshold"
    }
  ],
  "finalState": {
    "currentWeight": 75.3,
    "uncertainty": 0.2,
    "lastProcessedTimestamp": "2024-01-15T10:30:00Z",
    "totalMeasurements": 50,
    "adaptationState": "converged"
  }
}
```

#### 2.2.2 Process Measurements Endpoint
**Purpose**: Process multiple new measurements for a user. All measurements must be after the last processed timestamp, otherwise the entire batch is rejected with a historical conflict error.

```http
POST /api/v1/process/{userId}
```

**Request Body**:
```json
{
  "measurements": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440002",
      "weight": 76.0,
      "unit": "kg",
      "effectiveDateTime": "2024-01-16T08:00:00Z",
      "source": "patient-upload"
    },
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440003",
      "weight": 75.8,
      "unit": "kg",
      "effectiveDateTime": "2024-01-16T14:00:00Z",
      "source": "patient-device"
    }
  ],
  "options": {
    "failOnHistoricalConflict": true
  }
}
```

**Response - Success**:
```json
{
  "status": "processed",
  "processedCount": 2,
  "acceptedCount": 2,
  "rejectedCount": 0,
  "measurements": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440002",
      "accepted": true,
      "qualityScore": 0.78,
      "kalmanEstimate": 75.8,
      "kalmanUncertainty": 0.15,
      "resetTriggered": false
    },
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440003",
      "accepted": true,
      "qualityScore": 0.82,
      "kalmanEstimate": 75.7,
      "kalmanUncertainty": 0.12,
      "resetTriggered": false
    }
  ],
  "stateUpdate": {
    "previousWeight": 75.3,
    "currentWeight": 75.7,
    "lastProcessedTimestamp": "2024-01-16T14:00:00Z"
  }
}
```

**Response - Historical Conflict**:
```json
{
  "status": "historical_conflict",
  "error": "One or more measurements are before last processed timestamp",
  "details": {
    "earliestMeasurementTimestamp": "2024-01-14T08:00:00Z",
    "lastProcessedTimestamp": "2024-01-16T08:00:00Z",
    "replayRequired": true,
    "replayFromTimestamp": "2024-01-14T00:00:00Z",
    "snapshotAvailable": "2024-01-13T23:59:59Z",
    "conflictingMeasurements": ["550e8400-e29b-41d4-a716-446655440001"]
  }
}
```

#### 2.2.3 Replay Endpoint
**Purpose**: Replay historical measurements from a snapshot point.

```http
POST /api/v1/replay/{userId}
```

**Request Body**:
```json
{
  "replayFromTimestamp": "2024-01-14T00:00:00Z",
  "measurements": [
    {
      "uuid": "uuid-1",
      "weight": 75.2,
      "unit": "kg",
      "effectiveDateTime": "2024-01-14T08:00:00Z",
      "source": "patient-device"
    },
    {
      "uuid": "uuid-2",
      "weight": 75.5,
      "unit": "kg",
      "effectiveDateTime": "2024-01-15T10:30:00Z",
      "source": "patient-upload"
    }
  ],
  "options": {
    "useSnapshot": true,
    "createNewSnapshot": true
  }
}
```

**Response**:
```json
{
  "status": "replay_completed",
  "snapshotUsed": "2024-01-13T23:59:59Z",
  "measurements": [
    {
      "uuid": "uuid-1",
      "accepted": true,
      "qualityScore": 0.82
    },
    {
      "uuid": "uuid-2",
      "accepted": true,
      "qualityScore": 0.79
    }
  ],
  "stateChanges": {
    "beforeReplay": {
      "weight": 75.8,
      "timestamp": "2024-01-16T08:00:00Z"
    },
    "afterReplay": {
      "weight": 75.5,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
}
```

## 3. State Management

### 3.1 State Storage Schema

#### DynamoDB Schema
```python
{
  "TableName": "WeightProcessorState",
  "PartitionKey": "userId",
  "SortKey": "stateType",  # "current" or "snapshot_{timestamp}"
  "Attributes": {
    "userId": "string",
    "stateType": "string",
    "kalmanState": {
      "x": [[float]],  # State vector
      "P": [[float]],  # Covariance matrix
      "lastTimestamp": "ISO8601",
      "adaptationLevel": "float",
      "resetCount": "number",
      "measurementCount": "number"
    },
    "metadata": {
      "lastSource": "string",
      "lastRawWeight": "float",
      "qualityHistory": [float],
      "resetEvents": [{
        "timestamp": "ISO8601",
        "type": "string",
        "reason": "string"
      }]
    },
    "ttl": "number",  # For automatic snapshot cleanup
    "createdAt": "ISO8601",
    "updatedAt": "ISO8601"
  }
}
```

#### PostgreSQL Schema
```sql
CREATE TABLE kalman_states (
    user_id VARCHAR(255) NOT NULL,
    state_type VARCHAR(50) NOT NULL,
    kalman_x JSONB NOT NULL,
    kalman_p JSONB NOT NULL,
    last_timestamp TIMESTAMP WITH TIME ZONE,
    adaptation_level FLOAT,
    reset_count INTEGER DEFAULT 0,
    measurement_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ttl TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (user_id, state_type)
);

CREATE INDEX idx_user_timestamp ON kalman_states(user_id, last_timestamp);
CREATE INDEX idx_ttl ON kalman_states(ttl) WHERE ttl IS NOT NULL;
```

### 3.2 State Management Strategy

#### Snapshot Strategy
- Create snapshots every 24 hours of processing
- Keep last 7 snapshots per user
- Snapshots enable replay from any point
- Automatic cleanup of expired snapshots

#### Consistency Guarantees
- Use optimistic locking with version numbers
- Atomic state updates using transactions
- Idempotency keys for request deduplication

## 4. Infrastructure Requirements

### 4.1 Compute Options

#### Option A: AWS Lambda + API Gateway
**Pros**:
- Serverless, automatic scaling
- Pay-per-request pricing
- No infrastructure management
- Built-in API Gateway features

**Cons**:
- 15-minute timeout limit
- Cold start latency (300-500ms)
- Limited local storage (512MB /tmp)

**Configuration**:
```yaml
Runtime: Python 3.11
Memory: 1024 MB
Timeout: 60 seconds
Reserved Concurrency: 100
Environment Variables:
  - STATE_DB_TABLE
  - CONFIG_BUCKET
  - LOG_LEVEL
```

#### Option B: ECS Fargate
**Pros**:
- Long-running processes supported
- Better for batch operations
- Consistent performance
- More control over resources

**Cons**:
- Higher base cost
- Requires container management
- More complex deployment

**Configuration**:
```yaml
Task Definition:
  CPU: 1 vCPU
  Memory: 2 GB
  Container:
    Image: weight-processor:latest
    Port: 8080
    HealthCheck: /health
Auto Scaling:
  Min Tasks: 2
  Max Tasks: 20
  Target CPU: 70%
```

### 4.2 Database Selection

#### Recommended: DynamoDB
- Serverless, automatic scaling
- Single-digit millisecond latency
- Built-in backup and recovery
- Cost-effective for key-value access

#### Alternative: RDS PostgreSQL
- Better for complex queries
- ACID compliance
- Familiar SQL interface
- Point-in-time recovery

### 4.3 Supporting Services

```yaml
API Gateway:
  - Rate limiting: 1000 req/second
  - Throttling: 2000 burst
  - API keys for authentication

CloudWatch:
  - Custom metrics for processing
  - Alarms for failures
  - Log aggregation

S3:
  - Configuration storage
  - Batch result storage
  - Backup storage

SQS (Optional):
  - Async processing queue
  - Dead letter queue for failures

Secrets Manager:
  - API keys
  - Database credentials
```

## 5. Security Considerations

### 5.1 Authentication & Authorization
```python
# API Key validation
def validate_api_key(api_key: str) -> bool:
    # Verify against Secrets Manager
    # Check rate limits
    # Log access attempts
    pass

# JWT token validation (alternative)
def validate_jwt(token: str) -> dict:
    # Verify signature
    # Check expiration
    # Extract user claims
    pass
```

### 5.2 Data Protection
- Encryption at rest (DynamoDB/RDS)
- TLS 1.2+ for transit
- PII data masking in logs
- HIPAA compliance considerations

### 5.3 Access Controls
```yaml
IAM Policies:
  Lambda Execution Role:
    - DynamoDB: Read/Write on state table
    - CloudWatch: Write logs
    - Secrets Manager: Read secrets
    - S3: Read config bucket

  Java Backend Role:
    - API Gateway: Invoke APIs
    - CloudWatch: Read metrics
```

## 6. Monitoring & Observability

### 6.1 Key Metrics
```python
# Custom CloudWatch metrics
metrics = {
    "MeasurementsProcessed": "Count",
    "MeasurementsAccepted": "Count",
    "MeasurementsRejected": "Count",
    "ProcessingLatency": "Milliseconds",
    "QualityScoreDistribution": "Histogram",
    "ResetEvents": "Count",
    "ReplayOperations": "Count",
    "StateConflicts": "Count"
}
```

### 6.2 Logging Strategy
```python
import json
import logging
from aws_lambda_powertools import Logger

logger = Logger(service="weight-processor")

@logger.inject_lambda_context
def handler(event, context):
    logger.info("Processing measurement", extra={
        "userId": event["userId"],
        "source": event["source"],
        "correlationId": event.get("correlationId")
    })
```

### 6.3 Alarms
```yaml
CloudWatch Alarms:
  - HighRejectionRate:
      Threshold: > 30% rejected
      Period: 5 minutes
  - ProcessingErrors:
      Threshold: > 10 errors
      Period: 1 minute
  - HighLatency:
      Threshold: > 2000ms p99
      Period: 5 minutes
```

## 7. Deployment Strategy

### 7.1 CI/CD Pipeline
```yaml
GitHub Actions:
  - Test: pytest with coverage
  - Build: Docker image
  - Deploy:
    - Dev: Automatic
    - Staging: Manual approval
    - Production: Blue-green deployment

Stages:
  1. Code commit
  2. Run tests
  3. Build container
  4. Push to ECR
  5. Update Lambda/ECS
  6. Run integration tests
  7. Monitor metrics
  8. Rollback if needed
```

### 7.2 Environment Configuration
```python
# config/environments.py
ENVIRONMENTS = {
    "dev": {
        "state_table": "weight-state-dev",
        "log_level": "DEBUG",
        "snapshot_retention_days": 3
    },
    "staging": {
        "state_table": "weight-state-staging",
        "log_level": "INFO",
        "snapshot_retention_days": 7
    },
    "production": {
        "state_table": "weight-state-prod",
        "log_level": "WARNING",
        "snapshot_retention_days": 30
    }
}
```

## 8. Cost Estimation

### 8.1 Lambda Option (10,000 users, 100 req/day each)
```
API Gateway: $3.50/million requests = $105/month
Lambda: 1GB × 1s × 1M requests = $20/month
DynamoDB:
  - Storage: 10GB = $2.50/month
  - Reads: 30M/month = $7.50/month
  - Writes: 1M/month = $1.25/month
CloudWatch: $10/month
Total: ~$150/month
```

### 8.2 ECS Option
```
Fargate: 2 tasks × 1vCPU × 2GB = $73/month
ALB: $25/month
RDS: db.t3.small = $35/month
CloudWatch: $10/month
Total: ~$145/month
```

## 9. Migration Plan

### Phase 1: Core Service (Week 1-2)
- Port processing logic to Lambda/ECS
- Implement state management
- Create basic API endpoints

### Phase 2: Integration (Week 3-4)
- Java backend integration
- Authentication setup
- Error handling

### Phase 3: Features (Week 5-6)
- Replay functionality
- Snapshot management
- Monitoring setup

### Phase 4: Testing (Week 7-8)
- Load testing
- Integration testing
- Performance optimization

### Phase 5: Deployment (Week 9-10)
- Staging deployment
- Production rollout
- Documentation

## 10. Risk Mitigation

### Technical Risks
1. **State Consistency**: Use transactions and versioning
2. **Performance**: Cache frequently accessed states
3. **Scale**: Auto-scaling and rate limiting
4. **Data Loss**: Regular backups and snapshots

### Operational Risks
1. **Monitoring Gaps**: Comprehensive metrics and alarms
2. **Deployment Failures**: Blue-green deployments
3. **Cost Overruns**: Budget alerts and limits

## 11. Success Criteria

- API response time < 200ms p50, < 500ms p99
- Availability > 99.9%
- Processing accuracy matches current system
- Zero data loss during migration
- Cost within 20% of estimate

## Next Steps

1. Review and approve architecture
2. Set up AWS accounts and permissions
3. Create development environment
4. Begin Phase 1 implementation