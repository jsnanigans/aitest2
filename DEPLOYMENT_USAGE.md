# Weight Processor AWS Lambda - Deployment Usage Guide

## Overview

The Weight Processor is deployed as an AWS Lambda function that processes weight measurements using Kalman filtering and adaptive quality scoring. The function is invoked directly (no API Gateway) via AWS Lambda invoke API.

## Deployed Infrastructure

### Stack Information (dev-us environment)

```
Environment: dev
Region: us-east-1
Stack Status: UPDATE_COMPLETE
Last Updated: 2025-10-02T08:15:28.470000+00:00
```

### Key Resources

| Resource | Identifier |
|----------|-----------|
| **Lambda Function** | `weight-processor-dev-us` |
| **Function ARN** | `arn:aws:lambda:us-east-1:387257169268:function:weight-processor-dev-us` |
| **DynamoDB Table** | `weight-processor-state-dev-us` |
| **Invocation Role** | `arn:aws:iam::387257169268:role/weight-processor-invoker-dev-us` |
| **VPC** | `vpc-04624d1b017777654` |
| **KMS Key** | `arn:aws:kms:us-east-1:387257169268:key/39b3b276-ceb2-4c76-9ddd-2b8e796ebdd5` |

### Network Configuration

- **Private Subnets**:
  - `subnet-0ebe170f594c472fc` (Subnet 1)
  - `subnet-015948222172e4967` (Subnet 2)
- **Security Group**: `sg-0e00b0ac33aa37477`
- **VPC Endpoints**: DynamoDB and CloudWatch Logs

## Lambda Function Operations

The Lambda function supports the following operations via direct invocation:

### 1. Process Weight Measurements

**Event Payload:**
```json
{
  "operation": "process",
  "user_id": "user-123",
  "measurements": [
    {
      "uuid": "measurement-id-1",
      "weight": 185.5,
      "unit": "lb",
      "effectiveDateTime": "2025-10-02T10:30:00Z",
      "source": "smart_scale",
      "metadata": {}
    }
  ]
}
```

**Invocation:**
```bash
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{
    "operation": "process",
    "user_id": "user-123",
    "measurements": [{
      "uuid": "measurement-id-1",
      "weight": 185.5,
      "unit": "lb",
      "effectiveDateTime": "2025-10-02T10:30:00Z",
      "source": "smart_scale",
      "metadata": {}
    }]
  }' \
  response.json
```

**Response:**
```json
{
  "user_id": "user-123",
  "measurements_accepted": 1,
  "measurements_rejected": 0,
  "results": [
    {
      "uuid": "measurement-id-1",
      "accepted": true,
      "measured_at": "2025-10-02T10:30:00Z",
      "raw_weight": 185.5,
      "unit": "lb",
      "quality_score": 0.95,
      "kalman_estimate": 185.3,
      "kalman_variance": 0.25,
      "kalman_confidence_upper": 186.3,
      "kalman_confidence_lower": 184.3,
      "trend": -0.15,
      "trend_weekly": -1.2,
      "confidence": 0.95,
      "source": "smart_scale"
    }
  ],
  "state_updated": true,
  "timestamp": "2025-10-02T10:30:05.123Z"
}
```

### 2. Get Processing State

**Event Payload:**
```json
{
  "operation": "get_state",
  "user_id": "user-123",
  "device_id": "device-456"
}
```

**Invocation:**
```bash
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{
    "operation": "get_state",
    "user_id": "user-123",
    "device_id": "device-456"
  }' \
  response.json
```

**Response:**
```json
{
  "user_id": "user-123",
  "device_id": "device-456",
  "kalman_state": {
    "state": [185.3, -0.02],
    "covariance": [[0.25, 0.01], [0.01, 0.05]],
    "last_update": "2025-10-02T10:30:00Z"
  },
  "measurement_count": 45,
  "last_processed": "2025-10-02T10:30:00Z"
}
```

### 3. Reset Kalman Filter State

**Event Payload:**
```json
{
  "operation": "reset",
  "user_id": "user-123",
  "device_id": "device-456",
  "reason": "significant_weight_change"
}
```

**Invocation:**
```bash
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{
    "operation": "reset",
    "user_id": "user-123",
    "device_id": "device-456",
    "reason": "significant_weight_change"
  }' \
  response.json
```

**Response:**
```json
{
  "user_id": "user-123",
  "device_id": "device-456",
  "reset_successful": true,
  "timestamp": "2025-10-02T10:35:00.000Z"
}
```

### 4. Get Measurement History

**Event Payload:**
```json
{
  "operation": "get_history",
  "user_id": "user-123",
  "limit": 50,
  "start_date": "2025-09-01T00:00:00Z",
  "end_date": "2025-10-02T23:59:59Z"
}
```

**Parameters:**
- `user_id` (required): User identifier
- `limit` (optional): Maximum number of measurements to return (default: 100)
- `start_date` (optional): ISO format timestamp for range start
- `end_date` (optional): ISO format timestamp for range end

**Invocation:**
```bash
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{
    "operation": "get_history",
    "user_id": "user-123",
    "limit": 50
  }' \
  response.json
```

**Response:**
```json
{
  "user_id": "user-123",
  "measurements": [
    {
      "timestamp": "2025-10-02T10:30:00Z",
      "raw_weight": 185.5,
      "filtered_weight": 185.3,
      "accepted": true,
      "quality_score": 0.95,
      "source": "smart_scale"
    }
  ],
  "total_count": 45
}
```

### 5. Health Check

**Event Payload:**
```json
{
  "operation": "health"
}
```

**Invocation:**
```bash
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{"operation": "health"}' \
  response.json
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-02T12:00:00.000Z"
}
```

## Usage Patterns

### Single Measurement Processing

Process measurements one at a time for real-time filtering with automatic replay support:

```bash
# Process single measurement
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload file://measurement.json \
  response.json

# View results
cat response.json | jq '.'
```

### Batch Processing with Replay

Process multiple measurements sequentially. The service automatically:
1. Processes each measurement through Kalman filter
2. Checks if replay should trigger after each measurement
3. Executes replay to correct historical data if needed

**Important**: Process measurements one at a time (separate invocations) to allow replay triggers between measurements.

```bash
# Process measurements in chronological order
for file in measurements/*.json; do
  aws lambda invoke \
    --function-name weight-processor-dev-us \
    --region us-east-1 \
    --payload "file://$file" \
    "response_$(basename $file)"

  # Small delay to ensure sequential processing
  sleep 0.1
done
```

### State Management

Check processing state or reset when significant changes occur:

```bash
# Get current state
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{
    "operation": "get_state",
    "user_id": "user-123",
    "device_id": "device-456"
  }' \
  state.json

# Reset state (after pregnancy, medical procedure, etc.)
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --region us-east-1 \
  --payload '{
    "operation": "reset",
    "user_id": "user-123",
    "device_id": "device-456",
    "reason": "medical_event"
  }' \
  reset_response.json
```

## Configuration

### Supported Weight Units

The following units are accepted in the `unit` field:
- `kg`, `kilogram`, `kilograms`
- `lb`, `lbs`, `pound`, `pounds`
- `g`, `gram`, `grams`

### Quality Scoring Components

The service evaluates measurements using multiple criteria:
- **Plausibility**: Physiological limits (40-300 kg)
- **Temporal Consistency**: Change rate validation
- **Statistical Validation**: Outlier detection using Kalman innovation
- **Source Reliability**: Weighted by data source type

### Kalman Filter Parameters

Default configuration:
```json
{
  "kalman": {
    "process_noise": 0.01,
    "measurement_noise_base": 0.5,
    "velocity_process_noise": 0.001,
    "adaptive_noise": true
  },
  "quality": {
    "min_quality_threshold": 0.5,
    "max_innovation_threshold": 3.0
  },
  "replay": {
    "window_size": 30,
    "min_measurements_for_replay": 10,
    "check_interval_days": 7
  }
}
```

## Replay System

The weight processor includes an intelligent replay mechanism that automatically reprocesses historical data when outliers are detected.

### How Replay Works

1. **Continuous Monitoring**: After each measurement, the service checks if replay should trigger
2. **Outlier Detection**: Identifies statistical outliers using Kalman innovation
3. **Historical Reprocessing**: Re-filters measurements in the replay window (last 30 measurements)
4. **State Correction**: Updates acceptance decisions and filtered values

### Replay Trigger Conditions

Replay triggers when:
- Sufficient measurements exist in the buffer (≥10)
- Time since last replay exceeds threshold (≥7 days)
- Recent measurements show statistical anomalies

### Replay Response

When replay is triggered, the response includes additional fields:

```json
{
  "measurements_accepted": 1,
  "replay_triggered": true,
  "corrections_made": 3,
  "results": [...]
}
```

## Error Handling

### Common Error Responses

**Invalid Unit:**
```json
{
  "error": "Invalid unit",
  "message": "Unit 'm2' is not supported. Use: kg, lb, g, etc.",
  "code": "INVALID_UNIT"
}
```

**Missing Required Fields:**
```json
{
  "error": "Validation error",
  "message": "Missing required field: user_id",
  "code": "VALIDATION_ERROR"
}
```

**Invalid Weight Value:**
```json
{
  "error": "Invalid weight",
  "message": "Weight must be positive and within physiological limits",
  "code": "INVALID_WEIGHT"
}
```

## Monitoring & Observability

### CloudWatch Metrics

The Lambda function publishes the following metrics:

- **ProcessorErrors**: Error count (Alarm: `weight-processor-errors-dev-us`)
- **ProcessorThrottles**: Throttle count (Alarm: `weight-processor-throttles-dev-us`)
- **DatabaseThrottles**: DynamoDB throttle count (Alarm: `weight-processor-db-throttles-dev-us`)

### Log Groups

Logs are available in CloudWatch:
```
/aws/lambda/weight-processor-dev-us
```

### View Recent Logs

```bash
# Tail logs in real-time
aws logs tail /aws/lambda/weight-processor-dev-us \
  --follow \
  --region us-east-1

# View logs for specific time range
aws logs tail /aws/lambda/weight-processor-dev-us \
  --since 1h \
  --region us-east-1
```

### CloudWatch Insights Query

```bash
# Query accepted/rejected measurements
aws logs start-query \
  --log-group-name /aws/lambda/weight-processor-dev-us \
  --start-time $(date -u -d '1 hour ago' +%s) \
  --end-time $(date -u +%s) \
  --query-string 'fields @timestamp, @message | filter @message like /accepted|rejected/ | stats count() by accepted'
```

## Security

### Authentication & Authorization

- Use the **Invocation Role** to grant services permission to invoke the Lambda
- IAM Role ARN: `arn:aws:iam::387257169268:role/weight-processor-invoker-dev-us`

**Example IAM Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:us-east-1:387257169268:function:weight-processor-dev-us"
    }
  ]
}
```

### Data Encryption

- **At Rest**: DynamoDB table encrypted with KMS
  - Key ARN: `arn:aws:kms:us-east-1:387257169268:key/39b3b276-ceb2-4c76-9ddd-2b8e796ebdd5`
- **In Transit**: TLS for Lambda invocations and DynamoDB access

### Network Security

- Lambda runs in private VPC subnets (no direct internet access)
- VPC endpoints for AWS service communication (DynamoDB, CloudWatch)
- Security Group: `sg-0e00b0ac33aa37477`

## Best Practices

### 1. Sequential Processing
Process measurements in chronological order for optimal Kalman filtering. Sort by `effectiveDateTime` before invoking.

### 2. One-at-a-Time Invocation
For batch processing, invoke the Lambda once per measurement (separate invocations) to allow replay triggers between measurements.

**✓ Good: Allows replay between measurements**
```bash
for measurement in measurement_*.json; do
  aws lambda invoke --function-name weight-processor-dev-us \
    --payload "file://$measurement" "response_$measurement"
done
```

**✗ Avoid: Bypasses replay mechanism**
```bash
# Single invocation with all measurements prevents replay triggers
aws lambda invoke --payload '{"measurements": [...]}'
```

### 3. Unit Consistency
Always provide explicit units (no defaults):

**✓ Good:**
```json
{"weight": 185.5, "unit": "lb"}
```

**✗ Bad: Missing unit will be rejected**
```json
{"weight": 185.5}
```

### 4. State Reset
Reset state when users report significant weight changes (pregnancy, medical procedure, etc.)

### 5. Error Handling
Always check the response for errors before processing results.

## Support & Troubleshooting

### Check Function Status
```bash
aws lambda get-function \
  --function-name weight-processor-dev-us \
  --region us-east-1
```

### Test Function Invocation
```bash
# Basic health check
aws lambda invoke \
  --function-name weight-processor-dev-us \
  --payload '{"operation": "health"}' \
  --region us-east-1 \
  response.json

cat response.json
```

### Check Function Logs
```bash
# Get latest log stream
aws logs describe-log-streams \
  --log-group-name /aws/lambda/weight-processor-dev-us \
  --order-by LastEventTime \
  --descending \
  --max-items 1 \
  --region us-east-1

# View specific log stream
aws logs get-log-events \
  --log-group-name /aws/lambda/weight-processor-dev-us \
  --log-stream-name '<log-stream-name>' \
  --region us-east-1
```

### Check DynamoDB Table
```bash
# Describe table
aws dynamodb describe-table \
  --table-name weight-processor-state-dev-us \
  --region us-east-1

# Query user state
aws dynamodb query \
  --table-name weight-processor-state-dev-us \
  --key-condition-expression "user_id = :uid" \
  --expression-attribute-values '{":uid":{"S":"user-123"}}' \
  --region us-east-1
```

