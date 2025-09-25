# Weight Processor API Documentation

## Overview

The Weight Processor Service is a serverless API that processes weight measurements using Kalman filtering and intelligent quality scoring. It provides endpoints for processing measurements, managing user state, and performing one-time data cleanup operations.

## Base URL

```
https://{api-id}.execute-api.{region}.amazonaws.com/{stage}
```

- **Stages**: `dev`, `staging`, `prod`
- **Region**: Default is `us-east-1`

## Authentication

All endpoints require an API key to be provided in the `x-api-key` header:

```
x-api-key: your-api-key-here
```

## Endpoints

### 1. Process Measurements

Process a batch of weight measurements for a user.

**Endpoint**: `POST /api/v1/process/{userId}`

**Path Parameters**:
- `userId` (string, required): Unique identifier for the user

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
        "device": "scale-001",
        "location": "home"
      }
    }
  ],
  "options": {
    "fail_on_historical_conflict": true
  }
}
```

**Measurement Fields**:
- `uuid` (UUID, required): Unique identifier for the measurement
- `weight` (float, required): Weight value (must be > 0 and <= 1000)
- `unit` (string, required): Unit of measurement (`kg`, `lb`, `lbs`, `g`, `oz`)
- `effectiveDateTime` (datetime, required): ISO 8601 timestamp
- `source` (string, required): Data source identifier
- `metadata` (object, optional): Additional metadata

**Sources by Reliability** (lower noise = more reliable):
- `care-team-upload`: 0.5 noise multiplier (most reliable)
- `patient-upload`: 0.7
- `questionnaire`: 0.8
- `patient-device`: 1.0
- `connectivehealth.io`: 1.5
- `iglucose.com`: 3.0 (least reliable)

**Response** (200 OK):
```json
{
  "status": "processed",
  "processed_count": 2,
  "accepted_count": 1,
  "rejected_count": 1,
  "measurements": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "accepted": true,
      "quality_score": 0.85,
      "kalman_estimate": 75.4,
      "kalman_uncertainty": 0.5,
      "rejection_reason": null,
      "stage": "processed",
      "reset_triggered": false,
      "components": {
        "kalman": 0.9,
        "temporal": 0.8,
        "source": 0.85
      }
    }
  ],
  "state_update": {
    "previous_weight": null,
    "current_weight": 75.4,
    "last_processed_timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Error Response** (409 Conflict - Historical Conflict):
```json
{
  "status": "historical_conflict",
  "error": "One or more measurements are before last processed timestamp",
  "details": {
    "earliest_measurement_timestamp": "2024-01-10T10:00:00Z",
    "last_processed_timestamp": "2024-01-15T10:00:00Z",
    "replay_required": true,
    "replay_from_timestamp": "2024-01-10T10:00:00Z",
    "snapshot_available": null,
    "conflicting_measurements": [
      "550e8400-e29b-41d4-a716-446655440001"
    ]
  }
}
```

### 2. Cleanup User Data

Perform a one-time cleanup operation for a user's historical weight data. This reprocesses all measurements from scratch with the latest algorithms.

**Endpoint**: `POST /api/v1/cleanup/{userId}`

**Path Parameters**:
- `userId` (string, required): Unique identifier for the user

**Request Body**:
```json
{
  "measurements": [
    {
      "uuid": "650e8400-e29b-41d4-a716-446655440000",
      "weight": 80.0,
      "unit": "kg",
      "effectiveDateTime": "2024-01-01T00:00:00Z",
      "source": "care-team-upload"
    }
  ],
  "user_profile": {
    "height": 175.0,
    "height_unit": "cm",
    "date_of_birth": "1990-01-01",
    "gender": "male"
  },
  "options": {
    "reset_state": true,
    "include_quality_scores": true,
    "include_debug_info": false
  }
}
```

**Options**:
- `reset_state` (boolean): Whether to reset user state before processing (default: true)
- `include_quality_scores` (boolean): Include quality scores in response (default: true)
- `include_debug_info` (boolean): Include debug information (default: false)

**Response** (200 OK):
```json
{
  "user_id": "user-456",
  "processed_count": 5,
  "accepted_count": 4,
  "rejected_count": 1,
  "measurements": [
    {
      "uuid": "650e8400-e29b-41d4-a716-446655440000",
      "accepted": true,
      "quality_score": 0.95,
      "kalman_estimate": 79.8,
      "kalman_uncertainty": 0.3,
      "rejection_reason": null,
      "stage": "processed",
      "reset_triggered": false
    }
  ],
  "final_state": {
    "current_weight": 79.2,
    "uncertainty": 0.25,
    "last_processed_timestamp": "2024-01-05T00:00:00Z",
    "total_measurements": 5,
    "adaptation_state": "converged"
  }
}
```

### 3. Get User State

Retrieve the current Kalman filter state for a user.

**Endpoint**: `GET /api/v1/state/{userId}`

**Path Parameters**:
- `userId` (string, required): Unique identifier for the user

**Response** (200 OK):
```json
{
  "last_state": [75.4, 0.01],
  "last_covariance": [[0.5, 0.01], [0.01, 0.1]],
  "last_timestamp": "2024-01-15T10:30:00Z",
  "last_accepted_timestamp": "2024-01-15T10:30:00Z",
  "last_source": "patient-device",
  "last_raw_weight": 75.5,
  "measurements_since_reset": 10,
  "kalman_params": {
    "process_noise": 1.0,
    "observation_noise": 4.0
  },
  "adaptation_state": {
    "in_adaptation": false,
    "measurements_in_adaptation": 0
  }
}
```

**Response** (404 Not Found):
```json
{
  "error": "State not found for user user-123"
}
```

### 4. Delete User State

Delete all stored state for a user.

**Endpoint**: `DELETE /api/v1/state/{userId}`

**Path Parameters**:
- `userId` (string, required): Unique identifier for the user

**Response** (200 OK):
```json
{
  "message": "State deleted for user user-123"
}
```

**Response** (404 Not Found):
```json
{
  "error": "State not found for user user-123"
}
```

## Error Codes

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid request format or parameters |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Historical conflict detected |
| 500 | Internal Server Error |

## Rate Limits

- **Burst Limit**: 100 requests
- **Rate Limit**: 50 requests per second
- **Daily Quota**: 10,000 requests

## Quality Scoring

The system uses a unified quality scoring system that combines multiple factors:

1. **Kalman Deviation** (40% weight): How well the measurement fits the Kalman filter prediction
2. **Temporal Consistency** (30% weight): Consistency with recent measurements
3. **Source Reliability** (30% weight): Based on the data source

Quality scores range from 0.0 to 1.0, with higher scores indicating more reliable measurements.

## Kalman Filter Resets

The Kalman filter automatically resets in these scenarios:

1. **INITIAL**: First measurements for a user
2. **HARD**: Gap of 30+ days between measurements
3. **SOFT**: Manual entry from questionnaire source

During adaptation after a reset, the filter temporarily increases noise parameters to quickly converge to new weight ranges.

## Testing

### Local Testing with SAM

```bash
# Start local API
sam local start-api --env-vars env.json

# Test with curl
curl -X POST http://localhost:3000/api/v1/process/test-user \
  -H 'Content-Type: application/json' \
  -d @test_events/process_measurements.json
```

### Test Event Files

Example test events are provided in the `test_events/` directory:
- `process_measurements.json` - Standard measurement processing
- `cleanup_user.json` - Cleanup operation with multiple measurements
- `get_state.json` - Retrieve user state
- `historical_conflict.json` - Test historical conflict detection

### Running Integration Tests

```bash
# Run Lambda handler tests
make test-lambda

# Test all endpoints locally
./test_events/test_lambda_local.sh
```

## Migration from Batch Processing

### Before (CSV Batch):
```bash
python main.py data/weights.csv --output-dir output
```

### After (API):
```python
import requests
import json

api_url = "https://your-api.execute-api.region.amazonaws.com/prod"
api_key = "your-api-key"

# Process measurements
response = requests.post(
    f"{api_url}/api/v1/process/user-123",
    headers={"x-api-key": api_key},
    json={
        "measurements": [
            {
                "uuid": "unique-id",
                "weight": 75.5,
                "unit": "kg",
                "effectiveDateTime": "2024-01-15T10:30:00Z",
                "source": "patient-device"
            }
        ]
    }
)

result = response.json()
print(f"Accepted: {result['accepted_count']}, Rejected: {result['rejected_count']}")
```

## Performance Considerations

- Lambda cold start: ~1-2 seconds
- Warm invocation: ~100-300ms per request
- DynamoDB latency: ~10-20ms per operation
- Maximum payload size: 6MB
- Maximum timeout: 30 seconds

## Monitoring

CloudWatch metrics are automatically collected for:
- Lambda invocations, errors, and duration
- API Gateway requests and latency
- DynamoDB read/write capacity and throttles

Alarms are configured for:
- Error rate > 10 errors in 5 minutes
- Throttling > 5 throttles in 5 minutes
- DynamoDB capacity > 80% utilization