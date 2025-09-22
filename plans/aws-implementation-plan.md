# AWS Weight Processing Service - Implementation Plan

## 1. Service Architecture Implementation

### 1.1 Project Structure
```
weight-processor-service/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── handlers.py          # Lambda/API handlers
│   │   ├── schemas.py           # Pydantic models
│   │   └── validators.py        # Input validation
│   ├── core/
│   │   ├── __init__.py
│   │   ├── processor.py         # Core processing logic (from current)
│   │   ├── kalman.py           # Kalman filter (from current)
│   │   └── quality_scorer.py   # Quality scoring (from current)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── state_service.py    # State management service
│   │   ├── replay_service.py   # Replay orchestration
│   │   └── snapshot_service.py # Snapshot management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── dynamodb_client.py  # DynamoDB operations
│   │   ├── models.py           # Database models
│   │   └── repository.py       # Data access layer
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # CloudWatch metrics
│       └── logger.py           # Structured logging
├── tests/
├── deployment/
│   ├── terraform/              # Infrastructure as Code
│   ├── docker/                 # Container definitions
│   └── scripts/                # Deployment scripts
├── requirements.txt
├── Dockerfile
└── serverless.yml              # For Lambda deployment
```

### 1.2 Core Service Implementation

#### Handler Layer (Lambda/API)
```python
# src/api/handlers.py
import json
from typing import Dict, Any
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.logging import correlation_id

from .schemas import CleanupRequest, ProcessRequest, ReplayRequest
from .validators import validate_request
from ..services.state_service import StateService
from ..services.replay_service import ReplayService
from ..core.processor import process_measurement

logger = Logger()
metrics = Metrics()
tracer = Tracer()

@logger.inject_lambda_context(correlation_id_path="correlationId")
@metrics.log_metrics
@tracer.capture_lambda_handler
def cleanup_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for one-time cleanup operation.
    Processes all historical data for a user.
    """
    try:
        # Parse and validate request
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])
        request = CleanupRequest(**body)

        # Initialize services
        state_service = StateService()

        # Reset state if requested
        if request.options.reset_state:
            state_service.delete_state(user_id)
            logger.info(f"Reset state for user {user_id}")

        # Process all measurements
        results = []
        accepted_count = 0
        rejected_count = 0

        for measurement in request.measurements:
            try:
                # Process measurement using core logic
                result = process_measurement(
                    user_id=user_id,
                    weight=measurement.weight,
                    timestamp=measurement.effective_date_time,
                    source=measurement.source,
                    unit=measurement.unit,
                    config=get_config(),
                    db=state_service
                )

                # Format response
                measurement_result = {
                    "uuid": measurement.uuid,
                    "accepted": result.get("accepted", False),
                    "qualityScore": result.get("quality_score"),
                    "kalmanEstimate": result.get("kalman_estimate"),
                    "kalmanUncertainty": result.get("kalman_uncertainty")
                }

                if result.get("accepted"):
                    accepted_count += 1
                else:
                    rejected_count += 1
                    measurement_result["rejectionReason"] = result.get("reason")
                    measurement_result["stage"] = result.get("stage")

                results.append(measurement_result)

            except Exception as e:
                logger.error(f"Error processing measurement {measurement.uuid}: {str(e)}")
                results.append({
                    "uuid": measurement.uuid,
                    "accepted": False,
                    "rejectionReason": "Processing error",
                    "error": str(e)
                })
                rejected_count += 1

        # Get final state
        final_state = state_service.get_state(user_id)

        # Record metrics
        metrics.add_metric(name="CleanupProcessed", unit=MetricUnit.Count, value=len(results))
        metrics.add_metric(name="CleanupAccepted", unit=MetricUnit.Count, value=accepted_count)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "userId": user_id,
                "processedCount": len(results),
                "acceptedCount": accepted_count,
                "rejectedCount": rejected_count,
                "measurements": results,
                "finalState": format_state(final_state)
            })
        }

    except Exception as e:
        logger.exception("Error in cleanup handler")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }

@tracer.capture_lambda_handler
def process_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for processing multiple new measurements.
    All measurements must be after last processed timestamp.
    """
    try:
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])
        request = ProcessRequest(**body)

        # Initialize services
        state_service = StateService()

        # Get current state to check for conflicts
        current_state = state_service.get_state(user_id)

        # Sort measurements by timestamp
        sorted_measurements = sorted(
            request.measurements,
            key=lambda m: m.effective_date_time
        )

        # Check for historical conflicts
        if current_state and current_state.get("last_timestamp"):
            last_timestamp = parse_timestamp(current_state["last_timestamp"])
            earliest_measurement = sorted_measurements[0]

            if earliest_measurement.effective_date_time < last_timestamp:
                # Find all conflicting measurements
                conflicting_uuids = [
                    str(m.uuid) for m in sorted_measurements
                    if m.effective_date_time < last_timestamp
                ]

                snapshot_timestamp = state_service.get_nearest_snapshot(
                    user_id, earliest_measurement.effective_date_time
                )

                return {
                    "statusCode": 409,  # Conflict
                    "body": json.dumps({
                        "status": "historical_conflict",
                        "error": "One or more measurements are before last processed timestamp",
                        "details": {
                            "earliestMeasurementTimestamp": earliest_measurement.effective_date_time.isoformat(),
                            "lastProcessedTimestamp": last_timestamp.isoformat(),
                            "replayRequired": True,
                            "replayFromTimestamp": earliest_measurement.effective_date_time.isoformat(),
                            "snapshotAvailable": snapshot_timestamp.isoformat() if snapshot_timestamp else None,
                            "conflictingMeasurements": conflicting_uuids
                        }
                    })
                }

        # Process measurements in chronological order
        results = []
        accepted_count = 0
        rejected_count = 0
        previous_weight = current_state.get("last_raw_weight") if current_state else None

        for measurement in sorted_measurements:
            try:
                result = process_measurement(
                    user_id=user_id,
                    weight=measurement.weight,
                    timestamp=measurement.effective_date_time,
                    source=measurement.source,
                    unit=measurement.unit,
                    config=get_config(),
                    db=state_service
                )

                measurement_result = {
                    "uuid": str(measurement.uuid),
                    "accepted": result.get("accepted", False),
                    "qualityScore": result.get("quality_score"),
                    "kalmanEstimate": result.get("kalman_estimate"),
                    "kalmanUncertainty": result.get("kalman_uncertainty"),
                    "resetTriggered": result.get("reset_triggered", False)
                }

                if result.get("accepted"):
                    accepted_count += 1
                else:
                    rejected_count += 1
                    measurement_result["rejectionReason"] = result.get("reason")

                results.append(measurement_result)

            except Exception as e:
                logger.error(f"Error processing measurement {measurement.uuid}: {e}")
                results.append({
                    "uuid": str(measurement.uuid),
                    "accepted": False,
                    "rejectionReason": "Processing error",
                    "error": str(e)
                })
                rejected_count += 1

        # Get final state for response
        final_state = state_service.get_state(user_id)
        current_weight = final_state.get("last_raw_weight") if final_state else None

        # Create snapshot if needed
        state_service.maybe_create_snapshot(user_id)

        # Record metrics
        metrics.add_metric(name="BatchProcessed", unit=MetricUnit.Count, value=len(results))
        metrics.add_metric(name="BatchAccepted", unit=MetricUnit.Count, value=accepted_count)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "processed",
                "processedCount": len(results),
                "acceptedCount": accepted_count,
                "rejectedCount": rejected_count,
                "measurements": results,
                "stateUpdate": {
                    "previousWeight": previous_weight,
                    "currentWeight": current_weight,
                    "lastProcessedTimestamp": sorted_measurements[-1].effective_date_time.isoformat()
                }
            })
        }

    except Exception as e:
        logger.exception("Error in process handler")
        return error_response(500, str(e))

@tracer.capture_lambda_handler
def replay_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for replay operations.
    """
    try:
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])
        request = ReplayRequest(**body)

        # Initialize services
        replay_service = ReplayService()

        # Perform replay
        replay_result = replay_service.replay_from_timestamp(
            user_id=user_id,
            replay_from=request.replay_from_timestamp,
            measurements=request.measurements,
            use_snapshot=request.options.use_snapshot,
            create_snapshot=request.options.create_new_snapshot
        )

        return {
            "statusCode": 200,
            "body": json.dumps(replay_result)
        }

    except Exception as e:
        logger.exception("Error in replay handler")
        return error_response(500, str(e))
```

#### Schema Definitions
```python
# src/api/schemas.py
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID

class Measurement(BaseModel):
    """Single measurement data model."""
    uuid: UUID
    weight: float = Field(gt=0, le=1000)
    unit: str = Field(regex="^(kg|lbs|lb|g|oz)$")
    effective_date_time: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None

    @validator('weight')
    def validate_weight(cls, v, values):
        """Validate weight is within physiological bounds."""
        unit = values.get('unit', 'kg')
        if unit == 'kg' and (v < 10 or v > 500):
            raise ValueError(f"Weight {v}kg outside valid range")
        return v

class UserProfile(BaseModel):
    """User profile for validation."""
    height: Optional[float] = None
    height_unit: Optional[str] = "cm"
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None

class CleanupOptions(BaseModel):
    """Options for cleanup operation."""
    reset_state: bool = True
    include_quality_scores: bool = True
    include_debug_info: bool = False

class CleanupRequest(BaseModel):
    """Request body for cleanup endpoint."""
    measurements: List[Measurement]
    user_profile: Optional[UserProfile] = None
    options: Optional[CleanupOptions] = CleanupOptions()

class ProcessRequest(BaseModel):
    """Request body for process endpoint."""
    measurements: List[Measurement]
    options: Optional[Dict[str, Any]] = {}

class ReplayRequest(BaseModel):
    """Request body for replay endpoint."""
    replay_from_timestamp: datetime
    measurements: List[Measurement]
    options: Dict[str, Any] = {
        "use_snapshot": True,
        "create_new_snapshot": True
    }
```

### 1.3 State Management Service

```python
# src/services/state_service.py
import json
import boto3
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import numpy as np

from ..database.dynamodb_client import DynamoDBClient
from ..utils.logger import get_logger

logger = get_logger(__name__)

class StateService:
    """
    Service for managing Kalman filter states in DynamoDB.
    Handles current states and historical snapshots.
    """

    def __init__(self, table_name: str = None):
        self.client = DynamoDBClient(table_name)
        self.snapshot_retention_days = 7

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for a user."""
        try:
            response = self.client.get_item(
                Key={
                    'userId': user_id,
                    'stateType': 'current'
                }
            )

            if not response:
                return None

            # Deserialize numpy arrays
            state = self._deserialize_state(response)
            return state

        except Exception as e:
            logger.error(f"Error getting state for {user_id}: {e}")
            return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save current state for a user."""
        try:
            # Serialize numpy arrays
            serialized_state = self._serialize_state(state)

            # Add metadata
            serialized_state.update({
                'userId': user_id,
                'stateType': 'current',
                'updatedAt': datetime.utcnow().isoformat(),
                'version': serialized_state.get('version', 0) + 1
            })

            # Save with optimistic locking
            self.client.put_item(
                Item=serialized_state,
                ConditionExpression='attribute_not_exists(userId) OR version < :new_version',
                ExpressionAttributeValues={
                    ':new_version': serialized_state['version']
                }
            )

            logger.info(f"Saved state for {user_id}")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logger.warning(f"Version conflict saving state for {user_id}")
                return False
            raise
        except Exception as e:
            logger.error(f"Error saving state for {user_id}: {e}")
            return False

    def create_snapshot(self, user_id: str, timestamp: datetime = None) -> bool:
        """Create a snapshot of current state."""
        try:
            # Get current state
            current_state = self.get_state(user_id)
            if not current_state:
                logger.warning(f"No state to snapshot for {user_id}")
                return False

            # Create snapshot
            snapshot_time = timestamp or datetime.utcnow()
            snapshot_key = f"snapshot_{snapshot_time.isoformat()}"

            snapshot = current_state.copy()
            snapshot.update({
                'userId': user_id,
                'stateType': snapshot_key,
                'snapshotTime': snapshot_time.isoformat(),
                'ttl': int((snapshot_time + timedelta(days=self.snapshot_retention_days)).timestamp())
            })

            # Serialize and save
            serialized = self._serialize_state(snapshot)
            self.client.put_item(Item=serialized)

            logger.info(f"Created snapshot for {user_id} at {snapshot_time}")
            return True

        except Exception as e:
            logger.error(f"Error creating snapshot for {user_id}: {e}")
            return False

    def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get nearest snapshot before timestamp."""
        try:
            # Query snapshots before timestamp
            response = self.client.query(
                KeyConditionExpression='userId = :user_id AND begins_with(stateType, :prefix)',
                FilterExpression='snapshotTime <= :timestamp',
                ExpressionAttributeValues={
                    ':user_id': user_id,
                    ':prefix': 'snapshot_',
                    ':timestamp': timestamp.isoformat()
                },
                ScanIndexForward=False,  # Sort descending
                Limit=1
            )

            items = response.get('Items', [])
            if not items:
                return None

            return self._deserialize_state(items[0])

        except Exception as e:
            logger.error(f"Error getting snapshot for {user_id}: {e}")
            return None

    def restore_from_snapshot(self, user_id: str, snapshot: Dict[str, Any]) -> bool:
        """Restore state from a snapshot."""
        try:
            # Convert snapshot to current state
            current_state = snapshot.copy()
            current_state['stateType'] = 'current'
            current_state['restoredFrom'] = snapshot.get('snapshotTime')

            return self.save_state(user_id, current_state)

        except Exception as e:
            logger.error(f"Error restoring snapshot for {user_id}: {e}")
            return False

    def maybe_create_snapshot(self, user_id: str) -> bool:
        """Create snapshot if conditions are met (e.g., daily)."""
        try:
            # Check last snapshot time
            response = self.client.query(
                KeyConditionExpression='userId = :user_id AND begins_with(stateType, :prefix)',
                ExpressionAttributeValues={
                    ':user_id': user_id,
                    ':prefix': 'snapshot_'
                },
                ScanIndexForward=False,
                Limit=1
            )

            items = response.get('Items', [])

            # Create snapshot if no snapshots or last was > 24 hours ago
            should_create = True
            if items:
                last_snapshot = items[0]
                last_time = datetime.fromisoformat(last_snapshot['snapshotTime'])
                hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
                should_create = hours_since >= 24

            if should_create:
                return self.create_snapshot(user_id)

            return False

        except Exception as e:
            logger.error(f"Error checking snapshot for {user_id}: {e}")
            return False

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for DynamoDB storage."""
        serialized = state.copy()

        # Convert numpy arrays to lists
        if 'kalman_params' in state and state['kalman_params']:
            params = state['kalman_params']
            if 'x' in params and isinstance(params['x'], np.ndarray):
                serialized['kalman_params']['x'] = params['x'].tolist()
            if 'P' in params and isinstance(params['P'], np.ndarray):
                serialized['kalman_params']['P'] = params['P'].tolist()

        # Convert datetime objects to strings
        for key in ['last_timestamp', 'last_accepted_timestamp']:
            if key in state and isinstance(state[key], datetime):
                serialized[key] = state[key].isoformat()

        return serialized

    def _deserialize_state(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state from DynamoDB."""
        state = item.copy()

        # Convert lists back to numpy arrays
        if 'kalman_params' in state and state['kalman_params']:
            params = state['kalman_params']
            if 'x' in params and isinstance(params['x'], list):
                state['kalman_params']['x'] = np.array(params['x'])
            if 'P' in params and isinstance(params['P'], list):
                state['kalman_params']['P'] = np.array(params['P'])

        # Convert strings back to datetime objects
        for key in ['last_timestamp', 'last_accepted_timestamp']:
            if key in state and isinstance(state[key], str):
                state[key] = datetime.fromisoformat(state[key])

        return state

    # Implement ProcessorStateDB interface for compatibility
    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
        return {
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'last_timestamp': None,
            'last_accepted_timestamp': None,
            'last_source': None,
            'last_raw_weight': None,
            'measurement_history': [],
            'reset_events': [],
            'adaptation_state': {}
        }

    def delete_state(self, user_id: str) -> bool:
        """Delete all states for a user."""
        try:
            # Delete current state
            self.client.delete_item(
                Key={
                    'userId': user_id,
                    'stateType': 'current'
                }
            )

            # Delete all snapshots
            response = self.client.query(
                KeyConditionExpression='userId = :user_id AND begins_with(stateType, :prefix)',
                ExpressionAttributeValues={
                    ':user_id': user_id,
                    ':prefix': 'snapshot_'
                }
            )

            for item in response.get('Items', []):
                self.client.delete_item(
                    Key={
                        'userId': user_id,
                        'stateType': item['stateType']
                    }
                )

            logger.info(f"Deleted all states for {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting states for {user_id}: {e}")
            return False
```

### 1.4 Replay Service Implementation

```python
# src/services/replay_service.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .state_service import StateService
from ..core.processor import process_measurement
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ReplayService:
    """
    Service for handling replay operations.
    Manages state rollback and reprocessing of historical data.
    """

    def __init__(self, state_service: StateService = None):
        self.state_service = state_service or StateService()

    def replay_from_timestamp(
        self,
        user_id: str,
        replay_from: datetime,
        measurements: List[Dict[str, Any]],
        use_snapshot: bool = True,
        create_snapshot: bool = True
    ) -> Dict[str, Any]:
        """
        Replay measurements from a specific timestamp.

        Args:
            user_id: User identifier
            replay_from: Timestamp to replay from
            measurements: List of measurements to replay
            use_snapshot: Whether to use snapshot for rollback
            create_snapshot: Whether to create new snapshot after replay

        Returns:
            Replay results including state changes
        """
        try:
            # Save current state for comparison
            current_state = self.state_service.get_state(user_id)
            before_replay = {
                "weight": current_state.get("last_raw_weight") if current_state else None,
                "timestamp": current_state.get("last_timestamp") if current_state else None
            }

            # Rollback to snapshot if requested
            snapshot_used = None
            if use_snapshot:
                snapshot = self.state_service.get_snapshot(user_id, replay_from)
                if snapshot:
                    self.state_service.restore_from_snapshot(user_id, snapshot)
                    snapshot_used = snapshot.get("snapshotTime")
                    logger.info(f"Restored snapshot from {snapshot_used} for {user_id}")
                else:
                    # No snapshot available, reset state
                    self.state_service.delete_state(user_id)
                    logger.warning(f"No snapshot found for {user_id}, starting fresh")
            else:
                # Reset state without snapshot
                self.state_service.delete_state(user_id)

            # Sort measurements by timestamp
            sorted_measurements = sorted(
                measurements,
                key=lambda m: m.get("effective_date_time", datetime.min)
            )

            # Replay measurements
            results = []
            for measurement in sorted_measurements:
                try:
                    result = process_measurement(
                        user_id=user_id,
                        weight=measurement["weight"],
                        timestamp=measurement["effective_date_time"],
                        source=measurement.get("source", "unknown"),
                        unit=measurement.get("unit", "kg"),
                        config=self._get_config(),
                        db=self.state_service
                    )

                    results.append({
                        "uuid": measurement.get("uuid"),
                        "accepted": result.get("accepted", False),
                        "qualityScore": result.get("quality_score")
                    })

                except Exception as e:
                    logger.error(f"Error replaying measurement: {e}")
                    results.append({
                        "uuid": measurement.get("uuid"),
                        "accepted": False,
                        "error": str(e)
                    })

            # Get final state after replay
            final_state = self.state_service.get_state(user_id)
            after_replay = {
                "weight": final_state.get("last_raw_weight") if final_state else None,
                "timestamp": final_state.get("last_timestamp") if final_state else None
            }

            # Create new snapshot if requested
            if create_snapshot and final_state:
                self.state_service.create_snapshot(user_id)

            return {
                "status": "replay_completed",
                "snapshotUsed": snapshot_used,
                "measurements": results,
                "stateChanges": {
                    "beforeReplay": before_replay,
                    "afterReplay": after_replay
                }
            }

        except Exception as e:
            logger.exception(f"Error in replay for {user_id}")
            return {
                "status": "replay_failed",
                "error": str(e)
            }

    def _get_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        # This would be loaded from S3 or environment
        return {
            "kalman": {
                "enabled": True,
                "adaptive": True
            },
            "quality_scoring": {
                "enabled": True
            },
            "outlier_detection": {
                "enabled": True
            }
        }
```

## 2. Infrastructure as Code

### 2.1 Terraform Configuration

```hcl
# deployment/terraform/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "weight-processor-terraform-state"
    key    = "state/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# DynamoDB Table for State Storage
resource "aws_dynamodb_table" "state_table" {
  name           = "${var.environment}-weight-processor-state"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "userId"
  range_key      = "stateType"

  attribute {
    name = "userId"
    type = "S"
  }

  attribute {
    name = "stateType"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Environment = var.environment
    Service     = "weight-processor"
  }
}

# Lambda Function
resource "aws_lambda_function" "processor" {
  function_name = "${var.environment}-weight-processor"
  role         = aws_iam_role.lambda_role.arn

  runtime     = "python3.11"
  handler     = "src.api.handlers.main_handler"
  timeout     = 60
  memory_size = 1024

  environment {
    variables = {
      STATE_TABLE_NAME = aws_dynamodb_table.state_table.name
      ENVIRONMENT      = var.environment
      LOG_LEVEL        = var.log_level
    }
  }

  # Package the Lambda code
  filename         = "../../../dist/lambda.zip"
  source_code_hash = filebase64sha256("../../../dist/lambda.zip")
}

# API Gateway
resource "aws_apigatewayv2_api" "api" {
  name          = "${var.environment}-weight-processor-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = var.allowed_origins
    allow_methods = ["POST", "GET", "DELETE"]
    allow_headers = ["content-type", "x-api-key"]
    max_age       = 300
  }
}

# API Routes
resource "aws_apigatewayv2_route" "cleanup" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /api/v1/cleanup/{userId}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "process" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /api/v1/process/{userId}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "replay" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /api/v1/replay/{userId}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# Lambda Integration
resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.processor.invoke_arn
  integration_method = "POST"
}

# Lambda Permissions for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.processor.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.environment}-weight-processor-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for Lambda
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.environment}-weight-processor-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:UpdateItem"
        ]
        Resource = [
          aws_dynamodb_table.state_table.arn,
          "${aws_dynamodb_table.state_table.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "error_rate" {
  alarm_name          = "${var.environment}-weight-processor-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors lambda errors"

  dimensions = {
    FunctionName = aws_lambda_function.processor.function_name
  }
}

# Outputs
output "api_endpoint" {
  value = aws_apigatewayv2_api.api.api_endpoint
}

output "state_table_name" {
  value = aws_dynamodb_table.state_table.name
}
```

### 2.2 Docker Configuration

```dockerfile
# deployment/docker/Dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Set handler
CMD ["src.api.handlers.main_handler"]
```

## 3. Testing Strategy

### 3.1 Unit Tests
```python
# tests/test_handlers.py
import pytest
import json
from unittest.mock import Mock, patch
from src.api.handlers import cleanup_handler, process_handler

class TestHandlers:

    @patch('src.api.handlers.StateService')
    @patch('src.api.handlers.process_measurement')
    def test_cleanup_handler_success(self, mock_process, mock_state_service):
        """Test successful cleanup operation."""
        # Setup mocks
        mock_process.return_value = {
            "accepted": True,
            "quality_score": 0.85,
            "kalman_estimate": 75.5
        }

        # Create test event
        event = {
            "pathParameters": {"userId": "test-user"},
            "body": json.dumps({
                "measurements": [
                    {
                        "uuid": "test-uuid",
                        "weight": 75.5,
                        "unit": "kg",
                        "effectiveDateTime": "2024-01-01T10:00:00Z",
                        "source": "test"
                    }
                ]
            })
        }

        # Call handler
        response = cleanup_handler(event, None)

        # Verify response
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["processedCount"] == 1
        assert body["acceptedCount"] == 1

    def test_process_handler_historical_conflict(self):
        """Test historical conflict detection."""
        # Test implementation
        pass
```

### 3.2 Integration Tests
```python
# tests/integration/test_api.py
import requests
import pytest

class TestAPI:

    @pytest.fixture
    def api_endpoint(self):
        return "https://api.example.com"

    def test_cleanup_endpoint(self, api_endpoint):
        """Test full cleanup flow."""
        response = requests.post(
            f"{api_endpoint}/api/v1/cleanup/test-user",
            json={
                "measurements": [
                    {
                        "uuid": "test-uuid",
                        "weight": 75.5,
                        "unit": "kg",
                        "effectiveDateTime": "2024-01-01T10:00:00Z",
                        "source": "test"
                    }
                ]
            }
        )
        assert response.status_code == 200
```

## 4. Deployment Pipeline

### 4.1 GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy Weight Processor

on:
  push:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Package Lambda
        run: |
          pip install -r requirements.txt -t package/
          cp -r src package/
          cd package && zip -r ../lambda.zip .

      - name: Deploy to AWS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws lambda update-function-code \
            --function-name weight-processor \
            --zip-file fileb://lambda.zip
```

## 5. Migration Execution Plan

### Phase 1: Setup (Week 1)
- [ ] Set up AWS accounts and permissions
- [ ] Create development environment
- [ ] Set up CI/CD pipeline
- [ ] Create DynamoDB tables

### Phase 2: Core Implementation (Week 2-3)
- [ ] Port processing logic
- [ ] Implement state service
- [ ] Create API handlers
- [ ] Add authentication

### Phase 3: Integration (Week 4-5)
- [ ] Java client library
- [ ] Integration tests
- [ ] Performance testing
- [ ] Documentation

### Phase 4: Deployment (Week 6)
- [ ] Deploy to staging
- [ ] Run parallel testing
- [ ] Data migration
- [ ] Production deployment

## 6. Java Client Example

```java
// Example Java client for the service
public class WeightProcessorClient {
    private final String apiEndpoint;
    private final String apiKey;
    private final HttpClient httpClient;

    public WeightProcessorClient(String apiEndpoint, String apiKey) {
        this.apiEndpoint = apiEndpoint;
        this.apiKey = apiKey;
        this.httpClient = HttpClient.newHttpClient();
    }

    public CleanupResponse cleanup(String userId, List<Measurement> measurements) {
        CleanupRequest request = new CleanupRequest(measurements);

        HttpRequest httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(apiEndpoint + "/api/v1/cleanup/" + userId))
            .header("Content-Type", "application/json")
            .header("X-API-Key", apiKey)
            .POST(HttpRequest.BodyPublishers.ofString(toJson(request)))
            .build();

        HttpResponse<String> response = httpClient.send(httpRequest,
            HttpResponse.BodyHandlers.ofString());

        return fromJson(response.body(), CleanupResponse.class);
    }

    public ProcessResponse process(String userId, Measurement measurement) {
        // Implementation
    }

    public ReplayResponse replay(String userId, Instant replayFrom,
                                 List<Measurement> measurements) {
        // Implementation
    }
}
```