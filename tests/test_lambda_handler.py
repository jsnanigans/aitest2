"""Integration tests for AWS Lambda handler."""

import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.lambda_handler import handler, get_service
from src.factories.component_factory import ComponentFactory


@pytest.fixture
def api_gateway_event():
    """Create a sample API Gateway event."""
    return {
        "resource": "/api/v1/process/{userId}",
        "path": "/api/v1/process/test-user-123",
        "httpMethod": "POST",
        "headers": {"Content-Type": "application/json", "x-api-key": "test-key"},
        "pathParameters": {"userId": "test-user-123"},
        "body": json.dumps(
            {
                "measurements": [
                    {
                        "uuid": str(uuid4()),
                        "weight": 75.5,
                        "unit": "kg",
                        "effectiveDateTime": "2024-01-15T10:30:00Z",
                        "source": "patient-device",
                    },
                    {
                        "uuid": str(uuid4()),
                        "weight": 75.8,
                        "unit": "kg",
                        "effectiveDateTime": "2024-01-16T10:30:00Z",
                        "source": "patient-device",
                    },
                ]
            }
        ),
    }


@pytest.fixture
def lambda_context():
    """Create a mock Lambda context."""
    context = MagicMock()
    context.function_name = "weight-processor-test"
    context.function_version = "$LATEST"
    context.aws_request_id = str(uuid4())
    context.get_remaining_time_in_millis = MagicMock(return_value=30000)
    return context


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset factory before each test."""
    ComponentFactory.reset()
    yield
    ComponentFactory.reset()


class TestLambdaHandler:
    """Test suite for Lambda handler."""

    def test_process_measurements_success(self, api_gateway_event, lambda_context):
        """Test successful processing of measurements."""
        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            response = handler(api_gateway_event, lambda_context)

            assert response["statusCode"] == 200
            assert "body" in response

            body = json.loads(response["body"])
            assert body["status"] == "processed"
            assert body["processed_count"] == 2
            assert len(body["measurements"]) == 2

    def test_process_measurements_invalid_request(
        self, api_gateway_event, lambda_context
    ):
        """Test handling of invalid request."""
        # Invalid body
        api_gateway_event["body"] = '{"invalid": "data"}'

        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            response = handler(api_gateway_event, lambda_context)

            assert response["statusCode"] == 400
            assert "error" in json.loads(response["body"])

    def test_cleanup_endpoint(self, lambda_context):
        """Test cleanup endpoint."""
        event = {
            "resource": "/api/v1/cleanup/{userId}",
            "path": "/api/v1/cleanup/test-user-456",
            "httpMethod": "POST",
            "pathParameters": {"userId": "test-user-456"},
            "body": json.dumps(
                {
                    "measurements": [
                        {
                            "uuid": str(uuid4()),
                            "weight": 80.0,
                            "unit": "kg",
                            "effectiveDateTime": "2024-01-01T00:00:00Z",
                            "source": "care-team-upload",
                        }
                    ],
                    "options": {"reset_state": True},
                }
            ),
        }

        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            response = handler(event, lambda_context)

            assert response["statusCode"] == 200
            body = json.loads(response["body"])
            assert body["user_id"] == "test-user-456"
            assert body["processed_count"] == 1

    def test_get_state_endpoint(self, lambda_context):
        """Test get state endpoint."""
        # First, process some measurements
        process_event = {
            "resource": "/api/v1/process/{userId}",
            "path": "/api/v1/process/test-user-789",
            "httpMethod": "POST",
            "pathParameters": {"userId": "test-user-789"},
            "body": json.dumps(
                {
                    "measurements": [
                        {
                            "uuid": str(uuid4()),
                            "weight": 70.0,
                            "unit": "kg",
                            "effectiveDateTime": "2024-01-10T00:00:00Z",
                            "source": "patient-device",
                        }
                    ]
                }
            ),
        }

        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            # Process measurement first
            handler(process_event, lambda_context)

            # Now get state
            get_event = {
                "resource": "/api/v1/state/{userId}",
                "path": "/api/v1/state/test-user-789",
                "httpMethod": "GET",
                "pathParameters": {"userId": "test-user-789"},
            }

            response = handler(get_event, lambda_context)
            assert response["statusCode"] == 200

            state = json.loads(response["body"])
            assert state["last_raw_weight"] == 70.0

    def test_delete_state_endpoint(self, lambda_context):
        """Test delete state endpoint."""
        # First, process some measurements
        process_event = {
            "resource": "/api/v1/process/{userId}",
            "path": "/api/v1/process/test-user-999",
            "httpMethod": "POST",
            "pathParameters": {"userId": "test-user-999"},
            "body": json.dumps(
                {
                    "measurements": [
                        {
                            "uuid": str(uuid4()),
                            "weight": 65.0,
                            "unit": "kg",
                            "effectiveDateTime": "2024-01-05T00:00:00Z",
                            "source": "patient-device",
                        }
                    ]
                }
            ),
        }

        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            # Process measurement first
            handler(process_event, lambda_context)

            # Delete state
            delete_event = {
                "resource": "/api/v1/state/{userId}",
                "path": "/api/v1/state/test-user-999",
                "httpMethod": "DELETE",
                "pathParameters": {"userId": "test-user-999"},
            }

            response = handler(delete_event, lambda_context)
            assert response["statusCode"] == 200

            # Verify state is deleted
            get_event = {
                "resource": "/api/v1/state/{userId}",
                "path": "/api/v1/state/test-user-999",
                "httpMethod": "GET",
                "pathParameters": {"userId": "test-user-999"},
            }

            response = handler(get_event, lambda_context)
            assert response["statusCode"] == 404

    def test_404_not_found(self, lambda_context):
        """Test 404 response for unknown endpoint."""
        event = {
            "resource": "/api/v1/unknown",
            "path": "/api/v1/unknown",
            "httpMethod": "GET",
        }

        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            response = handler(event, lambda_context)
            assert response["statusCode"] == 404

    def test_historical_conflict_detection(self, lambda_context):
        """Test detection of historical conflict."""
        user_id = "test-user-conflict"

        # First measurement
        event1 = {
            "resource": "/api/v1/process/{userId}",
            "path": f"/api/v1/process/{user_id}",
            "httpMethod": "POST",
            "pathParameters": {"userId": user_id},
            "body": json.dumps(
                {
                    "measurements": [
                        {
                            "uuid": str(uuid4()),
                            "weight": 75.0,
                            "unit": "kg",
                            "effectiveDateTime": "2024-01-20T10:00:00Z",
                            "source": "patient-device",
                        }
                    ]
                }
            ),
        }

        # Second measurement with earlier timestamp (conflict)
        event2 = {
            "resource": "/api/v1/process/{userId}",
            "path": f"/api/v1/process/{user_id}",
            "httpMethod": "POST",
            "pathParameters": {"userId": user_id},
            "body": json.dumps(
                {
                    "measurements": [
                        {
                            "uuid": str(uuid4()),
                            "weight": 74.5,
                            "unit": "kg",
                            "effectiveDateTime": "2024-01-15T10:00:00Z",  # Earlier than first
                            "source": "patient-device",
                        }
                    ]
                }
            ),
        }

        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            # Process first measurement
            response1 = handler(event1, lambda_context)
            assert response1["statusCode"] == 200

            # Attempt to process earlier measurement (should conflict)
            response2 = handler(event2, lambda_context)
            assert response2["statusCode"] == 409  # Conflict

            conflict_data = json.loads(response2["body"])
            assert conflict_data["status"] == "historical_conflict"
            assert "details" in conflict_data

    def test_service_initialization(self):
        """Test service initialization with different backends."""
        with patch.dict("os.environ", {"DB_BACKEND": "memory"}):
            service = get_service()
            assert service is not None
            assert service.state_store is not None

        # Test with DynamoDB (mocked)
        with patch.dict(
            "os.environ",
            {
                "DB_BACKEND": "dynamodb",
                "DYNAMODB_TABLE_NAME": "test-table",
                "AWS_REGION": "us-east-1",
            },
        ):
            with patch("src.database.dynamodb_store.boto3"):
                service = get_service()
                assert service is not None
