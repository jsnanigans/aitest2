"""Mock Lambda events for local testing."""

import json
from datetime import datetime, timedelta
from uuid import uuid4


def create_api_gateway_event(
    path: str,
    method: str,
    path_parameters: dict = None,
    body: dict = None,
    headers: dict = None
) -> dict:
    """Create a mock API Gateway event."""
    event = {
        "resource": path,
        "path": path.replace("{userId}", path_parameters.get("userId", "test") if path_parameters else "test"),
        "httpMethod": method,
        "headers": headers or {
            "Content-Type": "application/json",
            "X-Api-Key": "test-api-key",
            "X-Correlation-Id": str(uuid4())
        },
        "multiValueHeaders": None,
        "queryStringParameters": None,
        "multiValueQueryStringParameters": None,
        "pathParameters": path_parameters or {"userId": "test-user"},
        "stageVariables": None,
        "requestContext": {
            "accountId": "123456789012",
            "apiId": "test-api",
            "protocol": "HTTP/1.1",
            "httpMethod": method,
            "path": path,
            "stage": "test",
            "requestId": str(uuid4()),
            "requestTime": datetime.utcnow().isoformat(),
            "requestTimeEpoch": int(datetime.utcnow().timestamp() * 1000),
            "identity": {
                "sourceIp": "127.0.0.1",
                "userAgent": "PostmanRuntime/7.28.4"
            }
        },
        "body": json.dumps(body) if body else None,
        "isBase64Encoded": False
    }
    return event


# Process endpoint events
def get_process_event_single():
    """Single measurement for process endpoint."""
    return create_api_gateway_event(
        path="/api/v1/process/{userId}",
        method="POST",
        path_parameters={"userId": "test-user-001"},
        body={
            "measurements": [
                {
                    "uuid": str(uuid4()),
                    "weight": 75.5,
                    "unit": "kg",
                    "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
                    "source": "patient-device",
                    "metadata": {
                        "deviceId": "scale-001",
                        "location": "home"
                    }
                }
            ]
        }
    )


def get_process_event_batch():
    """Multiple measurements for process endpoint."""
    base_time = datetime.utcnow()
    return create_api_gateway_event(
        path="/api/v1/process/{userId}",
        method="POST",
        path_parameters={"userId": "test-user-002"},
        body={
            "measurements": [
                {
                    "uuid": str(uuid4()),
                    "weight": 75.5,
                    "unit": "kg",
                    "effectiveDateTime": base_time.isoformat() + "Z",
                    "source": "patient-device"
                },
                {
                    "uuid": str(uuid4()),
                    "weight": 75.3,
                    "unit": "kg",
                    "effectiveDateTime": (base_time + timedelta(hours=1)).isoformat() + "Z",
                    "source": "care-team-upload"
                },
                {
                    "uuid": str(uuid4()),
                    "weight": 75.8,
                    "unit": "kg",
                    "effectiveDateTime": (base_time + timedelta(hours=2)).isoformat() + "Z",
                    "source": "patient-upload"
                }
            ]
        }
    )


def get_process_event_historical_conflict():
    """Event that will trigger historical conflict."""
    return create_api_gateway_event(
        path="/api/v1/process/{userId}",
        method="POST",
        path_parameters={"userId": "existing-user"},
        body={
            "measurements": [
                {
                    "uuid": str(uuid4()),
                    "weight": 74.0,
                    "unit": "kg",
                    "effectiveDateTime": "2023-01-01T10:00:00Z",  # Old date
                    "source": "patient-device"
                }
            ]
        }
    )


# Cleanup endpoint events
def get_cleanup_event():
    """Cleanup event with multiple historical measurements."""
    measurements = []
    base_date = datetime(2024, 1, 1)

    # Generate 100 measurements over 30 days
    for i in range(100):
        timestamp = base_date + timedelta(days=i/3.3, hours=i%24)
        weight = 75.0 + (i % 10) * 0.2 - 1.0  # Vary between 74-76
        measurements.append({
            "uuid": str(uuid4()),
            "weight": weight,
            "unit": "kg",
            "effectiveDateTime": timestamp.isoformat() + "Z",
            "source": ["patient-device", "care-team-upload", "patient-upload"][i % 3]
        })

    return create_api_gateway_event(
        path="/api/v1/cleanup/{userId}",
        method="POST",
        path_parameters={"userId": "cleanup-user"},
        body={
            "measurements": measurements,
            "userProfile": {
                "height": 175,
                "heightUnit": "cm",
                "dateOfBirth": "1990-01-01",
                "gender": "M"
            },
            "options": {
                "resetState": True,
                "includeQualityScores": True
            }
        }
    )


# Replay endpoint events
def get_replay_event():
    """Replay event with historical measurements."""
    base_time = datetime(2024, 1, 15)
    measurements = []

    for i in range(10):
        timestamp = base_time + timedelta(hours=i*6)
        measurements.append({
            "uuid": str(uuid4()),
            "weight": 75.0 + i * 0.1,
            "unit": "kg",
            "effectiveDateTime": timestamp.isoformat() + "Z",
            "source": "patient-device"
        })

    return create_api_gateway_event(
        path="/api/v1/replay/{userId}",
        method="POST",
        path_parameters={"userId": "replay-user"},
        body={
            "replayFromTimestamp": base_time.isoformat() + "Z",
            "measurements": measurements,
            "options": {
                "useSnapshot": True,
                "createNewSnapshot": True
            }
        }
    )


# State management events
def get_state_event():
    """Get state event."""
    return create_api_gateway_event(
        path="/api/v1/state/{userId}",
        method="GET",
        path_parameters={"userId": "test-user-001"}
    )


def get_delete_state_event():
    """Delete state event."""
    return create_api_gateway_event(
        path="/api/v1/state/{userId}",
        method="DELETE",
        path_parameters={"userId": "test-user-001"}
    )


# Edge cases and error scenarios
def get_invalid_weight_event():
    """Event with invalid weight value."""
    return create_api_gateway_event(
        path="/api/v1/process/{userId}",
        method="POST",
        path_parameters={"userId": "test-user"},
        body={
            "measurements": [
                {
                    "uuid": str(uuid4()),
                    "weight": -10,  # Invalid negative weight
                    "unit": "kg",
                    "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
                    "source": "patient-device"
                }
            ]
        }
    )


def get_missing_required_field_event():
    """Event missing required fields."""
    return create_api_gateway_event(
        path="/api/v1/process/{userId}",
        method="POST",
        path_parameters={"userId": "test-user"},
        body={
            "measurements": [
                {
                    "uuid": str(uuid4()),
                    # Missing weight
                    "unit": "kg",
                    "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
                    "source": "patient-device"
                }
            ]
        }
    )


def get_malformed_json_event():
    """Event with malformed JSON."""
    event = create_api_gateway_event(
        path="/api/v1/process/{userId}",
        method="POST",
        path_parameters={"userId": "test-user"},
        body=None
    )
    event["body"] = '{"measurements": [}}'  # Invalid JSON
    return event


# Utility functions
def get_all_test_events():
    """Get all test events for comprehensive testing."""
    return {
        "process_single": get_process_event_single(),
        "process_batch": get_process_event_batch(),
        "process_conflict": get_process_event_historical_conflict(),
        "cleanup": get_cleanup_event(),
        "replay": get_replay_event(),
        "get_state": get_state_event(),
        "delete_state": get_delete_state_event(),
        "invalid_weight": get_invalid_weight_event(),
        "missing_field": get_missing_required_field_event(),
        "malformed_json": get_malformed_json_event()
    }


def save_events_to_files(output_dir: str = "test-events"):
    """Save all test events to JSON files for manual testing."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    events = get_all_test_events()
    for name, event in events.items():
        filepath = os.path.join(output_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(event, f, indent=2, default=str)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    # Save all events to files
    save_events_to_files()

    # Print example event
    print("Example Process Event:")
    print(json.dumps(get_process_event_single(), indent=2, default=str))