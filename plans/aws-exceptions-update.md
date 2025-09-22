# Exceptions Module Update for AWS

## Update to `src/exceptions.py`

Add these new exceptions to your existing exceptions.py file:

```python
"""Custom exceptions for weight processor."""

from typing import Dict, Any, Optional
from datetime import datetime


# Keep existing exceptions...


class HistoricalConflictError(Exception):
    """
    Raised when measurements are older than the last processed timestamp.
    Indicates that replay is required.
    """

    def __init__(self, conflict_response: 'HistoricalConflictResponse'):
        """
        Initialize with conflict details.

        Args:
            conflict_response: HistoricalConflictResponse object with details
        """
        self.conflict_response = conflict_response
        super().__init__(conflict_response.error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return self.conflict_response.dict()

    @property
    def replay_from(self) -> datetime:
        """Get the timestamp from which replay should start."""
        return self.conflict_response.details.replay_from_timestamp

    @property
    def conflicting_measurements(self) -> list:
        """Get list of conflicting measurement UUIDs."""
        return self.conflict_response.details.conflicting_measurements


class StateNotFoundError(Exception):
    """Raised when user state is not found in the database."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"State not found for user: {user_id}")


class StateCorruptionError(Exception):
    """Raised when state data is corrupted or invalid."""

    def __init__(self, user_id: str, details: str):
        self.user_id = user_id
        self.details = details
        super().__init__(f"State corruption for user {user_id}: {details}")


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""

    def __init__(self, backend: str, details: str):
        self.backend = backend
        self.details = details
        super().__init__(f"Database connection failed ({backend}): {details}")


class SnapshotNotFoundError(Exception):
    """Raised when a required snapshot is not found."""

    def __init__(self, user_id: str, timestamp: datetime):
        self.user_id = user_id
        self.timestamp = timestamp
        super().__init__(
            f"No snapshot found for user {user_id} before {timestamp.isoformat()}"
        )


class ReplayError(Exception):
    """Base class for replay-related errors."""

    def __init__(self, message: str, user_id: str = None):
        self.user_id = user_id
        super().__init__(message)


class ReplayValidationError(ReplayError):
    """Raised when replay validation fails."""

    def __init__(self, user_id: str, reason: str):
        super().__init__(f"Replay validation failed: {reason}", user_id)


class ReplayExecutionError(ReplayError):
    """Raised when replay execution fails."""

    def __init__(self, user_id: str, step: str, details: str):
        self.step = step
        super().__init__(f"Replay failed at {step}: {details}", user_id)
```

## Usage Examples

### In Service Layer

```python
from ..exceptions import HistoricalConflictError, StateNotFoundError

class WeightProcessorService:

    def process_batch(self, user_id: str, measurements: List[Measurement]):
        # Check for conflicts
        conflict = self._check_historical_conflict(user_id, measurements)
        if conflict:
            raise HistoricalConflictError(conflict)

        # Get state
        state = self.state_store.get_state(user_id)
        if not state and self.config.get('require_existing_state'):
            raise StateNotFoundError(user_id)
```

### In Lambda Handler

```python
from .exceptions import (
    HistoricalConflictError,
    StateNotFoundError,
    ConfigurationError
)

def handle_process(event):
    try:
        # Process measurements
        response = service.process_batch(user_id, measurements)
        return success_response(response)

    except HistoricalConflictError as e:
        # Return 409 Conflict
        return {
            'statusCode': 409,
            'body': json.dumps(e.to_dict())
        }

    except StateNotFoundError as e:
        # Return 404 Not Found
        return {
            'statusCode': 404,
            'body': json.dumps({'error': str(e)})
        }

    except ConfigurationError as e:
        # Return 500 Internal Server Error
        logger.error(f"Configuration error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Service misconfiguration'})
        }
```

### In DynamoDB Store

```python
from ..exceptions import DatabaseConnectionError, StateCorruptionError

class DynamoDBStateStore(StateStore):

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.table.get_item(...)

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                raise DatabaseConnectionError(
                    'dynamodb',
                    f"Table {self.table_name} not found"
                )
            raise

        except Exception as e:
            raise StateCorruptionError(
                user_id,
                f"Failed to deserialize state: {e}"
            )
```

## Testing Exceptions

```python
import pytest
from datetime import datetime
from src.exceptions import HistoricalConflictError
from src.api.models import HistoricalConflictResponse, HistoricalConflictDetails

class TestExceptions:

    def test_historical_conflict_error(self):
        """Test HistoricalConflictError creation and properties."""
        # Create conflict response
        details = HistoricalConflictDetails(
            earliest_measurement_timestamp=datetime(2024, 1, 1),
            last_processed_timestamp=datetime(2024, 1, 10),
            replay_from_timestamp=datetime(2024, 1, 1),
            conflicting_measurements=['uuid1', 'uuid2']
        )

        response = HistoricalConflictResponse(
            error="Test error",
            details=details
        )

        # Create exception
        error = HistoricalConflictError(response)

        # Test properties
        assert error.replay_from == datetime(2024, 1, 1)
        assert error.conflicting_measurements == ['uuid1', 'uuid2']
        assert 'Test error' in str(error)

    def test_state_not_found_error(self):
        """Test StateNotFoundError."""
        from src.exceptions import StateNotFoundError

        error = StateNotFoundError('user123')
        assert error.user_id == 'user123'
        assert 'user123' in str(error)
```

## Error Response Utilities

Create a utility module for consistent error responses:

```python
# src/utils/error_responses.py

from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


def create_error_response(status_code: int, error: Exception) -> Dict[str, Any]:
    """
    Create standardized error response based on exception type.

    Args:
        status_code: HTTP status code
        error: Exception instance

    Returns:
        Lambda response dictionary
    """
    # Log the error
    if status_code >= 500:
        logger.error(f"Server error: {error}", exc_info=True)
    else:
        logger.warning(f"Client error: {error}")

    # Create response body
    if hasattr(error, 'to_dict'):
        body = error.to_dict()
    else:
        body = {
            'error': str(error),
            'type': error.__class__.__name__
        }

    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body, default=str)
    }


def get_status_code_for_exception(error: Exception) -> int:
    """
    Map exception types to HTTP status codes.

    Args:
        error: Exception instance

    Returns:
        Appropriate HTTP status code
    """
    from ..exceptions import (
        HistoricalConflictError,
        StateNotFoundError,
        ConfigurationError,
        DatabaseConnectionError,
        ValidationError
    )

    error_mapping = {
        HistoricalConflictError: 409,  # Conflict
        StateNotFoundError: 404,        # Not Found
        ValidationError: 400,            # Bad Request
        ValueError: 400,                 # Bad Request
        ConfigurationError: 500,         # Internal Server Error
        DatabaseConnectionError: 503,   # Service Unavailable
    }

    for error_type, status_code in error_mapping.items():
        if isinstance(error, error_type):
            return status_code

    # Default to 500 for unknown errors
    return 500
```

## Integration with Lambda Handler

Update the Lambda handler to use the new exception handling:

```python
# src/lambda_handler.py

from .utils.error_responses import create_error_response, get_status_code_for_exception
from .exceptions import HistoricalConflictError, StateNotFoundError

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler with proper error handling."""
    try:
        # Route to appropriate handler
        return route_request(event)

    except (HistoricalConflictError, StateNotFoundError, ValueError) as e:
        # Known client errors
        status_code = get_status_code_for_exception(e)
        return create_error_response(status_code, e)

    except Exception as e:
        # Unknown server errors
        logger.exception("Unhandled error in Lambda handler")
        return create_error_response(500, e)
```

## Benefits

1. **Type Safety**: Strong exception types make error handling explicit
2. **API Consistency**: Standardized error responses across all endpoints
3. **Debugging**: Better error tracking and logging
4. **Testing**: Easier to test error conditions
5. **Documentation**: Self-documenting error conditions