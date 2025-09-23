"""AWS Lambda handler for weight processor service."""

import json
import logging
import os
from typing import Dict, Any

import numpy as np

from .api.models import ProcessRequest, CleanupRequest, ReplayRequest
from .services.weight_processor_service import WeightProcessorService, HistoricalConflictError
from .config.config_manager import ConfigManager
from .database import get_state_db

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

# Initialize services (reused across invocations)
_service = None


def get_service() -> WeightProcessorService:
    """Get or create service instance."""
    global _service
    if _service is None:
        state_store = get_state_db(os.getenv('DB_BACKEND', 'memory'))
        config = ConfigManager.load_config('env' if os.getenv('AWS_LAMBDA_FUNCTION_NAME') else 'file')
        _service = WeightProcessorService(state_store, config)
    return _service


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler.

    Routes requests to appropriate handlers based on path and method.
    """
    try:
        # Log the event for debugging
        logger.debug(f"Received event: {json.dumps(event)}")

        # Extract routing information
        resource = event.get('resource', '')
        http_method = event.get('httpMethod', '')

        # Route to appropriate handler
        if resource == '/api/v1/process/{userId}' and http_method == 'POST':
            return handle_process(event)
        elif resource == '/api/v1/cleanup/{userId}' and http_method == 'POST':
            return handle_cleanup(event)
        elif resource == '/api/v1/replay/{userId}' and http_method == 'POST':
            return handle_replay(event)
        elif resource == '/api/v1/state/{userId}' and http_method == 'GET':
            return handle_get_state(event)
        elif resource == '/api/v1/state/{userId}' and http_method == 'DELETE':
            return handle_delete_state(event)
        else:
            return error_response(404, "Not Found")

    except Exception as e:
        logger.exception("Unhandled error in Lambda handler")
        return error_response(500, f"Internal server error: {str(e)}")


def handle_process(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle process endpoint."""
    try:
        # Extract user ID and request body
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])

        # Parse and validate request
        request = ProcessRequest(**body)

        # Process measurements
        service = get_service()
        response = service.process_batch(user_id, request.measurements)

        return success_response(response.model_dump())

    except HistoricalConflictError as e:
        return conflict_response(e.to_dict())
    except ValueError as e:
        return error_response(400, f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Error processing measurements for user")
        return error_response(500, f"Processing error: {str(e)}")


def handle_cleanup(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle cleanup endpoint."""
    try:
        # Extract user ID and request body
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])

        # Parse and validate request
        request = CleanupRequest(**body)

        # Perform cleanup
        service = get_service()
        response = service.cleanup(
            user_id,
            request.measurements,
            request.options.reset_state
        )

        return success_response(response.model_dump())

    except ValueError as e:
        return error_response(400, f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Error in cleanup for user")
        return error_response(500, f"Cleanup error: {str(e)}")


def handle_replay(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle replay endpoint."""
    try:
        # Extract user ID and request body
        user_id = event['pathParameters']['userId']
        body = json.loads(event['body'])

        # Parse and validate request
        request = ReplayRequest(**body)

        # For now, return not implemented
        # Replay service would be implemented separately
        return error_response(501, "Replay not yet implemented")

    except ValueError as e:
        return error_response(400, f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Error in replay for user")
        return error_response(500, f"Replay error: {str(e)}")


def handle_get_state(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get state endpoint."""
    try:
        user_id = event['pathParameters']['userId']
        state_store = get_state_db(os.getenv('DB_BACKEND', 'memory'))
        state = state_store.get_state(user_id)

        if state is None:
            return error_response(404, f"State not found for user {user_id}")

        # Convert any numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        state = convert_numpy(state)
        return success_response(state)

    except Exception as e:
        logger.exception(f"Error getting state for user")
        return error_response(500, f"Error retrieving state: {str(e)}")


def handle_delete_state(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle delete state endpoint."""
    try:
        user_id = event['pathParameters']['userId']
        state_store = get_state_db(os.getenv('DB_BACKEND', 'memory'))
        success = state_store.delete_state(user_id)

        if success:
            return success_response({"message": f"State deleted for user {user_id}"})
        else:
            return error_response(404, f"State not found for user {user_id}")

    except Exception as e:
        logger.exception(f"Error deleting state for user")
        return error_response(500, f"Error deleting state: {str(e)}")


def success_response(body: Any, status_code: int = 200) -> Dict[str, Any]:
    """Create successful response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body, default=str)
    }


def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create error response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'error': message
        })
    }


def conflict_response(conflict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create conflict response."""
    return {
        'statusCode': 409,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(conflict_data, default=str)
    }