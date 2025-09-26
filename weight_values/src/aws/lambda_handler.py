"""AWS Lambda handler - Weight Processor Service API v2."""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

from src.aws.api.models import (
    ProcessRequest, CleanupRequest, ReplayRequest,
    StandardResponse, ErrorResponse, HealthStatus,
    ProcessResponseData, CleanupResponseData, ReplayResponseData, StateInfo,
    MeasurementResult, ApiMeta,
    create_success_response, create_error_response
)
from src.aws.services.weight_processor_service import (
    WeightProcessorService,
    HistoricalConflictError,
)
from src.aws.config.config_manager import ConfigManager
from src.core.database import get_state_db

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Initialize services (reused across invocations)
_service = None


def get_service() -> WeightProcessorService:
    """Get or create service instance."""
    global _service
    if _service is None:
        state_store = get_state_db()
        config = ConfigManager.load_config(
            "env" if os.getenv("AWS_LAMBDA_FUNCTION_NAME") else "file"
        )
        _service = WeightProcessorService(state_store, config)
    return _service


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler v2 with improved error handling and consistent responses.

    Routes requests to appropriate handlers based on path and method.
    """
    request_id = generate_request_id()

    try:
        # Log the event for debugging
        logger.debug(f"[{request_id}] Received event: {json.dumps(event)}")

        # Extract routing information
        resource = event.get("resource", "")
        path = event.get("path", "")
        http_method = event.get("httpMethod", "")

        # Log the routing info
        logger.info(
            f"[{request_id}] Routing - resource: {resource}, path: {path}, method: {http_method}"
        )

        # Route to appropriate handler
        if (resource == "/api/v1/health" or path == "/api/v1/health") and http_method == "GET":
            return handle_health(event, request_id)
        elif resource == "/api/v1/process/{userId}" and http_method == "POST":
            return handle_process(event, request_id)
        elif resource == "/api/v1/cleanup/{userId}" and http_method == "POST":
            return handle_cleanup(event, request_id)
        elif resource == "/api/v1/replay/{userId}" and http_method == "POST":
            return handle_replay(event, request_id)
        elif resource == "/api/v1/state/{userId}" and http_method == "GET":
            return handle_get_state(event, request_id)
        elif resource == "/api/v1/state/{userId}" and http_method == "DELETE":
            return handle_delete_state(event, request_id)
        else:
            return format_error_response(
                404, "NOT_FOUND", "Endpoint not found", request_id=request_id
            )

    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error in Lambda handler")
        return format_error_response(
            500, "INTERNAL_ERROR", "An unexpected error occurred",
            details={"error_type": type(e).__name__},
            request_id=request_id
        )


def handle_health(event: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Handle health check endpoint with improved response format."""
    try:
        # Check database connectivity
        db_status = "healthy"
        db_backend = os.getenv("DB_BACKEND", "memory")

        try:
            state_store = get_state_db()
            _ = state_store.get_state("health_check_user")
        except Exception as e:
            db_status = "unhealthy"
            logger.warning(f"[{request_id}] Database health check failed: {e}")

        # Get configuration status
        config_status = "healthy"
        config_loaded = False
        try:
            config = ConfigManager.load_config(
                "env" if os.getenv("AWS_LAMBDA_FUNCTION_NAME") else "file"
            )
            config_loaded = bool(config)
        except Exception as e:
            config_status = "unhealthy"
            logger.warning(f"[{request_id}] Config health check failed: {e}")

        # Build health response
        health_data = HealthStatus(
            status="healthy" if db_status == "healthy" and config_status == "healthy" else "degraded",
            environment=os.getenv("ENVIRONMENT", "local"),
            components={
                "database": {"status": db_status, "backend": db_backend},
                "configuration": {"status": config_status, "loaded": config_loaded},
                "processing": {
                    "kalman_enabled": os.getenv("KALMAN_ENABLED", "true") == "true",
                    "quality_scoring_enabled": os.getenv("QUALITY_SCORING_ENABLED", "true") == "true",
                    "outlier_detection_enabled": os.getenv("OUTLIER_DETECTION_ENABLED", "true") == "true",
                    "replay_enabled": os.getenv("REPLAY_ENABLED", "true") == "true",
                },
            },
            runtime={
                "function_name": os.getenv("AWS_LAMBDA_FUNCTION_NAME", "local"),
                "function_version": os.getenv("AWS_LAMBDA_FUNCTION_VERSION", "local"),
                "region": os.getenv("AWS_REGION", "local"),
                "memory_size": os.getenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "local"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
            }
        )

        response = create_success_response(
            health_data.model_dump(),
            meta=ApiMeta(request_id=request_id)
        )
        return format_success_response(response)

    except Exception as e:
        logger.exception(f"[{request_id}] Error in health check")
        return format_error_response(
            503, "HEALTH_CHECK_FAILED", "Health check failed",
            details={"error": str(e)},
            request_id=request_id
        )


def handle_process(event: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Handle process endpoint with improved response format."""
    user_id = None

    try:
        # Extract user ID and request body
        user_id = event["pathParameters"]["userId"]
        body = json.loads(event.get("body", "{}"))

        # Parse and validate request
        try:
            request = ProcessRequest(**body)
        except Exception as e:
            return format_error_response(
                400, "VALIDATION_ERROR", "Invalid request format",
                details={"validation_errors": str(e)},
                suggestion="Check that all required fields are present and valid",
                request_id=request_id
            )

        # Process measurements
        service = get_service()

        # Fix: Wrap response properly to avoid NoneType issues
        try:
            response = service.process_batch(user_id, request.measurements)
        except AttributeError as e:
            if "NoneType" in str(e):
                # This is the outlier detection bug - provide better error
                logger.error(f"[{request_id}] Outlier detection error: {e}")
                return format_error_response(
                    500, "PROCESSING_ERROR", "Error in outlier detection module",
                    details={"error": "Internal state initialization error"},
                    suggestion="Try processing measurements one at a time",
                    request_id=request_id
                )
            raise

        # Build consistent response - using v2 attributes
        # The response from service is already ProcessResponseData, so we can use it directly
        response_data = response  # response is already ProcessResponseData from the service

        # If for some reason we need to reconstruct, ensure we use correct field names:
        # response_data = ProcessResponseData(
        #     user_id=user_id,
        #     measurements_processed=response.measurements_processed,
        #     measurements_accepted=response.measurements_accepted,
        #     measurements_rejected=response.measurements_rejected,
        #     results=response.results,
        #     state_update=response.state_update
        # )

        api_response = create_success_response(
            response_data.model_dump(),
            meta=ApiMeta(request_id=request_id)
        )
        return format_success_response(api_response)

    except HistoricalConflictError as e:
        return format_error_response(
            409, "HISTORICAL_CONFLICT", "Historical conflict detected",
            details=e.to_dict(),
            suggestion="Use replay endpoint to reprocess from the conflict point",
            request_id=request_id
        )
    except ValueError as e:
        return format_error_response(
            400, "INVALID_DATA", str(e),
            suggestion="Check measurement data format and values",
            request_id=request_id
        )
    except Exception as e:
        # Check for time gap issue
        if "502" in str(e) or "gap" in str(e).lower():
            logger.error(f"[{request_id}] Time gap processing error: {e}")
            return format_error_response(
                422, "TIME_GAP_ERROR", "Cannot process measurements with large time gaps",
                details={"error": str(e)},
                suggestion="Break measurements into smaller time windows or use replay mode",
                request_id=request_id
            )

        logger.exception(f"[{request_id}] Error processing measurements for user {user_id}")
        return format_error_response(
            500, "PROCESSING_ERROR", "Failed to process measurements",
            details={"error": str(e)},
            request_id=request_id
        )


def handle_cleanup(event: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Handle cleanup endpoint - no measurements required."""
    user_id = None

    try:
        # Extract user ID and request body
        user_id = event["pathParameters"]["userId"]
        body = json.loads(event.get("body", "{}"))

        # Parse and validate request - measurements NOT required
        try:
            request = CleanupRequest(**body)
        except Exception as e:
            return format_error_response(
                400, "VALIDATION_ERROR", "Invalid cleanup request",
                details={"validation_errors": str(e)},
                suggestion="Provide cleanup_type and optional settings",
                request_id=request_id
            )

        # Perform cleanup
        service = get_service()
        state_store = get_state_db()

        # Clear state based on cleanup type
        if request.cleanup_type == "reset_adaptive":
            # Reset adaptive parameters
            success = state_store.delete_state(user_id)
            message = "Adaptive parameters reset" if success else "No state to reset"
        elif request.cleanup_type == "clear_all":
            # Clear everything
            success = state_store.delete_state(user_id)
            message = "All state cleared" if success else "No state to clear"
        else:
            return format_error_response(
                400, "INVALID_CLEANUP_TYPE", f"Unknown cleanup type: {request.cleanup_type}",
                suggestion="Use 'reset_adaptive' or 'clear_all'",
                request_id=request_id
            )

        response_data = CleanupResponseData(
            user_id=user_id,
            cleanup_type=request.cleanup_type,
            state_cleared=success,
            message=message
        )

        api_response = create_success_response(
            response_data.model_dump(),
            meta=ApiMeta(request_id=request_id)
        )
        return format_success_response(api_response)

    except Exception as e:
        logger.exception(f"[{request_id}] Error in cleanup for user {user_id}")
        return format_error_response(
            500, "CLEANUP_ERROR", "Failed to perform cleanup",
            details={"error": str(e)},
            request_id=request_id
        )


def handle_replay(event: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Handle replay endpoint with correct field naming."""
    user_id = None

    try:
        # Extract user ID and request body
        user_id = event["pathParameters"]["userId"]
        body = json.loads(event.get("body", "{}"))

        # Parse and validate request - using correct field name
        try:
            request = ReplayRequest(**body)
        except Exception as e:
            return format_error_response(
                400, "VALIDATION_ERROR", "Invalid replay request",
                details={"validation_errors": str(e)},
                suggestion="Ensure 'replay_from_timestamp' field is present and valid",
                request_id=request_id
            )

        # Import replay service
        from src.aws.services.replay_service import replay_measurements

        # Get service dependencies
        state_store = get_state_db()
        config = ConfigManager.load_config(
            "env" if os.getenv("AWS_LAMBDA_FUNCTION_NAME") else "file"
        )

        # Run replay
        result = replay_measurements(
            user_id=user_id,
            measurements=request.measurements,
            replay_from=request.replay_from_timestamp,
            state_store=state_store,
            config=config,
        )

        if result["success"]:
            response_data = ReplayResponseData(
                user_id=user_id,
                replay_from=request.replay_from_timestamp,
                measurements_processed=result["processed_count"],
                measurements_accepted=result["accepted_count"],
                measurements_rejected=result["rejected_count"],
                results=[
                    MeasurementResult(
                        measurement_id=r.get("uuid"),
                        accepted=r.get("accepted", False),
                        quality_score=r.get("quality_score"),
                        rejection_reason=r.get("rejection_reason"),
                        processing_stage=r.get("stage"),
                        reset_triggered=r.get("reset_triggered", False)
                    ) for r in result.get("results", [])
                ]
            )

            api_response = create_success_response(
                response_data.model_dump(),
                meta=ApiMeta(request_id=request_id)
            )
            return format_success_response(api_response)
        else:
            return format_error_response(
                500, "REPLAY_FAILED", result.get("error", "Replay failed"),
                request_id=request_id
            )

    except Exception as e:
        logger.exception(f"[{request_id}] Error in replay for user {user_id}")
        return format_error_response(
            500, "REPLAY_ERROR", "Failed to replay measurements",
            details={"error": str(e)},
            request_id=request_id
        )


def handle_get_state(event: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Handle get state endpoint - includes user_id in response."""
    user_id = None

    try:
        user_id = event["pathParameters"]["userId"]
        state_store = get_state_db()
        state = state_store.get_state(user_id)

        if state is None:
            return format_error_response(
                404, "STATE_NOT_FOUND", f"No state found for user",
                request_id=request_id
            )

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

        # Build state info with user_id
        state_info = StateInfo(
            user_id=user_id,  # Include user_id in response
            current_weight=state.get("last_raw_weight"),
            previous_weight=None,  # Could extract from history if needed
            last_processed_at=state.get("last_timestamp"),
            measurements_count=int(state.get("measurements_since_reset", 0)),
            last_source=state.get("last_source"),
            adaptation_state=state.get("reset_type"),
            kalman_state={
                "state": state.get("last_state"),
                "covariance": state.get("last_covariance"),
                "parameters": state.get("kalman_params")
            }
        )

        # Include full state details
        response_data = {
            **state_info.model_dump(),
            "full_state": state  # Include original state for backward compatibility
        }

        api_response = create_success_response(
            response_data,
            meta=ApiMeta(request_id=request_id)
        )
        return format_success_response(api_response)

    except Exception as e:
        logger.exception(f"[{request_id}] Error getting state for user {user_id}")
        return format_error_response(
            500, "STATE_ERROR", "Failed to retrieve state",
            details={"error": str(e)},
            request_id=request_id
        )


def handle_delete_state(event: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Handle delete state endpoint with better response."""
    user_id = None

    try:
        user_id = event["pathParameters"]["userId"]
        state_store = get_state_db()

        # Check if state exists first
        existing_state = state_store.get_state(user_id)
        if existing_state is None:
            return format_error_response(
                404, "STATE_NOT_FOUND", "No state to delete",
                request_id=request_id
            )

        # Delete the state
        success = state_store.delete_state(user_id)

        response_data = {
            "user_id": user_id,
            "deleted": success,
            "message": f"State deleted for user {user_id}" if success else "Failed to delete state"
        }

        api_response = create_success_response(
            response_data,
            meta=ApiMeta(request_id=request_id)
        )
        return format_success_response(api_response, status_code=200 if success else 500)

    except Exception as e:
        logger.exception(f"[{request_id}] Error deleting state for user {user_id}")
        return format_error_response(
            500, "DELETE_ERROR", "Failed to delete state",
            details={"error": str(e)},
            request_id=request_id
        )


# ============= Response Formatting =============

def format_success_response(response: StandardResponse, status_code: int = 200) -> Dict[str, Any]:
    """Format successful API response for Lambda."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "X-Request-Id": response.meta.request_id or "",
        },
        "body": json.dumps(response.model_dump(), default=str),
    }


def format_error_response(
    status_code: int,
    error_code: str,
    message: str,
    field: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Format error response with helpful information."""

    error_response = create_error_response(
        code=error_code,
        message=message,
        field=field,
        details=details,
        suggestion=suggestion
    )

    if request_id:
        error_response.meta.request_id = request_id

    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "X-Request-Id": request_id or "",
        },
        "body": json.dumps(error_response.model_dump(), default=str),
    }