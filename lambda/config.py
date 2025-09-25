"""Configuration management for Lambda environment.

Reads configuration from environment variables for AWS Lambda deployment.
"""

import os
import json
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Configuration dictionary
    """
    # Base configuration structure
    config = {
        "kalman": {
            "process_noise": float(os.getenv("KALMAN_PROCESS_NOISE", "0.1")),
            "observation_noise": float(os.getenv("KALMAN_OBSERVATION_NOISE", "1.0")),
            "initial_covariance": float(os.getenv("KALMAN_INITIAL_COVARIANCE", "1.0")),
            "adaptation_enabled": os.getenv("KALMAN_ADAPTATION_ENABLED", "true").lower()
            == "true",
            "reset_window_hours": int(os.getenv("KALMAN_RESET_WINDOW_HOURS", "720")),
        },
        "quality_scoring": {
            "enabled": os.getenv("QUALITY_SCORING_ENABLED", "true").lower() == "true",
            "component_weights": {
                "kalman_deviation": float(os.getenv("QS_WEIGHT_KALMAN", "0.25")),
                "temporal_consistency": float(os.getenv("QS_WEIGHT_TEMPORAL", "0.20")),
                "source_reliability": float(os.getenv("QS_WEIGHT_SOURCE", "0.20")),
                "physiological_plausibility": float(
                    os.getenv("QS_WEIGHT_PHYSIO", "0.15")
                ),
                "statistical_position": float(
                    os.getenv("QS_WEIGHT_STATISTICAL", "0.10")
                ),
                "measurement_frequency": float(
                    os.getenv("QS_WEIGHT_FREQUENCY", "0.10")
                ),
            },
            "thresholds": {
                "high_quality": float(os.getenv("QS_THRESHOLD_HIGH", "0.8")),
                "medium_quality": float(os.getenv("QS_THRESHOLD_MEDIUM", "0.5")),
                "outlier_override": float(
                    os.getenv("QS_THRESHOLD_OUTLIER_OVERRIDE", "0.85")
                ),
            },
        },
        "processing": {
            "extreme_threshold": float(
                os.getenv("PROCESSING_EXTREME_THRESHOLD", "0.15")
            ),
            "max_daily_change_kg": float(
                os.getenv("PROCESSING_MAX_DAILY_CHANGE", "2.0")
            ),
            "min_weight_kg": float(os.getenv("PROCESSING_MIN_WEIGHT", "20.0")),
            "max_weight_kg": float(os.getenv("PROCESSING_MAX_WEIGHT", "500.0")),
        },
        "replay": {
            "enabled": os.getenv("REPLAY_ENABLED", "true").lower() == "true",
            "buffer_hours": int(os.getenv("REPLAY_BUFFER_HOURS", "72")),
            "trigger_mode": os.getenv("REPLAY_TRIGGER_MODE", "time_based"),
            "outlier_detection": {
                "methods": json.loads(
                    os.getenv("REPLAY_OUTLIER_METHODS", '["iqr", "mad"]')
                ),
                "iqr_multiplier": float(os.getenv("REPLAY_IQR_MULTIPLIER", "1.5")),
                "mad_threshold": float(os.getenv("REPLAY_MAD_THRESHOLD", "3.0")),
            },
            "safety": {
                "max_replay_attempts": int(os.getenv("REPLAY_MAX_ATTEMPTS", "3")),
                "min_measurements": int(os.getenv("REPLAY_MIN_MEASUREMENTS", "10")),
                "rollback_on_error": os.getenv(
                    "REPLAY_ROLLBACK_ON_ERROR", "true"
                ).lower()
                == "true",
            },
        },
        "circuit_breaker": {
            "enabled": os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true",
            "failure_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            "recovery_timeout": int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60")),
            "half_open_max_calls": int(os.getenv("CIRCUIT_BREAKER_HALF_OPEN", "3")),
        },
    }

    return config


def get_db_config() -> Dict[str, Any]:
    """Get database configuration.

    Returns:
        Database configuration dictionary
    """
    backend = os.getenv("DB_BACKEND", "dynamodb")

    if backend == "dynamodb":
        return {
            "backend": "dynamodb",
            "table_name": os.getenv("DYNAMODB_TABLE", "weight-processor-state"),
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "endpoint_url": os.getenv("DYNAMODB_ENDPOINT"),  # For local testing
        }
    else:
        return {
            "backend": "memory",
        }


def get_service_config() -> Dict[str, str]:
    """Get service-level configuration.

    Returns:
        Service configuration dictionary
    """
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "version": os.getenv("SERVICE_VERSION", "1.0.0"),
        "region": os.getenv("AWS_REGION", "us-east-1"),
    }
