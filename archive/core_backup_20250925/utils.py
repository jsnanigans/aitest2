"""
Core utility functions for weight stream processor.
Minimal utilities without external dependencies.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from enum import Enum


# ============================================================================
# Logging Utilities
# ============================================================================


class LogLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    METRIC = "METRIC"


class StructuredLogger:
    """Simple structured logger for production use."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        if not self.enabled:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "logger": self.name,
            "message": message,
            **kwargs,
        }

        if level == LogLevel.ERROR:
            print(json.dumps(log_entry), file=sys.stderr)
        elif level == LogLevel.METRIC:
            print(json.dumps(log_entry))

    def error(self, message: str, **kwargs):
        """Log an error."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info."""
        self._log(LogLevel.INFO, message, **kwargs)

    def metric(self, metric_name: str, value: float, **tags):
        """Log a metric."""
        self._log(
            LogLevel.METRIC,
            f"Metric: {metric_name}",
            metric=metric_name,
            value=value,
            tags=tags,
        )


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, logger: StructuredLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
            self.logger.metric(
                f"{self.operation}_duration_ms", duration_ms, operation=self.operation
            )


# Global logger instances
processor_logger = StructuredLogger("processor")
validation_logger = StructuredLogger("validation")
kalman_logger = StructuredLogger("kalman")


# ============================================================================
# Core Utility Functions
# ============================================================================


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration structure and required fields.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check for required top-level sections
    required_sections = ["kalman", "quality_scoring", "processing"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required config section: {section}")

    # Validate kalman config if present
    if "kalman" in config:
        kalman_config = config["kalman"]
        if "process_noise" not in kalman_config:
            errors.append("kalman.process_noise is required")
        if "observation_noise" not in kalman_config:
            errors.append("kalman.observation_noise is required")

    # Validate quality scoring config if present
    if "quality_scoring" in config:
        qs_config = config["quality_scoring"]
        if "component_weights" not in qs_config:
            errors.append("quality_scoring.component_weights is required")

    return len(errors) == 0, errors


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_timestamp(dt: datetime) -> str:
    """Format datetime for consistent string representation.

    Args:
        dt: Datetime to format

    Returns:
        ISO format string
    """
    return dt.isoformat() if dt else ""
