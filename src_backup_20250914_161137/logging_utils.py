"""
Structured logging utilities for weight processor.
Simple, focused logging without verbose debug output.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


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
            **kwargs
        }
        
        # For now, just print JSON to stderr for structured logging
        # In production, this would go to a logging service
        if level == LogLevel.ERROR:
            print(json.dumps(log_entry), file=sys.stderr)
        elif level == LogLevel.METRIC:
            # Metrics go to stdout for collection
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
        self._log(LogLevel.METRIC, f"Metric: {metric_name}", 
                 metric=metric_name, value=value, tags=tags)


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
            self.logger.metric(f"{self.operation}_duration_ms", duration_ms,
                             operation=self.operation)


# Global logger instances
processor_logger = StructuredLogger("processor")
validation_logger = StructuredLogger("validation")
kalman_logger = StructuredLogger("kalman")