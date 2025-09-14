"""
Utility functions for weight stream processor.
Consolidates logging and general utilities.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


# ============================================================================
# Logging Utilities (from logging_utils.py)
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
            **kwargs
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


# ============================================================================
# Visualization Logging (from viz_logger.py)
# ============================================================================

class VizLogger:
    """Simple logger for visualization modules."""
    
    def __init__(self, verbosity: Optional[int] = None):
        self.verbosity = verbosity if verbosity is not None else 0
        
    def debug(self, msg: str):
        if self.verbosity >= 2:
            print(f"[DEBUG] {msg}")
            
    def info(self, msg: str):
        if self.verbosity >= 1:
            print(f"[INFO] {msg}")
            
    def warning(self, msg: str):
        print(f"[WARNING] {msg}")
        
    def error(self, msg: str):
        print(f"[ERROR] {msg}", file=sys.stderr)


_viz_logger = None
_verbosity_level = 0


def get_logger() -> VizLogger:
    """Get the global visualization logger instance."""
    global _viz_logger
    if _viz_logger is None:
        _viz_logger = VizLogger(_verbosity_level)
    return _viz_logger


def set_verbosity(level: int):
    """Set the global verbosity level."""
    global _verbosity_level, _viz_logger
    _verbosity_level = level
    if _viz_logger is not None:
        _viz_logger.verbosity = level


# ============================================================================
# General Utilities
# ============================================================================

def format_timestamp(ts: Any) -> str:
    """Format timestamp for display."""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except:
            return str(ts)
    
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    
    return str(ts)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def calculate_percentage(part: float, whole: float, decimals: int = 1) -> str:
    """Calculate percentage with formatting."""
    if whole == 0:
        return "0.0%"
    percentage = (part / whole) * 100
    return f"{percentage:.{decimals}f}%"


def truncate_string(s: str, max_length: int = 80, suffix: str = "...") -> str:
    """Truncate string to maximum length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def ensure_list(value: Any) -> list:
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Merge two dictionaries, with dict2 values taking precedence."""
    result = dict1.copy()
    result.update(dict2)
    return result


# ============================================================================
# Export all utilities
# ============================================================================

__all__ = [
    # Logging
    'LogLevel',
    'StructuredLogger',
    'PerformanceTimer',
    'processor_logger',
    'validation_logger',
    'kalman_logger',
    
    # Visualization logging
    'VizLogger',
    'get_logger',
    'set_verbosity',
    
    # General utilities
    'format_timestamp',
    'safe_divide',
    'calculate_percentage',
    'truncate_string',
    'format_file_size',
    'ensure_list',
    'merge_dicts',
]