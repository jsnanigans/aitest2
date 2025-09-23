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


def validate_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate configuration structure and values.
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required sections
    required_sections = ['data', 'processing', 'kalman', 'visualization', 'logging', 'quality_scoring']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: [{section}]")
    
    # Validate data section
    if 'data' in config:
        data = config['data']
        if 'csv_file' not in data:
            errors.append("Missing required field: data.csv_file")
        if 'output_dir' not in data:
            errors.append("Missing required field: data.output_dir")
    
    # Validate processing section
    if 'processing' in config:
        processing = config['processing']
        if 'extreme_threshold' in processing:
            threshold = processing['extreme_threshold']
            if not (0 < threshold < 1):
                errors.append(f"Invalid extreme_threshold: {threshold} (must be between 0 and 1)")
    
    # Validate kalman section
    if 'kalman' in config:
        kalman = config['kalman']
        required_kalman = ['initial_variance', 'transition_covariance_weight', 
                          'transition_covariance_trend', 'observation_covariance']
        for field in required_kalman:
            if field not in kalman:
                errors.append(f"Missing required Kalman field: {field}")
            elif kalman[field] <= 0:
                errors.append(f"Invalid Kalman {field}: must be positive")
    
    # Validate quality scoring weights
    if 'quality_scoring' in config:
        qs = config['quality_scoring']
        if 'component_weights' in qs:
            weights = qs['component_weights']
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                errors.append(f"Quality scoring weights must sum to 1.0, got {total:.3f}")
            for name, weight in weights.items():
                if not (0 <= weight <= 1):
                    errors.append(f"Invalid weight for {name}: {weight} (must be 0-1)")
    
    # Validate visualization verbosity
    if 'visualization' in config:
        viz = config['visualization']
        if 'verbosity' in viz:
            valid_verbosity = ['silent', 'minimal', 'normal', 'verbose']
            if viz['verbosity'] not in valid_verbosity:
                errors.append(f"Invalid verbosity: {viz['verbosity']} (must be one of {valid_verbosity})")
    
    # Validate adaptive noise
    if 'adaptive_noise' in config:
        noise = config['adaptive_noise']
        if 'default_multiplier' in noise:
            multiplier = noise['default_multiplier']
            if not (0.5 <= multiplier <= 5.0):
                errors.append(f"Invalid default_multiplier: {multiplier} (should be between 0.5 and 5.0)")
    
    return len(errors) == 0, errors


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
    
    # Configuration validation
    'validate_config',
    
    # General utilities
    'format_timestamp',
    'safe_divide',
]