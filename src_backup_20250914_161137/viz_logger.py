"""
Centralized logging for visualization modules.
Controls verbosity and provides consistent output formatting.
"""

import os
from typing import Optional

class VizLogger:
    """Simple logger for visualization modules."""
    
    # Verbosity levels
    SILENT = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    
    def __init__(self, verbosity: Optional[int] = None):
        """Initialize logger with verbosity level."""
        if verbosity is None:
            # Check environment variable or default to MINIMAL
            verbosity = int(os.environ.get("VIZ_VERBOSITY", "1"))
        self.verbosity = verbosity
        self.suppress_next = False
    
    def debug(self, message: str):
        """Debug messages - only shown in VERBOSE mode."""
        if self.verbosity >= self.VERBOSE:
            print(f"[DEBUG] {message}")
    
    def info(self, message: str):
        """Info messages - shown in NORMAL and above."""
        if self.verbosity >= self.NORMAL:
            print(message)
    
    def progress(self, message: str):
        """Progress messages - shown in MINIMAL and above."""
        if self.verbosity >= self.MINIMAL:
            print(message, end="", flush=True)
    
    def success(self, message: str):
        """Success messages - always shown unless SILENT."""
        if self.verbosity > self.SILENT:
            print(message)
    
    def warning(self, message: str):
        """Warning messages - always shown unless SILENT."""
        if self.verbosity > self.SILENT:
            print(f"Warning: {message}")
    
    def error(self, message: str):
        """Error messages - always shown."""
        print(f"Error: {message}")
    
    def set_verbosity(self, level: int):
        """Change verbosity level."""
        self.verbosity = level

# Global logger instance
_logger = VizLogger()

def get_logger() -> VizLogger:
    """Get the global visualization logger."""
    return _logger

def set_verbosity(level: int):
    """Set global verbosity level."""
    _logger.set_verbosity(level)