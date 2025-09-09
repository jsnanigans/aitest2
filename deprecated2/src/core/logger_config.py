import logging
import logging.config
import os
import sys
from .config_loader import load_config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if sys.stdout.isatty():  # Only colorize for terminal output
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{log_color}{record.levelname:8}{self.RESET}"
        return super().format(record)


def setup_logging():
    """
    Set up logging configuration based on config.toml settings.
    Creates log directory if it doesn't exist.
    """
    config = load_config()
    log_config = config.get('logging', {})

    # Default logging configuration
    log_level = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file', 'logs/app.log')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stdout configuration
    stdout_enabled = log_config.get('stdout_enabled', True)
    stdout_level = log_config.get('stdout_level', 'INFO').upper()

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': log_format
            },
            'colored': {
                '()': ColoredFormatter,
                'format': '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                'datefmt': '%H:%M:%S'
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'level': log_level,
                'formatter': 'standard',
                'filename': log_file,
                'mode': 'w',
            },
        },
        'loggers': {
            '': {
                'handlers': ['file'],
                'level': log_level,
                'propagate': False
            }
        }
    }

    # Add console handler if stdout is enabled
    if stdout_enabled:
        # Use colored formatter for console if terminal supports it
        formatter = 'colored' if sys.stdout.isatty() else 'standard'
        logging_config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': stdout_level,
            'formatter': formatter,
            'stream': 'ext://sys.stdout'
        }
        logging_config['loggers']['']['handlers'].append('console')

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

    status_msg = f"Logging system initialized - File: {log_file} (Level: {log_level})"
    if stdout_enabled:
        status_msg += f", Stdout: enabled (Level: {stdout_level})"
    else:
        status_msg += ", Stdout: disabled"

    logger.info(status_msg)
    return logger

def get_logger(name):
    """
    Get a logger instance with the specified name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
