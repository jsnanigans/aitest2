"""Database module initialization."""

import os
from typing import Optional

from .base import StateStore

# Singleton instance
_db_instance: Optional[StateStore] = None


def get_state_db() -> StateStore:
    """
    Get or create state database instance.
    Always uses DynamoDB for consistency between local and production.

    Returns:
        StateStore instance
    """
    global _db_instance

    if _db_instance is None:
        # Always use DynamoDB for consistency
        try:
            from .dynamodb_store import DynamoDBStateStore
            _db_instance = DynamoDBStateStore()
        except ImportError as e:
            import logging
            logging.error("DynamoDB is required. Please install boto3: pip install boto3")
            raise ImportError(
                "boto3 is required for database operations. "
                "Install it with: pip install boto3 or uv pip install boto3"
            ) from e

    return _db_instance


def reset_db_instance():
    """Reset the singleton instance (for testing)."""
    global _db_instance
    # Clean up existing instance before resetting
    if _db_instance and hasattr(_db_instance, 'close_connections'):
        _db_instance.close_connections()
    _db_instance = None


# For backward compatibility - alias the old name
def reset_db():
    """Reset the global database instance (useful for testing)."""
    reset_db_instance()