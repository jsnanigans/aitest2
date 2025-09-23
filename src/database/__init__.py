"""Database module initialization."""

import os
from typing import Optional

from .base import StateStore
from .memory_store import InMemoryStateStore
from .database import ProcessorStateDB  # For backward compatibility

# Singleton instance
_db_instance: Optional[StateStore] = None


def get_state_db(backend: str = None) -> StateStore:
    """
    Get or create state database instance.

    Args:
        backend: 'memory', 'dynamodb', or None for auto-detection

    Returns:
        StateStore instance
    """
    global _db_instance

    if _db_instance is None:
        if backend is None:
            backend = os.getenv('DB_BACKEND', 'memory')

        if backend == 'dynamodb':
            # Import only when needed to avoid AWS SDK dependency
            try:
                from .dynamodb_store import DynamoDBStateStore
                _db_instance = DynamoDBStateStore()
            except ImportError:
                # Fallback to memory if DynamoDB not available
                _db_instance = InMemoryStateStore()
        else:
            _db_instance = InMemoryStateStore()

    return _db_instance


def reset_db_instance():
    """Reset the singleton instance (for testing)."""
    global _db_instance
    _db_instance = None


# For backward compatibility - alias the old name
def reset_db():
    """Reset the global database instance (useful for testing)."""
    reset_db_instance()