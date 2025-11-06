"""Database module initialization."""

import os
from typing import Optional

from .base import StateStore
from .memory_store import InMemoryStore

# Singleton instance
_db_instance: Optional[StateStore] = None


def get_state_db(backend: Optional[str] = None) -> StateStore:
    """
    Get or create state database instance.

    The backend is selected in this order:
    1. Explicit backend parameter
    2. DB_BACKEND environment variable
    3. Default to DynamoDB

    Args:
        backend: Database backend to use. Options:
            - "memory": In-memory store (fast, no persistence)
            - "dynamodb": DynamoDB store (default)
            - None: Use environment variable or default to DynamoDB

    Returns:
        StateStore instance

    Environment Variables:
        DB_BACKEND: Set to "memory" or "dynamodb"
        DYNAMODB_ENDPOINT: DynamoDB endpoint (for local development)
        DYNAMODB_TABLE_NAME: DynamoDB table name

    Examples:
        >>> # Use in-memory store
        >>> db = get_state_db(backend="memory")

        >>> # Use DynamoDB (production)
        >>> db = get_state_db(backend="dynamodb")

        >>> # Use environment variable
        >>> os.environ["DB_BACKEND"] = "memory"
        >>> db = get_state_db()
    """
    global _db_instance

    if _db_instance is None:
        # Determine backend
        selected_backend = backend or os.getenv("DB_BACKEND", "dynamodb")

        if selected_backend == "memory":
            import logging
            logging.info("Using InMemoryStore for state storage")
            _db_instance = InMemoryStore()
        elif selected_backend == "dynamodb":
            try:
                from .dynamodb_store import DynamoDBStateStore
                _db_instance = DynamoDBStateStore()
            except ImportError as e:
                import logging
                logging.error(
                    "DynamoDB is required but boto3 not found. "
                    "Install with: pip install boto3 or use backend='memory'"
                )
                raise ImportError(
                    "boto3 is required for DynamoDB operations. "
                    "Install it with: pip install boto3 or use backend='memory' "
                    "for in-memory storage"
                ) from e
        else:
            raise ValueError(
                f"Unknown database backend: {selected_backend}. "
                "Valid options: 'memory', 'dynamodb'"
            )

    return _db_instance


def set_state_db(db: StateStore) -> None:
    """
    Set the state database instance explicitly.

    Useful for testing or when you want to provide your own StateStore implementation.

    Args:
        db: StateStore instance to use

    Example:
        >>> from weight_processor_lib.core.database import InMemoryStore, set_state_db
        >>> custom_db = InMemoryStore()
        >>> set_state_db(custom_db)
    """
    global _db_instance
    # Clean up existing instance before replacing
    if _db_instance and hasattr(_db_instance, "close_connections"):
        _db_instance.close_connections()
    _db_instance = db


def reset_db_instance():
    """Reset the singleton instance (for testing)."""
    global _db_instance
    # Clean up existing instance before resetting
    if _db_instance and hasattr(_db_instance, "close_connections"):
        _db_instance.close_connections()
    _db_instance = None


# For backward compatibility - alias the old name
def reset_db():
    """Reset the global database instance (useful for testing)."""
    reset_db_instance()


__all__ = [
    "StateStore",
    "InMemoryStore",
    "get_state_db",
    "set_state_db",
    "reset_db_instance",
    "reset_db",
]
