"""Database stores for Lambda deployment."""

from .base import StateStore
from .memory import MemoryStateStore
from .dynamodb import DynamoDBStateStore


def get_state_db(backend: str = "dynamodb") -> StateStore:
    """Factory function to get appropriate state store.

    Args:
        backend: Storage backend type ('dynamodb' or 'memory')

    Returns:
        StateStore instance
    """
    if backend == "dynamodb":
        return DynamoDBStateStore()
    elif backend == "memory":
        return MemoryStateStore()
    else:
        raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    'StateStore',
    'MemoryStateStore',
    'DynamoDBStateStore',
    'get_state_db',
]