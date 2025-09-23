"""Local database implementations."""

from .sqlite_store import SQLiteStateStore


def get_state_db(backend: str = "sqlite") -> "SQLiteStateStore":
    """Get local database instance.

    Args:
        backend: Database backend (always 'sqlite' for local)

    Returns:
        SQLiteStateStore instance
    """
    return SQLiteStateStore()


__all__ = [
    'SQLiteStateStore',
    'get_state_db',
]