"""
Processor State Database
Manages persistent Kalman state for all users
Initially in-memory with JSON serialization, can be backed by files or database later
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


class ProcessorStateDB:
    """
    State database for weight processor.
    Stores and retrieves Kalman state for each user.

    State includes:
    - Kalman filter parameters
    - Last state vector
    - Last covariance matrix
    - Last timestamp
    - Adapted parameters
    - Initialization buffer
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize state database.

        Args:
            storage_path: Optional path for persistent storage (future feature)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.states = {}  # In-memory store: {user_id: state_dict}

        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve state for a user.

        Returns:
            State dictionary or None if user not found
        """
        state = self.states.get(user_id)
        if state:
            # Return a deserialized copy so modifications don't affect stored state
            return self._deserialize(state.copy())
        return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """
        Save state for a user.

        Args:
            user_id: User identifier
            state: State dictionary to save
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_state = self._make_serializable(state.copy())
        self.states[user_id] = serializable_state

        # Optionally persist to disk
        if self.storage_path:
            self._save_user_state(user_id, serializable_state)

    def delete_state(self, user_id: str) -> bool:
        """
        Delete state for a user.

        Returns:
            True if deleted, False if user not found
        """
        if user_id in self.states:
            del self.states[user_id]
            if self.storage_path:
                user_file = self.storage_path / f"{user_id}.json"
                if user_file.exists():
                    user_file.unlink()
            return True
        return False

    def create_initial_state(self) -> Dict[str, Any]:
        """
        Create an empty initial state for a new user.
        """
        return {
            'initialized': False,
            'last_state': None,
            'last_covariance': None,
            'last_timestamp': None,
            'kalman_params': None,
        }

    def _make_serializable(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays and datetime objects to JSON-serializable format."""
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state[key] = {
                    '_type': 'ndarray',
                    'data': value.tolist(),
                    'shape': value.shape
                }
            elif isinstance(value, datetime):
                state[key] = {
                    '_type': 'datetime',
                    'data': value.isoformat()
                }

            elif isinstance(value, dict):
                state[key] = self._make_serializable(value)
        return state

    def _deserialize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON-serialized format back to numpy arrays and datetime objects."""
        for key, value in state.items():
            if isinstance(value, dict):
                if value.get('_type') == 'ndarray':
                    state[key] = np.array(value['data']).reshape(value['shape'])
                elif value.get('_type') == 'datetime':
                    state[key] = datetime.fromisoformat(value['data'])
                else:
                    state[key] = self._deserialize(value)

        return state

    def _save_user_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """Save a single user's state to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        user_file = self.storage_path / f"{user_id}.json"

        with open(user_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _load_from_disk(self) -> None:
        """Load all user states from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        for user_file in self.storage_path.glob("*.json"):
            user_id = user_file.stem
            try:
                with open(user_file, 'r') as f:
                    state = json.load(f)
                    self.states[user_id] = self._deserialize(state)
            except Exception as e:
                print(f"Error loading state for {user_id}: {e}")

    def get_all_users(self) -> list[str]:
        """Get list of all users with saved state."""
        return list(self.states.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        initialized_count = sum(1 for s in self.states.values() if s.get('initialized'))

        return {
            'total_users': len(self.states),
            'initialized_users': initialized_count,
            'storage_path': str(self.storage_path) if self.storage_path else 'memory-only'
        }

    def create_snapshot(self, user_id: str, timestamp: datetime) -> str:
        """
        Create a snapshot of current state for rollback.

        Args:
            user_id: User identifier
            timestamp: Timestamp for the snapshot

        Returns:
            Snapshot ID
        """
        if user_id not in self.states:
            return None

        snapshot_id = f"{user_id}_{timestamp.isoformat()}"

        if not hasattr(self, 'snapshots'):
            self.snapshots = {}

        self.snapshots[snapshot_id] = self._make_serializable(
            self.states[user_id].copy()
        )

        return snapshot_id

    def restore_snapshot(self, user_id: str, snapshot_id: str) -> bool:
        """
        Restore state from a snapshot.

        Args:
            user_id: User identifier
            snapshot_id: Snapshot ID to restore

        Returns:
            True if restored, False if snapshot not found
        """
        if not hasattr(self, 'snapshots'):
            return False

        if snapshot_id not in self.snapshots:
            return False

        self.states[user_id] = self.snapshots[snapshot_id].copy()
        return True

    def get_state_before_date(self, user_id: str, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Get the last known state before a specific date.
        Note: This is a placeholder - full implementation would need event history.

        Args:
            user_id: User identifier
            date: Date to get state before

        Returns:
            State dictionary or None
        """
        current_state = self.get_state(user_id)

        if not current_state:
            return None

        if current_state.get('last_timestamp'):
            if current_state['last_timestamp'] < date:
                return current_state

        return None

    def clear_state(self, user_id: str) -> None:
        """
        Clear state for a user (start fresh).

        Args:
            user_id: User identifier
        """
        self.states[user_id] = self.create_initial_state()

    def get_last_timestamp(self, user_id: str) -> Optional[datetime]:
        """
        Get the last processed timestamp for a user.

        Args:
            user_id: User identifier

        Returns:
            Last timestamp or None
        """
        state = self.get_state(user_id)
        if state:
            return state.get('last_timestamp')
        return None


# Alias for compatibility
ProcessorDatabase = ProcessorStateDB

# Global instance for easy access (can be replaced with dependency injection)
_db_instance = None

def get_state_db(storage_path: Optional[str] = None) -> ProcessorStateDB:
    """Get or create the global state database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ProcessorStateDB(storage_path)
    return _db_instance
