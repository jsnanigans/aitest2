"""In-memory implementation of StateStore for testing and development."""

import copy
import csv
import logging
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional, List

from .base import StateStore

logger = logging.getLogger(__name__)


class InMemoryStore(StateStore):
    """
    In-memory state storage for testing and development.

    This implementation stores all data in memory using dictionaries.
    Data is NOT persisted and will be lost when the process ends.

    Features:
    - Thread-safe operations using locks
    - Fast for testing and development
    - No external dependencies
    - Supports snapshots for replay functionality

    Usage:
        >>> from weight_processor_lib.core.database import InMemoryStore
        >>> db = InMemoryStore()
        >>> state = db.create_initial_state()
        >>> db.save_state("user123", state)
        >>> retrieved = db.get_state("user123")
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._states: Dict[str, Dict[str, Any]] = {}
        self._snapshots: Dict[str, List[tuple[datetime, Dict[str, Any]]]] = {}
        self._lock = Lock()
        logger.info("Initialized InMemoryStore")

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve state for a user.

        Args:
            user_id: User identifier

        Returns:
            State dict if found, None otherwise
        """
        with self._lock:
            state = self._states.get(user_id)
            # Return a deep copy to prevent external modifications
            return copy.deepcopy(state) if state is not None else None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """
        Save state for a user.

        Args:
            user_id: User identifier
            state: State dictionary to save

        Returns:
            True if successful
        """
        with self._lock:
            # Store a deep copy to prevent external modifications
            self._states[user_id] = copy.deepcopy(state)
            return True

    def delete_state(self, user_id: str) -> bool:
        """
        Delete state for a user.

        Args:
            user_id: User identifier

        Returns:
            True if state was deleted, False if it didn't exist
        """
        with self._lock:
            if user_id in self._states:
                del self._states[user_id]
                # Also delete snapshots
                if user_id in self._snapshots:
                    del self._snapshots[user_id]
                return True
            return False

    def create_initial_state(self) -> Dict[str, Any]:
        """
        Create an empty initial state.

        Returns:
            Initial state dictionary with all fields set to defaults
        """
        return {
            "kalman_params": None,
            "last_state": None,
            "last_covariance": None,
            "last_timestamp": None,
            "last_accepted_timestamp": None,
            "last_source": None,
            "last_raw_weight": None,
            "measurement_history": [],
            "reset_events": [],
            "measurements_since_reset": 0,
            "adaptation_state": {},
            "version": 0,
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """
        Save a snapshot of current state.

        Args:
            user_id: User identifier
            timestamp: Timestamp for the snapshot

        Returns:
            True if successful, False if no current state exists
        """
        with self._lock:
            current_state = self._states.get(user_id)
            if current_state is None:
                logger.warning(f"Cannot save snapshot for {user_id}: no current state")
                return False

            # Initialize snapshots list if needed
            if user_id not in self._snapshots:
                self._snapshots[user_id] = []

            # Add snapshot (store as tuple of timestamp and deep copy of state)
            snapshot = (timestamp, copy.deepcopy(current_state))
            self._snapshots[user_id].append(snapshot)

            # Sort snapshots by timestamp
            self._snapshots[user_id].sort(key=lambda x: x[0])

            logger.debug(
                f"Saved snapshot for {user_id} at {timestamp.isoformat()} "
                f"(total snapshots: {len(self._snapshots[user_id])})"
            )
            return True

    def restore_state_snapshot(self, user_id: str) -> bool:
        """
        Restore state from the latest snapshot.

        Args:
            user_id: User identifier

        Returns:
            True if restored, False if no snapshot exists
        """
        with self._lock:
            if user_id not in self._snapshots or not self._snapshots[user_id]:
                logger.warning(f"Cannot restore snapshot for {user_id}: no snapshots exist")
                return False

            # Get the latest snapshot
            _, snapshot_state = self._snapshots[user_id][-1]

            # Restore state (deep copy to prevent modifications)
            self._states[user_id] = copy.deepcopy(snapshot_state)

            logger.debug(
                f"Restored latest snapshot for {user_id} "
                f"(from {len(self._snapshots[user_id])} available)"
            )
            return True

    def get_snapshot(
        self, user_id: str, timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get the nearest snapshot before the given timestamp.

        Args:
            user_id: User identifier
            timestamp: Target timestamp

        Returns:
            Snapshot state if found, None otherwise
        """
        with self._lock:
            if user_id not in self._snapshots or not self._snapshots[user_id]:
                return None

            # Find the latest snapshot before or at the timestamp
            matching_snapshot = None
            for snap_time, snap_state in self._snapshots[user_id]:
                if snap_time <= timestamp:
                    matching_snapshot = snap_state
                else:
                    break  # List is sorted, so we can stop here

            # Return deep copy to prevent modifications
            return copy.deepcopy(matching_snapshot) if matching_snapshot is not None else None

    def get_latest_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent snapshot for a user.

        Args:
            user_id: User identifier

        Returns:
            Latest snapshot state if found, None otherwise
        """
        with self._lock:
            if user_id not in self._snapshots or not self._snapshots[user_id]:
                return None

            # Get the latest snapshot
            _, snapshot_state = self._snapshots[user_id][-1]

            # Return deep copy to prevent modifications
            return copy.deepcopy(snapshot_state)

    def check_and_restore_snapshot(
        self, user_id: str, buffer_start_time: datetime
    ) -> dict:
        """
        Check if a snapshot exists and restore it atomically.

        This is used for replay functionality to restore state to a point
        before the replay buffer started.

        Args:
            user_id: User identifier
            buffer_start_time: Start time of the buffer to replay from

        Returns:
            Dict with:
                - snapshot_found: bool
                - snapshot_restored: bool
                - snapshot_timestamp: datetime or None
        """
        with self._lock:
            result = {
                "snapshot_found": False,
                "snapshot_restored": False,
                "snapshot_timestamp": None,
            }

            # Check if we have snapshots for this user
            if user_id not in self._snapshots or not self._snapshots[user_id]:
                return result

            # Find the nearest snapshot before buffer_start_time
            matching_snapshot = None
            matching_timestamp = None

            for snap_time, snap_state in self._snapshots[user_id]:
                if snap_time <= buffer_start_time:
                    matching_snapshot = snap_state
                    matching_timestamp = snap_time
                else:
                    break  # List is sorted

            if matching_snapshot is None:
                return result

            # Found a snapshot
            result["snapshot_found"] = True
            result["snapshot_timestamp"] = matching_timestamp

            # Restore it
            self._states[user_id] = copy.deepcopy(matching_snapshot)
            result["snapshot_restored"] = True

            logger.info(
                f"Restored snapshot for {user_id} from {matching_timestamp.isoformat()} "
                f"for buffer starting at {buffer_start_time.isoformat()}"
            )

            return result

    def export_to_csv(self, filepath: str) -> int:
        """
        Export all states to CSV.

        Args:
            filepath: Path to write CSV file

        Returns:
            Number of states exported
        """
        with self._lock:
            if not self._states:
                logger.warning("No states to export")
                return 0

            # Prepare CSV data
            fieldnames = ["user_id", "last_timestamp", "last_raw_weight",
                         "measurements_since_reset", "version"]

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for user_id, state in self._states.items():
                    row = {
                        "user_id": user_id,
                        "last_timestamp": state.get("last_timestamp"),
                        "last_raw_weight": state.get("last_raw_weight"),
                        "measurements_since_reset": state.get("measurements_since_reset", 0),
                        "version": state.get("version", 0),
                    }
                    writer.writerow(row)

            logger.info(f"Exported {len(self._states)} states to {filepath}")
            return len(self._states)

    # Additional helper methods

    def clear_all(self) -> None:
        """
        Clear all states and snapshots.

        Useful for testing to reset to a clean state.
        """
        with self._lock:
            self._states.clear()
            self._snapshots.clear()
            logger.info("Cleared all states and snapshots")

    def list_users(self) -> List[str]:
        """
        Get list of all user IDs with stored states.

        Returns:
            List of user IDs
        """
        with self._lock:
            return list(self._states.keys())

    def get_snapshot_count(self, user_id: str) -> int:
        """
        Get number of snapshots stored for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of snapshots
        """
        with self._lock:
            if user_id not in self._snapshots:
                return 0
            return len(self._snapshots[user_id])

    def clear_snapshots(self, user_id: str) -> int:
        """
        Clear all snapshots for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of snapshots cleared
        """
        with self._lock:
            if user_id not in self._snapshots:
                return 0
            count = len(self._snapshots[user_id])
            del self._snapshots[user_id]
            logger.debug(f"Cleared {count} snapshots for {user_id}")
            return count

    def __repr__(self) -> str:
        """String representation showing stored data counts."""
        with self._lock:
            return (
                f"InMemoryStore(states={len(self._states)}, "
                f"users_with_snapshots={len(self._snapshots)})"
            )
