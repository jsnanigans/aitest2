"""
Simple in-memory state database for weight processor.
Stores Kalman filter states without persistence.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
import logging
import copy

logger = logging.getLogger(__name__)


class ProcessorStateDB:
    """
    In-memory state storage for weight processor.
    Stores and retrieves Kalman state for each user.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize in-memory state database."""
        self.states = {}
        self._snapshots = {}  # For replay functionality: user_id -> list of snapshots

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve state for a user.

        Returns:
            State dictionary or None if user not found
        """
        if user_id in self.states:
            # Return a deep copy to prevent external modifications
            return copy.deepcopy(self.states[user_id])
        return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """
        Save state for a user.

        Args:
            user_id: User identifier
            state: State dictionary to save
        """
        # Store a deep copy to prevent external modifications
        self.states[user_id] = copy.deepcopy(state)

    def delete_state(self, user_id: str) -> bool:
        """
        Delete state for a user.

        Returns:
            True if deleted, False if user not found
        """
        if user_id in self.states:
            del self.states[user_id]
            if user_id in self._snapshots:
                del self._snapshots[user_id]
            return True
        return False

    def create_initial_state(self) -> Dict[str, Any]:
        """
        Create an empty initial state.

        Returns:
            Empty state dictionary with required fields
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
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """
        Save a snapshot of current state (for replay functionality).

        Args:
            user_id: User identifier
            timestamp: Timestamp for the snapshot

        Returns:
            True if snapshot saved successfully
        """
        if user_id in self.states:
            if user_id not in self._snapshots:
                self._snapshots[user_id] = []

            snapshot = {
                "timestamp": timestamp,
                "snapshotTime": timestamp.isoformat(),
                "state": copy.deepcopy(self.states[user_id]),
            }
            self._snapshots[user_id].append(snapshot)

            # Keep only last 10 snapshots (10 days with 24-hour intervals)
            self._snapshots[user_id] = sorted(
                self._snapshots[user_id], key=lambda s: s["timestamp"]
            )[-10:]

            return True
        return False

    def get_latest_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent snapshot for a user.

        Used by periodic snapshot logic to determine when to create next snapshot.

        Args:
            user_id: User identifier

        Returns:
            Latest snapshot dict or None if no snapshots exist
        """
        if user_id not in self._snapshots or not self._snapshots[user_id]:
            return None

        # Get the most recent snapshot (list is kept sorted)
        latest = self._snapshots[user_id][-1]
        return copy.deepcopy(latest["state"])

    def get_snapshot(
        self, user_id: str, timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get the nearest snapshot before the given timestamp.

        Args:
            user_id: User identifier
            timestamp: Find snapshot before this time

        Returns:
            Snapshot state dict or None if no suitable snapshot exists
        """
        if user_id not in self._snapshots or not self._snapshots[user_id]:
            return None

        # Find the most recent snapshot before or at the timestamp
        suitable_snapshots = [
            s for s in self._snapshots[user_id] if s["timestamp"] <= timestamp
        ]

        if not suitable_snapshots:
            return None

        # Return the most recent one
        latest = max(suitable_snapshots, key=lambda s: s["timestamp"])
        return copy.deepcopy(latest["state"])

    def restore_state_snapshot(self, user_id: str) -> bool:
        """
        Restore state from the latest snapshot.

        Returns:
            True if restored, False if no snapshot found
        """
        latest_snapshot_state = self.get_latest_snapshot(user_id)
        if latest_snapshot_state:
            self.states[user_id] = copy.deepcopy(latest_snapshot_state)
            return True
        return False

    def get_measurements_in_window(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """
        Get measurements for a user within a time window.

        Used by replay trigger logic to find measurements in the 72-hour window.

        Args:
            user_id: User identifier
            start_time: Window start time (inclusive)
            end_time: Window end time (exclusive)

        Returns:
            List of measurement dicts with keys:
            - timestamp: datetime
            - weight: float
            - source: str
            - unit: str
            - metadata: dict
        """
        state = self.get_state(user_id)
        if not state or "measurement_history" not in state:
            return []

        measurements = []
        for m in state["measurement_history"]:
            timestamp = m.get("timestamp")
            if timestamp is None:
                continue

            # Ensure timestamp is datetime
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Check if in window
            if start_time <= timestamp < end_time:
                measurements.append({
                    "timestamp": timestamp,
                    "weight": m.get("weight"),
                    "source": m.get("source", "unknown"),
                    "unit": m.get("unit", "kg"),
                    "metadata": m.get("metadata", {})
                })

        return measurements

    def check_and_restore_snapshot(
        self, user_id: str, buffer_start_time: datetime
    ) -> dict:
        """
        Check if a snapshot exists before buffer_start_time and restore it atomically.

        Args:
            user_id: User identifier
            buffer_start_time: Find snapshot before this time

        Returns:
            Dictionary with success status and snapshot details
        """
        snapshot_state = self.get_snapshot(user_id, buffer_start_time)
        if snapshot_state:
            # Restore the state
            self.states[user_id] = copy.deepcopy(snapshot_state)
            return {
                "success": True,
                "snapshot": snapshot_state,
                "snapshot_timestamp": snapshot_state.get(
                    "last_timestamp", buffer_start_time
                ),
                "user_id": user_id,
            }
        else:
            return {
                "success": False,
                "error": f"No snapshot found for user {user_id} before {buffer_start_time}",
                "user_id": user_id,
            }

    def export_to_csv(self, filepath: str) -> int:
        """
        Export all states to CSV (simplified version).

        Args:
            filepath: Path to CSV file

        Returns:
            Number of users exported
        """
        import csv

        users_exported = 0
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "user_id",
                    "last_weight",
                    "last_trend",
                    "last_timestamp",
                    "measurements_since_reset",
                    "last_source",
                ],
            )
            writer.writeheader()

            for user_id, state in self.states.items():
                row = {
                    "user_id": user_id,
                    "last_weight": None,
                    "last_trend": None,
                    "last_timestamp": state.get("last_timestamp"),
                    "measurements_since_reset": state.get(
                        "measurements_since_reset", 0
                    ),
                    "last_source": state.get("last_source"),
                }

                # Extract weight and trend from last_state if available
                last_state = state.get("last_state")
                if last_state is not None:
                    if isinstance(last_state, np.ndarray):
                        if last_state.ndim == 1 and last_state.size >= 2:
                            row["last_weight"] = float(last_state[0])
                            row["last_trend"] = float(last_state[1])
                        elif last_state.ndim == 2:
                            row["last_weight"] = float(last_state[-1][0])
                            row["last_trend"] = float(last_state[-1][1])
                    elif isinstance(last_state, list) and len(last_state) >= 2:
                        # Handle list format [[weight], [trend]]
                        if isinstance(last_state[0], list):
                            row["last_weight"] = float(last_state[0][0])
                            row["last_trend"] = float(last_state[1][0])
                        else:
                            row["last_weight"] = float(last_state[0])
                            row["last_trend"] = float(last_state[1])

                writer.writerow(row)
                users_exported += 1

        return users_exported


# Global instance - now delegating to the new factory
_db_instance = None


def get_state_db() -> ProcessorStateDB:
    """
    Get the global state database instance.
    This is kept for backward compatibility - new code should use
    the factory from __init__.py
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = ProcessorStateDB()
    return _db_instance


def reset_db() -> None:
    """Reset the global database instance (useful for testing)."""
    global _db_instance
    _db_instance = None
    # Also reset the new instance
    from . import reset_db_instance

    reset_db_instance()
