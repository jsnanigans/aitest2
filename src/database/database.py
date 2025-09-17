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
        self._snapshots = {}  # For replay functionality

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
            'kalman_params': None,
            'last_state': None,
            'last_covariance': None,
            'last_timestamp': None,
            'last_accepted_timestamp': None,
            'last_source': None,
            'last_raw_weight': None,
            'measurement_history': [],
            'reset_events': [],
            'measurements_since_reset': 0
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> None:
        """
        Save a snapshot of current state (for replay functionality).

        Args:
            user_id: User identifier
            timestamp: Timestamp for the snapshot
        """
        if user_id in self.states:
            self._snapshots[user_id] = {
                'timestamp': timestamp,
                'state': copy.deepcopy(self.states[user_id])
            }

    def restore_state_snapshot(self, user_id: str) -> bool:
        """
        Restore state from snapshot.

        Returns:
            True if restored, False if no snapshot found
        """
        if user_id in self._snapshots:
            self.states[user_id] = copy.deepcopy(self._snapshots[user_id]['state'])
            return True
        return False

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
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'user_id', 'last_weight', 'last_trend', 'last_timestamp',
                'measurements_since_reset', 'last_source'
            ])
            writer.writeheader()

            for user_id, state in self.states.items():
                row = {
                    'user_id': user_id,
                    'last_weight': None,
                    'last_trend': None,
                    'last_timestamp': state.get('last_timestamp'),
                    'measurements_since_reset': state.get('measurements_since_reset', 0),
                    'last_source': state.get('last_source')
                }

                # Extract weight and trend from last_state if available
                last_state = state.get('last_state')
                if last_state is not None:
                    if isinstance(last_state, np.ndarray):
                        if last_state.ndim == 1 and last_state.size >= 2:
                            row['last_weight'] = float(last_state[0])
                            row['last_trend'] = float(last_state[1])
                        elif last_state.ndim == 2:
                            row['last_weight'] = float(last_state[-1][0])
                            row['last_trend'] = float(last_state[-1][1])
                    elif isinstance(last_state, list) and len(last_state) >= 2:
                        # Handle list format [[weight], [trend]]
                        if isinstance(last_state[0], list):
                            row['last_weight'] = float(last_state[0][0])
                            row['last_trend'] = float(last_state[1][0])
                        else:
                            row['last_weight'] = float(last_state[0])
                            row['last_trend'] = float(last_state[1])

                writer.writerow(row)
                users_exported += 1

        return users_exported


# Global instance
_db_instance = None


def get_state_db() -> ProcessorStateDB:
    """Get the global state database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ProcessorStateDB()
    return _db_instance


def reset_db() -> None:
    """Reset the global database instance (useful for testing)."""
    global _db_instance
    _db_instance = None