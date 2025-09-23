"""In-memory implementation of StateStore."""

import copy
import csv
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List

import numpy as np

from .base import StateStore

logger = logging.getLogger(__name__)


class InMemoryStateStore(StateStore):
    """
    In-memory state storage for weight processor.
    This is the refactored version of ProcessorStateDB.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize in-memory state database."""
        self.states = {}
        self._snapshots = {}
        self.storage_path = storage_path  # For future file persistence

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user."""
        if user_id in self.states:
            return copy.deepcopy(self.states[user_id])
        return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Save state for a user."""
        try:
            self.states[user_id] = copy.deepcopy(state)
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def delete_state(self, user_id: str) -> bool:
        """Delete state for a user."""
        if user_id in self.states:
            del self.states[user_id]
            if user_id in self._snapshots:
                del self._snapshots[user_id]
            return True
        return False

    def create_initial_state(self) -> Dict[str, Any]:
        """Create an empty initial state."""
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
            'measurements_since_reset': 0,
            'adaptation_state': {}
        }

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """Save a snapshot of current state."""
        try:
            if user_id in self.states:
                self._snapshots[user_id] = {
                    'timestamp': timestamp,
                    'state': copy.deepcopy(self.states[user_id])
                }
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return False

    def restore_state_snapshot(self, user_id: str) -> bool:
        """Restore state from snapshot."""
        if user_id in self._snapshots:
            self.states[user_id] = copy.deepcopy(self._snapshots[user_id]['state'])
            return True
        return False

    def get_snapshot(self, user_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get the nearest snapshot before the given timestamp."""
        if user_id in self._snapshots:
            snapshot = self._snapshots[user_id]
            # In this simple implementation, we only store one snapshot per user
            # Return it if it's before the requested timestamp
            if snapshot.get('timestamp') and snapshot['timestamp'] <= timestamp:
                return copy.deepcopy(snapshot['state'])
        return None

    def check_and_restore_snapshot(self, user_id: str, buffer_start_time: datetime) -> dict:
        """Check if a snapshot exists and restore it atomically."""
        if user_id in self._snapshots:
            snapshot = self._snapshots[user_id]
            # Restore the state
            self.states[user_id] = copy.deepcopy(snapshot['state'])
            return {
                'success': True,
                'snapshot': snapshot,
                'snapshot_timestamp': snapshot.get('timestamp', buffer_start_time),
                'user_id': user_id
            }
        else:
            return {
                'success': False,
                'error': f'No snapshot found for user {user_id}',
                'user_id': user_id
            }

    def export_to_csv(self, filepath: str) -> int:
        """Export all states to CSV."""
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