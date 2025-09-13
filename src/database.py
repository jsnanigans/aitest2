"""
Processor State Database
Manages persistent Kalman state for all users
Initially in-memory with JSON serialization, can be backed by files or database later
"""

import csv
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
        active_count = sum(1 for s in self.states.values() if s.get('kalman_params') is not None)

        return {
            'total_users': len(self.states),
            'active_users': active_count,
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

    def export_to_csv(self, filepath: str) -> int:
        """
        Export all user states to CSV format.
        
        Args:
            filepath: Path to save the CSV file
            
        Returns:
            Number of users exported
        """
        csv_path = Path(filepath)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'user_id',
                'last_timestamp',
                'last_weight',
                'last_trend',
                'has_kalman_params',
                'process_noise',
                'measurement_noise',
                'initial_uncertainty',
                'has_adapted_params',
                'adapted_process_noise',
                'adapted_measurement_noise',
                'state_reset_count',
                'last_reset_timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            users_exported = 0
            for user_id in sorted(self.states.keys()):
                state = self.get_state(user_id)
                if not state:
                    continue
                    
                row = {
                    'user_id': user_id,
                    'last_timestamp': '',
                    'last_weight': '',
                    'last_trend': '',
                    'has_kalman_params': 'false',
                    'process_noise': '',
                    'measurement_noise': '',
                    'initial_uncertainty': '',
                    'has_adapted_params': 'false',
                    'adapted_process_noise': '',
                    'adapted_measurement_noise': '',
                    'state_reset_count': '0',
                    'last_reset_timestamp': ''
                }
                
                if state.get('last_timestamp'):
                    row['last_timestamp'] = state['last_timestamp'].isoformat() if isinstance(state['last_timestamp'], datetime) else str(state['last_timestamp'])
                
                if state.get('last_state') is not None:
                    try:
                        last_state = state['last_state']
                        if isinstance(last_state, np.ndarray):
                            # Handle 2D array (multiple states)
                            if len(last_state.shape) > 1:
                                # Get the most recent state (last row)
                                current_state = last_state[-1]
                                row['last_weight'] = f"{float(current_state[0]):.2f}"
                                row['last_trend'] = f"{float(current_state[1]):.4f}"
                            else:
                                # 1D array
                                row['last_weight'] = f"{float(last_state[0]):.2f}"
                                row['last_trend'] = f"{float(last_state[1]):.4f}"
                        elif isinstance(last_state, (list, tuple)) and len(last_state) >= 2:
                            # Handle list/tuple
                            if isinstance(last_state[0], (list, tuple, np.ndarray)):
                                # Nested structure - get last element
                                current_state = last_state[-1]
                                row['last_weight'] = f"{float(current_state[0]):.2f}"
                                row['last_trend'] = f"{float(current_state[1]):.4f}"
                            else:
                                # Flat structure
                                row['last_weight'] = f"{float(last_state[0]):.2f}"
                                row['last_trend'] = f"{float(last_state[1]):.4f}"
                    except (TypeError, IndexError, ValueError):
                        pass
                
                if state.get('kalman_params'):
                    row['has_kalman_params'] = 'true'
                    params = state['kalman_params']
                    
                    if isinstance(params, dict):
                        # Extract transition covariance (process noise)
                        trans_cov = params.get('transition_covariance', '')
                        if trans_cov and isinstance(trans_cov, (list, np.ndarray)):
                            try:
                                # Get the weight component of process noise
                                row['process_noise'] = str(trans_cov[0][0] if isinstance(trans_cov[0], (list, np.ndarray)) else trans_cov[0])
                            except:
                                row['process_noise'] = str(trans_cov)
                        else:
                            row['process_noise'] = str(trans_cov)
                        
                        # Extract observation covariance (measurement noise)
                        obs_cov = params.get('observation_covariance', '')
                        if obs_cov and isinstance(obs_cov, (list, np.ndarray)):
                            try:
                                # Get the scalar value
                                row['measurement_noise'] = str(obs_cov[0][0] if isinstance(obs_cov[0], (list, np.ndarray)) else obs_cov[0])
                            except:
                                row['measurement_noise'] = str(obs_cov)
                        else:
                            row['measurement_noise'] = str(obs_cov)
                        
                        # Extract initial state covariance
                        init_cov = params.get('initial_state_covariance', '')
                        if init_cov and isinstance(init_cov, (list, np.ndarray)):
                            try:
                                # Get the weight uncertainty component
                                row['initial_uncertainty'] = str(init_cov[0][0] if isinstance(init_cov[0], (list, np.ndarray)) else init_cov[0])
                            except:
                                row['initial_uncertainty'] = str(init_cov)
                        else:
                            row['initial_uncertainty'] = str(init_cov)
                
                if state.get('adapted_params'):
                    row['has_adapted_params'] = 'true'
                    adapted = state['adapted_params']
                    
                    if isinstance(adapted, dict):
                        row['adapted_process_noise'] = str(adapted.get('process_noise', ''))
                        row['adapted_measurement_noise'] = str(adapted.get('measurement_noise', ''))
                
                if state.get('state_reset_count'):
                    row['state_reset_count'] = str(state['state_reset_count'])
                
                if state.get('last_reset_timestamp'):
                    row['last_reset_timestamp'] = state['last_reset_timestamp'].isoformat() if isinstance(state['last_reset_timestamp'], datetime) else str(state['last_reset_timestamp'])
                
                writer.writerow(row)
                users_exported += 1
        
        return users_exported


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
