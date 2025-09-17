"""
Processor State Database
Manages persistent Kalman state for all users
State Schema:
- last_state: np.array - Kalman state [weight, trend]
- last_covariance: np.array - Kalman covariance matrix
- last_timestamp: datetime - Last measurement timestamp
- kalman_params: dict - Filter parameters (always present after first measurement)
- last_source: str - Source of last measurement
- last_raw_weight: float - Raw weight before filtering
- measurement_history: list - Last 30 measurements for quality scoring
"""

import csv
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from src.processing.kalman_state_validator import KalmanStateValidator
from src.exceptions import DataCorruptionError, StateValidationError
from src.feature_manager import FeatureManager

logger = logging.getLogger(__name__)


class ProcessorStateDB:
    """
    State database for weight processor.
    Stores and retrieves Kalman state for each user.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize state database.

        Args:
            storage_path: Optional path for persistent storage (future feature)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.states = {}
        self._transaction_active = False
        self._transaction_backup = None

        # Initialize validator and feature manager
        self.validator = KalmanStateValidator()
        self.feature_manager = FeatureManager()
        self.enable_validation = self.feature_manager.is_enabled('kalman_state_validation')

        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    @contextmanager
    def transaction(self):
        """
        Context manager for atomic operations.
        Provides rollback capability on failure.
        """
        if self._transaction_active:
            # Nested transaction - just yield
            yield self
            return

        # Start transaction
        self._transaction_active = True
        self._transaction_backup = {
            user_id: state.copy() for user_id, state in self.states.items()
        }

        try:
            yield self
            # Commit (implicit - changes already made)
            self._transaction_active = False
            self._transaction_backup = None
        except Exception as e:
            # Rollback
            logger.error(f"Transaction failed, rolling back: {e}")
            self.states = self._transaction_backup
            self._transaction_active = False
            self._transaction_backup = None
            raise

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve state for a user.

        Returns:
            State dictionary or None if user not found
        """
        state = self.states.get(user_id)
        if state:
            return self._deserialize(state.copy())
        return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """
        Save state for a user.

        Args:
            user_id: User identifier
            state: State dictionary to save
        """
        serializable_state = self._make_serializable(state.copy())
        self.states[user_id] = serializable_state

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
            'last_source': None,
            'last_raw_weight': None,
            'measurement_history': [],
            'last_accepted_timestamp': None,
            'reset_events': [],
            'state_history': [],  # New: State snapshots for retrospective processing
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

        # Validate Kalman-specific state components if enabled
        if self.enable_validation and self._contains_kalman_state(state):
            state = self._validate_kalman_components(state)

        return state

    def _contains_kalman_state(self, state: Dict[str, Any]) -> bool:
        """Check if state contains Kalman filter components."""
        return any(key in state for key in ['last_state', 'last_covariance', 'kalman_params'])

    def _validate_kalman_components(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and potentially recover Kalman state components.

        Args:
            state: Deserialized state dictionary

        Returns:
            Validated state dictionary

        Raises:
            DataCorruptionError: If state is corrupted beyond recovery
        """
        # Build validation dict for Kalman components
        kalman_components = {}

        # Map database fields to Kalman validator fields
        if state.get('last_state') is not None:
            # last_state should be a 2D array but we only care about the weight part
            # Extract the state vector [weight, trend]
            last_state = state['last_state']

            # Try to handle various input types
            if not isinstance(last_state, np.ndarray):
                # Try to convert to numpy array (handles lists, tuples, etc.)
                try:
                    last_state = np.array(last_state)
                except:
                    # If conversion fails, mark as invalid
                    kalman_components['x'] = last_state  # Pass the invalid value for validation to catch

            if isinstance(last_state, np.ndarray) and last_state.size >= 1:
                # For validation purposes, we'll validate just the weight component
                if last_state.ndim == 2:
                    kalman_components['x'] = last_state[0:1, 0:1]
                elif last_state.ndim == 1 and last_state.size >= 1:
                    # 1D array - take first element only
                    kalman_components['x'] = np.array([[last_state[0]]])
                else:
                    # Scalar or other
                    kalman_components['x'] = np.array([[last_state.flat[0]]])
            elif 'x' not in kalman_components:
                # Handle invalid or empty arrays
                kalman_components['x'] = last_state

        if state.get('last_covariance') is not None:
            # For now, validate covariance as P (even though it's larger)
            # We'll just validate the weight variance component
            last_cov = state['last_covariance']

            # Try to handle various input types
            if not isinstance(last_cov, np.ndarray):
                try:
                    last_cov = np.array(last_cov)
                except:
                    kalman_components['P'] = last_cov  # Pass invalid value for validation

            if isinstance(last_cov, np.ndarray) and last_cov.size >= 1:
                kalman_components['P'] = last_cov[0:1, 0:1] if last_cov.ndim > 1 else np.array([[last_cov.flat[0]]])
            elif 'P' not in kalman_components:
                kalman_components['P'] = last_cov

        # Add default F, H, Q, R for validation (these are typically constants)
        kalman_components['F'] = np.array([[1.0]])  # State transition
        kalman_components['H'] = np.array([[1.0]])  # Observation model
        kalman_components['Q'] = np.array([[0.1]])  # Process noise (default)
        kalman_components['R'] = np.array([[1.0]])  # Observation noise (default)

        # Extract actual Q and R from kalman_params if available
        if state.get('kalman_params'):
            params = state['kalman_params']
            if isinstance(params, dict):
                if 'transition_covariance' in params:
                    trans_cov = params['transition_covariance']
                    if isinstance(trans_cov, (list, np.ndarray)) and len(trans_cov) > 0:
                        if isinstance(trans_cov[0], (list, np.ndarray)) and len(trans_cov[0]) > 0:
                            kalman_components['Q'] = np.array([[trans_cov[0][0]]])

                if 'observation_covariance' in params:
                    obs_cov = params['observation_covariance']
                    if isinstance(obs_cov, (list, np.ndarray)) and len(obs_cov) > 0:
                        if isinstance(obs_cov[0], (list, np.ndarray)) and len(obs_cov[0]) > 0:
                            kalman_components['R'] = np.array([[obs_cov[0][0]]])
                        else:
                            kalman_components['R'] = np.array([[obs_cov[0]]])

        # Validate the components
        try:
            validated = self.validator.validate_and_fix(kalman_components)

            if validated is None:
                logger.error(
                    f"Kalman state validation failed. "
                    f"Metrics: {self.validator.get_validation_metrics()}"
                )

                # For now, log error but don't raise exception to maintain backward compatibility
                # In production, this should raise DataCorruptionError
                if self.feature_manager.is_enabled('strict_validation'):
                    raise DataCorruptionError("Kalman state corrupted and unrecoverable")
                else:
                    logger.warning("Kalman validation failed but continuing due to backward compatibility mode")

        except StateValidationError as e:
            logger.error(f"Kalman state validation error: {e}")
            if self.feature_manager.is_enabled('strict_validation'):
                raise DataCorruptionError(f"Kalman state validation failed: {e}")
            else:
                logger.warning(f"Kalman validation error but continuing: {e}")

        return state

    def _legacy_deserialize(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy deserialization without validation for backward compatibility."""
        return self._deserialize_without_validation(state_dict)

    def _deserialize_without_validation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize without validation (for legacy support)."""
        for key, value in state.items():
            if isinstance(value, dict):
                if value.get('_type') == 'ndarray':
                    state[key] = np.array(value['data']).reshape(value['shape'])
                elif value.get('_type') == 'datetime':
                    state[key] = datetime.fromisoformat(value['data'])
                else:
                    state[key] = self._deserialize_without_validation(value)
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

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> bool:
        """
        Save a state snapshot for retrospective processing.
        Maintains up to 100 historical snapshots per user.

        Args:
            user_id: User identifier
            timestamp: Timestamp of the snapshot

        Returns:
            True if snapshot saved successfully
        """
        state = self.get_state(user_id)
        if not state or not state.get('kalman_params'):
            return False

        snapshot = {
            'timestamp': timestamp,
            'kalman_state': state['last_state'].copy() if state.get('last_state') is not None else None,
            'kalman_covariance': state['last_covariance'].copy() if state.get('last_covariance') is not None else None,
            'kalman_params': state['kalman_params'].copy() if state.get('kalman_params') else None,
            'measurements_count': len(state.get('measurement_history', []))
        }

        # Add to state history
        history = state.get('state_history', [])
        history.append(snapshot)

        # Keep only last 100 snapshots
        if len(history) > 100:
            history = history[-100:]

        # Update state and save
        state['state_history'] = history
        self.save_state(user_id, state)
        return True

    def get_state_snapshot_before(self, user_id: str, before_timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Get the most recent state snapshot before the given timestamp.

        Args:
            user_id: User identifier
            before_timestamp: Find snapshot before this time

        Returns:
            State snapshot dictionary or None if not found
        """
        state = self.get_state(user_id)
        if not state or not state.get('state_history'):
            return None

        history = state['state_history']
        # Find the most recent snapshot before the timestamp
        matching_snapshots = [
            snapshot for snapshot in history
            if snapshot['timestamp'] < before_timestamp
        ]

        if not matching_snapshots:
            return None

        # Return the most recent one
        return max(matching_snapshots, key=lambda x: x['timestamp'])

    def restore_state_from_snapshot(self, user_id: str, snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Restore user state from a historical snapshot with comprehensive validation.
        WARNING: This modifies the current state - use with caution.

        Args:
            user_id: User identifier
            snapshot: Snapshot dictionary from get_state_snapshot_before

        Returns:
            Dict with 'success' bool and 'error' string if failed
        """
        # Comprehensive validation
        if snapshot is None:
            error_msg = f"Cannot restore state for {user_id}: snapshot is None"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        if not isinstance(snapshot, dict):
            error_msg = f"Cannot restore state for {user_id}: invalid snapshot type {type(snapshot)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        # Validate required fields
        required_fields = ['kalman_state', 'kalman_covariance', 'timestamp']
        missing_fields = [f for f in required_fields if f not in snapshot]

        if missing_fields:
            error_msg = f"Cannot restore state for {user_id}: missing required fields {missing_fields}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        # Validate field types
        if not isinstance(snapshot.get('kalman_state'), (list, np.ndarray)):
            error_msg = f"Cannot restore state for {user_id}: invalid kalman_state type"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        try:
            state = self.get_state(user_id)
            if not state:
                state = self.create_initial_state()
                logger.info(f"Creating new state for {user_id} during restoration")

            # Restore key state components from snapshot
            state['last_state'] = snapshot.get('kalman_state')
            state['last_covariance'] = snapshot.get('kalman_covariance')
            state['kalman_params'] = snapshot.get('kalman_params')
            state['last_timestamp'] = snapshot.get('timestamp')

            # Clear measurement history to start fresh from restore point
            # Keep state_history intact for potential rollbacks
            state['measurement_history'] = []

            self.save_state(user_id, state)

            logger.info(f"Successfully restored state for {user_id} to timestamp {snapshot.get('timestamp')}")
            return {'success': True, 'snapshot_timestamp': snapshot.get('timestamp')}

        except Exception as e:
            error_msg = f"Failed to restore state for {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': error_msg}

    def check_and_restore_snapshot(self, user_id: str, before_timestamp: datetime) -> Dict[str, Any]:
        """
        Atomic check and restore operation.
        Combines snapshot lookup and restoration in a single transaction.

        Args:
            user_id: User identifier
            before_timestamp: Timestamp to restore to

        Returns:
            Dict with 'success' bool, 'error' string if failed, and 'snapshot' if successful
        """
        with self.transaction():
            try:
                # Log the attempt
                logger.info(f"Attempting atomic restore for {user_id} to time before {before_timestamp}")

                # Get snapshot before the specified time
                snapshot = self.get_state_snapshot_before(user_id, before_timestamp)

                if not snapshot:
                    error_msg = f"No snapshot found for {user_id} before {before_timestamp}"
                    logger.warning(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'timestamp': before_timestamp,
                        'user_id': user_id
                    }

                # Log snapshot found
                logger.debug(f"Found snapshot for {user_id} at {snapshot.get('timestamp')}")

                # Attempt restoration
                result = self.restore_state_from_snapshot(user_id, snapshot)

                if result['success']:
                    # Add snapshot to successful result
                    result['snapshot'] = snapshot
                    result['user_id'] = user_id
                    logger.info(f"Atomic restore successful for {user_id}")
                else:
                    # Log failure details
                    logger.error(f"Atomic restore failed for {user_id}: {result.get('error')}")

                return result

            except Exception as e:
                error_msg = f"Atomic restore failed for {user_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                # Transaction will be rolled back automatically
                return {
                    'success': False,
                    'error': error_msg,
                    'user_id': user_id,
                    'timestamp': before_timestamp
                }

    def get_state_history_count(self, user_id: str) -> int:
        """
        Get the number of state snapshots stored for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of snapshots (0 if user not found)
        """
        state = self.get_state(user_id)
        if not state:
            return 0
        return len(state.get('state_history', []))

    def export_to_csv(self, filepath: str) -> int:
        """
        Export all user states to CSV format.
        Only exports users with actual measurements (kalman_params not None).
        
        Args:
            filepath: Path to save the CSV file
            
        Returns:
            Number of users exported
        """
        csv_path = Path(filepath)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as csvfile:
            # More meaningful columns - removed has_kalman_params
            fieldnames = [
                'user_id',
                'last_timestamp',
                'last_weight',
                'last_trend',
                'last_source',
                'last_raw_weight',
                'weight_change',  # New: change from raw to filtered
                'measurement_count',  # New: how many measurements in history
                'process_noise',
                'measurement_noise',
                'initial_uncertainty',
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            users_exported = 0
            for user_id in sorted(self.states.keys()):
                state = self.get_state(user_id)
                if not state or not state.get('kalman_params'):
                    # Skip users with no measurements
                    continue
                    
                row = {
                    'user_id': user_id,
                    'last_timestamp': '',
                    'last_weight': '',
                    'last_trend': '',
                    'last_source': '',
                    'last_raw_weight': '',
                    'weight_change': '',
                    'measurement_count': '0',
                    'process_noise': '',
                    'measurement_noise': '',
                    'initial_uncertainty': '',
                }
                
                if state.get('last_timestamp'):
                    row['last_timestamp'] = state['last_timestamp'].isoformat() if isinstance(state['last_timestamp'], datetime) else str(state['last_timestamp'])
                
                if state.get('last_source'):
                    row['last_source'] = str(state['last_source'])
                    
                if state.get('last_raw_weight') is not None:
                    row['last_raw_weight'] = f"{float(state['last_raw_weight']):.2f}"
                
                # Count measurements in history
                if state.get('measurement_history'):
                    row['measurement_count'] = str(len(state['measurement_history']))
                
                if state.get('last_state') is not None:
                    try:
                        last_state = state['last_state']
                        if isinstance(last_state, np.ndarray):
                            if len(last_state.shape) > 1:
                                current_state = last_state[-1]
                                row['last_weight'] = f"{float(current_state[0]):.2f}"
                                row['last_trend'] = f"{float(current_state[1]):.4f}"
                            else:
                                row['last_weight'] = f"{float(last_state[0]):.2f}"
                                row['last_trend'] = f"{float(last_state[1]):.4f}"
                        elif isinstance(last_state, (list, tuple)) and len(last_state) >= 2:
                            if isinstance(last_state[0], (list, tuple, np.ndarray)):
                                current_state = last_state[-1]
                                row['last_weight'] = f"{float(current_state[0]):.2f}"
                                row['last_trend'] = f"{float(current_state[1]):.4f}"
                            else:
                                row['last_weight'] = f"{float(last_state[0]):.2f}"
                                row['last_trend'] = f"{float(last_state[1]):.4f}"
                        
                        # Calculate weight change
                        if row['last_weight'] and row['last_raw_weight']:
                            filtered = float(row['last_weight'])
                            raw = float(row['last_raw_weight'])
                            row['weight_change'] = f"{filtered - raw:.2f}"
                            
                    except (TypeError, IndexError, ValueError):
                        pass
                
                # Extract Kalman parameters
                params = state['kalman_params']
                if isinstance(params, dict):
                    trans_cov = params.get('transition_covariance', '')
                    if trans_cov and isinstance(trans_cov, (list, np.ndarray)):
                        try:
                            row['process_noise'] = str(trans_cov[0][0] if isinstance(trans_cov[0], (list, np.ndarray)) else trans_cov[0])
                        except:
                            row['process_noise'] = str(trans_cov)
                    else:
                        row['process_noise'] = str(trans_cov)
                    
                    obs_cov = params.get('observation_covariance', '')
                    if obs_cov and isinstance(obs_cov, (list, np.ndarray)):
                        try:
                            row['measurement_noise'] = str(obs_cov[0][0] if isinstance(obs_cov[0], (list, np.ndarray)) else obs_cov[0])
                        except:
                            row['measurement_noise'] = str(obs_cov)
                    else:
                        row['measurement_noise'] = str(obs_cov)
                    
                    init_cov = params.get('initial_state_covariance', '')
                    if init_cov and isinstance(init_cov, (list, np.ndarray)):
                        try:
                            row['initial_uncertainty'] = str(init_cov[0][0] if isinstance(init_cov[0], (list, np.ndarray)) else init_cov[0])
                        except:
                            row['initial_uncertainty'] = str(init_cov)
                    else:
                        row['initial_uncertainty'] = str(init_cov)
                
                writer.writerow(row)
                users_exported += 1
        
        return users_exported


ProcessorDatabase = ProcessorStateDB

_db_instance = None

def get_state_db(storage_path: Optional[str] = None) -> ProcessorStateDB:
    """Get or create the global state database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ProcessorStateDB(storage_path)
    return _db_instance
