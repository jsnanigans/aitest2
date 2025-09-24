"""SQLite implementation of StateStore for local persistent storage."""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base import StateStore

logger = logging.getLogger(__name__)


class SQLiteStateStore(StateStore):
    """SQLite-based state storage for local development."""

    def __init__(self, db_path: str = None):
        """
        Initialize SQLite state store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or os.getenv('DB_PATH', '/tmp/weight-processor.db')
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create main state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_states (
                user_id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create snapshots table for replay functionality
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state_snapshots (
                user_id TEXT,
                snapshot_time TIMESTAMP,
                state_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, snapshot_time)
            )
        ''')

        # Create buffer table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS replay_buffers (
                user_id TEXT PRIMARY KEY,
                buffer_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"SQLite database initialized at: {self.db_path}")

    def _serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state to JSON string."""
        # Convert numpy arrays and datetime objects to JSON-serializable format
        def convert_value(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        serializable_state = {}
        for key, value in state.items():
            if key in ['last_state', 'last_covariance'] and value is not None:
                serializable_state[key] = convert_value(value)
            elif key == 'measurement_history' and isinstance(value, list):
                serializable_state[key] = [
                    {k: convert_value(v) for k, v in item.items()}
                    for item in value
                ]
            elif key == 'reset_events' and isinstance(value, list):
                serializable_state[key] = [
                    {k: convert_value(v) for k, v in item.items()}
                    for item in value
                ]
            else:
                serializable_state[key] = convert_value(value)

        return json.dumps(serializable_state)

    def _deserialize_state(self, state_json: str) -> Dict[str, Any]:
        """Deserialize state from JSON string."""
        state = json.loads(state_json)

        # Convert ISO format strings back to datetime objects where needed
        if 'last_timestamp' in state and state['last_timestamp']:
            state['last_timestamp'] = datetime.fromisoformat(state['last_timestamp'])
        if 'last_accepted_timestamp' in state and state['last_accepted_timestamp']:
            state['last_accepted_timestamp'] = datetime.fromisoformat(state['last_accepted_timestamp'])

        # Convert measurement_history timestamps
        if 'measurement_history' in state and state['measurement_history']:
            for measurement in state['measurement_history']:
                if 'timestamp' in measurement and measurement['timestamp']:
                    measurement['timestamp'] = datetime.fromisoformat(measurement['timestamp'])

        # Convert reset_events timestamps
        if 'reset_events' in state and state['reset_events']:
            for event in state['reset_events']:
                if 'timestamp' in event and event['timestamp']:
                    event['timestamp'] = datetime.fromisoformat(event['timestamp'])

        return state

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state for a user from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT state_data FROM user_states WHERE user_id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._deserialize_state(row[0])
            return None

        except Exception as e:
            logger.error(f"Error getting state for user {user_id}: {e}")
            return None

    def save_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """Save state for a user to SQLite."""
        try:
            state_json = self._serialize_state(state)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO user_states (user_id, state_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, state_json))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving state for user {user_id}: {e}")
            raise

    def delete_state(self, user_id: str) -> bool:
        """Delete state for a user from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete from user_states
            cursor.execute('DELETE FROM user_states WHERE user_id = ?', (user_id,))
            deleted = cursor.rowcount > 0

            # Also delete snapshots
            cursor.execute('DELETE FROM state_snapshots WHERE user_id = ?', (user_id,))

            # Delete replay buffer
            cursor.execute('DELETE FROM replay_buffers WHERE user_id = ?', (user_id,))

            conn.commit()
            conn.close()

            return deleted

        except Exception as e:
            logger.error(f"Error deleting state for user {user_id}: {e}")
            return False

    def save_buffer(self, user_id: str, buffer_data: List[Dict[str, Any]]) -> None:
        """Save replay buffer for a user."""
        try:
            buffer_json = json.dumps(buffer_data, default=str)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO replay_buffers (user_id, buffer_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, buffer_json))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving buffer for user {user_id}: {e}")
            raise

    def get_buffer(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get replay buffer for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT buffer_data FROM replay_buffers WHERE user_id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
            return None

        except Exception as e:
            logger.error(f"Error getting buffer for user {user_id}: {e}")
            return None

    def clear_buffer(self, user_id: str) -> bool:
        """Clear replay buffer for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM replay_buffers WHERE user_id = ?', (user_id,))
            cleared = cursor.rowcount > 0

            conn.commit()
            conn.close()

            return cleared

        except Exception as e:
            logger.error(f"Error clearing buffer for user {user_id}: {e}")
            return False

    def save_state_snapshot(self, user_id: str, timestamp: datetime) -> None:
        """Save a snapshot of current state for replay functionality."""
        try:
            # Get current state
            state = self.get_state(user_id)
            if not state:
                logger.warning(f"No state found for user {user_id} to snapshot")
                return

            state_json = self._serialize_state(state)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO state_snapshots (user_id, snapshot_time, state_data)
                VALUES (?, ?, ?)
            ''', (user_id, timestamp.isoformat(), state_json))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving snapshot for user {user_id}: {e}")
            raise

    def restore_state_snapshot(self, user_id: str) -> bool:
        """Restore state from latest snapshot."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get latest snapshot
            cursor.execute('''
                SELECT state_data FROM state_snapshots
                WHERE user_id = ?
                ORDER BY snapshot_time DESC
                LIMIT 1
            ''', (user_id,))

            row = cursor.fetchone()

            if row:
                # Restore the state
                state = self._deserialize_state(row[0])
                conn.close()
                self.save_state(user_id, state)
                return True

            conn.close()
            return False

        except Exception as e:
            logger.error(f"Error restoring snapshot for user {user_id}: {e}")
            return False