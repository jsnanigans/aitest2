"""
Database Wrapper for Analysis Module

Provides simplified database access for the analysis report generator.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .database import ProcessorStateDB


class Database:
    """Simplified database interface for analysis module"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database wrapper

        Args:
            db_path: Path to database file (optional)
        """
        self.db_path = db_path
        self.processor_db = ProcessorStateDB(db_path)
        self.logger = logging.getLogger(__name__)

    def get_user_measurements(self, user_id: str) -> List[Dict]:
        """
        Get all measurements for a user from the database

        Args:
            user_id: User identifier

        Returns:
            List of measurement dictionaries
        """
        state = self.processor_db.get_state(user_id)

        if not state:
            return []

        measurements = []

        # Get measurement history if available
        if state.get('measurement_history'):
            for measurement in state['measurement_history']:
                measurements.append({
                    'timestamp': measurement.get('timestamp'),
                    'weight': measurement.get('weight'),
                    'raw_weight': measurement.get('weight'),  # Use same value for now
                    'filtered_weight': measurement.get('filtered_weight', measurement.get('weight')),
                    'source': measurement.get('source', 'unknown'),
                    'quality_score': measurement.get('quality_score', 0.5),
                    'is_outlier': measurement.get('is_outlier', False)
                })

        # Add the last measurement if not in history
        if state.get('last_timestamp') and state.get('last_state'):
            last_measurement = {
                'timestamp': state['last_timestamp'],
                'weight': state.get('last_raw_weight'),
                'raw_weight': state.get('last_raw_weight'),
                'source': state.get('last_source', 'unknown'),
                'quality_score': 0.5,  # Default quality
                'is_outlier': False
            }

            # Get filtered weight from Kalman state
            last_state = state['last_state']
            if isinstance(last_state, (list, tuple)) and len(last_state) >= 2:
                last_measurement['filtered_weight'] = float(last_state[0])
            elif hasattr(last_state, '__len__') and len(last_state) >= 2:
                last_measurement['filtered_weight'] = float(last_state[0])

            # Check if this measurement is already in the list
            if not measurements or measurements[-1].get('timestamp') != last_measurement['timestamp']:
                measurements.append(last_measurement)

        return measurements

    def get_user_signup_date(self, user_id: str) -> Optional[datetime]:
        """
        Get the signup date for a user (first measurement or initial-questionnaire)

        Args:
            user_id: User identifier

        Returns:
            Signup datetime or None
        """
        measurements = self.get_user_measurements(user_id)

        if not measurements:
            return None

        # Look for initial-questionnaire source
        for m in measurements:
            if m.get('source') == 'initial-questionnaire':
                return m['timestamp']

        # Fallback to first measurement
        return measurements[0]['timestamp'] if measurements else None

    def get_all_users(self) -> List[str]:
        """
        Get list of all users in the database

        Returns:
            List of user IDs
        """
        return list(self.processor_db.states.keys())

    def get_all_users_with_counts(self) -> List[Dict]:
        """
        Get all users with their measurement counts

        Returns:
            List of dictionaries with user_id and measurement_count
        """
        users = []
        for user_id in self.get_all_users():
            measurements = self.get_user_measurements(user_id)
            users.append({
                'user_id': user_id,
                'measurement_count': len(measurements)
            })
        return users

    def get_measurements_in_window(self, user_id: str, start: datetime, end: datetime) -> List[Dict]:
        """
        Get measurements for a user within a time window

        Args:
            user_id: User identifier
            start: Start datetime
            end: End datetime

        Returns:
            List of measurements in the window
        """
        all_measurements = self.get_user_measurements(user_id)

        filtered = []
        for m in all_measurements:
            if m['timestamp'] and start <= m['timestamp'] <= end:
                filtered.append(m)

        return filtered