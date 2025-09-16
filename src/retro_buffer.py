"""
RetroBuffer - Thread-safe 72-hour measurement buffering

Manages in-memory storage of measurements for retrospective analysis.
Automatically manages buffer windows and provides thread-safe operations.

Key features:
- Thread-safe buffer management
- Automatic 72-hour window rotation
- Memory-efficient storage with limits
- Buffer state tracking and cleanup
"""

import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import time


logger = logging.getLogger(__name__)


class RetroBuffer:
    """
    Thread-safe buffer for storing measurements in 72-hour windows.
    Each user has their own measurement buffer with automatic window management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize retrospective buffer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Configuration
        self.buffer_hours = self.config.get('buffer_hours', 72)
        self.max_buffer_measurements = self.config.get('max_buffer_measurements', 100)
        self.trigger_mode = self.config.get('trigger_mode', 'time_based')

        # Buffer storage: user_id -> buffer_data
        self.buffers = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'total_measurements_buffered': 0,
            'buffers_created': 0,
            'buffers_triggered': 0,
            'buffers_cleaned': 0,
            'last_cleanup_time': datetime.now()
        }

    def add_measurement(self, user_id: str, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a measurement to the user's buffer.

        Args:
            user_id: User identifier
            measurement: Measurement dictionary with 'weight', 'timestamp', etc.

        Returns:
            Result dictionary with buffer status and trigger information
        """
        with self._lock:
            try:
                # Ensure buffer exists for user
                if user_id not in self.buffers:
                    self._create_user_buffer(user_id)

                buffer_data = self.buffers[user_id]
                timestamp = measurement['timestamp']

                # Add measurement to buffer
                measurement_copy = measurement.copy()
                buffer_data['measurements'].append(measurement_copy)

                # Update buffer timestamps
                if not buffer_data['first_timestamp'] or timestamp < buffer_data['first_timestamp']:
                    buffer_data['first_timestamp'] = timestamp
                if not buffer_data['last_timestamp'] or timestamp > buffer_data['last_timestamp']:
                    buffer_data['last_timestamp'] = timestamp

                # Update statistics
                self._stats['total_measurements_buffered'] += 1

                # Check buffer limits
                self._enforce_buffer_limits(user_id)

                # Check if buffer should be triggered for processing
                trigger_result = self._check_buffer_trigger(user_id)

                logger.debug(f"Added measurement for user {user_id}, buffer size: {len(buffer_data['measurements'])}")

                return {
                    'success': True,
                    'user_id': user_id,
                    'buffer_size': len(buffer_data['measurements']),
                    'buffer_ready': trigger_result['should_trigger'],
                    'trigger_reason': trigger_result.get('reason', 'not_ready'),
                    'buffer_window_start': buffer_data['first_timestamp'],
                    'buffer_window_end': buffer_data['last_timestamp']
                }

            except Exception as e:
                logger.error(f"Failed to add measurement for user {user_id}: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'user_id': user_id
                }

    def get_buffer_measurements(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get all buffered measurements for a user.

        Args:
            user_id: User identifier

        Returns:
            List of measurements or None if no buffer exists
        """
        with self._lock:
            if user_id not in self.buffers:
                return None

            # Return copy to prevent external modification
            measurements = self.buffers[user_id]['measurements']
            return [m.copy() for m in measurements]

    def clear_buffer(self, user_id: str) -> bool:
        """
        Clear buffer for a user (typically after processing).

        Args:
            user_id: User identifier

        Returns:
            True if buffer was cleared
        """
        with self._lock:
            if user_id in self.buffers:
                old_size = len(self.buffers[user_id]['measurements'])
                self._create_user_buffer(user_id)  # Reset to empty buffer
                self._stats['buffers_cleaned'] += 1
                logger.info(f"Cleared buffer for user {user_id} (was {old_size} measurements)")
                return True
            return False

    def get_buffer_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a user's buffer.

        Args:
            user_id: User identifier

        Returns:
            Buffer information dictionary or None
        """
        with self._lock:
            if user_id not in self.buffers:
                return None

            buffer_data = self.buffers[user_id]
            measurements = buffer_data['measurements']

            info = {
                'user_id': user_id,
                'measurement_count': len(measurements),
                'first_timestamp': buffer_data['first_timestamp'],
                'last_timestamp': buffer_data['last_timestamp'],
                'buffer_age_hours': 0,
                'is_ready_for_processing': False,
                'trigger_reason': None
            }

            # Calculate buffer age
            if buffer_data['first_timestamp']:
                age_delta = datetime.now() - buffer_data['first_timestamp']
                info['buffer_age_hours'] = age_delta.total_seconds() / 3600

            # Check if ready for processing
            trigger_result = self._check_buffer_trigger(user_id)
            info['is_ready_for_processing'] = trigger_result['should_trigger']
            info['trigger_reason'] = trigger_result.get('reason')

            return info

    def get_ready_buffers(self) -> List[str]:
        """
        Get list of user IDs with buffers ready for processing.

        Returns:
            List of user IDs ready for retrospective processing
        """
        with self._lock:
            ready_users = []

            for user_id in self.buffers:
                trigger_result = self._check_buffer_trigger(user_id)
                if trigger_result['should_trigger']:
                    ready_users.append(user_id)

            return ready_users

    def cleanup_old_buffers(self, max_age_hours: Optional[float] = None) -> int:
        """
        Clean up old buffers that haven't been active.

        Args:
            max_age_hours: Maximum age in hours (defaults to 2x buffer_hours)

        Returns:
            Number of buffers cleaned up
        """
        if max_age_hours is None:
            max_age_hours = self.buffer_hours * 2

        with self._lock:
            current_time = datetime.now()
            users_to_remove = []

            for user_id, buffer_data in self.buffers.items():
                last_timestamp = buffer_data['last_timestamp']
                if last_timestamp:
                    age_hours = (current_time - last_timestamp).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        users_to_remove.append(user_id)

            # Remove old buffers
            for user_id in users_to_remove:
                del self.buffers[user_id]

            self._stats['buffers_cleaned'] += len(users_to_remove)
            self._stats['last_cleanup_time'] = current_time

            if users_to_remove:
                logger.info(f"Cleaned up {len(users_to_remove)} old buffers")

            return len(users_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'active_buffers': len(self.buffers),
                'total_buffered_measurements': sum(len(b['measurements']) for b in self.buffers.values()),
                'ready_for_processing': len(self.get_ready_buffers()),
                'config': {
                    'buffer_hours': self.buffer_hours,
                    'max_buffer_measurements': self.max_buffer_measurements,
                    'trigger_mode': self.trigger_mode
                }
            })
            return stats

    def _create_user_buffer(self, user_id: str) -> None:
        """
        Create empty buffer for user.

        Args:
            user_id: User identifier
        """
        self.buffers[user_id] = {
            'measurements': [],
            'first_timestamp': None,
            'last_timestamp': None,
            'created_at': datetime.now()
        }
        self._stats['buffers_created'] += 1
        logger.debug(f"Created buffer for user {user_id}")

    def _enforce_buffer_limits(self, user_id: str) -> None:
        """
        Enforce buffer size limits by removing oldest measurements.

        Args:
            user_id: User identifier
        """
        buffer_data = self.buffers[user_id]
        measurements = buffer_data['measurements']

        if len(measurements) > self.max_buffer_measurements:
            # Sort by timestamp and keep most recent
            measurements.sort(key=lambda x: x['timestamp'])
            buffer_data['measurements'] = measurements[-self.max_buffer_measurements:]

            # Update first timestamp
            if buffer_data['measurements']:
                buffer_data['first_timestamp'] = buffer_data['measurements'][0]['timestamp']

            logger.debug(f"Enforced buffer limit for user {user_id}, kept {len(buffer_data['measurements'])} measurements")

    def _check_buffer_trigger(self, user_id: str) -> Dict[str, Any]:
        """
        Check if buffer should be triggered for processing.

        Args:
            user_id: User identifier

        Returns:
            Trigger result dictionary
        """
        buffer_data = self.buffers[user_id]
        measurements = buffer_data['measurements']

        if not measurements or not buffer_data['first_timestamp']:
            return {'should_trigger': False, 'reason': 'empty_buffer'}

        # Check minimum measurements requirement for meaningful analysis
        outlier_config = self.config.get('outlier_detection', {})
        min_measurements = outlier_config.get('min_measurements_for_analysis', 5)

        if len(measurements) < min_measurements:
            return {
                'should_trigger': False,
                'reason': 'insufficient_measurements',
                'measurement_count': len(measurements),
                'min_required': min_measurements
            }

        # Calculate buffer age
        buffer_age = datetime.now() - buffer_data['first_timestamp']
        age_hours = buffer_age.total_seconds() / 3600

        if self.trigger_mode == 'time_based':
            # Trigger when buffer reaches configured age AND has enough measurements
            if age_hours >= self.buffer_hours:
                return {
                    'should_trigger': True,
                    'reason': 'time_based_trigger',
                    'buffer_age_hours': age_hours,
                    'trigger_threshold_hours': self.buffer_hours,
                    'measurement_count': len(measurements)
                }

        elif self.trigger_mode == 'measurement_count':
            # Trigger when buffer reaches measurement limit
            if len(measurements) >= self.max_buffer_measurements:
                return {
                    'should_trigger': True,
                    'reason': 'measurement_count_trigger',
                    'measurement_count': len(measurements),
                    'trigger_threshold': self.max_buffer_measurements
                }

        return {
            'should_trigger': False,
            'reason': 'threshold_not_reached',
            'buffer_age_hours': age_hours,
            'measurement_count': len(measurements)
        }

    def force_trigger_buffer(self, user_id: str) -> bool:
        """
        Force trigger a buffer for immediate processing (for testing/debugging).

        Args:
            user_id: User identifier

        Returns:
            True if buffer exists and can be triggered
        """
        with self._lock:
            if user_id in self.buffers and self.buffers[user_id]['measurements']:
                logger.info(f"Force triggered buffer for user {user_id}")
                return True
            return False


# Global buffer instance
_buffer_instance = None


def get_retro_buffer(config: Optional[Dict[str, Any]] = None) -> RetroBuffer:
    """Get or create the global retro buffer instance."""
    global _buffer_instance
    if _buffer_instance is None:
        _buffer_instance = RetroBuffer(config)
    return _buffer_instance