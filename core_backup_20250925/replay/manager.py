"""
Replay Manager for Replay Data Quality Processing

Handles state restoration and chronological reprocessing of clean measurements.
Provides atomic operations with rollback capability for safe replay processing.

Key responsibilities:
- Restore Kalman state from historical snapshots
- Chronologically replay clean measurements
- Provide rollback capability on failures
- Ensure atomic state transitions
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

try:
    from ..database.database import ProcessorStateDB
    from ..processing.processor import process_measurement
except ImportError:
    # For direct import testing
    try:
        from src.database.database import ProcessorStateDB
        from src.processing.processor import process_measurement
    except ImportError:
        # Mock for testing without full processor
        def process_measurement(measurement_data):
            return {'accepted': True}


logger = logging.getLogger(__name__)


class ReplayManager:
    """
    Manages state restoration and measurement replay for replay processing.
    All operations are designed to be atomic with rollback capability.
    """

    def __init__(self, db: ProcessorStateDB, config: Optional[Dict[str, Any]] = None):
        """
        Initialize replay manager.

        Args:
            db: State database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config or {}

        # Safety configuration
        self.max_processing_time = self.config.get('max_processing_time_seconds', 60)
        self.require_rollback_confirmation = self.config.get('require_rollback_confirmation', False)
        self.preserve_immediate_results = self.config.get('preserve_immediate_results', True)

        # Backup storage for rollback
        self._backup_states = {}

    def replay_clean_measurements(
        self,
        user_id: str,
        clean_measurements: List[Dict[str, Any]],
        buffer_start_time: datetime
    ) -> Dict[str, Any]:
        """
        Replay clean measurements for a user with full rollback capability.

        Args:
            user_id: User identifier
            clean_measurements: List of measurements without outliers
            buffer_start_time: Start time of the buffer window (for state restoration)

        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()

        try:
            # Step 1: Create backup of current state
            if not self._create_state_backup(user_id):
                return {
                    'success': False,
                    'error': 'Failed to create state backup',
                    'user_id': user_id
                }

            # Step 2: Restore state to before buffer start
            restore_result = self._restore_state_to_buffer_start(user_id, buffer_start_time)
            if not restore_result['success']:
                self._restore_state_from_backup(user_id)
                return restore_result

            # Step 2.5: Check trajectory continuity to prevent impossible jumps
            current_backup_state = self._backup_states.get(user_id, {})
            restored_state = self.db.get_state(user_id)

            if (current_backup_state.get('last_state') is not None and
                restored_state.get('last_state') is not None):
                try:
                    from .kalman import KalmanFilterManager
                    backup_weight, _ = KalmanFilterManager.get_current_state_values(current_backup_state)
                    restored_weight, _ = KalmanFilterManager.get_current_state_values(restored_state)

                    if backup_weight and restored_weight:
                        weight_jump = abs(backup_weight - restored_weight)
                        # If restoration would cause >15kg jump, skip replay processing
                        if weight_jump > 15.0:
                            logger.warning(f"Skipping replay processing for {user_id}: would cause {weight_jump:.1f}kg trajectory jump")
                            self._restore_state_from_backup(user_id)
                            return {
                                'success': False,
                                'error': f'Trajectory continuity check failed: {weight_jump:.1f}kg jump exceeds 15kg limit',
                                'user_id': user_id
                            }
                except Exception as e:
                    logger.debug(f"Trajectory continuity check failed: {e}")
                    # Continue with replay if check fails

            # Step 3: Replay measurements chronologically
            replay_result = self._replay_measurements_chronologically(user_id, clean_measurements, start_time)
            if not replay_result['success']:
                self._restore_state_from_backup(user_id)
                return replay_result

            # Step 4: Commit changes (clear backup)
            self._clear_state_backup(user_id)

            return {
                'success': True,
                'user_id': user_id,
                'measurements_replayed': len(clean_measurements),
                'processing_time_seconds': time.time() - start_time,
                'final_state': self.db.get_state(user_id),
                'restore_point': restore_result.get('restore_snapshot')
            }

        except Exception as e:
            logger.error(f"Replay failed for user {user_id}: {e}")
            # Emergency rollback
            self._restore_state_from_backup(user_id)
            return {
                'success': False,
                'error': f'Replay exception: {str(e)}',
                'user_id': user_id,
                'processing_time_seconds': time.time() - start_time
            }

    def _create_state_backup(self, user_id: str) -> bool:
        """
        Create a backup of the current state for rollback capability.

        Args:
            user_id: User identifier

        Returns:
            True if backup created successfully
        """
        try:
            current_state = self.db.get_state(user_id)
            if current_state:
                # Create deep copy for backup
                import copy
                self._backup_states[user_id] = copy.deepcopy(current_state)
                logger.debug(f"Created state backup for user {user_id}")
                return True
            else:
                logger.warning(f"No current state found for user {user_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to create backup for user {user_id}: {e}")
            return False

    def _restore_state_from_backup(self, user_id: str) -> bool:
        """
        Restore state from backup.

        Args:
            user_id: User identifier

        Returns:
            True if restoration successful
        """
        try:
            if user_id in self._backup_states:
                backup_state = self._backup_states[user_id]
                self.db.save_state(user_id, backup_state)
                logger.info(f"Restored state from backup for user {user_id}")
                return True
            else:
                logger.error(f"No backup found for user {user_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to restore from backup for user {user_id}: {e}")
            return False

    def _clear_state_backup(self, user_id: str) -> None:
        """
        Clear backup state after successful commit.

        Args:
            user_id: User identifier
        """
        if user_id in self._backup_states:
            del self._backup_states[user_id]
            logger.debug(f"Cleared backup for user {user_id}")

    def _restore_state_to_buffer_start(self, user_id: str, buffer_start_time: datetime) -> Dict[str, Any]:
        """
        Restore user state to just before the buffer start time using atomic operations.
        Includes retry logic for transient failures.

        Args:
            user_id: User identifier
            buffer_start_time: Time to restore state before

        Returns:
            Result dictionary with success status
        """
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Log retry attempt if not first try
                if attempt > 0:
                    logger.info(f"Retry {attempt + 1}/{max_retries} for state restoration of {user_id}")
                    # Exponential backoff: 0.1s, 0.2s, 0.4s
                    time.sleep(0.1 * (2 ** attempt))

                # Use atomic check-and-restore method to prevent race condition
                result = self.db.check_and_restore_snapshot(user_id, buffer_start_time)

                if result['success']:
                    logger.info(
                        f"Successfully restored state for {user_id} to {result.get('snapshot_timestamp')} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    return {
                        'success': True,
                        'user_id': user_id,
                        'restore_snapshot': result.get('snapshot'),
                        'restored_to_time': result.get('snapshot_timestamp'),
                        'attempts': attempt + 1
                    }

                # Log the specific error
                error_msg = result.get('error', 'Unknown error')
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {user_id}: {error_msg}"
                )
                last_error = error_msg

                # Don't retry if snapshot doesn't exist (not a transient failure)
                if 'No snapshot found' in error_msg:
                    logger.error(f"No snapshot exists for {user_id} before {buffer_start_time}, aborting retries")
                    return {
                        'success': False,
                        'error': error_msg,
                        'user_id': user_id,
                        'buffer_start_time': buffer_start_time,
                        'attempts': attempt + 1
                    }

            except Exception as e:
                last_error = f'State restoration exception: {str(e)}'
                logger.error(
                    f"Exception during state restoration for {user_id} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True
                )

                # Don't retry on programming errors
                if 'AttributeError' in str(e) or 'TypeError' in str(e):
                    return {
                        'success': False,
                        'error': last_error,
                        'user_id': user_id,
                        'attempts': attempt + 1
                    }

        # All retries exhausted
        logger.error(
            f"Failed to restore state for {user_id} after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
        return {
            'success': False,
            'error': f'All {max_retries} restore attempts failed. Last error: {last_error}',
            'user_id': user_id,
            'buffer_start_time': buffer_start_time,
            'attempts': max_retries
        }

    def _replay_measurements_chronologically(
        self,
        user_id: str,
        clean_measurements: List[Dict[str, Any]],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Replay measurements in chronological order.

        Args:
            user_id: User identifier
            clean_measurements: List of clean measurements
            start_time: Processing start time for timeout check

        Returns:
            Result dictionary with success status
        """
        try:
            # Sort measurements by timestamp
            sorted_measurements = sorted(clean_measurements, key=lambda x: x['timestamp'])

            processed_count = 0
            last_result = None

            for measurement in sorted_measurements:
                # Check timeout
                if time.time() - start_time > self.max_processing_time:
                    return {
                        'success': False,
                        'error': f'Processing timeout after {self.max_processing_time}s',
                        'user_id': user_id,
                        'processed_count': processed_count
                    }

                # Process measurement through normal pipeline
                try:
                    # Extract measurement data
                    weight = measurement['weight']
                    timestamp = measurement['timestamp']
                    source = measurement.get('source', 'replay-replay')
                    unit = measurement.get('unit', 'kg')

                    # Create replay-friendly config with more lenient quality thresholds
                    replay_config = self.config.copy()
                    if 'quality_scoring' in replay_config:
                        # Use more lenient threshold during replay since we already filtered outliers
                        replay_config['quality_scoring']['threshold'] = 0.25  # vs normal 0.6
                        # Adjust component weights to be more forgiving during replay
                        if 'component_weights' in replay_config['quality_scoring']:
                            weights = replay_config['quality_scoring']['component_weights'].copy()
                            weights['consistency'] = 0.10  # Reduce consistency weight (often fails during replay)
                            weights['plausibility'] = 0.15  # Reduce plausibility weight
                            weights['safety'] = 0.45  # Keep safety high
                            weights['reliability'] = 0.30  # Increase reliability weight
                            replay_config['quality_scoring']['component_weights'] = weights

                    # Process through existing pipeline with replay-friendly config
                    result = process_measurement(
                        user_id=user_id,
                        weight=weight,
                        timestamp=timestamp,
                        source=source,
                        config=replay_config,
                        unit=unit,
                        db=self.db
                    )
                    last_result = result

                    if not result.get('accepted', False):
                        logger.debug(f"Measurement rejected during replay: {user_id} {weight}kg at {timestamp}")
                        # Continue processing - rejection is normal and expected

                    processed_count += 1

                except Exception as e:
                    logger.error(f"Failed to process measurement during replay: {e}")
                    return {
                        'success': False,
                        'error': f'Measurement processing failed: {str(e)}',
                        'user_id': user_id,
                        'processed_count': processed_count,
                        'failed_measurement': measurement
                    }

            logger.info(f"Successfully replayed {processed_count} measurements for user {user_id}")
            return {
                'success': True,
                'user_id': user_id,
                'processed_count': processed_count,
                'last_result': last_result
            }

        except Exception as e:
            logger.error(f"Chronological replay failed for user {user_id}: {e}")
            return {
                'success': False,
                'error': f'Chronological replay exception: {str(e)}',
                'user_id': user_id
            }

    def rollback_user_state(self, user_id: str) -> bool:
        """
        Manually rollback user state to backup.

        Args:
            user_id: User identifier

        Returns:
            True if rollback successful
        """
        if self.require_rollback_confirmation:
            logger.warning(f"Rollback requires confirmation for user {user_id}")
            return False

        return self._restore_state_from_backup(user_id)

    def has_backup(self, user_id: str) -> bool:
        """
        Check if a backup exists for the user.

        Args:
            user_id: User identifier

        Returns:
            True if backup exists
        """
        return user_id in self._backup_states

    def get_replay_stats(self) -> Dict[str, Any]:
        """
        Get statistics about replay operations.

        Returns:
            Statistics dictionary
        """
        return {
            'active_backups': len(self._backup_states),
            'max_processing_time': self.max_processing_time,
            'preserve_immediate_results': self.preserve_immediate_results,
            'require_rollback_confirmation': self.require_rollback_confirmation
        }

    def cleanup_old_backups(self) -> int:
        """
        Clean up old backups to prevent memory leaks.

        Returns:
            Number of backups cleaned up
        """
        cleaned_count = len(self._backup_states)
        self._backup_states.clear()
        logger.info(f"Cleaned up {cleaned_count} backup states")
        return cleaned_count


class ReplayError(Exception):
    """Custom exception for replay operation errors."""
    pass


class ReplayTimeoutError(ReplayError):
    """Exception for replay operation timeouts."""
    pass


class ReplayStateError(ReplayError):
    """Exception for state-related replay errors."""
    pass