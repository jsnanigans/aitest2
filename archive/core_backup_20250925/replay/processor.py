"""
Replay Processor - Integrates enhanced replay analysis with main processing flow.

Handles the complete replay workflow:
1. Buffer analysis with enhanced scoring
2. Reset re-evaluation and correction
3. State restoration and reprocessing
4. Metrics tracking
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from .enhanced_replay_analyzer import EnhancedReplayAnalyzer
from .replay_manager import ReplayManager

logger = logging.getLogger(__name__)


class ReplayProcessor:
    """
    Orchestrates replay processing with enhanced analysis capabilities.
    """

    def __init__(self, db, config: Optional[Dict[str, Any]] = None):
        """
        Initialize replay processor.

        Args:
            db: Database instance
            config: Replay configuration
        """
        self.db = db
        self.config = config or {}

        # Initialize components
        self.analyzer = EnhancedReplayAnalyzer(db, config.get('analysis', {}))
        self.replay_manager = ReplayManager(db, config.get('safety', {}))

        # Metrics tracking
        self.metrics = {
            'buffers_processed': 0,
            'measurements_analyzed': 0,
            'outliers_found': 0,
            'resets_changed': 0,
            'corrections_made': 0,
            'replay_time_total': 0.0
        }

    def process_buffer(
        self,
        user_id: str,
        buffered_measurements: List[Dict[str, Any]],
        buffer_start_time: datetime
    ) -> Dict[str, Any]:
        """
        Process a buffer of measurements with enhanced analysis.

        Args:
            user_id: User identifier
            buffered_measurements: List of measurements from buffer
            buffer_start_time: Start time of the buffer window

        Returns:
            Processing result dictionary
        """
        start_time = time.time()

        try:
            # Update metrics
            self.metrics['buffers_processed'] += 1
            self.metrics['measurements_analyzed'] += len(buffered_measurements)

            # Step 1: Analyze measurements with enhanced scoring
            logger.info(f"Analyzing {len(buffered_measurements)} measurements for {user_id}")

            clean_measurements, analysis = self.analyzer.analyze_measurements_with_reset_context(
                buffered_measurements,
                user_id,
                buffer_start_time
            )

            # Update metrics from analysis
            self.metrics['outliers_found'] += analysis.get('outliers_found', 0)

            # Step 2: Check for reset changes
            reset_changes = analysis.get('reset_changes')
            if reset_changes and reset_changes.get('should_change'):
                self.metrics['resets_changed'] += 1
                logger.warning(f"Reset change recommended for {user_id}: {reset_changes['reason']}")

                # Handle reset change
                reset_result = self._handle_reset_change(
                    user_id, reset_changes, clean_measurements, buffer_start_time
                )

                if not reset_result['success']:
                    return {
                        'success': False,
                        'error': f"Failed to handle reset change: {reset_result['error']}",
                        'analysis': analysis
                    }

            # Step 3: Replay clean measurements if we found outliers
            if analysis.get('outliers_found', 0) > 0:
                logger.info(f"Replaying {len(clean_measurements)} clean measurements for {user_id}")

                replay_result = self.replay_manager.replay_clean_measurements(
                    user_id=user_id,
                    clean_measurements=clean_measurements,
                    buffer_start_time=buffer_start_time
                )

                if replay_result['success']:
                    self.metrics['corrections_made'] += 1

                # Add analysis to result
                replay_result['analysis'] = analysis

                # Track processing time
                processing_time = time.time() - start_time
                self.metrics['replay_time_total'] += processing_time
                replay_result['processing_time'] = processing_time

                return replay_result
            else:
                logger.info(f"No outliers found for {user_id}, skipping replay")

                return {
                    'success': True,
                    'user_id': user_id,
                    'message': 'No outliers found, replay not needed',
                    'analysis': analysis,
                    'processing_time': time.time() - start_time
                }

        except Exception as e:
            logger.error(f"Error processing buffer for {user_id}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'user_id': user_id,
                'processing_time': time.time() - start_time
            }

    def _handle_reset_change(
        self,
        user_id: str,
        reset_changes: Dict[str, Any],
        clean_measurements: List[Dict[str, Any]],
        buffer_start_time: datetime
    ) -> Dict[str, Any]:
        """
        Handle a recommended reset change.

        This involves:
        1. Restoring to before the original reset
        2. Applying the new reset at the better anchor point
        3. Reprocessing from that point

        Args:
            user_id: User identifier
            reset_changes: Reset change recommendations
            clean_measurements: Clean measurements list
            buffer_start_time: Buffer start time

        Returns:
            Result dictionary
        """
        try:
            original_reset = reset_changes['original_reset']
            new_anchor = reset_changes['new_anchor']

            logger.info(
                f"Changing reset anchor for {user_id} from "
                f"{original_reset['weight']:.1f}kg to {new_anchor['weight']:.1f}kg"
            )

            # Create a modified measurement list with the new reset point
            modified_measurements = []

            for i, measurement in enumerate(clean_measurements):
                measurement_copy = measurement.copy()

                # Mark the new reset point
                if i == new_anchor['index']:
                    measurement_copy['force_reset'] = True
                    measurement_copy['reset_type'] = 'corrected'

                # Skip the original reset point if it's now an outlier
                if i == original_reset['index'] and new_anchor['index'] != original_reset['index']:
                    continue

                modified_measurements.append(measurement_copy)

            # Now replay with the modified measurements
            replay_result = self.replay_manager.replay_clean_measurements(
                user_id=user_id,
                clean_measurements=modified_measurements,
                buffer_start_time=buffer_start_time
            )

            if replay_result['success']:
                replay_result['reset_changed'] = True
                replay_result['reset_change_details'] = reset_changes

            return replay_result

        except Exception as e:
            logger.error(f"Failed to handle reset change for {user_id}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'user_id': user_id
            }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get replay processing metrics.

        Returns:
            Metrics dictionary
        """
        metrics = self.metrics.copy()

        # Calculate derived metrics
        if metrics['buffers_processed'] > 0:
            metrics['avg_processing_time'] = (
                metrics['replay_time_total'] / metrics['buffers_processed']
            )
            metrics['correction_rate'] = (
                metrics['corrections_made'] / metrics['buffers_processed']
            )
        else:
            metrics['avg_processing_time'] = 0.0
            metrics['correction_rate'] = 0.0

        if metrics['measurements_analyzed'] > 0:
            metrics['outlier_rate'] = (
                metrics['outliers_found'] / metrics['measurements_analyzed']
            )
        else:
            metrics['outlier_rate'] = 0.0

        return metrics

    def reset_metrics(self):
        """Reset all metrics to zero."""
        self.metrics = {
            'buffers_processed': 0,
            'measurements_analyzed': 0,
            'outliers_found': 0,
            'resets_changed': 0,
            'corrections_made': 0,
            'replay_time_total': 0.0
        }