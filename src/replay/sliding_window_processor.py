"""
Sliding Window Processor for continuous replay analysis.

Instead of waiting for full buffer windows, this processor:
1. Maintains overlapping windows for faster detection
2. Triggers immediate replay for suspicious patterns
3. Allows for proactive correction before full buffer
"""

import logging
from typing import Dict, Any, List, Optional, Deque
from datetime import datetime, timedelta
from collections import deque
import threading

from .enhanced_replay_analyzer import EnhancedReplayAnalyzer

logger = logging.getLogger(__name__)


class SlidingWindowProcessor:
    """
    Processes measurements in sliding windows for continuous quality monitoring.
    """

    def __init__(self, db, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sliding window processor.

        Args:
            db: Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config or {}

        # Window configuration
        self.window_size = config.get('window_size', 10)  # Number of measurements
        self.slide_interval = config.get('slide_interval', 3)  # Slide every N measurements
        self.min_window_size = config.get('min_window_size', 5)  # Minimum for analysis
        self.immediate_trigger_threshold = config.get('immediate_trigger_threshold', 0.2)  # Score for immediate action

        # User windows - maintains sliding windows per user
        self.user_windows: Dict[str, Deque[Dict[str, Any]]] = {}

        # Analyzer
        self.analyzer = EnhancedReplayAnalyzer(db, config.get('analysis', {}))

        # Thread safety
        self._lock = threading.RLock()

        # Tracking
        self.measurements_processed = 0
        self.windows_analyzed = 0
        self.immediate_triggers = 0

    def add_measurement(
        self,
        user_id: str,
        measurement: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Add a measurement and check if window analysis should trigger.

        Args:
            user_id: User identifier
            measurement: Measurement dictionary

        Returns:
            Analysis result if triggered, None otherwise
        """
        with self._lock:
            # Initialize window if needed
            if user_id not in self.user_windows:
                self.user_windows[user_id] = deque(maxlen=self.window_size)

            window = self.user_windows[user_id]
            window.append(measurement.copy())
            self.measurements_processed += 1

            # Check if we should analyze
            if len(window) >= self.min_window_size:
                # Check for immediate trigger conditions
                if self._check_immediate_trigger(list(window)):
                    self.immediate_triggers += 1
                    return self._analyze_window(user_id, list(window), immediate=True)

                # Check for regular sliding interval
                if len(window) % self.slide_interval == 0:
                    return self._analyze_window(user_id, list(window), immediate=False)

            return None

    def _check_immediate_trigger(self, window: List[Dict[str, Any]]) -> bool:
        """
        Check if the window contains patterns requiring immediate analysis.

        Triggers on:
        - Large sudden changes (>20% in one measurement)
        - Multiple rejected measurements
        - Reset patterns that look incorrect

        Args:
            window: List of measurements

        Returns:
            True if immediate analysis is needed
        """
        if len(window) < 2:
            return False

        # Check for large jumps
        for i in range(1, len(window)):
            prev_weight = window[i-1].get('weight', 0)
            curr_weight = window[i].get('weight', 0)

            if prev_weight > 0:
                change = abs(curr_weight - prev_weight) / prev_weight
                if change > 0.2:  # 20% change
                    logger.info(f"Immediate trigger: {change:.1%} weight change detected")
                    return True

        # Check for multiple rejections
        rejected_count = sum(
            1 for m in window[-3:]  # Last 3 measurements
            if m.get('metadata', {}).get('accepted') == False
        )
        if rejected_count >= 2:
            logger.info(f"Immediate trigger: {rejected_count} recent rejections")
            return True

        # Check for potential bad reset
        # If first measurement has very different weight from rest
        if len(window) >= 3:
            first_weight = window[0].get('weight', 0)
            avg_rest = sum(m.get('weight', 0) for m in window[1:]) / (len(window) - 1)

            if first_weight > 0 and avg_rest > 0:
                deviation = abs(first_weight - avg_rest) / first_weight
                if deviation > 0.15:  # 15% deviation
                    logger.info(f"Immediate trigger: Potential bad anchor detected")
                    return True

        return False

    def _analyze_window(
        self,
        user_id: str,
        window: List[Dict[str, Any]],
        immediate: bool
    ) -> Dict[str, Any]:
        """
        Analyze a window of measurements.

        Args:
            user_id: User identifier
            window: Window of measurements
            immediate: Whether this is an immediate trigger

        Returns:
            Analysis result
        """
        self.windows_analyzed += 1

        try:
            # Get window time bounds
            window_start = min(m['timestamp'] for m in window)
            window_end = max(m['timestamp'] for m in window)

            # Run enhanced analysis
            clean_measurements, analysis = self.analyzer.analyze_measurements_with_reset_context(
                window,
                user_id,
                window_start
            )

            # Create result
            result = {
                'user_id': user_id,
                'window_size': len(window),
                'window_start': window_start,
                'window_end': window_end,
                'immediate_trigger': immediate,
                'analysis': analysis,
                'action_needed': self._determine_action(analysis, immediate)
            }

            # Log significant findings
            if analysis.get('outliers_found', 0) > 0:
                logger.info(
                    f"Window analysis for {user_id}: "
                    f"{analysis['outliers_found']} outliers in {len(window)} measurements"
                )

            if analysis.get('reset_changes'):
                logger.warning(
                    f"Reset change recommended for {user_id}: "
                    f"{analysis['reset_changes'].get('reason', 'unknown')}"
                )

            return result

        except Exception as e:
            logger.error(f"Error analyzing window for {user_id}: {e}")
            return {
                'user_id': user_id,
                'error': str(e),
                'window_size': len(window)
            }

    def _determine_action(
        self,
        analysis: Dict[str, Any],
        immediate: bool
    ) -> str:
        """
        Determine what action to take based on analysis.

        Returns:
            Action recommendation string
        """
        if analysis.get('reset_changes', {}).get('should_change'):
            return 'reset_correction_needed'

        outliers = analysis.get('outliers_found', 0)
        total = analysis.get('total_measurements', 1)

        if outliers == 0:
            return 'no_action'

        outlier_rate = outliers / total if total > 0 else 0

        if immediate and outlier_rate > 0.3:
            return 'immediate_replay_recommended'
        elif outlier_rate > 0.5:
            return 'replay_recommended'
        elif outlier_rate > 0.2:
            return 'monitor_closely'
        else:
            return 'minor_correction'

    def get_window_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a user's window.

        Args:
            user_id: User identifier

        Returns:
            Window statistics or None
        """
        with self._lock:
            if user_id not in self.user_windows:
                return None

            window = self.user_windows[user_id]

            if not window:
                return {
                    'user_id': user_id,
                    'window_size': 0,
                    'measurements': []
                }

            measurements = list(window)
            weights = [m.get('weight', 0) for m in measurements]

            return {
                'user_id': user_id,
                'window_size': len(window),
                'oldest_timestamp': measurements[0].get('timestamp'),
                'newest_timestamp': measurements[-1].get('timestamp'),
                'weight_range': (min(weights), max(weights)) if weights else (0, 0),
                'weight_mean': sum(weights) / len(weights) if weights else 0
            }

    def clear_window(self, user_id: str) -> bool:
        """
        Clear a user's sliding window.

        Args:
            user_id: User identifier

        Returns:
            True if window was cleared
        """
        with self._lock:
            if user_id in self.user_windows:
                self.user_windows[user_id].clear()
                return True
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get processor metrics.

        Returns:
            Metrics dictionary
        """
        with self._lock:
            active_windows = sum(
                1 for window in self.user_windows.values()
                if len(window) > 0
            )

            total_measurements = sum(
                len(window) for window in self.user_windows.values()
            )

            return {
                'measurements_processed': self.measurements_processed,
                'windows_analyzed': self.windows_analyzed,
                'immediate_triggers': self.immediate_triggers,
                'active_windows': active_windows,
                'total_users': len(self.user_windows),
                'total_measurements_in_windows': total_measurements,
                'trigger_rate': (
                    self.immediate_triggers / self.windows_analyzed
                    if self.windows_analyzed > 0 else 0
                )
            }