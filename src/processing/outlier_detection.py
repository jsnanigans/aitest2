"""
Batch Outlier Detection for Retrospective Data Quality Processing

Implements statistical methods to identify outliers in buffered measurement batches:
- IQR Method: Interquartile Range-based outlier detection
- Modified Z-Score: Median Absolute Deviation-based detection
- Temporal Consistency: Rate-of-change based detection

All methods are designed for batch analysis (not streaming) and focus on
identifying measurements that would cause Kalman filter instability.
"""

import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime, timedelta
import statistics
from ..feature_manager import FeatureManager


class OutlierDetector:
    """
    Statistical outlier detection for batch measurement analysis.
    Designed to identify problematic measurements before Kalman processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, db=None):
        """
        Initialize outlier detector with configuration.

        Args:
            config: Configuration dict with thresholds and method settings
            db: Database instance for accessing Kalman states
        """
        self.config = config or {}
        self.db = db

        # Get feature manager
        self.feature_manager = config.get('feature_manager') if config else None
        if not self.feature_manager:
            self.feature_manager = FeatureManager(config)

        # Default thresholds (can be overridden by config)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        self.z_score_threshold = self.config.get('z_score_threshold', 3.0)
        self.temporal_max_change_percent = self.config.get('temporal_max_change_percent', 0.30)
        self.min_measurements_for_analysis = self.config.get('min_measurements_for_analysis', 5)

        # Quality score threshold - measurements with quality > this are never outliers
        self.quality_score_threshold = self.config.get('quality_score_threshold', 0.7)

        # Kalman prediction deviation threshold (as percentage)
        self.kalman_deviation_threshold = self.config.get('kalman_deviation_threshold', 0.15)

    def detect_outliers(self, measurements: List[Dict[str, Any]], user_id: Optional[str] = None) -> Set[int]:
        """
        Detect outliers in a batch of measurements using multiple methods.
        Respects quality scores and uses Kalman prediction deviation.

        Args:
            measurements: List of measurement dictionaries with 'weight', 'timestamp', 'metadata'
            user_id: Optional user identifier for accessing Kalman state

        Returns:
            Set of indices that are considered outliers
        """
        # If outlier detection is completely disabled, return empty set
        if not self.feature_manager.is_enabled('outlier_detection'):
            return set()

        if len(measurements) < self.min_measurements_for_analysis:
            return set()

        # Extract weights and sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])
        weights = [m['weight'] for m in sorted_measurements]

        # First, identify high-quality measurements that should never be marked as outliers
        protected_indices = set()

        # Only apply quality override if feature is enabled
        if self.feature_manager.is_enabled('quality_override'):
            for i, measurement in enumerate(sorted_measurements):
                metadata = measurement.get('metadata', {})

                # Check if measurement has quality score
                quality_score = metadata.get('quality_score')
                if quality_score is not None and quality_score > self.quality_score_threshold:
                    protected_indices.add(i)

                # Also protect measurements that were explicitly accepted
                if metadata.get('accepted', False):
                    protected_indices.add(i)

        # Collect potential outliers from statistical methods
        statistical_outliers = set()

        # Method 1: IQR-based detection
        if self.feature_manager.is_enabled('outlier_iqr'):
            iqr_outliers = self._detect_iqr_outliers(weights)
            statistical_outliers.update(iqr_outliers)

        # Method 2: Modified Z-score detection
        if self.feature_manager.is_enabled('outlier_mad'):
            zscore_outliers = self._detect_zscore_outliers(weights)
            statistical_outliers.update(zscore_outliers)

        # Method 3: Temporal consistency check
        if self.feature_manager.is_enabled('outlier_temporal'):
            temporal_outliers = self._detect_temporal_outliers(sorted_measurements)
            statistical_outliers.update(temporal_outliers)

        # Method 4: Kalman prediction deviation (if database and user_id available)
        kalman_outliers = set()
        if self.db and user_id and self.feature_manager.is_enabled('kalman_deviation_check'):
            kalman_outliers = self._detect_kalman_outliers(sorted_measurements, user_id)

        # AND logic: A measurement is only an outlier if:
        # 1. It's NOT protected by high quality score, AND
        # 2. It fails BOTH statistical tests AND Kalman prediction (if available)
        final_outliers = set()

        for idx in range(len(sorted_measurements)):
            # Skip if protected by quality score
            if idx in protected_indices:
                continue

            # Check if it fails statistical tests
            if idx not in statistical_outliers:
                continue

            # If we have Kalman predictions, also require it to fail that test
            if kalman_outliers:
                if idx not in kalman_outliers:
                    continue

            # This measurement is an outlier
            final_outliers.add(idx)

        return final_outliers

    def _detect_iqr_outliers(self, weights: List[float]) -> Set[int]:
        """
        Detect outliers using Interquartile Range method.

        Args:
            weights: List of weight values

        Returns:
            Set of indices that are IQR outliers
        """
        if len(weights) < 4:  # Need at least 4 points for quartiles
            return set()

        q1 = np.percentile(weights, 25)
        q3 = np.percentile(weights, 75)
        iqr = q3 - q1

        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)

        outliers = set()
        for i, weight in enumerate(weights):
            if weight < lower_bound or weight > upper_bound:
                outliers.add(i)

        return outliers

    def _detect_zscore_outliers(self, weights: List[float]) -> Set[int]:
        """
        Detect outliers using Modified Z-Score method (median-based).
        More robust than standard z-score for datasets with outliers.

        Args:
            weights: List of weight values

        Returns:
            Set of indices that are Z-score outliers
        """
        if len(weights) < 3:
            return set()

        weights_array = np.array(weights)
        median = np.median(weights_array)

        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(weights_array - median))

        if mad == 0:  # All values are identical
            return set()

        # Modified Z-scores
        modified_z_scores = 0.6745 * (weights_array - median) / mad

        outliers = set()
        for i, z_score in enumerate(modified_z_scores):
            if abs(z_score) > self.z_score_threshold:
                outliers.add(i)

        return outliers

    def _detect_temporal_outliers(self, sorted_measurements: List[Dict[str, Any]]) -> Set[int]:
        """
        Detect outliers based on temporal consistency (rate of change).
        Identifies measurements with impossible rate of change.

        Args:
            sorted_measurements: Measurements sorted by timestamp

        Returns:
            Set of indices that are temporal outliers
        """
        if len(sorted_measurements) < 2:
            return set()

        outliers = set()

        for i in range(1, len(sorted_measurements)):
            prev_measurement = sorted_measurements[i-1]
            curr_measurement = sorted_measurements[i]

            weight_diff = abs(curr_measurement['weight'] - prev_measurement['weight'])
            time_diff = curr_measurement['timestamp'] - prev_measurement['timestamp']

            # Skip if measurements are very close in time (< 1 hour)
            if time_diff.total_seconds() < 3600:
                continue

            # Calculate percentage change
            prev_weight = prev_measurement['weight']
            if prev_weight > 0:
                percent_change = weight_diff / prev_weight

                # Flag if change exceeds threshold
                if percent_change > self.temporal_max_change_percent:
                    outliers.add(i)

        return outliers

    def _detect_kalman_outliers(self, sorted_measurements: List[Dict[str, Any]], user_id: str) -> Set[int]:
        """
        Detect outliers based on deviation from Kalman filter predictions.

        Args:
            sorted_measurements: Measurements sorted by timestamp
            user_id: User identifier for accessing Kalman state

        Returns:
            Set of indices that deviate significantly from Kalman predictions
        """
        if not self.db:
            return set()

        outliers = set()

        # Get user's Kalman state
        user_state = self.db.get_state(user_id)
        if not user_state:
            # No user state available
            return set()

        last_state = user_state.get('last_state')
        if last_state is None:
            # No Kalman state available, can't do prediction-based detection
            return set()

        # Get state history if available for more accurate predictions
        state_history = user_state.get('state_history', [])

        for i, measurement in enumerate(sorted_measurements):
            weight = measurement['weight']
            timestamp = measurement['timestamp']

            # Find the closest state snapshot before this measurement
            predicted_weight = None

            if state_history:
                # Look for state snapshot just before this measurement
                for snapshot in reversed(state_history):
                    snapshot_ts = snapshot.get('timestamp')
                    if snapshot_ts is not None:
                        # Handle numpy arrays and datetime comparison properly
                        try:
                            if hasattr(snapshot_ts, 'item'):
                                snapshot_ts = snapshot_ts.item()
                            if hasattr(timestamp, 'item'):
                                timestamp_val = timestamp.item()
                            else:
                                timestamp_val = timestamp

                            if snapshot_ts < timestamp_val:
                                if snapshot.get('state'):
                                    # Use the weight component of the state (first element)
                                    predicted_weight = snapshot['state'][0] if isinstance(snapshot['state'], (list, np.ndarray)) else None
                                    break
                        except (AttributeError, TypeError):
                            # If comparison fails, skip this snapshot
                            continue

            # Fall back to last state if no snapshot found
            if predicted_weight is None:
                if isinstance(last_state, np.ndarray) and len(last_state) > 0:
                    predicted_weight = last_state[0]
                elif isinstance(last_state, list) and len(last_state) > 0:
                    predicted_weight = last_state[0]

            # Check deviation from prediction
            if predicted_weight is not None:
                # Handle numpy arrays
                try:
                    weight_val = float(predicted_weight)
                    if weight_val > 0:
                        deviation = abs(weight - weight_val) / weight_val
                        if deviation > self.kalman_deviation_threshold:
                            outliers.add(i)
                except (TypeError, ValueError):
                    # If can't convert to float, skip this measurement
                    continue

        return outliers

    def analyze_outliers(self, measurements: List[Dict[str, Any]], outlier_indices: Set[int]) -> Dict[str, Any]:
        """
        Analyze detected outliers and provide detailed information.

        Args:
            measurements: Original measurements list
            outlier_indices: Set of outlier indices

        Returns:
            Analysis report dictionary
        """
        if not outlier_indices or not measurements:
            return {
                'total_measurements': len(measurements),
                'outlier_count': 0,
                'outlier_percentage': 0.0,
                'outlier_details': []
            }

        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])
        weights = [m['weight'] for m in sorted_measurements]

        outlier_details = []
        for idx in sorted(outlier_indices):
            if idx < len(sorted_measurements):
                measurement = sorted_measurements[idx]
                weight = measurement['weight']

                detail = {
                    'index': idx,
                    'timestamp': measurement['timestamp'],
                    'weight': weight,
                    'deviation_from_median': weight - np.median(weights),
                    'source': measurement.get('source', 'unknown')
                }

                # Add context from neighboring measurements
                if idx > 0:
                    prev_weight = sorted_measurements[idx-1]['weight']
                    detail['change_from_previous'] = weight - prev_weight
                    detail['percent_change_from_previous'] = abs(weight - prev_weight) / prev_weight if prev_weight > 0 else 0

                outlier_details.append(detail)

        return {
            'total_measurements': len(measurements),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(measurements)) * 100,
            'outlier_details': outlier_details,
            'median_weight': np.median(weights),
            'weight_std': np.std(weights),
            'weight_range': (min(weights), max(weights))
        }

    def get_clean_measurements(self, measurements: List[Dict[str, Any]], user_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Set[int]]:
        """
        Get measurements with outliers removed.

        Args:
            measurements: Original measurements list
            user_id: Optional user identifier for Kalman-based outlier detection

        Returns:
            Tuple of (clean_measurements, outlier_indices)
        """
        outlier_indices = self.detect_outliers(measurements, user_id)

        if not outlier_indices:
            return measurements.copy(), set()

        # Sort by timestamp to maintain chronological order
        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])

        clean_measurements = []
        for i, measurement in enumerate(sorted_measurements):
            if i not in outlier_indices:
                clean_measurements.append(measurement.copy())

        return clean_measurements, outlier_indices

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update detector configuration.

        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)

        # Update thresholds
        self.iqr_multiplier = self.config.get('iqr_multiplier', self.iqr_multiplier)
        self.z_score_threshold = self.config.get('z_score_threshold', self.z_score_threshold)
        self.temporal_max_change_percent = self.config.get('temporal_max_change_percent', self.temporal_max_change_percent)
        self.min_measurements_for_analysis = self.config.get('min_measurements_for_analysis', self.min_measurements_for_analysis)

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Current configuration dictionary
        """
        return {
            'iqr_multiplier': self.iqr_multiplier,
            'z_score_threshold': self.z_score_threshold,
            'temporal_max_change_percent': self.temporal_max_change_percent,
            'min_measurements_for_analysis': self.min_measurements_for_analysis
        }