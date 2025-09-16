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


class OutlierDetector:
    """
    Statistical outlier detection for batch measurement analysis.
    Designed to identify problematic measurements before Kalman processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize outlier detector with configuration.

        Args:
            config: Configuration dict with thresholds and method settings
        """
        self.config = config or {}

        # Default thresholds (can be overridden by config)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        self.z_score_threshold = self.config.get('z_score_threshold', 3.0)
        self.temporal_max_change_percent = self.config.get('temporal_max_change_percent', 0.30)
        self.min_measurements_for_analysis = self.config.get('min_measurements_for_analysis', 5)

    def detect_outliers(self, measurements: List[Dict[str, Any]]) -> Set[int]:
        """
        Detect outliers in a batch of measurements using multiple methods.

        Args:
            measurements: List of measurement dictionaries with 'weight', 'timestamp'

        Returns:
            Set of indices that are considered outliers
        """
        if len(measurements) < self.min_measurements_for_analysis:
            return set()

        # Extract weights and sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda x: x['timestamp'])
        weights = [m['weight'] for m in sorted_measurements]

        outliers = set()

        # Method 1: IQR-based detection
        iqr_outliers = self._detect_iqr_outliers(weights)
        outliers.update(iqr_outliers)

        # Method 2: Modified Z-score detection
        zscore_outliers = self._detect_zscore_outliers(weights)
        outliers.update(zscore_outliers)

        # Method 3: Temporal consistency check
        temporal_outliers = self._detect_temporal_outliers(sorted_measurements)
        outliers.update(temporal_outliers)

        return outliers

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

    def get_clean_measurements(self, measurements: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Set[int]]:
        """
        Get measurements with outliers removed.

        Args:
            measurements: Original measurements list

        Returns:
            Tuple of (clean_measurements, outlier_indices)
        """
        outlier_indices = self.detect_outliers(measurements)

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