"""
Weight Selector Module

Intelligent selection of weight measurements within time windows,
considering data quality, source reliability, and proximity to target dates.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class WeightSelector:
    """Selects optimal weight measurements within time windows"""

    # Source reliability rankings (lower = more reliable)
    SOURCE_PRIORITY = {
        'care-team-upload': 0,
        'patient-upload': 1,
        'questionnaire': 2,
        'patient-device': 3,
        'initial-questionnaire': 4,
        'connectivehealth.io': 5,
        'iglucose.com': 6
    }

    def __init__(self, window_days: int = 7):
        """
        Initialize weight selector

        Args:
            window_days: ±days tolerance for selecting measurements
        """
        self.window_days = window_days

    def select_interval_weight(self,
                              measurements: pd.DataFrame,
                              target_date: datetime,
                              prefer_filtered: bool = False) -> Optional[Dict]:
        """
        Select the most appropriate weight for a given interval

        Args:
            measurements: DataFrame with weight measurements
            target_date: Target date for the interval
            prefer_filtered: Whether to prefer filtered over raw weights

        Returns:
            Dictionary with selected weight info or None
        """
        if measurements.empty:
            return None

        # Filter to window
        window_start = target_date - timedelta(days=self.window_days)
        window_end = target_date + timedelta(days=self.window_days)

        in_window = measurements[
            (measurements['timestamp'] >= window_start) &
            (measurements['timestamp'] <= window_end)
        ].copy()

        if in_window.empty:
            return None

        # Calculate selection criteria
        in_window = self._calculate_selection_scores(in_window, target_date)

        # Sort by composite score
        in_window = in_window.sort_values('selection_score')

        # Get best measurement
        best = in_window.iloc[0]

        return self._create_weight_record(best, target_date, prefer_filtered)

    def _calculate_selection_scores(self,
                                   data: pd.DataFrame,
                                   target_date: datetime) -> pd.DataFrame:
        """Calculate composite selection scores for measurements"""
        # Distance from target (normalized to 0-1)
        data['distance_days'] = abs((data['timestamp'] - target_date).dt.days)
        data['distance_score'] = data['distance_days'] / self.window_days

        # Source reliability score (normalized to 0-1)
        data['source_score'] = data['source'].map(
            lambda x: self.SOURCE_PRIORITY.get(x, 10) / 10
        )

        # Quality score (if available)
        if 'quality_score' in data.columns:
            data['quality_rank'] = 1 - data['quality_score']
        else:
            data['quality_rank'] = 0.5  # Neutral if not available

        # Composite selection score (lower is better)
        data['selection_score'] = (
            data['distance_score'] * 0.5 +  # 50% weight on proximity
            data['source_score'] * 0.3 +     # 30% weight on source
            data['quality_rank'] * 0.2       # 20% weight on quality
        )

        return data

    def _create_weight_record(self,
                            measurement: pd.Series,
                            target_date: datetime,
                            prefer_filtered: bool) -> Dict:
        """Create a weight record from selected measurement"""
        # Determine which weight to use
        weight_value = measurement['weight']
        weight_type = 'raw'

        if prefer_filtered and 'filtered_weight' in measurement:
            if pd.notna(measurement['filtered_weight']):
                weight_value = measurement['filtered_weight']
                weight_type = 'filtered'

        record = {
            'weight': weight_value,
            'weight_type': weight_type,
            'timestamp': measurement['timestamp'],
            'source': measurement['source'],
            'distance_days': measurement['distance_days'],
            'selection_score': measurement['selection_score']
        }

        # Add optional fields if present
        optional_fields = ['quality_score', 'bmi', 'is_outlier']
        for field in optional_fields:
            if field in measurement and pd.notna(measurement[field]):
                record[field] = measurement[field]

        return record

    def select_baseline_weight(self,
                              measurements: pd.DataFrame,
                              sources: List[str] = None) -> Optional[Dict]:
        """
        Select baseline weight, preferring specific sources

        Args:
            measurements: DataFrame with weight measurements
            sources: Preferred sources for baseline (e.g., ['initial-questionnaire'])

        Returns:
            Dictionary with baseline weight info or None
        """
        if measurements.empty:
            return None

        # Filter by preferred sources if specified
        if sources:
            filtered = measurements[measurements['source'].isin(sources)]
            if not filtered.empty:
                measurements = filtered

        # Sort by timestamp (earliest first) and source priority
        measurements = measurements.copy()
        measurements['source_rank'] = measurements['source'].map(
            lambda x: self.SOURCE_PRIORITY.get(x, 99)
        )
        measurements = measurements.sort_values(['timestamp', 'source_rank'])

        # Return first (earliest, most reliable) measurement
        baseline = measurements.iloc[0]

        return {
            'weight': baseline['weight'],
            'timestamp': baseline['timestamp'],
            'source': baseline['source'],
            'quality_score': baseline.get('quality_score', None)
        }

    def calculate_weight_change(self,
                              initial_weight: Optional[float],
                              current_weight: Optional[float]) -> Dict:
        """
        Calculate weight change metrics

        Args:
            initial_weight: Starting weight
            current_weight: Current weight

        Returns:
            Dictionary with change metrics
        """
        if initial_weight is None or current_weight is None:
            return {
                'absolute_change': None,
                'percentage_change': None,
                'is_loss': None,
                'is_gain': None
            }

        absolute_change = current_weight - initial_weight
        percentage_change = (absolute_change / initial_weight) * 100

        return {
            'absolute_change': round(absolute_change, 2),
            'percentage_change': round(percentage_change, 2),
            'is_loss': absolute_change < 0,
            'is_gain': absolute_change > 0
        }

    def identify_data_gaps(self,
                         measurements: pd.DataFrame,
                         expected_frequency_days: int = 7) -> List[Dict]:
        """
        Identify gaps in measurement data

        Args:
            measurements: DataFrame with weight measurements
            expected_frequency_days: Expected days between measurements

        Returns:
            List of gap periods
        """
        if measurements.empty or len(measurements) < 2:
            return []

        # Sort by timestamp
        sorted_data = measurements.sort_values('timestamp')
        gaps = []

        for i in range(len(sorted_data) - 1):
            current = sorted_data.iloc[i]
            next_measurement = sorted_data.iloc[i + 1]

            days_between = (next_measurement['timestamp'] - current['timestamp']).days

            if days_between > expected_frequency_days * 2:
                gaps.append({
                    'start_date': current['timestamp'],
                    'end_date': next_measurement['timestamp'],
                    'gap_days': days_between,
                    'severity': self._classify_gap_severity(days_between)
                })

        return gaps

    def _classify_gap_severity(self, gap_days: int) -> str:
        """Classify the severity of a data gap"""
        if gap_days <= 14:
            return 'minor'
        elif gap_days <= 30:
            return 'moderate'
        elif gap_days <= 60:
            return 'significant'
        else:
            return 'severe'