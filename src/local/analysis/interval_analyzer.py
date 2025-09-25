"""
Interval Analysis Module

Calculates weight measurements at regular intervals (e.g., 30-day)
from each user's signup date.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging


class IntervalAnalyzer:
    """Analyzes weight measurements at regular intervals from signup"""

    def __init__(self, database, interval_days: int = 30, window_days: int = 7):
        """
        Initialize interval analyzer

        Args:
            database: Database connection object
            interval_days: Days between each interval (default: 30)
            window_days: ±days tolerance for finding measurements (default: 7)
        """
        self.db = database
        self.interval_days = interval_days
        self.window_days = window_days
        self.logger = logging.getLogger(__name__)

    def calculate_all_users(self,
                          raw_data: pd.DataFrame,
                          filtered_data: pd.DataFrame) -> Dict:
        """
        Calculate interval weights for all users

        Args:
            raw_data: DataFrame with raw measurements
            filtered_data: DataFrame with filtered measurements

        Returns:
            Dictionary mapping user_id to interval data
        """
        results = {}

        # Get unique users
        all_users = set(raw_data['user_id'].unique())
        if not filtered_data.empty:
            all_users.update(filtered_data['user_id'].unique())

        self.logger.info(f"Analyzing {len(all_users)} users")

        for user_id in all_users:
            user_raw = raw_data[raw_data['user_id'] == user_id]
            user_filtered = filtered_data[filtered_data['user_id'] == user_id] if not filtered_data.empty else pd.DataFrame()

            # Find signup date (initial-questionnaire)
            signup_date = self._find_signup_date(user_raw)
            if signup_date is None:
                self.logger.debug(f"No signup date found for user {user_id}")
                continue

            # Calculate intervals
            intervals = self._calculate_user_intervals(
                user_id, signup_date, user_raw, user_filtered
            )

            if intervals:
                results[user_id] = {
                    'user_id': user_id,
                    'signup_date': signup_date,
                    'intervals': intervals,
                    'num_raw_measurements': len(user_raw),
                    'num_filtered_measurements': len(user_filtered) if not user_filtered.empty else 0
                }

        self.logger.info(f"Calculated intervals for {len(results)} users")
        return results

    def _find_signup_date(self, user_data: pd.DataFrame) -> Optional[datetime]:
        """Find user's signup date from initial-questionnaire source"""
        signup_rows = user_data[user_data['source'] == 'initial-questionnaire']

        if signup_rows.empty:
            # Fallback to first measurement
            if not user_data.empty:
                return user_data.iloc[0]['timestamp']
            return None

        return signup_rows.iloc[0]['timestamp']

    def _calculate_user_intervals(self,
                                 user_id: str,
                                 signup_date: datetime,
                                 raw_data: pd.DataFrame,
                                 filtered_data: pd.DataFrame) -> List[Dict]:
        """Calculate interval measurements for a single user"""
        intervals = []

        # Determine maximum interval based on available data
        max_date_raw = raw_data['timestamp'].max() if not raw_data.empty else signup_date
        max_date_filtered = filtered_data['timestamp'].max() if not filtered_data.empty else signup_date
        max_date = max(max_date_raw, max_date_filtered)

        # Calculate number of intervals to check
        days_elapsed = (max_date - signup_date).days
        num_intervals = min((days_elapsed // self.interval_days) + 1, 12)  # Cap at 12 intervals (1 year)

        # Get baseline weight (day 0)
        baseline_raw = self._select_weight_at_date(
            raw_data, signup_date, window_days=3  # Tighter window for baseline
        )
        baseline_filtered = self._select_weight_at_date(
            filtered_data, signup_date, window_days=3
        ) if not filtered_data.empty else None

        # Add baseline as interval 0
        intervals.append({
            'interval_num': 0,
            'interval_days': 0,
            'target_date': signup_date,
            'raw_weight': baseline_raw['weight'] if baseline_raw else None,
            'filtered_weight': baseline_filtered['weight'] if baseline_filtered else None,
            'raw_source': baseline_raw['source'] if baseline_raw else None,
            'filtered_source': baseline_filtered['source'] if baseline_filtered else None,
            'raw_distance_days': baseline_raw['distance_days'] if baseline_raw else None,
            'filtered_distance_days': baseline_filtered['distance_days'] if baseline_filtered else None
        })

        # Calculate subsequent intervals
        for i in range(1, num_intervals + 1):
            interval_date = signup_date + timedelta(days=i * self.interval_days)

            # Skip if interval date is in the future
            if interval_date > datetime.now():
                break

            # Select weights for this interval
            raw_weight = self._select_weight_at_date(raw_data, interval_date, self.window_days)
            filtered_weight = self._select_weight_at_date(
                filtered_data, interval_date, self.window_days
            ) if not filtered_data.empty else None

            intervals.append({
                'interval_num': i,
                'interval_days': i * self.interval_days,
                'target_date': interval_date,
                'raw_weight': raw_weight['weight'] if raw_weight else None,
                'filtered_weight': filtered_weight['weight'] if filtered_weight else None,
                'raw_source': raw_weight['source'] if raw_weight else None,
                'filtered_source': filtered_weight['source'] if filtered_weight else None,
                'raw_distance_days': raw_weight['distance_days'] if raw_weight else None,
                'filtered_distance_days': filtered_weight['distance_days'] if filtered_weight else None
            })

        return intervals

    def _select_weight_at_date(self,
                              data: pd.DataFrame,
                              target_date: datetime,
                              window_days: int) -> Optional[Dict]:
        """
        Select the best weight measurement near a target date

        Args:
            data: DataFrame with weight measurements
            target_date: Target date for measurement
            window_days: ±days tolerance

        Returns:
            Dictionary with weight info or None if no measurement found
        """
        if data.empty:
            return None

        # Filter to window
        window_start = target_date - timedelta(days=window_days)
        window_end = target_date + timedelta(days=window_days)

        in_window = data[
            (data['timestamp'] >= window_start) &
            (data['timestamp'] <= window_end)
        ].copy()

        if in_window.empty:
            return None

        # Calculate distance from target for each measurement
        in_window['distance_days'] = abs((in_window['timestamp'] - target_date).dt.days)

        # Sort by multiple criteria
        # 1. Distance from target (closest first)
        # 2. Quality score if available (highest first)
        # 3. Prefer certain sources
        sort_columns = ['distance_days']
        if 'quality_score' in in_window.columns:
            in_window['quality_rank'] = -in_window['quality_score']
            sort_columns.append('quality_rank')

        # Add source preference ranking
        source_priority = {
            'care-team-upload': 0,
            'patient-upload': 1,
            'questionnaire': 2,
            'patient-device': 3,
            'initial-questionnaire': 4,
            'connectivehealth.io': 5,
            'iglucose.com': 6
        }
        in_window['source_rank'] = in_window['source'].map(
            lambda x: source_priority.get(x, 99)
        )
        sort_columns.append('source_rank')

        in_window = in_window.sort_values(sort_columns)

        # Return best match
        best = in_window.iloc[0]
        return {
            'weight': best['weight'],
            'timestamp': best['timestamp'],
            'source': best['source'],
            'distance_days': best['distance_days'],
            'quality_score': best.get('quality_score', None)
        }

    def get_interval_summary(self, user_intervals: Dict) -> pd.DataFrame:
        """Generate summary statistics for intervals"""
        summary_data = []

        # Collect data for each interval number
        max_intervals = max(
            len(data['intervals'])
            for data in user_intervals.values()
        )

        for interval_num in range(max_intervals):
            raw_weights = []
            filtered_weights = []

            for user_data in user_intervals.values():
                if interval_num < len(user_data['intervals']):
                    interval = user_data['intervals'][interval_num]
                    if interval['raw_weight'] is not None:
                        raw_weights.append(interval['raw_weight'])
                    if interval['filtered_weight'] is not None:
                        filtered_weights.append(interval['filtered_weight'])

            if raw_weights or filtered_weights:
                summary_data.append({
                    'interval_num': interval_num,
                    'interval_days': interval_num * self.interval_days,
                    'raw_count': len(raw_weights),
                    'filtered_count': len(filtered_weights),
                    'raw_mean': pd.Series(raw_weights).mean() if raw_weights else None,
                    'filtered_mean': pd.Series(filtered_weights).mean() if filtered_weights else None,
                    'raw_std': pd.Series(raw_weights).std() if len(raw_weights) > 1 else None,
                    'filtered_std': pd.Series(filtered_weights).std() if len(filtered_weights) > 1 else None
                })

        return pd.DataFrame(summary_data)